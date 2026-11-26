// x86-32 -> C++ lifter.
//
// Each guest function becomes one C++ function. Block starts become labels,
// branches become gotos, every instruction is wrapped in its own { } scope so
// gotos never jump over initialisations. Arithmetic flags are only computed
// when a later instruction can observe them (per-function liveness).
#include "lifter.h"

#include <cstdio>
#include <deque>
#include <initializer_list>
#include <set>
#include <string_view>
#include <vector>

namespace vc3 {

std::set<uint32_t> g_hook_entries;  // functions that call vc3::guest_hook(c, entry) first

std::string fn_name(uint32_t va) {
    char b[32];
    std::snprintf(b, sizeof b, "fn_%08x", va);
    return b;
}

namespace {

// EFLAGS bits (identical to ZYDIS_CPUFLAG_*).
constexpr uint32_t CF = 1u << 0, PF = 1u << 2, AF = 1u << 4, ZF = 1u << 6, SF = 1u << 7, OF = 1u << 11;
constexpr uint32_t ALL = CF | PF | AF | ZF | SF | OF;

// "$0 + $1" style substitution; avoids escaping braces in generated C++.
std::string F(std::string_view tpl, std::initializer_list<std::string> args) {
    std::vector<std::string> a(args);
    std::string out;
    for (size_t i = 0; i < tpl.size(); ++i) {
        if (tpl[i] == '$' && i + 1 < tpl.size() && tpl[i + 1] >= '0' && tpl[i + 1] <= '9') {
            out += a.at(size_t(tpl[i + 1] - '0'));
            ++i;
        } else {
            out += tpl[i];
        }
    }
    return out;
}

std::string hex(uint64_t v) {
    char b[32];
    std::snprintf(b, sizeof b, "0x%llxu", static_cast<unsigned long long>(v));
    return b;
}
std::string hex8(uint32_t v) {
    char b[16];
    std::snprintf(b, sizeof b, "%08x", v);
    return b;
}
std::string label(uint32_t va) { return "L_" + hex8(va); }
std::string ut(unsigned bits) { return bits == 8 ? "u8" : bits == 16 ? "u16" : bits == 64 ? "u64" : "u32"; }
std::string st(unsigned bits) { return bits == 8 ? "s8" : bits == 16 ? "s16" : bits == 64 ? "s64" : "s32"; }
uint64_t mask(unsigned bits) { return bits >= 64 ? ~0ull : (1ull << bits) - 1; }

std::string cstr_escape(const std::string& s) {
    std::string o;
    for (char ch : s) {
        if (ch == '"' || ch == '\\') o += '\\';
        o += ch;
    }
    return o;
}

bool is_shift_or_rotate(ZydisMnemonic m) {
    switch (m) {
        case ZYDIS_MNEMONIC_SHL: case ZYDIS_MNEMONIC_SHR: case ZYDIS_MNEMONIC_SAR:
        case ZYDIS_MNEMONIC_ROL: case ZYDIS_MNEMONIC_ROR: case ZYDIS_MNEMONIC_RCL: case ZYDIS_MNEMONIC_RCR:
        case ZYDIS_MNEMONIC_SHLD: case ZYDIS_MNEMONIC_SHRD:
            return true;
        default:
            return false;
    }
}

// Condition expression for jcc / setcc / cmovcc / loop variants, by mnemonic suffix.
std::string cond_expr(std::string_view suffix) {
    if (suffix == "o") return "c.of";
    if (suffix == "no") return "!c.of";
    if (suffix == "b") return "c.cf";
    if (suffix == "nb") return "!c.cf";
    if (suffix == "z") return "c.zf";
    if (suffix == "nz") return "!c.zf";
    if (suffix == "be") return "(c.cf || c.zf)";
    if (suffix == "nbe") return "(!c.cf && !c.zf)";
    if (suffix == "s") return "c.sf";
    if (suffix == "ns") return "!c.sf";
    if (suffix == "p") return "c.pf";
    if (suffix == "np") return "!c.pf";
    if (suffix == "l") return "(c.sf != c.of)";
    if (suffix == "nl") return "(c.sf == c.of)";
    if (suffix == "le") return "(c.zf || c.sf != c.of)";
    if (suffix == "nle") return "(!c.zf && c.sf == c.of)";
    return "";
}

class FunctionLifter {
public:
    FunctionLifter(const Analyzer& an, const Function& f, LiftStats& stats) : an_(an), f_(f), stats_(stats) {}

    std::string run() {
        compute_liveness();
        std::vector<std::pair<uint32_t, std::string>> bodies;
        for (const auto& [va, in] : f_.insns) {
            in_ = &in;
            next_ = va + in.insn.length;
            live_ = live_out_[va];
            out_.clear();
            ++stats_.total;
            const bool ok = lift();
            if (ok) {
                ++stats_.lifted;
            } else {
                ++stats_.unlifted[ZydisMnemonicGetString(in.insn.mnemonic)];
                out_ = F("vc3::unimplemented_insn(c, $0, \"$1\");", {hex(va), cstr_escape(disasm(in))});
                falls_ = false;
            }
            if (falls_) emit_fallthrough(va);
            bodies.emplace_back(va, out_);
        }

        std::string s = "void " + fn_name(f_.entry) + "(vc3::Cpu& c) {\n";
        if (g_hook_entries.count(f_.entry)) s += "    vc3::guest_hook(c, " + hex(f_.entry) + ");\n";
        if (f_.insns.empty() || f_.insns.begin()->first != f_.entry) {
            labels_.insert(f_.entry);
            s += "    goto " + label(f_.entry) + ";\n";
        }
        for (const auto& [va, body] : bodies) {
            if (labels_.count(va)) s += label(va) + ":\n";
            s += "    // " + hex8(va) + "  " + disasm(f_.insns.at(va)) + "\n";
            s += "    { " + body + " }\n";
        }
        s += "}\n";
        return s;
    }

private:
    // ---------------------------------------------------------------- flow
    enum class Flow { Fall, Cond, Jump, Table, Indirect, Exit };

    Flow flow(const Instruction& in) const {
        const auto cat = in.insn.meta.category;
        const auto m = in.insn.mnemonic;
        if (cat == ZYDIS_CATEGORY_COND_BR) return Flow::Cond;
        if (cat == ZYDIS_CATEGORY_UNCOND_BR) {
            if (in.ops[0].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) return Flow::Jump;
            return an_.read_jump_table(in).empty() ? Flow::Indirect : Flow::Table;
        }
        if (cat == ZYDIS_CATEGORY_RET || m == ZYDIS_MNEMONIC_INT3 || m == ZYDIS_MNEMONIC_HLT || m == ZYDIS_MNEMONIC_UD2)
            return Flow::Exit;
        return Flow::Fall;
    }

    static bool is_int2d(const Instruction& in) {
        return in.insn.mnemonic == ZYDIS_MNEMONIC_INT && in.ops[0].type == ZYDIS_OPERAND_TYPE_IMMEDIATE &&
               in.ops[0].imm.value.u == 0x2D;
    }

    static uint32_t rel_target(const Instruction& in) {
        ZyanU64 abs = 0;
        ZydisCalcAbsoluteAddress(&in.insn, &in.ops[0], in.addr, &abs);
        return uint32_t(abs);
    }

    // Intra-function successors (targets outside the function are exits).
    std::vector<uint32_t> successors(const Instruction& in) const {
        std::vector<uint32_t> s;
        const uint32_t next = in.addr + in.insn.length;
        auto add = [&](uint32_t t) { if (f_.insns.count(t)) s.push_back(t); };
        switch (flow(in)) {
            case Flow::Fall: add(is_int2d(in) ? next + 1 : next); break;
            case Flow::Cond: add(rel_target(in)); add(next); break;
            case Flow::Jump: if (!f_.tail_calls.count(rel_target(in))) add(rel_target(in)); break;
            case Flow::Table: for (uint32_t t : an_.read_jump_table(in)) add(t); break;
            default: break;
        }
        return s;
    }

    void compute_liveness() {
        std::map<uint32_t, uint32_t> live_in;
        bool changed = true;
        while (changed) {
            changed = false;
            for (auto it = f_.insns.rbegin(); it != f_.insns.rend(); ++it) {
                const auto& in = it->second;
                uint32_t out = 0;
                for (uint32_t s : successors(in)) out |= live_in[s];
                uint32_t use = 0, def = 0;
                if (in.insn.meta.category == ZYDIS_CATEGORY_CALL) {
                    def = ALL;  // callee clobbers flags
                } else if (in.insn.cpu_flags) {
                    const auto* fl = in.insn.cpu_flags;
                    use = fl->tested & ALL;
                    def = (fl->modified | fl->set_0 | fl->set_1 | fl->undefined) & ALL;
                    // Shifts by a variable (or zero) count leave flags alone when count == 0.
                    if (is_shift_or_rotate(in.insn.mnemonic)) {
                        const auto& cnt = in.ops[in.insn.operand_count_visible - 1];
                        if (cnt.type != ZYDIS_OPERAND_TYPE_IMMEDIATE || (cnt.imm.value.u & 31) == 0) {
                            use |= def;
                            def = 0;
                        }
                    }
                }
                const uint32_t in_live = use | (out & ~def);
                live_out_[in.addr] = out;
                if (live_in[in.addr] != in_live) {
                    live_in[in.addr] = in_live;
                    changed = true;
                }
            }
        }
    }

    void emit_fallthrough(uint32_t va) {
        auto it = f_.insns.find(va);
        auto nx = std::next(it);
        if (nx != f_.insns.end() && nx->first == next_) return;  // natural fallthrough
        out_ += " " + jump_to(next_);
    }

    std::string jump_to(uint32_t t) {
        if (f_.insns.count(t) && !f_.tail_calls.count(t)) {
            labels_.insert(t);
            // Backward jumps are loops: give other guest threads a chance to run.
            return std::string(t <= in_->addr ? "VC3_BACKEDGE(c); " : "") + "goto " + label(t) + ";";
        }
        return tail_call(t);
    }
    std::string tail_call(uint32_t t) const {
        if (an_.is_function(t)) return fn_name(t) + "(c); return;";
        return F("vc3::jump_indirect(c, $0); return;", {hex(t)});
    }

    // ------------------------------------------------------------ operands
    const ZydisDecodedOperand& op(int i) const { return in_->ops[i]; }
    int nops() const { return in_->insn.operand_count_visible; }
    unsigned bits(int i) const { return op(i).size; }

    static bool is_gpr(ZydisRegister r) {
        const auto cls = ZydisRegisterGetClass(r);
        return cls == ZYDIS_REGCLASS_GPR8 || cls == ZYDIS_REGCLASS_GPR16 || cls == ZYDIS_REGCLASS_GPR32;
    }
    static std::string gpr32(ZydisRegister r) {
        static const char* names[] = {"eax", "ecx", "edx", "ebx", "esp", "ebp", "esi", "edi"};
        const ZydisRegister big = ZydisRegisterGetLargestEnclosing(ZYDIS_MACHINE_MODE_LEGACY_32, r);
        return std::string("c.") + names[big - ZYDIS_REGISTER_EAX];
    }
    static bool is_high8(ZydisRegister r) {
        return r == ZYDIS_REGISTER_AH || r == ZYDIS_REGISTER_CH || r == ZYDIS_REGISTER_DH || r == ZYDIS_REGISTER_BH;
    }
    static std::string reg_read(ZydisRegister r) {
        const auto w = ZydisRegisterGetWidth(ZYDIS_MACHINE_MODE_LEGACY_32, r);
        const std::string g = gpr32(r);
        if (w == 32) return g;
        if (w == 16) return "u16(" + g + ")";
        return is_high8(r) ? "u8(" + g + " >> 8)" : "u8(" + g + ")";
    }
    static std::string reg_write(ZydisRegister r, const std::string& v) {
        const auto w = ZydisRegisterGetWidth(ZYDIS_MACHINE_MODE_LEGACY_32, r);
        const std::string g = gpr32(r);
        if (w == 32) return g + " = u32(" + v + ");";
        if (w == 16) return "vc3::set16(" + g + ", u16(" + v + "));";
        return std::string(is_high8(r) ? "vc3::set8h(" : "vc3::set8l(") + g + ", u8(" + v + "));";
    }

    bool addr_expr(const ZydisDecodedOperand& o, std::string& out) const {
        std::string e;
        if (o.mem.base != ZYDIS_REGISTER_NONE) {
            if (!is_gpr(o.mem.base) || ZydisRegisterGetWidth(ZYDIS_MACHINE_MODE_LEGACY_32, o.mem.base) != 32) return false;
            e = gpr32(o.mem.base);
        }
        if (o.mem.index != ZYDIS_REGISTER_NONE) {
            if (!is_gpr(o.mem.index)) return false;
            if (!e.empty()) e += " + ";
            e += gpr32(o.mem.index);
            if (o.mem.scale > 1) e += " * " + std::to_string(o.mem.scale);
        }
        if (o.mem.disp.has_displacement && o.mem.disp.value != 0) {
            const int64_t d = o.mem.disp.value;
            if (e.empty()) e = hex(uint32_t(d));
            else if (d < 0) e += " - " + hex(uint32_t(-d));
            else e += " + " + hex(uint32_t(d));
        }
        if (e.empty()) e = "0u";
        switch (o.mem.segment) {
            case ZYDIS_REGISTER_FS: e = "c.fs_base + " + e; break;
            case ZYDIS_REGISTER_DS: case ZYDIS_REGISTER_SS: case ZYDIS_REGISTER_ES: case ZYDIS_REGISTER_CS:
            case ZYDIS_REGISTER_NONE: break;
            default: return false;
        }
        out = "u32(" + e + ")";
        return true;
    }

    // Read operand i as an unsigned value of `as_bits` (defaults to its own size).
    bool rd(int i, std::string& out, unsigned as_bits = 0) const {
        const auto& o = op(i);
        if (!as_bits) as_bits = o.size;
        switch (o.type) {
            case ZYDIS_OPERAND_TYPE_REGISTER:
                if (!is_gpr(o.reg.value)) return false;
                out = reg_read(o.reg.value);
                return true;
            case ZYDIS_OPERAND_TYPE_MEMORY:
                out = "vc3::rd<" + ut(o.size) + ">(ea)";
                return true;
            case ZYDIS_OPERAND_TYPE_IMMEDIATE:
                out = ut(as_bits) + "(" + hex(o.imm.value.u & mask(as_bits)) + ")";
                return true;
            default:
                return false;
        }
    }
    bool wrt(int i, const std::string& v, std::string& out) const {
        const auto& o = op(i);
        if (o.type == ZYDIS_OPERAND_TYPE_REGISTER) {
            if (!is_gpr(o.reg.value)) return false;
            out = reg_write(o.reg.value, v);
            return true;
        }
        if (o.type == ZYDIS_OPERAND_TYPE_MEMORY) {
            out = "vc3::wr<" + ut(o.size) + ">(ea, " + ut(o.size) + "(" + v + "));";
            return true;
        }
        return false;
    }

    // Declares `ea` for the first explicit memory operand.
    bool prelude() {
        for (int i = 0; i < nops(); ++i) {
            if (op(i).type == ZYDIS_OPERAND_TYPE_MEMORY && op(i).mem.type == ZYDIS_MEMOP_TYPE_MEM) {
                std::string a;
                if (!addr_expr(op(i), a)) return false;
                out_ += "const u32 ea = " + a + "; ";
                return true;
            }
        }
        return true;
    }

    bool need(uint32_t fl) const { return (live_ & fl) != 0; }

    // Flags for r = a OP b; variables a, b, r of type T are in scope (cin for adc/sbb).
    std::string arith_flags(const char* kind, unsigned b) const {
        const std::string T = ut(b), sh = std::to_string(b - 1);
        const std::string k = kind;
        std::string s;
        if (need(CF)) {
            if (k == "add") s += "c.cf = r < a; ";
            else if (k == "adc") s += "c.cf = cin ? (r <= a) : (r < a); ";
            else if (k == "sub") s += "c.cf = a < b; ";
            else if (k == "sbb") s += "c.cf = cin ? (a <= b) : (a < b); ";
            else if (k == "logic") s += "c.cf = false; ";
            else if (k == "neg") s += "c.cf = a != 0; ";
        }
        if (need(OF)) {
            if (k == "add" || k == "adc") s += "c.of = ((~(a ^ b) & (a ^ r)) >> " + sh + ") & 1; ";
            else if (k == "sub" || k == "sbb") s += "c.of = (((a ^ b) & (a ^ r)) >> " + sh + ") & 1; ";
            else if (k == "logic") s += "c.of = false; ";
            else if (k == "inc") s += "c.of = r == " + T + "(" + hex(1ull << (b - 1)) + "); ";
            else if (k == "dec") s += "c.of = a == " + T + "(" + hex(1ull << (b - 1)) + "); ";
            else if (k == "neg") s += "c.of = a == " + T + "(" + hex(1ull << (b - 1)) + "); ";
        }
        if (need(AF) && k != "logic") {
            if (k == "inc" || k == "dec") s += "c.af = ((a ^ r) >> 4) & 1; ";
            else if (k == "neg") s += "c.af = (a & 0xF) != 0; ";
            else s += "c.af = ((a ^ b ^ r) >> 4) & 1; ";
        }
        if (need(AF) && k == "logic") s += "c.af = false; ";
        s += szp();
        return s;
    }
    std::string szp(const char* var = "r") const {
        std::string s, v = var;
        if (need(ZF)) s += "c.zf = " + v + " == 0; ";
        if (need(SF)) s += "c.sf = vc3::msb(" + v + "); ";
        if (need(PF)) s += "c.pf = vc3::parity(" + v + "); ";
        return s;
    }

    // ------------------------------------------------------------ lifting
    bool lift() {
        falls_ = true;
        const auto m = in_->insn.mnemonic;
        const std::string name = ZydisMnemonicGetString(m);
        if (in_->insn.meta.category == ZYDIS_CATEGORY_STRINGOP) return lift_string();
        if (in_->insn.meta.isa_ext == ZYDIS_ISA_EXT_X87) return lift_x87() == 1;
        if (const auto r = lift_simd(); r >= 0) return r;

        switch (in_->insn.meta.category) {
            case ZYDIS_CATEGORY_COND_BR: return lift_jcc(name);
            case ZYDIS_CATEGORY_UNCOND_BR: return lift_jmp();
            case ZYDIS_CATEGORY_CALL: return lift_call();
            case ZYDIS_CATEGORY_RET: return lift_ret();
            default: break;
        }
        if (name.rfind("set", 0) == 0 && !cond_expr(name.substr(3)).empty()) {
            if (!prelude()) return false;
            std::string w;
            if (!wrt(0, cond_expr(name.substr(3)), w)) return false;
            out_ += w;
            return true;
        }
        if (name.rfind("cmov", 0) == 0 && !cond_expr(name.substr(4)).empty()) {
            std::string v, w;
            if (!prelude() || !rd(1, v) || !wrt(0, v, w)) return false;
            out_ += "if (" + cond_expr(name.substr(4)) + ") " + w;
            return true;
        }

        switch (m) {
            case ZYDIS_MNEMONIC_NOP: case ZYDIS_MNEMONIC_FWAIT: case ZYDIS_MNEMONIC_PREFETCHT0:
            case ZYDIS_MNEMONIC_PREFETCHT1: case ZYDIS_MNEMONIC_PREFETCHT2: case ZYDIS_MNEMONIC_PREFETCHNTA:
            case ZYDIS_MNEMONIC_SFENCE: case ZYDIS_MNEMONIC_PAUSE: case ZYDIS_MNEMONIC_EMMS:
            // Interrupt masking / cache control: interrupts are delivered only while the
            // interrupted thread is outside guest code (guest lock), so these are no-ops.
            case ZYDIS_MNEMONIC_CLI: case ZYDIS_MNEMONIC_STI: case ZYDIS_MNEMONIC_WBINVD: case ZYDIS_MNEMONIC_INVD:
                return true;
            case ZYDIS_MNEMONIC_MOV: return lift_mov();
            case ZYDIS_MNEMONIC_MOVZX: case ZYDIS_MNEMONIC_MOVSX: return lift_movx(m == ZYDIS_MNEMONIC_MOVSX);
            case ZYDIS_MNEMONIC_LEA: return lift_lea();
            case ZYDIS_MNEMONIC_PUSH: return lift_push();
            case ZYDIS_MNEMONIC_POP: return lift_pop();
            case ZYDIS_MNEMONIC_ADD: return lift_alu("add", "a + b", true);
            case ZYDIS_MNEMONIC_SUB: return lift_alu("sub", "a - b", true);
            case ZYDIS_MNEMONIC_CMP: return lift_alu("sub", "a - b", false);
            case ZYDIS_MNEMONIC_AND: return lift_alu("logic", "a & b", true);
            case ZYDIS_MNEMONIC_OR: return lift_alu("logic", "a | b", true);
            case ZYDIS_MNEMONIC_XOR: return lift_alu("logic", "a ^ b", true);
            case ZYDIS_MNEMONIC_TEST: return lift_alu("logic", "a & b", false);
            case ZYDIS_MNEMONIC_ADC: return lift_alu("adc", "a + b + cin", true);
            case ZYDIS_MNEMONIC_SBB: return lift_alu("sbb", "a - b - cin", true);
            case ZYDIS_MNEMONIC_INC: return lift_unary("inc", "a + 1");
            case ZYDIS_MNEMONIC_DEC: return lift_unary("dec", "a - 1");
            case ZYDIS_MNEMONIC_NEG: return lift_unary("neg", "0 - a");
            case ZYDIS_MNEMONIC_NOT: return lift_unary("not", "~a");
            case ZYDIS_MNEMONIC_SHL: case ZYDIS_MNEMONIC_SHR: case ZYDIS_MNEMONIC_SAR:
            case ZYDIS_MNEMONIC_ROL: case ZYDIS_MNEMONIC_ROR: case ZYDIS_MNEMONIC_RCL: case ZYDIS_MNEMONIC_RCR:
                return lift_shift(m);
            case ZYDIS_MNEMONIC_SHLD: case ZYDIS_MNEMONIC_SHRD: return lift_shxd(m == ZYDIS_MNEMONIC_SHLD);
            case ZYDIS_MNEMONIC_IMUL: return lift_imul();
            case ZYDIS_MNEMONIC_MUL: return lift_mul();
            case ZYDIS_MNEMONIC_DIV: case ZYDIS_MNEMONIC_IDIV: return lift_div(m == ZYDIS_MNEMONIC_IDIV);
            case ZYDIS_MNEMONIC_CDQ: out_ += "c.edx = u32(s32(c.eax) >> 31);"; return true;
            case ZYDIS_MNEMONIC_CWDE: out_ += "c.eax = u32(s32(s16(c.eax)));"; return true;
            case ZYDIS_MNEMONIC_CBW: out_ += "vc3::set16(c.eax, u16(s16(s8(c.eax))));"; return true;
            case ZYDIS_MNEMONIC_CWD: out_ += "vc3::set16(c.edx, u16(s16(c.eax) >> 15));"; return true;
            case ZYDIS_MNEMONIC_XCHG: return lift_xchg();
            case ZYDIS_MNEMONIC_XADD: return lift_xadd();
            case ZYDIS_MNEMONIC_CMPXCHG: return lift_cmpxchg();
            case ZYDIS_MNEMONIC_BSWAP: {
                std::string v, w;
                if (!rd(0, v) || !wrt(0, "vc3::bswap32(" + v + ")", w)) return false;
                out_ += w;
                return true;
            }
            case ZYDIS_MNEMONIC_BSF: case ZYDIS_MNEMONIC_BSR: return lift_bitscan(m == ZYDIS_MNEMONIC_BSR);
            case ZYDIS_MNEMONIC_BT: case ZYDIS_MNEMONIC_BTS: case ZYDIS_MNEMONIC_BTR: case ZYDIS_MNEMONIC_BTC:
                return lift_bt(m);
            case ZYDIS_MNEMONIC_LEAVE: out_ += "c.esp = c.ebp; c.ebp = vc3::pop32(c);"; return true;
            case ZYDIS_MNEMONIC_CLD: out_ += "c.df = false;"; return true;
            case ZYDIS_MNEMONIC_STD: out_ += "c.df = true;"; return true;
            case ZYDIS_MNEMONIC_CLC: out_ += "c.cf = false;"; return true;
            case ZYDIS_MNEMONIC_STC: out_ += "c.cf = true;"; return true;
            case ZYDIS_MNEMONIC_CMC: out_ += "c.cf = !c.cf;"; return true;
            case ZYDIS_MNEMONIC_SAHF: out_ += "vc3::sahf(c);"; return true;
            case ZYDIS_MNEMONIC_LAHF: out_ += "vc3::lahf(c);"; return true;
            case ZYDIS_MNEMONIC_PUSHFD: out_ += "vc3::push32(c, vc3::pack_eflags(c));"; return true;
            case ZYDIS_MNEMONIC_POPFD: out_ += "vc3::unpack_eflags(c, vc3::pop32(c));"; return true;
            case ZYDIS_MNEMONIC_PUSHAD: out_ += "vc3::pushad(c);"; return true;
            case ZYDIS_MNEMONIC_POPAD: out_ += "vc3::popad(c);"; return true;
            case ZYDIS_MNEMONIC_IN: case ZYDIS_MNEMONIC_OUT: {  // port I/O: in al|ax|eax, dx|imm8 / out dx|imm8, al|ax|eax
                const bool in = m == ZYDIS_MNEMONIC_IN;
                const auto& port_op = op(in ? 1 : 0);
                const auto& data_op = op(in ? 0 : 1);
                if (data_op.type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
                const std::string port = port_op.type == ZYDIS_OPERAND_TYPE_IMMEDIATE ? hex(port_op.imm.value.u & 0xFF) : "u16(c.edx)";
                const std::string size = std::to_string(data_op.size / 8);
                if (in) out_ += reg_write(data_op.reg.value, "vc3::port_in(c, " + port + ", " + size + ")");
                else out_ += "vc3::port_out(c, " + port + ", " + reg_read(data_op.reg.value) + ", " + size + ");";
                return true;
            }
            case ZYDIS_MNEMONIC_INT:
                if (!is_int2d(*in_)) return false;
                // Debug service (eax = 1 print, ecx = ANSI_STRING*); execution resumes after the trailing int3.
                out_ += "vc3::debug_service(c); " + jump_to(next_ + 1);
                falls_ = false;
                return true;
            case ZYDIS_MNEMONIC_RDTSC: out_ += "vc3::rdtsc(c);"; return true;
            case ZYDIS_MNEMONIC_CPUID: out_ += "vc3::cpuid(c);"; return true;
            case ZYDIS_MNEMONIC_INT3:
                out_ += F("vc3::guest_trap(c, $0, \"int3\");", {hex(in_->addr)});
                falls_ = false;
                return true;
            case ZYDIS_MNEMONIC_HLT: case ZYDIS_MNEMONIC_UD2:
                out_ += F("vc3::guest_trap(c, $0, \"$1\");", {hex(in_->addr), name});
                falls_ = false;
                return true;
            default:
                return false;
        }
    }

    bool lift_mov() {
        std::string v, w;
        if (!prelude() || !rd(1, v, bits(0)) || !wrt(0, v, w)) return false;
        out_ += w;
        return true;
    }
    bool lift_movx(bool sign) {
        std::string v, w;
        if (!prelude() || !rd(1, v)) return false;
        if (sign) v = st(bits(0)) + "(" + st(bits(1)) + "(" + v + "))";
        if (!wrt(0, v, w)) return false;
        out_ += w;
        return true;
    }
    bool lift_lea() {
        std::string a, w;
        if (!addr_expr(op(1), a)) return false;
        if (op(1).mem.segment == ZYDIS_REGISTER_FS) return false;
        if (!wrt(0, a, w)) return false;
        out_ += w;
        return true;
    }
    bool lift_push() {
        if (bits(0) != 32 && op(0).type != ZYDIS_OPERAND_TYPE_IMMEDIATE) return false;
        std::string v;
        if (!prelude() || !rd(0, v, 32)) return false;
        if (op(0).type == ZYDIS_OPERAND_TYPE_IMMEDIATE) v = hex(uint32_t(op(0).imm.value.s));
        out_ += "const u32 v = " + v + "; vc3::push32(c, v);";
        return true;
    }
    bool lift_pop() {
        if (bits(0) != 32) return false;
        out_ += "const u32 v = vc3::pop32(c); ";
        if (!prelude()) return false;  // pop [mem] computes the address after esp += 4
        std::string w;
        if (!wrt(0, "v", w)) return false;
        out_ += w;
        return true;
    }

    bool lift_alu(const char* kind, const char* expr, bool writeback) {
        const unsigned b = bits(0);
        const std::string T = ut(b);
        std::string a, bv, w;
        if (!prelude() || !rd(0, a) || !rd(1, bv, b)) return false;
        const std::string k = kind;
        if (k == "adc" || k == "sbb") out_ += "const " + T + " cin = c.cf; ";
        out_ += "const " + T + " a = " + a + ", b = " + bv + "; const " + T + " r = " + T + "(" + expr + "); ";
        out_ += arith_flags(kind, b);
        if (writeback) {
            if (!wrt(0, "r", w)) return false;
            out_ += w;
        }
        return true;
    }
    bool lift_unary(const char* kind, const char* expr) {
        const unsigned b = bits(0);
        const std::string T = ut(b);
        std::string a, w;
        if (!prelude() || !rd(0, a) || !wrt(0, "r", w)) return false;
        out_ += "const " + T + " a = " + a + "; const " + T + " r = " + T + "(" + expr + "); ";
        if (std::string(kind) != "not") out_ += arith_flags(kind, b);
        out_ += w;
        return true;
    }

    bool lift_shift(ZydisMnemonic m) {
        const unsigned b = bits(0);
        const std::string T = ut(b), S = st(b), B = std::to_string(b);
        std::string a, n, w;
        if (!prelude() || !rd(0, a) || !rd(1, n, 8) || !wrt(0, "r", w)) return false;
        out_ += "const " + T + " a = " + a + "; const unsigned n = unsigned(" + n + ") & 31; if (n) { ";
        std::string cf, of;  // expressions using a, r, n
        switch (m) {
            case ZYDIS_MNEMONIC_SHL:
                out_ += "const u64 wide = u64(a) << n; const " + T + " r = " + T + "(wide); ";
                cf = "(wide >> " + B + ") & 1"; of = "vc3::msb(r) != c.cf";
                break;
            case ZYDIS_MNEMONIC_SHR:
                out_ += "const " + T + " r = " + T + "(u32(a) >> n); ";
                cf = "(u32(a) >> (n - 1)) & 1"; of = "vc3::msb(a)";
                break;
            case ZYDIS_MNEMONIC_SAR:
                out_ += "const s32 sa = s32(" + S + "(a)); const " + T + " r = " + T + "(sa >> n); ";
                cf = "(sa >> (n - 1)) & 1"; of = "false";
                break;
            case ZYDIS_MNEMONIC_ROL:
                out_ += "const " + T + " r = std::rotl(a, int(n % " + B + ")); ";
                cf = "r & 1"; of = "vc3::msb(r) != c.cf";
                break;
            case ZYDIS_MNEMONIC_ROR:
                out_ += "const " + T + " r = std::rotr(a, int(n % " + B + ")); ";
                cf = "vc3::msb(r)"; of = "vc3::msb(r) != ((r >> " + std::to_string(b - 2) + ") & 1)";
                break;
            case ZYDIS_MNEMONIC_RCL: case ZYDIS_MNEMONIC_RCR: {
                const bool left = m == ZYDIS_MNEMONIC_RCL;
                out_ += T + " r = a; bool cf = c.cf; for (unsigned i = 0; i < n % " + std::to_string(b + 1) +
                        "; ++i) { ";
                out_ += left ? "const bool o = vc3::msb(r); r = " + T + "((r << 1) | cf); cf = o; } "
                             : "const bool o = r & 1; r = " + T + "((r >> 1) | (" + T + "(cf) << " +
                                   std::to_string(b - 1) + ")); cf = o; } ";
                cf = "cf";
                of = left ? "vc3::msb(r) != cf" : "vc3::msb(r) != ((r >> " + std::to_string(b - 2) + ") & 1)";
                break;
            }
            default: return false;
        }
        // CF first: OF for shl/rol is defined in terms of the new CF.
        const bool want_cf = need(CF) || (need(OF) && of.find("c.cf") != std::string::npos);
        if (want_cf) out_ += "c.cf = " + cf + "; ";
        if (need(OF)) out_ += "c.of = " + of + "; ";
        if (m != ZYDIS_MNEMONIC_ROL && m != ZYDIS_MNEMONIC_ROR && m != ZYDIS_MNEMONIC_RCL && m != ZYDIS_MNEMONIC_RCR) {
            out_ += szp();
            if (need(AF)) out_ += "c.af = false; ";
        }
        out_ += w + " }";
        return true;
    }
    bool lift_shxd(bool left) {
        if (bits(0) != 32) return false;
        std::string a, s, n, w;
        if (!prelude() || !rd(0, a) || !rd(1, s) || !rd(2, n, 8) || !wrt(0, "r", w)) return false;
        out_ += "const u32 a = " + a + ", b = " + s + "; const unsigned n = unsigned(" + n + ") & 31; if (n) { ";
        if (left) out_ += "const u32 r = (a << n) | (b >> (32 - n)); c.cf = (a >> (32 - n)) & 1; ";
        else out_ += "const u32 r = (a >> n) | (b << (32 - n)); c.cf = (a >> (n - 1)) & 1; ";
        out_ += "c.of = vc3::msb(r) != vc3::msb(a); " + szp() + w + " }";
        return true;
    }

    bool lift_imul() {
        if (nops() == 1) {
            const unsigned b = bits(0);
            std::string s;
            if (!prelude() || !rd(0, s)) return false;
            if (b == 32) {
                out_ += "const s64 p = s64(s32(c.eax)) * s64(s32(" + s + ")); c.eax = u32(p); c.edx = u32(u64(p) >> 32); ";
                out_ += "c.cf = c.of = p != s64(s32(p));";
            } else if (b == 16) {
                out_ += "const s32 p = s32(s16(c.eax)) * s32(s16(" + s + ")); vc3::set16(c.eax, u16(p)); vc3::set16(c.edx, u16(u32(p) >> 16)); ";
                out_ += "c.cf = c.of = p != s32(s16(p));";
            } else {
                out_ += "const s16 p = s16(s16(s8(c.eax)) * s16(s8(" + s + "))); vc3::set16(c.eax, u16(p)); ";
                out_ += "c.cf = c.of = p != s16(s8(p));";
            }
            return true;
        }
        const unsigned b = bits(0);
        const std::string T = ut(b), S = st(b);
        std::string x, y, w;
        if (!prelude()) return false;
        if (nops() == 2) { if (!rd(0, x) || !rd(1, y, b)) return false; }
        else { if (!rd(1, x) || !rd(2, y, b)) return false; }
        if (!wrt(0, "r", w)) return false;
        out_ += "const s64 p = s64(" + S + "(" + x + ")) * s64(" + S + "(" + y + ")); const " + T + " r = " + T + "(p); ";
        if (need(CF | OF)) out_ += "c.cf = c.of = p != s64(" + S + "(r)); ";
        out_ += szp() + w;
        return true;
    }
    bool lift_mul() {
        const unsigned b = bits(0);
        std::string s;
        if (!prelude() || !rd(0, s)) return false;
        if (b == 32) out_ += "const u64 p = u64(c.eax) * u64(" + s + "); c.eax = u32(p); c.edx = u32(p >> 32); c.cf = c.of = c.edx != 0;";
        else if (b == 16) out_ += "const u32 p = u32(u16(c.eax)) * u32(" + s + "); vc3::set16(c.eax, u16(p)); vc3::set16(c.edx, u16(p >> 16)); c.cf = c.of = (p >> 16) != 0;";
        else out_ += "const u16 p = u16(u16(u8(c.eax)) * u16(" + s + ")); vc3::set16(c.eax, p); c.cf = c.of = (p >> 8) != 0;";
        return true;
    }
    bool lift_div(bool sign) {
        const unsigned b = bits(0);
        std::string s;
        if (!prelude() || !rd(0, s)) return false;
        const std::string trap = F("vc3::guest_trap(c, $0, \"divide error\");", {hex(in_->addr)});
        if (b == 32) {
            if (sign) out_ += "const s64 n = s64((u64(c.edx) << 32) | c.eax); const s32 d = s32(" + s + "); if (d == 0 || (d == -1 && n == INT64_MIN)) " + trap +
                              " const s64 q = n / d; if (q != s32(q)) " + trap +
                              " c.eax = u32(q); c.edx = u32(n % d);";
            else out_ += "const u64 n = (u64(c.edx) << 32) | c.eax; const u32 d = " + s + "; if (d == 0) " + trap +
                         " const u64 q = n / d; if (q > 0xFFFFFFFFu) " + trap + " c.eax = u32(q); c.edx = u32(n % d);";
        } else if (b == 16) {
            if (sign) out_ += "const s32 n = s32((u32(u16(c.edx)) << 16) | u16(c.eax)); const s32 d = s16(" + s + "); if (d == 0) " + trap +
                              " const s32 q = n / d; if (q != s16(q)) " + trap + " vc3::set16(c.eax, u16(q)); vc3::set16(c.edx, u16(n % d));";
            else out_ += "const u32 n = (u32(u16(c.edx)) << 16) | u16(c.eax); const u32 d = " + s + "; if (d == 0) " + trap +
                         " const u32 q = n / d; if (q > 0xFFFFu) " + trap + " vc3::set16(c.eax, u16(q)); vc3::set16(c.edx, u16(n % d));";
        } else {
            if (sign) out_ += "const s32 n = s16(c.eax); const s32 d = s8(" + s + "); if (d == 0) " + trap +
                              " const s32 q = n / d; if (q != s8(q)) " + trap + " vc3::set8l(c.eax, u8(q)); vc3::set8h(c.eax, u8(n % d));";
            else out_ += "const u32 n = u16(c.eax); const u32 d = " + s + "; if (d == 0) " + trap +
                         " const u32 q = n / d; if (q > 0xFFu) " + trap + " vc3::set8l(c.eax, u8(q)); vc3::set8h(c.eax, u8(n % d));";
        }
        return true;
    }
    bool lift_xchg() {
        std::string a, b, w0, w1;
        if (!prelude() || !rd(0, a) || !rd(1, b) || !wrt(0, "y", w0) || !wrt(1, "x", w1)) return false;
        const std::string T = ut(bits(0));
        out_ += "const " + T + " x = " + a + ", y = " + b + "; " + w0 + " " + w1;
        return true;
    }
    bool lift_xadd() {
        const unsigned b = bits(0);
        const std::string T = ut(b);
        std::string a, bv, w0, w1;
        if (!prelude() || !rd(0, a) || !rd(1, bv) || !wrt(0, "r", w0) || !wrt(1, "a", w1)) return false;
        out_ += "const " + T + " a = " + a + ", b = " + bv + "; const " + T + " r = " + T + "(a + b); ";
        out_ += arith_flags("add", b) + w1 + " " + w0;
        return true;
    }
    bool lift_cmpxchg() {
        const unsigned b = bits(0);
        if (b != 32) return false;
        std::string d, s, w;
        if (!prelude() || !rd(0, d) || !rd(1, s) || !wrt(0, "src", w)) return false;
        out_ += "const u32 dst = " + d + ", src = " + s + "; vc3::flags_sub<u32>(c, c.eax, dst); ";
        out_ += "if (c.eax == dst) { " + w + " } else { c.eax = dst; }";
        return true;
    }
    bool lift_bitscan(bool reverse) {
        const unsigned b = bits(0);
        if (b != 32) return false;
        std::string s, w;
        const std::string v = reverse ? "31 - __builtin_clz(s)" : "__builtin_ctz(s)";
        if (!prelude() || !rd(1, s) || !wrt(0, v, w)) return false;
        out_ += "const u32 s = " + s + "; c.zf = s == 0; if (s) { " + w + " }";
        return true;
    }
    bool lift_bt(ZydisMnemonic m) {
        const unsigned b = bits(0);
        if (b != 32 && b != 16) return false;
        const std::string T = ut(b), S = st(b);
        std::string off;
        if (!prelude() || !rd(1, off, b)) return false;
        out_ += "const " + T + " off = " + off + "; ";
        std::string a, w;
        if (op(0).type == ZYDIS_OPERAND_TYPE_MEMORY && op(1).type == ZYDIS_OPERAND_TYPE_REGISTER) {
            // bit-string addressing: register offset may reach outside the operand
            out_ += "const u32 ea2 = ea + u32(" + S + "(off) >> " + (b == 32 ? "5" : "4") + ") * " +
                    std::to_string(b / 8) + "; const unsigned bit = off & " + std::to_string(b - 1) + "; ";
            a = "vc3::rd<" + T + ">(ea2)";
            w = "vc3::wr<" + T + ">(ea2, r);";
        } else {
            out_ += "const unsigned bit = off & " + std::to_string(b - 1) + "; ";
            if (!rd(0, a) || !wrt(0, "r", w)) return false;
        }
        out_ += "const " + T + " a = " + a + "; c.cf = (a >> bit) & 1; ";
        if (m == ZYDIS_MNEMONIC_BTS) out_ += "const " + T + " r = " + T + "(a | (" + T + "(1) << bit)); " + w;
        if (m == ZYDIS_MNEMONIC_BTR) out_ += "const " + T + " r = " + T + "(a & ~(" + T + "(1) << bit)); " + w;
        if (m == ZYDIS_MNEMONIC_BTC) out_ += "const " + T + " r = " + T + "(a ^ (" + T + "(1) << bit)); " + w;
        return true;
    }

    bool lift_string() {
        const auto m = in_->insn.mnemonic;
        const auto at = in_->insn.attributes;
        const bool rep = at & (ZYDIS_ATTRIB_HAS_REP | ZYDIS_ATTRIB_HAS_REPE | ZYDIS_ATTRIB_HAS_REPNE);
        const char* repmode = (at & ZYDIS_ATTRIB_HAS_REPNE) ? "vc3::kRepNE" : (at & ZYDIS_ATTRIB_HAS_REPE) ? "vc3::kRepE" : "vc3::kNoRep";
        std::string fn, T;
        switch (m) {
            case ZYDIS_MNEMONIC_MOVSB: fn = "movs"; T = "u8"; break;
            case ZYDIS_MNEMONIC_MOVSW: fn = "movs"; T = "u16"; break;
            case ZYDIS_MNEMONIC_MOVSD: fn = "movs"; T = "u32"; break;
            case ZYDIS_MNEMONIC_STOSB: fn = "stos"; T = "u8"; break;
            case ZYDIS_MNEMONIC_STOSW: fn = "stos"; T = "u16"; break;
            case ZYDIS_MNEMONIC_STOSD: fn = "stos"; T = "u32"; break;
            case ZYDIS_MNEMONIC_LODSB: fn = "lods"; T = "u8"; break;
            case ZYDIS_MNEMONIC_LODSW: fn = "lods"; T = "u16"; break;
            case ZYDIS_MNEMONIC_LODSD: fn = "lods"; T = "u32"; break;
            case ZYDIS_MNEMONIC_SCASB: fn = "scas"; T = "u8"; break;
            case ZYDIS_MNEMONIC_SCASW: fn = "scas"; T = "u16"; break;
            case ZYDIS_MNEMONIC_SCASD: fn = "scas"; T = "u32"; break;
            case ZYDIS_MNEMONIC_CMPSB: fn = "cmps"; T = "u8"; break;
            case ZYDIS_MNEMONIC_CMPSW: fn = "cmps"; T = "u16"; break;
            case ZYDIS_MNEMONIC_CMPSD: fn = "cmps"; T = "u32"; break;
            default: return false;
        }
        // Segment overrides on the source (e.g. fs:) are not supported.
        if (in_->insn.attributes & ZYDIS_ATTRIB_HAS_SEGMENT_FS) return false;
        if (fn == "scas" || fn == "cmps") out_ += "vc3::str_" + fn + "<" + T + ">(c, " + repmode + ");";
        else out_ += "vc3::str_" + fn + "<" + T + ">(c, " + (rep ? "true" : "false") + ");";
        return true;
    }

    // ------------------------------------------------------- control flow
    bool lift_jcc(const std::string& name) {
        const uint32_t t = rel_target(*in_);
        std::string cond;
        if (name == "jecxz") cond = "c.ecx == 0";
        else if (name == "jcxz") cond = "u16(c.ecx) == 0";
        else if (name == "loop") { out_ += "--c.ecx; "; cond = "c.ecx != 0"; }
        else if (name == "loope") { out_ += "--c.ecx; "; cond = "c.ecx != 0 && c.zf"; }
        else if (name == "loopne") { out_ += "--c.ecx; "; cond = "c.ecx != 0 && !c.zf"; }
        else cond = cond_expr(std::string_view(name).substr(1));
        if (cond.empty()) return false;
        out_ += "if (" + cond + ") { " + jump_to(t) + " }";
        return true;
    }
    bool lift_jmp() {
        falls_ = false;
        if (op(0).type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
            out_ += jump_to(rel_target(*in_));
            return true;
        }
        std::string t;
        if (!prelude() || !rd(0, t)) return false;
        out_ += "const u32 t = " + t + "; ";
        if (auto tbl = an_.read_jump_table(*in_); !tbl.empty()) {
            out_ += "switch (t) { ";
            std::set<uint32_t> seen;
            for (uint32_t x : tbl)
                if (seen.insert(x).second) out_ += "case " + hex(x) + ": " + jump_to(x) + " ";
            out_ += "default: vc3::jump_indirect(c, t); return; }";
            return true;
        }
        out_ += "vc3::jump_indirect(c, t); return;";
        return true;
    }
    bool lift_call() {
        if (op(0).type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
            const uint32_t t = rel_target(*in_);
            if (t == next_) {  // `call $+5; pop reg` get-EIP idiom: just push the address
                out_ += "vc3::push32(c, " + hex(next_) + ");";
                return true;
            }
            if (an_.is_function(t)) out_ += "vc3::push32(c, " + hex(next_) + "); " + fn_name(t) + "(c);";
            else out_ += F("vc3::call_indirect(c, $0, $1);", {hex(t), hex(next_)});
            return true;
        }
        std::string t;
        if (!prelude() || !rd(0, t)) return false;
        out_ += "const u32 t = " + t + "; vc3::call_indirect(c, t, " + hex(next_) + ");";
        return true;
    }
    bool lift_ret() {
        falls_ = false;
        uint32_t n = 4;
        if (nops() == 1 && op(0).type == ZYDIS_OPERAND_TYPE_IMMEDIATE) n += uint32_t(op(0).imm.value.u);
        out_ += "c.esp += " + std::to_string(n) + "; return;";
        return true;
    }

    // ---------------------------------------------------------------- x87
    static int st_index(ZydisRegister r) {
        return (r >= ZYDIS_REGISTER_ST0 && r <= ZYDIS_REGISTER_ST7) ? int(r - ZYDIS_REGISTER_ST0) : -1;
    }
    std::string stx(int i) const { return "vc3::st(c, " + std::to_string(i) + ")"; }

    // Floating-point value of operand i (st register or m32/m64/m80).
    bool fval(int i, std::string& out) const {
        const auto& o = op(i);
        if (o.type == ZYDIS_OPERAND_TYPE_REGISTER && st_index(o.reg.value) >= 0) { out = stx(st_index(o.reg.value)); return true; }
        if (o.type != ZYDIS_OPERAND_TYPE_MEMORY) return false;
        if (o.size == 32) out = "double(vc3::rdf32(ea))";
        else if (o.size == 64) out = "vc3::rdf64(ea)";
        else if (o.size == 80) out = "vc3::rdf80(ea)";
        else return false;
        return true;
    }
    bool fival(int i, std::string& out) const {
        const auto& o = op(i);
        if (o.type != ZYDIS_OPERAND_TYPE_MEMORY) return false;
        out = "double(" + st(o.size) + "(vc3::rd<" + ut(o.size) + ">(ea)))";
        return o.size == 16 || o.size == 32 || o.size == 64;
    }

    // Returns 1 lifted, 0 not handled.
    int lift_x87() {
        const auto m = in_->insn.mnemonic;
        std::string name = ZydisMnemonicGetString(m);
        if (!prelude()) return 0;
        std::string v;

        auto arith = [&](const std::string& base, bool integer, bool pop) -> int {
            std::string d, s;
            if (nops() == 0) { d = stx(1); s = stx(0); }
            else if (nops() == 1) { d = stx(0); if (!(integer ? fival(0, s) : fval(0, s))) return 0; }
            else { if (!fval(0, d) || !fval(1, s)) return 0; }
            std::string e;
            if (base == "add") e = "d + s";
            else if (base == "sub") e = "d - s";
            else if (base == "subr") e = "s - d";
            else if (base == "mul") e = "d * s";
            else if (base == "div") e = "d / s";
            else if (base == "divr") e = "s / d";
            else return 0;
            out_ += "const double s = " + s + "; double& d = " + d + "; d = " + e + ";";
            if (pop) out_ += " vc3::fpop(c);";
            return 1;
        };
        static const char* kArith[] = {"add", "sub", "subr", "mul", "div", "divr"};
        for (const char* a : kArith) {
            const std::string A = a;
            if (name == "f" + A) return arith(A, false, false);
            if (name == "f" + A + "p") return arith(A, false, true);
            if (name == "fi" + A) return arith(A, true, false);
        }

        auto compare = [&](bool integer, int pops) -> int {
            std::string s = stx(1);
            if (nops() >= 1) {
                const int i = nops() == 2 ? 1 : 0;
                if (!(integer ? fival(i, s) : fval(i, s))) return 0;
            }
            out_ += "vc3::fcom(c, " + stx(0) + ", " + s + ");";
            for (int i = 0; i < pops; ++i) out_ += " vc3::fpop(c);";
            return 1;
        };
        if (name == "fcom" || name == "fucom") return compare(false, 0);
        if (name == "fcomp" || name == "fucomp") return compare(false, 1);
        if (name == "fcompp" || name == "fucompp") return compare(false, 2);
        if (name == "ficom") return compare(true, 0);
        if (name == "ficomp") return compare(true, 1);
        if (name == "fcomi" || name == "fucomi" || name == "fcomip" || name == "fucomip") {
            std::string s;
            if (nops() != 2 || !fval(1, s)) return 0;
            out_ += "vc3::fcomi(c, " + stx(0) + ", " + s + ");";
            if (name.back() == 'p') out_ += " vc3::fpop(c);";
            return 1;
        }

        switch (m) {
            case ZYDIS_MNEMONIC_FLD:
                if (!fval(0, v)) return 0;
                out_ += "const double v = " + v + "; vc3::fpush(c, v);";
                return 1;
            case ZYDIS_MNEMONIC_FILD:
                if (!fival(0, v)) return 0;
                out_ += "vc3::fpush(c, " + v + ");";
                return 1;
            case ZYDIS_MNEMONIC_FST: case ZYDIS_MNEMONIC_FSTP: {
                const auto& o = op(0);
                if (o.type == ZYDIS_OPERAND_TYPE_REGISTER) {
                    const int i = st_index(o.reg.value);
                    if (i < 0) return 0;
                    out_ += stx(i) + " = " + stx(0) + ";";
                } else if (o.size == 32) out_ += "vc3::wrf32(ea, " + stx(0) + ");";
                else if (o.size == 64) out_ += "vc3::wrf64(ea, " + stx(0) + ");";
                else if (o.size == 80) out_ += "vc3::wrf80(ea, " + stx(0) + ");";
                else return 0;
                if (m == ZYDIS_MNEMONIC_FSTP) out_ += " vc3::fpop(c);";
                return 1;
            }
            case ZYDIS_MNEMONIC_FIST: case ZYDIS_MNEMONIC_FISTP: {
                const unsigned b = op(0).size;
                if (b != 16 && b != 32 && b != 64) return 0;
                out_ += "vc3::wr<" + ut(b) + ">(ea, " + ut(b) + "(vc3::fist<" + st(b) + ">(c, " + stx(0) + ")));";
                if (m == ZYDIS_MNEMONIC_FISTP) out_ += " vc3::fpop(c);";
                return 1;
            }
            case ZYDIS_MNEMONIC_FXCH: {
                const int i = nops() ? st_index(op(nops() - 1).reg.value) : 1;
                if (i < 0) return 0;
                out_ += "std::swap(" + stx(0) + ", " + stx(i == 0 && nops() == 2 ? st_index(op(0).reg.value) : i) + ");";
                return 1;
            }
            case ZYDIS_MNEMONIC_FLD1: out_ += "vc3::fpush(c, 1.0);"; return 1;
            case ZYDIS_MNEMONIC_FLDZ: out_ += "vc3::fpush(c, 0.0);"; return 1;
            case ZYDIS_MNEMONIC_FLDPI: out_ += "vc3::fpush(c, 3.14159265358979323846);"; return 1;
            case ZYDIS_MNEMONIC_FLDL2E: out_ += "vc3::fpush(c, 1.44269504088896340736);"; return 1;
            case ZYDIS_MNEMONIC_FLDL2T: out_ += "vc3::fpush(c, 3.32192809488736234787);"; return 1;
            case ZYDIS_MNEMONIC_FLDLN2: out_ += "vc3::fpush(c, 0.69314718055994530942);"; return 1;
            case ZYDIS_MNEMONIC_FLDLG2: out_ += "vc3::fpush(c, 0.30102999566398119521);"; return 1;
            case ZYDIS_MNEMONIC_FCHS: out_ += stx(0) + " = -" + stx(0) + ";"; return 1;
            case ZYDIS_MNEMONIC_FABS: out_ += stx(0) + " = std::fabs(" + stx(0) + ");"; return 1;
            case ZYDIS_MNEMONIC_FSQRT: out_ += stx(0) + " = std::sqrt(" + stx(0) + ");"; return 1;
            case ZYDIS_MNEMONIC_FSIN: out_ += stx(0) + " = std::sin(" + stx(0) + "); c.fpu_sw &= ~0x0400;"; return 1;
            case ZYDIS_MNEMONIC_FCOS: out_ += stx(0) + " = std::cos(" + stx(0) + "); c.fpu_sw &= ~0x0400;"; return 1;
            case ZYDIS_MNEMONIC_FSINCOS:
                out_ += "const double v = " + stx(0) + "; " + stx(0) + " = std::sin(v); vc3::fpush(c, std::cos(v)); c.fpu_sw &= ~0x0400;";
                return 1;
            case ZYDIS_MNEMONIC_FPTAN:
                out_ += stx(0) + " = std::tan(" + stx(0) + "); vc3::fpush(c, 1.0); c.fpu_sw &= ~0x0400;";
                return 1;
            case ZYDIS_MNEMONIC_FPATAN:
                out_ += stx(1) + " = std::atan2(" + stx(1) + ", " + stx(0) + "); vc3::fpop(c);";
                return 1;
            case ZYDIS_MNEMONIC_FYL2X:
                out_ += stx(1) + " = " + stx(1) + " * std::log2(" + stx(0) + "); vc3::fpop(c);";
                return 1;
            case ZYDIS_MNEMONIC_F2XM1: out_ += stx(0) + " = std::exp2(" + stx(0) + ") - 1.0;"; return 1;
            case ZYDIS_MNEMONIC_FSCALE:
                out_ += stx(0) + " = std::ldexp(" + stx(0) + ", int(std::trunc(" + stx(1) + ")));";
                return 1;
            case ZYDIS_MNEMONIC_FRNDINT: out_ += stx(0) + " = vc3::fround(c, " + stx(0) + ");"; return 1;
            case ZYDIS_MNEMONIC_FPREM:
                out_ += stx(0) + " = std::fmod(" + stx(0) + ", " + stx(1) + "); c.fpu_sw &= ~0x0400;";
                return 1;
            case ZYDIS_MNEMONIC_FPREM1:
                out_ += stx(0) + " = std::remainder(" + stx(0) + ", " + stx(1) + "); c.fpu_sw &= ~0x0400;";
                return 1;
            case ZYDIS_MNEMONIC_FTST: out_ += "vc3::fcom(c, " + stx(0) + ", 0.0);"; return 1;
            case ZYDIS_MNEMONIC_FNSTSW:
                if (op(0).type == ZYDIS_OPERAND_TYPE_REGISTER) out_ += "vc3::set16(c.eax, vc3::fnstsw(c));";
                else out_ += "vc3::wr<u16>(ea, vc3::fnstsw(c));";
                return 1;
            case ZYDIS_MNEMONIC_FNSTCW: out_ += "vc3::wr<u16>(ea, c.fpu_cw);"; return 1;
            case ZYDIS_MNEMONIC_FLDCW: out_ += "c.fpu_cw = vc3::rd<u16>(ea);"; return 1;
            case ZYDIS_MNEMONIC_FNCLEX: out_ += "c.fpu_sw &= u16(~0x80FFu);"; return 1;
            case ZYDIS_MNEMONIC_FNINIT: out_ += "vc3::fninit(c);"; return 1;
            case ZYDIS_MNEMONIC_FFREE: case ZYDIS_MNEMONIC_FWAIT: case ZYDIS_MNEMONIC_FNOP: return 1;
            case ZYDIS_MNEMONIC_FINCSTP: out_ += "c.fpu_top = (c.fpu_top + 1) & 7;"; return 1;
            case ZYDIS_MNEMONIC_FDECSTP: out_ += "c.fpu_top = (c.fpu_top - 1) & 7;"; return 1;
            default:
                return 0;
        }
    }

    // ---------------------------------------------------------- SSE / MMX
    static int xmm_index(const ZydisDecodedOperand& o) {
        if (o.type != ZYDIS_OPERAND_TYPE_REGISTER) return -1;
        const auto r = o.reg.value;
        return (r >= ZYDIS_REGISTER_XMM0 && r <= ZYDIS_REGISTER_XMM7) ? int(r - ZYDIS_REGISTER_XMM0) : -1;
    }
    static int mm_index(const ZydisDecodedOperand& o) {
        if (o.type != ZYDIS_OPERAND_TYPE_REGISTER) return -1;
        const auto r = o.reg.value;
        return (r >= ZYDIS_REGISTER_MM0 && r <= ZYDIS_REGISTER_MM7) ? int(r - ZYDIS_REGISTER_MM0) : -1;
    }
    bool v4(int i, std::string& out) const {  // packed source
        if (int x = xmm_index(op(i)); x >= 0) { out = "vc3::ld_x(c, " + std::to_string(x) + ")"; return true; }
        if (op(i).type == ZYDIS_OPERAND_TYPE_MEMORY) { out = "vc3::ld_v4(ea)"; return true; }
        return false;
    }
    bool f1(int i, std::string& out) const {  // scalar float source
        if (int x = xmm_index(op(i)); x >= 0) { out = "vc3::xf(c, " + std::to_string(x) + ")[0]"; return true; }
        if (op(i).type == ZYDIS_OPERAND_TYPE_MEMORY) { out = "vc3::rdf32(ea)"; return true; }
        return false;
    }

    int lift_simd() {
        const auto m = in_->insn.mnemonic;
        const std::string name = ZydisMnemonicGetString(m);
        bool any_vec = false;
        for (int i = 0; i < nops(); ++i) any_vec |= xmm_index(op(i)) >= 0 || mm_index(op(i)) >= 0;
        if (!any_vec && in_->insn.meta.isa_ext != ZYDIS_ISA_EXT_SSE) return -1;
        if (!prelude()) return 0;
        const int d = nops() ? xmm_index(op(0)) : -1;
        const std::string D = std::to_string(d);
        std::string s;

        // packed lane-wise ops
        struct Lane { ZydisMnemonic m; const char* expr; };
        static const Lane kPacked[] = {
            {ZYDIS_MNEMONIC_ADDPS, "a + b"}, {ZYDIS_MNEMONIC_SUBPS, "a - b"}, {ZYDIS_MNEMONIC_MULPS, "a * b"},
            {ZYDIS_MNEMONIC_DIVPS, "a / b"}, {ZYDIS_MNEMONIC_MINPS, "a < b ? a : b"}, {ZYDIS_MNEMONIC_MAXPS, "a > b ? a : b"},
            {ZYDIS_MNEMONIC_SQRTPS, "std::sqrt(b)"}, {ZYDIS_MNEMONIC_RSQRTPS, "vc3::rsqrt(b)"}, {ZYDIS_MNEMONIC_RCPPS, "1.0f / b"},
        };
        static const Lane kScalar[] = {
            {ZYDIS_MNEMONIC_ADDSS, "a + b"}, {ZYDIS_MNEMONIC_SUBSS, "a - b"}, {ZYDIS_MNEMONIC_MULSS, "a * b"},
            {ZYDIS_MNEMONIC_DIVSS, "a / b"}, {ZYDIS_MNEMONIC_MINSS, "a < b ? a : b"}, {ZYDIS_MNEMONIC_MAXSS, "a > b ? a : b"},
            {ZYDIS_MNEMONIC_SQRTSS, "std::sqrt(b)"}, {ZYDIS_MNEMONIC_RSQRTSS, "vc3::rsqrt(b)"}, {ZYDIS_MNEMONIC_RCPSS, "1.0f / b"},
        };
        for (const auto& l : kPacked)
            if (l.m == m) {
                if (d < 0 || !v4(1, s)) return 0;
                out_ += "vc3::V4 x = vc3::ld_x(c, " + D + "); const vc3::V4 y = " + s +
                        "; for (int i = 0; i < 4; ++i) { const float a = x.f[i], b = y.f[i]; (void)a; x.f[i] = " + l.expr +
                        "; } vc3::st_x(c, " + D + ", x);";
                return 1;
            }
        for (const auto& l : kScalar)
            if (l.m == m) {
                if (d < 0 || !f1(1, s)) return 0;
                out_ += "const float a = vc3::xf(c, " + D + ")[0], b = " + s + "; (void)a; vc3::xf(c, " + D + ")[0] = " + l.expr + ";";
                return 1;
            }

        if (const int r = lift_mmx(m); r >= 0) return r;

        switch (m) {
            case ZYDIS_MNEMONIC_MOVAPS: case ZYDIS_MNEMONIC_MOVUPS: case ZYDIS_MNEMONIC_MOVNTPS:
                if (d >= 0) { if (!v4(1, s)) return 0; out_ += "vc3::st_x(c, " + D + ", " + s + ");"; return 1; }
                if (xmm_index(op(1)) < 0) return 0;
                out_ += "vc3::st_v4(ea, vc3::ld_x(c, " + std::to_string(xmm_index(op(1))) + "));";
                return 1;
            case ZYDIS_MNEMONIC_MOVSS:
                if (d >= 0 && op(1).type == ZYDIS_OPERAND_TYPE_MEMORY) {
                    out_ += "vc3::st_x(c, " + D + ", vc3::V4{{vc3::rdf32(ea), 0, 0, 0}});";
                    return 1;
                }
                if (d >= 0) { if (!f1(1, s)) return 0; out_ += "vc3::xf(c, " + D + ")[0] = " + s + ";"; return 1; }
                if (xmm_index(op(1)) < 0) return 0;
                out_ += "vc3::wr<float>(ea, vc3::xf(c, " + std::to_string(xmm_index(op(1))) + ")[0]);";
                return 1;
            case ZYDIS_MNEMONIC_SHUFPS:
                if (d < 0 || !v4(1, s)) return 0;
                out_ += "vc3::st_x(c, " + D + ", vc3::shufps(vc3::ld_x(c, " + D + "), " + s + ", " +
                        hex(op(2).imm.value.u & 0xFF) + "));";
                return 1;
            case ZYDIS_MNEMONIC_UNPCKLPS: case ZYDIS_MNEMONIC_UNPCKHPS: {
                if (d < 0 || !v4(1, s)) return 0;
                const char* k = m == ZYDIS_MNEMONIC_UNPCKLPS ? "0" : "2";
                const char* k1 = m == ZYDIS_MNEMONIC_UNPCKLPS ? "1" : "3";
                out_ += std::string("const vc3::V4 a = vc3::ld_x(c, ") + D + "), b = " + s + "; vc3::st_x(c, " + D +
                        ", vc3::V4{{a.f[" + k + "], b.f[" + k + "], a.f[" + k1 + "], b.f[" + k1 + "]}});";
                return 1;
            }
            case ZYDIS_MNEMONIC_MOVLPS: case ZYDIS_MNEMONIC_MOVHPS: {
                const std::string off = m == ZYDIS_MNEMONIC_MOVLPS ? "0" : "8";
                if (d >= 0) { out_ += "std::memcpy(c.xmm[" + D + "] + " + off + ", vc3::host_ptr(ea), 8);"; return 1; }
                const int x = xmm_index(op(1));
                if (x < 0) return 0;
                out_ += "std::memcpy(vc3::host_ptr(ea), c.xmm[" + std::to_string(x) + "] + " + off + ", 8);";
                return 1;
            }
            case ZYDIS_MNEMONIC_MOVHLPS: case ZYDIS_MNEMONIC_MOVLHPS: {
                const int x = xmm_index(op(1));
                if (d < 0 || x < 0) return 0;
                const bool hl = m == ZYDIS_MNEMONIC_MOVHLPS;
                out_ += "std::memcpy(c.xmm[" + D + "] + " + (hl ? "0" : "8") + ", c.xmm[" + std::to_string(x) + "] + " +
                        (hl ? "8" : "0") + ", 8);";
                return 1;
            }
            case ZYDIS_MNEMONIC_ANDPS: case ZYDIS_MNEMONIC_ANDNPS: case ZYDIS_MNEMONIC_ORPS: case ZYDIS_MNEMONIC_XORPS: {
                if (d < 0 || !v4(1, s)) return 0;
                const char* e = m == ZYDIS_MNEMONIC_ANDPS ? "a & b" : m == ZYDIS_MNEMONIC_ANDNPS ? "~a & b"
                              : m == ZYDIS_MNEMONIC_ORPS ? "a | b" : "a ^ b";
                out_ += "const vc3::V4 y = " + s + "; u32 bv[4]; std::memcpy(bv, y.f, 16); u32* xv = vc3::xu(c, " + D +
                        "); for (int i = 0; i < 4; ++i) { const u32 a = xv[i], b = bv[i]; xv[i] = " + e + "; }";
                return 1;
            }
            case ZYDIS_MNEMONIC_COMISS: case ZYDIS_MNEMONIC_UCOMISS:
                if (d < 0 || !f1(1, s)) return 0;
                out_ += "vc3::fcomi(c, vc3::xf(c, " + D + ")[0], " + s + ");";
                return 1;
            case ZYDIS_MNEMONIC_CVTSI2SS: {
                std::string v;
                if (d < 0 || !rd(1, v)) return 0;
                out_ += "vc3::xf(c, " + D + ")[0] = float(s32(" + v + "));";
                return 1;
            }
            case ZYDIS_MNEMONIC_CVTSS2SI: case ZYDIS_MNEMONIC_CVTTSS2SI: {
                std::string w;
                if (!f1(1, s)) return 0;
                const std::string e = m == ZYDIS_MNEMONIC_CVTSS2SI ? "vc3::cvt_f2i_mxcsr(c, " + s + ")" : "vc3::cvt_f2i_trunc(" + s + ")";
                if (!wrt(0, "u32(" + e + ")", w)) return 0;
                out_ += w;
                return 1;
            }
            case ZYDIS_MNEMONIC_MOVMSKPS: {
                const int x = xmm_index(op(1));
                std::string w;
                if (x < 0 || !wrt(0, "msk", w)) return 0;
                out_ += "const u32* xv = vc3::xu(c, " + std::to_string(x) + "); const u32 msk = (xv[0] >> 31) | ((xv[1] >> 31) << 1) | ((xv[2] >> 31) << 2) | ((xv[3] >> 31) << 3); " + w;
                return 1;
            }
            case ZYDIS_MNEMONIC_SFENCE: case ZYDIS_MNEMONIC_PREFETCHNTA: case ZYDIS_MNEMONIC_PREFETCHT0:
            case ZYDIS_MNEMONIC_PREFETCHT1: case ZYDIS_MNEMONIC_PREFETCHT2:
                return 1;
            case ZYDIS_MNEMONIC_LDMXCSR: out_ += "vc3::ldmxcsr(c, vc3::rd<u32>(ea));"; return 1;
            case ZYDIS_MNEMONIC_STMXCSR: out_ += "vc3::wr<u32>(ea, c.mxcsr);"; return 1;

            // MMX (only data movement shows up: memcpy-style loops)
            case ZYDIS_MNEMONIC_MOVQ: case ZYDIS_MNEMONIC_MOVNTQ: {
                const int a = mm_index(op(0)), b = mm_index(op(1));
                if (a >= 0 && b >= 0) out_ += "c.mm[" + std::to_string(a) + "] = c.mm[" + std::to_string(b) + "];";
                else if (a >= 0 && op(1).type == ZYDIS_OPERAND_TYPE_MEMORY) out_ += "c.mm[" + std::to_string(a) + "] = vc3::rd<u64>(ea);";
                else if (b >= 0 && op(0).type == ZYDIS_OPERAND_TYPE_MEMORY) out_ += "vc3::wr<u64>(ea, c.mm[" + std::to_string(b) + "]);";
                else return 0;
                return 1;
            }
            case ZYDIS_MNEMONIC_MOVD: {
                const int a = mm_index(op(0)), b = mm_index(op(1));
                std::string v, w;
                if (a >= 0) { if (!rd(1, v)) return 0; out_ += "c.mm[" + std::to_string(a) + "] = u64(u32(" + v + "));"; return 1; }
                if (b >= 0) { if (!wrt(0, "u32(c.mm[" + std::to_string(b) + "])", w)) return 0; out_ += w; return 1; }
                return 0;
            }
            default:
                return 0;
        }
    }

    // MMX packed-integer ops on c.mm[] (and the SSE<->MMX conversions). -1 = not MMX.
    int lift_mmx(ZydisMnemonic m) {
        // Every MMX operation maps to a named helper (vc3::mmx, SSE2-backed); a = dest, b = source.
        struct Whole { ZydisMnemonic m; const char* e; };
        static const Whole kWhole[] = {
            {ZYDIS_MNEMONIC_PADDB, "vc3::mmx::paddb(a, b)"}, {ZYDIS_MNEMONIC_PADDW, "vc3::mmx::paddw(a, b)"},
            {ZYDIS_MNEMONIC_PADDD, "vc3::mmx::paddd(a, b)"}, {ZYDIS_MNEMONIC_PSUBB, "vc3::mmx::psubb(a, b)"},
            {ZYDIS_MNEMONIC_PSUBW, "vc3::mmx::psubw(a, b)"}, {ZYDIS_MNEMONIC_PSUBD, "vc3::mmx::psubd(a, b)"},
            {ZYDIS_MNEMONIC_PADDSB, "vc3::mmx::paddsb(a, b)"}, {ZYDIS_MNEMONIC_PADDSW, "vc3::mmx::paddsw(a, b)"},
            {ZYDIS_MNEMONIC_PSUBSB, "vc3::mmx::psubsb(a, b)"}, {ZYDIS_MNEMONIC_PSUBSW, "vc3::mmx::psubsw(a, b)"},
            {ZYDIS_MNEMONIC_PADDUSB, "vc3::mmx::paddusb(a, b)"}, {ZYDIS_MNEMONIC_PADDUSW, "vc3::mmx::paddusw(a, b)"},
            {ZYDIS_MNEMONIC_PSUBUSB, "vc3::mmx::psubusb(a, b)"}, {ZYDIS_MNEMONIC_PSUBUSW, "vc3::mmx::psubusw(a, b)"},
            {ZYDIS_MNEMONIC_PMULLW, "vc3::mmx::pmullw(a, b)"}, {ZYDIS_MNEMONIC_PMULHW, "vc3::mmx::pmulhw(a, b)"},
            {ZYDIS_MNEMONIC_PMULHUW, "vc3::mmx::pmulhuw(a, b)"},
            {ZYDIS_MNEMONIC_PCMPEQB, "vc3::mmx::pcmpeqb(a, b)"}, {ZYDIS_MNEMONIC_PCMPEQW, "vc3::mmx::pcmpeqw(a, b)"},
            {ZYDIS_MNEMONIC_PCMPEQD, "vc3::mmx::pcmpeqd(a, b)"}, {ZYDIS_MNEMONIC_PCMPGTB, "vc3::mmx::pcmpgtb(a, b)"},
            {ZYDIS_MNEMONIC_PCMPGTW, "vc3::mmx::pcmpgtw(a, b)"}, {ZYDIS_MNEMONIC_PCMPGTD, "vc3::mmx::pcmpgtd(a, b)"},
            {ZYDIS_MNEMONIC_PAVGB, "vc3::mmx::pavgb(a, b)"}, {ZYDIS_MNEMONIC_PAVGW, "vc3::mmx::pavgw(a, b)"},
            {ZYDIS_MNEMONIC_PMINUB, "vc3::mmx::pminub(a, b)"}, {ZYDIS_MNEMONIC_PMAXUB, "vc3::mmx::pmaxub(a, b)"},
            {ZYDIS_MNEMONIC_PMINSW, "vc3::mmx::pminsw(a, b)"}, {ZYDIS_MNEMONIC_PMAXSW, "vc3::mmx::pmaxsw(a, b)"},
            {ZYDIS_MNEMONIC_PAND, "a & b"}, {ZYDIS_MNEMONIC_POR, "a | b"}, {ZYDIS_MNEMONIC_PXOR, "a ^ b"},
            {ZYDIS_MNEMONIC_PANDN, "~a & b"}, {ZYDIS_MNEMONIC_PADDQ, "a + b"}, {ZYDIS_MNEMONIC_PSUBQ, "a - b"},
            {ZYDIS_MNEMONIC_PMADDWD, "vc3::mmx::pmaddwd_x(a, b)"}, {ZYDIS_MNEMONIC_PSADBW, "vc3::mmx::psadbw_x(a, b)"},
            {ZYDIS_MNEMONIC_PACKSSWB, "vc3::mmx::packsswb(a, b)"}, {ZYDIS_MNEMONIC_PACKSSDW, "vc3::mmx::packssdw(a, b)"},
            {ZYDIS_MNEMONIC_PACKUSWB, "vc3::mmx::packuswb(a, b)"},
            {ZYDIS_MNEMONIC_PUNPCKLBW, "vc3::mmx::punpcklbw(a, b)"}, {ZYDIS_MNEMONIC_PUNPCKHBW, "vc3::mmx::punpckhbw(a, b)"},
            {ZYDIS_MNEMONIC_PUNPCKLWD, "vc3::mmx::punpcklwd(a, b)"}, {ZYDIS_MNEMONIC_PUNPCKHWD, "vc3::mmx::punpckhwd(a, b)"},
            {ZYDIS_MNEMONIC_PUNPCKLDQ, "vc3::mmx::punpckldq(a, b)"}, {ZYDIS_MNEMONIC_PUNPCKHDQ, "vc3::mmx::punpckhdq(a, b)"},
            {ZYDIS_MNEMONIC_PSLLW, "vc3::mmx::psllw(a, b)"}, {ZYDIS_MNEMONIC_PSLLD, "vc3::mmx::pslld(a, b)"},
            {ZYDIS_MNEMONIC_PSLLQ, "b >= 64 ? 0 : a << b"}, {ZYDIS_MNEMONIC_PSRLW, "vc3::mmx::psrlw(a, b)"},
            {ZYDIS_MNEMONIC_PSRLD, "vc3::mmx::psrld(a, b)"}, {ZYDIS_MNEMONIC_PSRLQ, "b >= 64 ? 0 : a >> b"},
            {ZYDIS_MNEMONIC_PSRAW, "vc3::mmx::psraw(a, b)"}, {ZYDIS_MNEMONIC_PSRAD, "vc3::mmx::psrad(a, b)"},
        };
        const int d = mm_index(op(0));
        auto mmsrc = [&](int i, std::string& out) -> bool {
            if (const int x = mm_index(op(i)); x >= 0) { out = "c.mm[" + std::to_string(x) + "]"; return true; }
            if (op(i).type == ZYDIS_OPERAND_TYPE_MEMORY) { out = "vc3::rd<u64>(ea)"; return true; }
            if (op(i).type == ZYDIS_OPERAND_TYPE_IMMEDIATE) { out = "u64(" + hex(op(i).imm.value.u & 0xFF) + ")"; return true; }
            return false;
        };
        std::string src;
        for (const auto& w : kWhole)
            if (w.m == m) {
                if (d < 0 || !mmsrc(1, src)) return 0;
                const std::string D = "c.mm[" + std::to_string(d) + "]";
                out_ += "const u64 a = " + D + ", b = " + src + "; " + D + " = u64(" + w.e + ");";
                return 1;
            }
        switch (m) {
            case ZYDIS_MNEMONIC_PSHUFW:
                if (d < 0 || !mmsrc(1, src)) return 0;
                out_ += "c.mm[" + std::to_string(d) + "] = vc3::mmx::pshufw(" + src + ", " + hex(op(2).imm.value.u & 0xFF) + ");";
                return 1;
            case ZYDIS_MNEMONIC_PEXTRW: {
                const int s = mm_index(op(1));
                std::string w;
                if (s < 0) return xmm_index(op(1)) >= 0 ? 0 : -1;
                if (!wrt(0, "u32((c.mm[" + std::to_string(s) + "] >> (16 * (" + hex(op(2).imm.value.u & 3) + "))) & 0xFFFF)", w)) return 0;
                out_ += w;
                return 1;
            }
            case ZYDIS_MNEMONIC_PINSRW: {
                std::string v;
                if (d < 0 || !rd(1, v)) return d < 0 ? -1 : 0;
                const std::string sh = std::to_string(16 * (op(2).imm.value.u & 3));
                const std::string D = "c.mm[" + std::to_string(d) + "]";
                out_ += D + " = (" + D + " & ~(0xFFFFull << " + sh + ")) | (u64(u16(" + v + ")) << " + sh + ");";
                return 1;
            }
            case ZYDIS_MNEMONIC_PMOVMSKB: {
                const int s = mm_index(op(1));
                std::string w;
                if (s < 0) return -1;
                if (!wrt(0, "vc3::mmx::pmovmskb(c.mm[" + std::to_string(s) + "])", w)) return 0;
                out_ += w;
                return 1;
            }
            case ZYDIS_MNEMONIC_CVTPS2PI: case ZYDIS_MNEMONIC_CVTTPS2PI: {
                if (d < 0) return 0;
                const bool t = m == ZYDIS_MNEMONIC_CVTTPS2PI;
                std::string f0, f1;
                if (const int x = xmm_index(op(1)); x >= 0) {
                    f0 = "vc3::xf(c, " + std::to_string(x) + ")[0]";
                    f1 = "vc3::xf(c, " + std::to_string(x) + ")[1]";
                } else {
                    f0 = "vc3::rdf32(ea)";
                    f1 = "vc3::rdf32(ea + 4)";
                }
                out_ += "c.mm[" + std::to_string(d) + "] = vc3::mmx::cvtps2pi(c, " + f0 + ", " + f1 + ", " + (t ? "true" : "false") + ");";
                return 1;
            }
            case ZYDIS_MNEMONIC_CVTPI2PS: {
                const int x = xmm_index(op(0));
                if (x < 0) return 0;
                std::string v;
                if (const int s = mm_index(op(1)); s >= 0) v = "c.mm[" + std::to_string(s) + "]";
                else v = "vc3::rd<u64>(ea)";
                out_ += "const u64 v = " + v + "; vc3::xf(c, " + std::to_string(x) + ")[0] = float(s32(u32(v))); vc3::xf(c, " +
                        std::to_string(x) + ")[1] = float(s32(u32(v >> 32)));";
                return 1;
            }
            default:
                return -1;
        }
    }

    static std::string disasm(const Instruction& in) {
        static ZydisFormatter fmt = [] {
            ZydisFormatter f;
            ZydisFormatterInit(&f, ZYDIS_FORMATTER_STYLE_INTEL);
            return f;
        }();
        char buf[256];
        ZydisFormatterFormatInstruction(&fmt, &in.insn, in.ops, in.insn.operand_count_visible, buf, sizeof buf,
                                        in.addr, nullptr);
        return buf;
    }

    const Analyzer& an_;
    const Function& f_;
    LiftStats& stats_;
    const Instruction* in_ = nullptr;
    uint32_t next_ = 0, live_ = 0;
    bool falls_ = true;
    std::string out_;
    std::set<uint32_t> labels_;
    std::map<uint32_t, uint32_t> live_out_;
};

}  // namespace

std::string lift_function(const Analyzer& an, const Function& f, LiftStats& stats) {
    return FunctionLifter(an, f, stats).run();
}

}  // namespace vc3
