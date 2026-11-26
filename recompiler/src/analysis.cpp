#include "analysis.h"

#include <algorithm>
#include <cstdio>
#include <deque>

namespace vc3 {

Analyzer::Analyzer(const Xbe& xbe, AnalysisOptions opts) : xbe_(xbe), opts_(std::move(opts)) {
    ZydisDecoderInit(&decoder_, ZYDIS_MACHINE_MODE_LEGACY_32, ZYDIS_STACK_WIDTH_32);
}

bool Analyzer::is_code(uint32_t va) const {
    const auto* s = xbe_.section_at(va);
    return s && va < s->vaddr + s->rawsize && opts_.code_sections.count(s->name);
}

bool Analyzer::decode(uint32_t va, Instruction& out) const {
    size_t avail = 0;
    const uint8_t* p = xbe_.ptr(va, &avail);
    if (!p || !is_code(va)) return false;
    out.addr = va;
    return ZYAN_SUCCESS(ZydisDecoderDecodeFull(&decoder_, p, avail, &out.insn, out.ops));
}

static bool mn_is_int2d(const Instruction& in) {
    return in.insn.mnemonic == ZYDIS_MNEMONIC_INT && in.ops[0].type == ZYDIS_OPERAND_TYPE_IMMEDIATE &&
           in.ops[0].imm.value.u == 0x2D;
}

static Function make_function(uint32_t entry, Function::Source src) {
    Function f;
    f.entry = entry;
    f.source = src;
    return f;
}

static bool branch_target(const Instruction& in, uint32_t& target) {
    const auto& op = in.ops[0];
    if (op.type != ZYDIS_OPERAND_TYPE_IMMEDIATE || !op.imm.is_relative) return false;
    ZyanU64 abs = 0;
    if (!ZYAN_SUCCESS(ZydisCalcAbsoluteAddress(&in.insn, &op, in.addr, &abs))) return false;
    target = uint32_t(abs);
    return true;
}

void Analyzer::scan_pointers() {
    // Function pointers / vtables live in data sections. Accept a dword as a
    // function entry only if it points at code that looks like a function start
    // (16-byte aligned, or preceded by int3/nop/ret padding) and decodes.
    for (const auto& s : xbe_.sections) {
        if (opts_.code_sections.count(s.name)) continue;  // skip code: jump tables point mid-function
        size_t avail = 0;
        const uint8_t* p = xbe_.ptr(s.vaddr, &avail);
        if (!p) continue;
        for (size_t o = 0; o + 4 <= avail; o += 4) {
            uint32_t v = p[o] | p[o + 1] << 8 | p[o + 2] << 16 | uint32_t(p[o + 3]) << 24;
            if (!is_code(v) || funcs_.count(v)) continue;
            if (looks_like_function_start(v)) {
                funcs_[v] = make_function(v, Function::Source::PointerScan);
                ++ptr_hits_;
            }
        }
    }
}

// A plausible function entry: 16-byte aligned or preceded by int3/nop/ret
// padding, and decodable.
bool Analyzer::looks_like_function_start(uint32_t va) const {
    if (!is_code(va)) return false;
    const uint8_t* prev = xbe_.ptr(va - 1);
    const bool aligned = (va & 15) == 0;
    const uint8_t* prev3 = xbe_.ptr(va - 3);
    const bool padded = prev && (*prev == 0xCC || *prev == 0x90 || *prev == 0xC3);
    const bool after_ret_n = prev3 && *prev3 == 0xC2;  // ret imm16
    size_t avail = 0;
    const uint8_t* p = xbe_.ptr(va, &avail);
    // push ebp; mov ebp, esp  (also the 8B FF hot-patch form)
    const bool prologue = p && avail >= 5 &&
                          ((p[0] == 0x55 && p[1] == 0x8B && p[2] == 0xEC) ||
                           (p[0] == 0x8B && p[1] == 0xFF && p[2] == 0x55 && p[3] == 0x8B && p[4] == 0xEC));
    Instruction tmp;
    return (aligned || padded || after_ret_n || prologue) && decode(va, tmp);
}

std::vector<uint32_t> Analyzer::read_jump_table(const Instruction& in) const {
    // MSVC switch: jmp dword ptr [reg*4 + table]
    std::vector<uint32_t> out;
    const auto& op = in.ops[0];
    if (op.type != ZYDIS_OPERAND_TYPE_MEMORY || op.mem.base != ZYDIS_REGISTER_NONE ||
        op.mem.index == ZYDIS_REGISTER_NONE || op.mem.scale != 4 || !op.mem.disp.has_displacement)
        return out;
    uint32_t table = uint32_t(op.mem.disp.value);
    // Entries before the displacement are reachable with negative indices
    // (MSVC memcpy: `neg ecx; jmp [ecx*4 + UnwindUpVec + 0x1C]`).
    for (uint32_t i = 1; i <= 256; ++i) {
        size_t avail = 0;
        if (!xbe_.ptr(table - i * 4, &avail) || avail < 4) break;
        const uint32_t t = xbe_.read32(table - i * 4);
        if (!is_code(t)) break;
        out.insert(out.begin(), t);
    }
    for (uint32_t i = 0; i < 1024; ++i) {
        size_t avail = 0;
        if (!xbe_.ptr(table + i * 4, &avail) || avail < 4) break;
        uint32_t t = xbe_.read32(table + i * 4);
        // Entry 0 may be unused and overlap other bytes (MSVC memcpy's misaligned-copy
        // tables are indexed 1..3 and start right after the jmp: entry 0 is the jmp's
        // own tail). Skip it rather than giving up on the table.
        if (!is_code(t)) {
            if (i == 0 && out.empty()) continue;
            break;
        }
        out.push_back(t);
    }
    return out;
}

void Analyzer::trace(Function& f) {
    std::deque<uint32_t> work{f.entry};
    while (!work.empty()) {
        uint32_t va = work.front();
        work.pop_front();
        f.block_starts.insert(va);
        while (!f.insns.count(va)) {
            Instruction in;
            if (!decode(va, in)) {
                f.decode_error = true;
                break;
            }
            f.insns.emplace(va, in);
            // `int 2Dh` (debug service) resumes one byte later, skipping the int3 that follows it.
            const bool dbg_service = mn_is_int2d(in);
            const uint32_t next = va + in.insn.length + (dbg_service ? 1 : 0);
            const auto mn = in.insn.mnemonic;
            const auto cat = in.insn.meta.category;
            uint32_t tgt = 0;

            // Code addresses used as immediates (push offset ThreadProc, mov [x], offset Callback).
            for (int i = 0; i < in.insn.operand_count_visible; ++i) {
                const auto& op = in.ops[i];
                if (op.type != ZYDIS_OPERAND_TYPE_IMMEDIATE || op.imm.is_relative) continue;
                const uint32_t v = uint32_t(op.imm.value.u);
                if (!funcs_.count(v) && looks_like_function_start(v))
                    funcs_[v] = make_function(v, Function::Source::ImmediateRef);
            }

            if (cat == ZYDIS_CATEGORY_CALL) {
                if (branch_target(in, tgt) && is_code(tgt) && !funcs_.count(tgt))
                    funcs_[tgt] = make_function(tgt, Function::Source::CallTarget);
            } else if (cat == ZYDIS_CATEGORY_COND_BR) {
                if (branch_target(in, tgt)) work.push_back(tgt);
                work.push_back(next);
                break;
            } else if (cat == ZYDIS_CATEGORY_UNCOND_BR) {
                if (branch_target(in, tgt)) {
                    if (tgt != f.entry && funcs_.count(tgt)) f.tail_calls.insert(tgt);
                    else work.push_back(tgt);
                } else if (auto tbl = read_jump_table(in); !tbl.empty()) {
                    f.jump_tables.push_back(va);
                    // Remember where the table lives so the code-section pointer scan skips it.
                    const uint32_t disp = uint32_t(in.ops[0].mem.disp.value);
                    uint32_t lo = disp, hi = disp;
                    while (lo > disp - 256 * 4 && is_code(xbe_.read32(lo - 4))) lo -= 4;
                    if (!is_code(xbe_.read32(hi))) hi += 4;  // unused entry 0 (see read_jump_table)
                    while (hi < disp + 1024 * 4 && is_code(xbe_.read32(hi))) hi += 4;
                    table_ranges_.insert({lo, hi});
                    for (uint32_t t : tbl) work.push_back(t);
                }
                break;
            } else if (cat == ZYDIS_CATEGORY_RET || mn == ZYDIS_MNEMONIC_INT3 ||
                       mn == ZYDIS_MNEMONIC_HLT || mn == ZYDIS_MNEMONIC_UD2) {
                break;
            }
            va = next;
            if (funcs_.count(va) && va != f.entry) {  // fell through into another function
                f.tail_calls.insert(va);
                break;
            }
        }
    }
}

// Function pointers stored inside code sections (e.g. XPP's init table). Only
// dwords not covered by decoded instructions or jump tables are considered.
size_t Analyzer::scan_code_section_pointers() {
    std::vector<std::pair<uint32_t, uint32_t>> covered(table_ranges_.begin(), table_ranges_.end());
    for (const auto& [va, f] : funcs_)
        for (const auto& [ia, in] : f.insns) covered.push_back({ia, ia + in.insn.length});
    std::sort(covered.begin(), covered.end());
    auto is_covered = [&](uint32_t a) {
        auto it = std::upper_bound(covered.begin(), covered.end(), std::make_pair(a, 0xFFFFFFFFu));
        // any range starting at or before a+3 that ends after a overlaps the dword
        for (auto r = it; r != covered.begin();) {
            --r;
            if (r->second > a) return true;
            if (a - r->first > 64) break;
        }
        return false;
    };
    size_t added = 0;
    // Immediates naming code outside every decoded instruction (callback registrations
    // such as `mov [g_cb], offset Func` where Func follows a no-return call without padding).
    std::vector<uint32_t> imm_targets;
    for (const auto& [va, f] : funcs_)
        for (const auto& [ia, in] : f.insns)
            for (int i = 0; i < in.insn.operand_count_visible; ++i) {
                const auto& op = in.ops[i];
                if (op.type == ZYDIS_OPERAND_TYPE_IMMEDIATE && !op.imm.is_relative && op.size == 32)
                    imm_targets.push_back(uint32_t(op.imm.value.u));
            }
    for (uint32_t v : imm_targets) {
        Instruction tmp;
        if (is_code(v) && !funcs_.count(v) && !is_covered(v) && decode(v, tmp)) {
            funcs_[v] = make_function(v, Function::Source::ImmediateRef);
            ++added;
        }
    }
    for (const auto& s : xbe_.sections) {
        if (!opts_.code_sections.count(s.name)) continue;
        for (uint32_t va = s.vaddr; va + 4 <= s.vaddr + s.rawsize; va += 4) {
            const uint32_t v = xbe_.read32(va);
            if (!is_code(v) || funcs_.count(v) || is_covered(va) || is_covered(va + 3)) continue;
            if (looks_like_function_start(v)) {
                funcs_[v] = make_function(v, Function::Source::PointerScan);
                ++ptr_hits_;
                ++added;
            }
        }
    }
    return added;
}

void Analyzer::run() {
    funcs_[xbe_.entry] = make_function(xbe_.entry, Function::Source::Entry);
    for (uint32_t e : opts_.extra_entries)
        funcs_.try_emplace(e, make_function(e, Function::Source::Manual));
    if (opts_.pointer_scan) scan_pointers();

    // Iterate until no tracing discovers new call targets.
    std::set<uint32_t> traced;
    auto trace_pending = [&] {
        bool changed = true;
        while (changed) {
            changed = false;
            std::vector<uint32_t> pending;
            for (auto& [va, f] : funcs_)
                if (!traced.count(va)) pending.push_back(va);
            for (uint32_t va : pending) {
                trace(funcs_.at(va));
                traced.insert(va);
                changed = true;
            }
        }
    };
    trace_pending();
    if (opts_.pointer_scan)
        while (scan_code_section_pointers()) trace_pending();

    // Pointer-scan candidates that run into undecodable bytes were data, not code.
    // (Call targets they discovered are kept; TODO: prune those too.)
    for (auto it = funcs_.begin(); it != funcs_.end();) {
        const auto src = it->second.source;
        if ((src == Function::Source::PointerScan || src == Function::Source::ImmediateRef) && it->second.decode_error) {
            if (src == Function::Source::PointerScan) --ptr_hits_;
            it = funcs_.erase(it);
        } else {
            ++it;
        }
    }
}

}  // namespace vc3
