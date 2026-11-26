#include "mmio.h"

#include <Zydis/Zydis.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#include <signal.h>
#include <ucontext.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <vector>

#include "vc3/helpers.h"

namespace vc3::mmio {

namespace {

struct Range {
    uint32_t base, size;
    const char* name;
    ReadFn rd;
    WriteFn wr;
    void* ctx;
};
std::vector<Range> g_ranges;
ZydisDecoder g_dec;

// The faulting thread's registers: Linux ucontext_t or Windows CONTEXT. GPRs by index in
// x86 order (rax rcx rdx rbx rsp rbp rsi rdi r8..r15).
#ifdef _WIN32
struct HostCtx {
    CONTEXT* c;
    unsigned long long& gpr(int i) {
        unsigned long long* r[16] = {&c->Rax, &c->Rcx, &c->Rdx, &c->Rbx, &c->Rsp, &c->Rbp, &c->Rsi, &c->Rdi,
                           &c->R8,  &c->R9,  &c->R10, &c->R11, &c->R12, &c->R13, &c->R14, &c->R15};
        return *r[i];
    }
    uint64_t flags() const { return c->EFlags; }
    void set_flags(uint64_t f) { c->EFlags = DWORD(f); }
    unsigned long long& rip() { return c->Rip; }
    uint32_t* xmm(int i) { return reinterpret_cast<uint32_t*>(&c->Xmm0 + i); }
};
#else
struct HostCtx {
    ucontext_t* uc;
    unsigned long long& gpr(int i) {
        static const int map[16] = {REG_RAX, REG_RCX, REG_RDX, REG_RBX, REG_RSP, REG_RBP, REG_RSI, REG_RDI,
                                    REG_R8,  REG_R9,  REG_R10, REG_R11, REG_R12, REG_R13, REG_R14, REG_R15};
        return reinterpret_cast<unsigned long long&>(uc->uc_mcontext.gregs[map[i]]);
    }
    uint64_t flags() const { return uint64_t(uc->uc_mcontext.gregs[REG_EFL]); }
    void set_flags(uint64_t f) { uc->uc_mcontext.gregs[REG_EFL] = greg_t(f); }
    unsigned long long& rip() { return reinterpret_cast<unsigned long long&>(uc->uc_mcontext.gregs[REG_RIP]); }
    uint32_t* xmm(int i) { return reinterpret_cast<uint32_t*>(&uc->uc_mcontext.fpregs->_xmm[i]); }
};
#endif

const Range* find(uint32_t a) {
    for (const auto& r : g_ranges)
        if (a - r.base < r.size) return &r;
    return nullptr;
}

int greg_index(ZydisRegister r) {
    const ZydisRegister big = ZydisRegisterGetLargestEnclosing(ZYDIS_MACHINE_MODE_LONG_64, r);
    switch (big) {
        case ZYDIS_REGISTER_RAX: return 0;  case ZYDIS_REGISTER_RCX: return 1;
        case ZYDIS_REGISTER_RDX: return 2;  case ZYDIS_REGISTER_RBX: return 3;
        case ZYDIS_REGISTER_RSP: return 4;  case ZYDIS_REGISTER_RBP: return 5;
        case ZYDIS_REGISTER_RSI: return 6;  case ZYDIS_REGISTER_RDI: return 7;
        case ZYDIS_REGISTER_R8: return 8;   case ZYDIS_REGISTER_R9: return 9;
        case ZYDIS_REGISTER_R10: return 10; case ZYDIS_REGISTER_R11: return 11;
        case ZYDIS_REGISTER_R12: return 12; case ZYDIS_REGISTER_R13: return 13;
        case ZYDIS_REGISTER_R14: return 14; case ZYDIS_REGISTER_R15: return 15;
        default: return -1;
    }
}

bool is_high8(ZydisRegister r) {
    return r == ZYDIS_REGISTER_AH || r == ZYDIS_REGISTER_CH || r == ZYDIS_REGISTER_DH || r == ZYDIS_REGISTER_BH;
}

uint64_t get_reg(HostCtx& uc, ZydisRegister r) {
    const int i = greg_index(r);
    const uint64_t v = uc.gpr(i);
    return is_high8(r) ? (v >> 8) & 0xFF : v;
}

void set_reg(HostCtx& uc, ZydisRegister r, uint64_t v) {
    const int i = greg_index(r);
    unsigned long long& g = uc.gpr(i);
    const auto w = ZydisRegisterGetWidth(ZYDIS_MACHINE_MODE_LONG_64, r);
    if (w == 64) g = v;
    else if (w == 32) g = uint32_t(v);  // 32-bit writes zero-extend
    else if (w == 16) g = (g & ~0xFFFFull) | (v & 0xFFFF);
    else if (is_high8(r)) g = (g & ~0xFF00ull) | ((v & 0xFF) << 8);
    else g = (g & ~0xFFull) | (v & 0xFF);
}

void set_flags_from(HostCtx& uc, uint64_t a, uint64_t b, uint64_t r, unsigned bits, bool sub, bool logic) {
    const uint64_t m = bits == 64 ? ~0ull : (1ull << bits) - 1, sign = 1ull << (bits - 1);
    a &= m; b &= m; r &= m;
    uint64_t f = uc.flags() & ~0x8D5ull;
    if (!logic) {
        if (sub ? a < b : r < a) f |= 0x1;
        if (sub ? ((a ^ b) & (a ^ r) & sign) : (~(a ^ b) & (a ^ r) & sign)) f |= 0x800;
        if ((a ^ b ^ r) & 0x10) f |= 0x10;
    }
    if (r == 0) f |= 0x40;
    if (r & sign) f |= 0x80;
    if (!__builtin_parity(r & 0xFF)) f |= 0x4;
    uc.set_flags(f);
}

[[noreturn]] void fatal(const char* why, uint32_t guest, HostCtx& uc) {
    const uint64_t rip = uc.rip();
#ifdef _WIN32
    const char* fn = "?";
#else
    Dl_info di{};
    const char* fn = (dladdr(reinterpret_cast<void*>(rip), &di) && di.dli_sname) ? di.dli_sname : "?";
#endif
    char text[128] = "?";
    ZydisDecodedInstruction in;
    ZydisDecodedOperand ops[ZYDIS_MAX_OPERAND_COUNT];
    if (ZYAN_SUCCESS(ZydisDecoderDecodeFull(&g_dec, reinterpret_cast<void*>(rip), 15, &in, ops))) {
        ZydisFormatter f;
        ZydisFormatterInit(&f, ZYDIS_FORMATTER_STYLE_INTEL);
        ZydisFormatterFormatInstruction(&f, &in, ops, in.operand_count_visible, text, sizeof text, rip, nullptr);
    }
    std::fprintf(stderr, "[mmio] %s: guest address %08x, host insn \"%s\" in %s\n", why, guest, text, fn);
    std::abort();
}

// Emulates the host instruction at the faulting RIP against the MMIO range and moves past it.
void emulate(HostCtx& uc, uint32_t guest) {
    const Range* r = find(guest);
    if (!r) fatal("access to unmapped guest memory", guest, uc);
    const auto rip = reinterpret_cast<const uint8_t*>(uc.rip());
    ZydisDecodedInstruction in;
    ZydisDecodedOperand ops[ZYDIS_MAX_OPERAND_COUNT];
    if (!ZYAN_SUCCESS(ZydisDecoderDecodeFull(&g_dec, rip, 15, &in, ops))) fatal("undecodable host instruction", guest, uc);

    int mem = -1;
    for (int i = 0; i < in.operand_count_visible; ++i)
        if (ops[i].type == ZYDIS_OPERAND_TYPE_MEMORY) mem = i;
    if (mem < 0) fatal("no memory operand", guest, uc);
    const unsigned size = ops[mem].size / 8;
    const int other = mem == 0 ? 1 : 0;
    auto other_value = [&]() -> uint64_t {
        if (ops[other].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) return ops[other].imm.value.u;
        return get_reg(uc, ops[other].reg.value);
    };

    switch (in.mnemonic) {
        case ZYDIS_MNEMONIC_MOV:
            if (mem == 1) set_reg(uc, ops[0].reg.value, r->rd(r->ctx, guest, size));
            else r->wr(r->ctx, guest, uint32_t(other_value()), size);
            break;
        case ZYDIS_MNEMONIC_MOVD: case ZYDIS_MNEMONIC_MOVQ: case ZYDIS_MNEMONIC_MOVSS: {
            // XMM <-> memory (the host compiler sometimes routes 4/8-byte copies through SSE).
            const auto& regop = ops[other];
            if (regop.type != ZYDIS_OPERAND_TYPE_REGISTER || regop.reg.value < ZYDIS_REGISTER_XMM0 ||
                regop.reg.value > ZYDIS_REGISTER_XMM15) {
                if (regop.type == ZYDIS_OPERAND_TYPE_REGISTER && greg_index(regop.reg.value) >= 0) goto gpr_mov;
                fatal("unsupported SSE MMIO operand", guest, uc);
            }
            uint32_t* x = uc.xmm(regop.reg.value - ZYDIS_REGISTER_XMM0);
            if (mem == 1) {  // load
                const bool keep_upper = in.mnemonic == ZYDIS_MNEMONIC_MOVSS && false;
                (void)keep_upper;
                x[0] = r->rd(r->ctx, guest, 4);
                x[1] = size == 8 ? r->rd(r->ctx, guest + 4, 4) : 0;
                x[2] = x[3] = 0;
            } else {
                r->wr(r->ctx, guest, x[0], 4);
                if (size == 8) r->wr(r->ctx, guest + 4, x[1], 4);
            }
            break;
        }
        gpr_mov:
        case ZYDIS_MNEMONIC_MOVZX:
            set_reg(uc, ops[0].reg.value, r->rd(r->ctx, guest, size));
            break;
        case ZYDIS_MNEMONIC_MOVSX: case ZYDIS_MNEMONIC_MOVSXD: {
            const uint32_t v = r->rd(r->ctx, guest, size);
            const int64_t s = size == 1 ? int8_t(v) : size == 2 ? int16_t(v) : int32_t(v);
            set_reg(uc, ops[0].reg.value, uint64_t(s));
            break;
        }
        case ZYDIS_MNEMONIC_CMP: case ZYDIS_MNEMONIC_TEST: {
            const uint64_t m = r->rd(r->ctx, guest, size), o = other_value();
            const uint64_t a = mem == 0 ? m : o, b = mem == 0 ? o : m;
            const bool test = in.mnemonic == ZYDIS_MNEMONIC_TEST;
            set_flags_from(uc, a, b, test ? (a & b) : (a - b), size * 8, true, test);
            break;
        }
        case ZYDIS_MNEMONIC_ADD: case ZYDIS_MNEMONIC_SUB: case ZYDIS_MNEMONIC_AND:
        case ZYDIS_MNEMONIC_OR: case ZYDIS_MNEMONIC_XOR: {
            const uint64_t m = r->rd(r->ctx, guest, size), o = other_value();
            const uint64_t a = mem == 0 ? m : get_reg(uc, ops[0].reg.value), b = mem == 0 ? o : m;
            uint64_t res = 0;
            bool sub = false, logic = false;
            switch (in.mnemonic) {
                case ZYDIS_MNEMONIC_ADD: res = a + b; break;
                case ZYDIS_MNEMONIC_SUB: res = a - b; sub = true; break;
                case ZYDIS_MNEMONIC_AND: res = a & b; logic = true; break;
                case ZYDIS_MNEMONIC_OR: res = a | b; logic = true; break;
                default: res = a ^ b; logic = true; break;
            }
            set_flags_from(uc, a, b, res, size * 8, sub, logic);
            if (mem == 0) r->wr(r->ctx, guest, uint32_t(res), size);
            else set_reg(uc, ops[0].reg.value, res);
            break;
        }
        default:
            fatal("unsupported host instruction for MMIO", guest, uc);
    }
    uc.rip() += in.length;
}

#ifdef _WIN32
LONG CALLBACK veh(EXCEPTION_POINTERS* ep) {
    if (ep->ExceptionRecord->ExceptionCode != EXCEPTION_ACCESS_VIOLATION || ep->ExceptionRecord->NumberParameters < 2)
        return EXCEPTION_CONTINUE_SEARCH;
    const auto host = reinterpret_cast<uint8_t*>(ep->ExceptionRecord->ExceptionInformation[1]);
    if (!g_mem || host < g_mem || host >= g_mem + (uint64_t(1) << 32)) return EXCEPTION_CONTINUE_SEARCH;
    HostCtx uc{ep->ContextRecord};
    emulate(uc, uint32_t(host - g_mem));
    return EXCEPTION_CONTINUE_EXECUTION;
}
#else
void handler(int sig, siginfo_t* si, void* vuc) {
    const auto host = reinterpret_cast<uint8_t*>(si->si_addr);
    if (!g_mem || host < g_mem || host >= g_mem + (uint64_t(1) << 32)) {
        signal(sig, SIG_DFL);  // not a guest access: crash normally
        return;
    }
    HostCtx uc{static_cast<ucontext_t*>(vuc)};
    emulate(uc, uint32_t(host - g_mem));
}
#endif

}  // namespace

std::vector<Range> g_ports;

void register_ports(uint32_t base, uint32_t count, const char* name, ReadFn rd, WriteFn wr, void* ctx) {
    g_ports.push_back(Range{base, count, name, rd, wr, ctx});
}

static const Range* find_port(uint32_t p) {
    for (const auto& r : g_ports)
        if (p - r.base < r.size) return &r;
    return nullptr;
}

static void unknown_port(const char* what, uint32_t port, uint32_t value) {
    static std::mutex m;
    static std::set<uint32_t> seen;
    std::lock_guard lk(m);
    if (seen.insert(port).second) std::fprintf(stderr, "[io] unhandled port %s %#x (%#x)\n", what, port, value);
}

void register_range(uint32_t base, uint32_t size, const char* name, ReadFn rd, WriteFn wr, void* ctx) {
    g_ranges.push_back(Range{base, size, name, rd, wr, ctx});
}

void install() {
    ZydisDecoderInit(&g_dec, ZYDIS_MACHINE_MODE_LONG_64, ZYDIS_STACK_WIDTH_64);
#ifdef _WIN32
    AddVectoredExceptionHandler(1, veh);
#else
    struct sigaction sa{};
    sa.sa_sigaction = handler;
    sa.sa_flags = SA_SIGINFO | SA_NODEFER;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGSEGV, &sa, nullptr);
    sigaction(SIGBUS, &sa, nullptr);
#endif
}

}  // namespace vc3::mmio

namespace vc3 {

u32 port_in(Cpu&, u32 port, unsigned size) {
    if (const auto* r = mmio::find_port(port)) return r->rd(r->ctx, port, size);
    mmio::unknown_port("read", port, 0);
    return size == 1 ? 0xFF : size == 2 ? 0xFFFF : 0xFFFFFFFF;
}

void port_out(Cpu&, u32 port, u32 value, unsigned size) {
    if (const auto* r = mmio::find_port(port)) return r->wr(r->ctx, port, value, size);
    mmio::unknown_port("write", port, value);
}

}  // namespace vc3
