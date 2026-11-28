#include <time.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "kernel/ke.h"
#include "kernel/kernel.h"
#include "vc3/helpers.h"
#include "vc3/recomp.h"

namespace vc3 {

GuestFunc lookup_function(uint32_t addr) {
    const FuncEntry* end = vc3_func_table + vc3_func_table_size;
    const FuncEntry* it = std::lower_bound(vc3_func_table, end, addr,
                                           [](const FuncEntry& e, uint32_t a) { return e.addr < a; });
    return (it != end && it->addr == addr) ? it->fn : nullptr;
}

static void dump_regs(const Cpu& c) {
    std::fprintf(stderr, "         eax=%08x ecx=%08x edx=%08x ebx=%08x esp=%08x ebp=%08x esi=%08x edi=%08x\n",
                 c.eax, c.ecx, c.edx, c.ebx, c.esp, c.ebp, c.esi, c.edi);
}

[[noreturn]] static void unknown_target(Cpu& c, uint32_t target, const char* how) {
    std::fprintf(stderr, "[dispatch] %s to unknown guest address %08x (return address on stack %08x)\n", how,
                 target, rd<uint32_t>(c.esp));
    std::fprintf(stderr, "           if this is real code, add it with: vc3recomp ... --entry 0x%08x\n", target);
    dump_regs(c);
    std::abort();
}

// Shared by call/jmp: the return address is already on the guest stack.
static void enter(Cpu& c, uint32_t target, const char* how) {
    if (target >= kKernelThunkBase) {
        kernel::call(c, target - kKernelThunkBase);
        return;
    }
    if (GuestFunc fn = lookup_function(target)) {
        fn(c);
        return;
    }
    unknown_target(c, target, how);
}

void call_indirect(Cpu& c, uint32_t target, uint32_t ret_addr) {
    push32(c, ret_addr);
    enter(c, target, "call");
}

void jump_indirect(Cpu& c, uint32_t target) { enter(c, target, "jump"); }

void unimplemented(Cpu& c, uint32_t func_addr, const char* file, int line) {
    std::fprintf(stderr, "[recomp] fn_%08x is not lifted yet (%s:%d)\n", func_addr, file, line);
    dump_regs(c);
    std::abort();
}

void unimplemented_insn(Cpu& c, u32 addr, const char* text) {
    std::fprintf(stderr, "[recomp] unlifted instruction at %08x: %s\n", addr, text);
    dump_regs(c);
    std::abort();
}

void guest_trap(Cpu& c, u32 addr, const char* what) {
    std::fprintf(stderr, "[recomp] guest trap at %08x: %s\n", addr, what);
    dump_regs(c);
    std::abort();
}

void debug_service(Cpu& c) {
    if (c.eax == 1 && c.ecx) {  // BREAKPOINT_PRINT: ecx = ANSI_STRING*
        const u16 len = rd<u16>(c.ecx);
        const u32 buf = rd<u32>(c.ecx + 4);
        std::string s(reinterpret_cast<const char*>(host_ptr(buf)), len);
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
        std::fprintf(stderr, "[guest] %s\n", s.c_str());
    }
    c.eax = 0;
}

void rdtsc(Cpu& c) {
    // Chihiro's Pentium III runs at 733 MHz. The host TSC rate is unrelated, so
    // derive a 733 MHz counter from the monotonic clock.
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    u64 t = u64(ts.tv_sec) * 733333333ull + u64(ts.tv_nsec) * 733333333ull / 1000000000ull;
    t -= u64((unsigned __int128)u64(kernel::ke::paused_ns()) * 733333333ull / 1000000000ull);  // (pause menu)
    c.eax = u32(t);
    c.edx = u32(t >> 32);
}

void cpuid(Cpu& c) {
    // Pentium III (Coppermine), family 6 model 8: MMX, SSE, CMOV, FXSR, TSC.
    switch (c.eax) {
        case 0: c.eax = 2; c.ebx = 0x756E6547; c.edx = 0x49656E69; c.ecx = 0x6C65746E; break;  // "GenuineIntel"
        case 1: c.eax = 0x00000683; c.ebx = 0; c.ecx = 0; c.edx = 0x0383F9FF; break;
        default: c.eax = c.ebx = c.ecx = c.edx = 0; break;
    }
}

}  // namespace vc3
