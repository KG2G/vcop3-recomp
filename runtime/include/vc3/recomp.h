// Interface between generated code and the runtime.
//
// Conventions
// -----------
// * Every recompiled guest function is `void fn_XXXXXXXX(vc3::Cpu&)`.
// * The guest stack lives in guest memory and keeps its real layout:
//   a guest `call` pushes the guest return address, then calls the host
//   function; a guest `ret n` pops it (and n bytes). Code that reads its
//   arguments at [esp+4] therefore works unchanged.
// * Guest pointers are 32-bit offsets into one 4 GiB host reservation
//   (vc3::g_mem), so a guest address converts to host with g_mem + addr.
// * Indirect calls/jumps go through vc3::call_indirect(), which looks the
//   target up in the generated function table. Kernel imports resolve to
//   sentinel addresses (kKernelThunkBase + ordinal) and route to HLE.
#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace vc3 {

struct Cpu {
    union {
        uint32_t r[8];
        struct { uint32_t eax, ecx, edx, ebx, esp, ebp, esi, edi; };
    };
    // Arithmetic flags kept unpacked; pack/unpack only for pushf/popf/lahf.
    bool cf = false, pf = false, af = false, zf = false, sf = false, of = false, df = false;

    // x87 (MSVC code for the Pentium III still leans heavily on it).
    double st[8] = {};
    unsigned fpu_top = 0;
    uint16_t fpu_cw = 0x027F, fpu_sw = 0, fpu_tw = 0xFFFF;

    // SSE (Pentium III: SSE1 only).
    alignas(16) uint8_t xmm[8][16] = {};
    uint32_t mxcsr = 0x1F80;

    uint64_t mm[8] = {};  // MMX (not aliased onto st[] — code mixing the two is not supported)

    uint32_t fs_base = 0;  // guest KPCR / TIB (fs:[0] SEH chain, fs:[0x28] TLS)
    uint32_t last_eip = 0;  // diagnostic only
};

using GuestFunc = void (*)(Cpu&);

struct FuncEntry {
    uint32_t addr;
    GuestFunc fn;
};

// --- guest memory ---------------------------------------------------------
extern uint8_t* g_mem;

inline uint8_t* host_ptr(uint32_t a) { return g_mem + a; }

template <typename T>
inline T rd(uint32_t a) {
    T v;
    std::memcpy(&v, g_mem + a, sizeof(T));
    return v;
}
template <typename T>
inline void wr(uint32_t a, T v) {
    std::memcpy(g_mem + a, &v, sizeof(T));
}

inline void push32(Cpu& c, uint32_t v) { c.esp -= 4; wr<uint32_t>(c.esp, v); }
inline uint32_t pop32(Cpu& c) { uint32_t v = rd<uint32_t>(c.esp); c.esp += 4; return v; }

// --- control flow ---------------------------------------------------------
constexpr uint32_t kKernelThunkBase = 0xFFFF0000;  // + ordinal

// Call the guest function at `target` as if by `call target` from `ret_addr`.
void call_indirect(Cpu& c, uint32_t target, uint32_t ret_addr);
// Called at the start of functions listed in config/vc3_hooks.txt (runtime/src/hooks.cpp).
void guest_hook(Cpu& c, uint32_t entry);
// Once per guest vblank (video timing thread): cabinet settings applied to game state.
void per_vblank();
// Player p (0, 1) is in a game (playing or at the continue countdown).
bool player_in_game(int p);
GuestFunc lookup_function(uint32_t addr);

[[noreturn]] void unimplemented(Cpu& c, uint32_t func_addr, const char* file, int line);

// Cooperative preemption: checked by generated code on loop back-edges.
extern std::atomic<bool> g_yield_request;
void yield_point(Cpu& c);

}  // namespace vc3

#define VC3_UNIMPLEMENTED(c, addr) ::vc3::unimplemented((c), (addr), __FILE__, __LINE__)
#define VC3_BACKEDGE(c)                                                                  \
    do {                                                                                 \
        if (__builtin_expect(::vc3::g_yield_request.load(std::memory_order_relaxed), 0)) \
            ::vc3::yield_point(c);                                                       \
    } while (0)

// Provided by the generated code (or runtime/src/generated_stub.cpp).
extern const vc3::FuncEntry vc3_func_table[];
extern const size_t vc3_func_table_size;
extern const uint32_t vc3_entry_point;
