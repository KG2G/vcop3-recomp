// Inline helpers used by generated code: sub-registers, flags, string ops,
// x87 stack, SSE/MMX lanes. Everything here must stay header-only so the
// optimizer can fold it into the generated functions.
#pragma once
#include <emmintrin.h>
#include <xmmintrin.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <type_traits>
#include <utility>

#include "vc3/recomp.h"

namespace vc3 {

using u8 = uint8_t;   using s8 = int8_t;
using u16 = uint16_t; using s16 = int16_t;
using u32 = uint32_t; using s32 = int32_t;
using u64 = uint64_t; using s64 = int64_t;

// --- sub-registers --------------------------------------------------------
inline void set16(u32& r, u16 v) { r = (r & 0xFFFF0000u) | v; }
inline void set8l(u32& r, u8 v) { r = (r & 0xFFFFFF00u) | v; }
inline void set8h(u32& r, u8 v) { r = (r & 0xFFFF00FFu) | (u32(v) << 8); }

// --- flags ------------------------------------------------------------------
inline bool parity(u32 r) { return !__builtin_parity(r & 0xFF); }

template <typename T> constexpr unsigned kBits = sizeof(T) * 8;
template <typename T> inline bool msb(T v) { return (v >> (kBits<T> - 1)) & 1; }

template <typename T>
inline void flags_szp(Cpu& c, T r) { c.zf = r == 0; c.sf = msb(r); c.pf = parity(r); }

template <typename T>
inline void flags_sub(Cpu& c, T a, T b) {  // full flags for a - b (cmp/scas/cmps)
    T r = T(a - b);
    c.cf = a < b;
    c.of = msb(T((a ^ b) & (a ^ r)));
    c.af = ((a ^ b ^ r) >> 4) & 1;
    flags_szp(c, r);
}

inline u32 pack_eflags(const Cpu& c) {
    return 0x2u | u32(c.cf) | u32(c.pf) << 2 | u32(c.af) << 4 | u32(c.zf) << 6 | u32(c.sf) << 7 |
           0x200u /*IF*/ | u32(c.df) << 10 | u32(c.of) << 11;
}
inline void unpack_eflags(Cpu& c, u32 f) {
    c.cf = f & 1; c.pf = (f >> 2) & 1; c.af = (f >> 4) & 1; c.zf = (f >> 6) & 1;
    c.sf = (f >> 7) & 1; c.df = (f >> 10) & 1; c.of = (f >> 11) & 1;
}
inline void sahf(Cpu& c) { unpack_eflags(c, (pack_eflags(c) & ~0xFFu) | ((c.eax >> 8) & 0xD5)); }
inline void lahf(Cpu& c) { set8h(c.eax, u8(pack_eflags(c))); }

inline void pushad(Cpu& c) {
    u32 sp = c.esp;
    push32(c, c.eax); push32(c, c.ecx); push32(c, c.edx); push32(c, c.ebx);
    push32(c, sp);    push32(c, c.ebp); push32(c, c.esi); push32(c, c.edi);
}
inline void popad(Cpu& c) {
    c.edi = pop32(c); c.esi = pop32(c); c.ebp = pop32(c); c.esp += 4;
    c.ebx = pop32(c); c.edx = pop32(c); c.ecx = pop32(c); c.eax = pop32(c);
}

// --- string instructions ----------------------------------------------------
enum Rep { kNoRep = 0, kRepE = 1, kRepNE = 2 };

template <typename T>
inline void str_movs(Cpu& c, bool rep) {
    const s32 d = c.df ? -s32(sizeof(T)) : s32(sizeof(T));
    if (rep && !c.df && c.ecx) {
        // A forward element-by-element copy equals memmove unless the destination
        // starts inside the source range (that case replicates a pattern instead).
        const u32 n = c.ecx * u32(sizeof(T));
        if (c.edi <= c.esi || c.edi - c.esi >= n) {
            std::memmove(host_ptr(c.edi), host_ptr(c.esi), n);
            c.esi += n; c.edi += n; c.ecx = 0;
            return;
        }
    }
    do {
        if (rep && c.ecx == 0) break;
        wr<T>(c.edi, rd<T>(c.esi));
        c.esi += d; c.edi += d;
        if (rep) --c.ecx;
    } while (rep);
}

template <typename T>
inline void str_stos(Cpu& c, bool rep) {
    const s32 d = c.df ? -s32(sizeof(T)) : s32(sizeof(T));
    const T v = T(c.eax);
    if (rep && !c.df && sizeof(T) == 1) {
        std::memset(host_ptr(c.edi), v, c.ecx);
        c.edi += c.ecx; c.ecx = 0;
        return;
    }
    do {
        if (rep && c.ecx == 0) break;
        wr<T>(c.edi, v);
        c.edi += d;
        if (rep) --c.ecx;
    } while (rep);
}

template <typename T>
inline void str_lods(Cpu& c, bool rep) {
    const s32 d = c.df ? -s32(sizeof(T)) : s32(sizeof(T));
    do {
        if (rep && c.ecx == 0) break;
        T v = rd<T>(c.esi);
        if constexpr (sizeof(T) == 1) set8l(c.eax, v);
        else if constexpr (sizeof(T) == 2) set16(c.eax, v);
        else c.eax = v;
        c.esi += d;
        if (rep) --c.ecx;
    } while (rep);
}

template <typename T>
inline void str_scas(Cpu& c, Rep rep) {
    const s32 d = c.df ? -s32(sizeof(T)) : s32(sizeof(T));
    do {
        if (rep && c.ecx == 0) break;
        flags_sub<T>(c, T(c.eax), rd<T>(c.edi));
        c.edi += d;
        if (rep) {
            --c.ecx;
            if ((rep == kRepE && !c.zf) || (rep == kRepNE && c.zf)) break;
        }
    } while (rep);
}

template <typename T>
inline void str_cmps(Cpu& c, Rep rep) {
    const s32 d = c.df ? -s32(sizeof(T)) : s32(sizeof(T));
    do {
        if (rep && c.ecx == 0) break;
        flags_sub<T>(c, rd<T>(c.esi), rd<T>(c.edi));
        c.esi += d; c.edi += d;
        if (rep) {
            --c.ecx;
            if ((rep == kRepE && !c.zf) || (rep == kRepNE && c.zf)) break;
        }
    } while (rep);
}

// --- x87 -------------------------------------------------------------------
// Registers are kept as host doubles (MSVC sets 53-bit precision control).
inline double& st(Cpu& c, unsigned i) { return c.st[(c.fpu_top + i) & 7]; }
inline void fpush(Cpu& c, double v) { c.fpu_top = (c.fpu_top - 1) & 7; c.st[c.fpu_top] = v; }
inline void fpop(Cpu& c) { c.fpu_top = (c.fpu_top + 1) & 7; }

inline float rdf32(u32 a) { return rd<float>(a); }
inline double rdf64(u32 a) { return rd<double>(a); }
inline double rdf80(u32 a) { long double v = 0; std::memcpy(&v, host_ptr(a), 10); return double(v); }
inline void wrf32(u32 a, double v) { wr<float>(a, float(v)); }
inline void wrf64(u32 a, double v) { wr<double>(a, v); }
inline void wrf80(u32 a, double v) { long double x = v; std::memcpy(host_ptr(a), &x, 10); }

// fcom/fucom/ftst: C3 (bit 14), C2 (bit 10), C0 (bit 8)
inline void fcom(Cpu& c, double a, double b) {
    u16 cc;
    if (std::isnan(a) || std::isnan(b)) cc = 0x4500;
    else if (a > b) cc = 0;
    else if (a < b) cc = 0x0100;
    else cc = 0x4000;
    c.fpu_sw = u16((c.fpu_sw & ~0x4700) | cc);
}
// fcomi/fucomi/comiss: ZF, PF, CF
inline void fcomi(Cpu& c, double a, double b) {
    c.of = c.sf = c.af = false;
    if (std::isnan(a) || std::isnan(b)) { c.zf = c.pf = c.cf = true; return; }
    c.zf = a == b; c.pf = false; c.cf = a < b;
}
inline u16 fnstsw(const Cpu& c) { return u16((c.fpu_sw & ~0x3800) | ((c.fpu_top & 7) << 11)); }

// Round per the control word's RC field (used by fist/fistp/frndint).
inline double fround(const Cpu& c, double v) {
    switch ((c.fpu_cw >> 10) & 3) {
        case 0: return std::nearbyint(v);  // host default mode is round-to-nearest-even
        case 1: return std::floor(v);
        case 2: return std::ceil(v);
        default: return std::trunc(v);
    }
}
template <typename T>
inline T fist(const Cpu& c, double v) {
    double r = fround(c, v);
    constexpr T kIndefinite = T(T(1) << (sizeof(T) * 8 - 1));
    if (!(r >= -std::ldexp(1.0, sizeof(T) * 8 - 1) && r < std::ldexp(1.0, sizeof(T) * 8 - 1))) return kIndefinite;
    return T(r);
}

inline void fninit(Cpu& c) { c.fpu_cw = 0x037F; c.fpu_sw = 0; c.fpu_tw = 0xFFFF; c.fpu_top = 0; }

// --- SSE / MMX ---------------------------------------------------------------
inline float* xf(Cpu& c, unsigned i) { return reinterpret_cast<float*>(c.xmm[i]); }
inline u32* xu(Cpu& c, unsigned i) { return reinterpret_cast<u32*>(c.xmm[i]); }

struct V4 { float f[4]; };
inline V4 ld_v4(u32 a) { V4 v; std::memcpy(v.f, host_ptr(a), 16); return v; }
inline V4 ld_x(Cpu& c, unsigned i) { V4 v; std::memcpy(v.f, c.xmm[i], 16); return v; }
inline void st_v4(u32 a, const V4& v) { std::memcpy(host_ptr(a), v.f, 16); }
inline void st_x(Cpu& c, unsigned i, const V4& v) { std::memcpy(c.xmm[i], v.f, 16); }

inline V4 shufps(const V4& d, const V4& s, u8 imm) {
    return V4{{d.f[imm & 3], d.f[(imm >> 2) & 3], s.f[(imm >> 4) & 3], s.f[(imm >> 6) & 3]}};
}
inline float rsqrt(float x) { return 1.0f / std::sqrt(x); }

// Mirror the guest MXCSR onto the host thread (exceptions stay masked) so lifted
// SSE arithmetic uses the guest's rounding mode and FTZ/DAZ settings.
inline void ldmxcsr(Cpu& c, u32 v) {
    c.mxcsr = v;
    _mm_setcsr((v & 0xFFFFu) | 0x1F80u);
}

inline s32 cvt_f2i_mxcsr(const Cpu& c, float v) {  // cvtss2si honours MXCSR.RC
    double r;
    switch ((c.mxcsr >> 13) & 3) {
        case 0: r = std::nearbyint(v); break;
        case 1: r = std::floor(v); break;
        case 2: r = std::ceil(v); break;
        default: r = std::trunc(v); break;
    }
    return (r >= -2147483648.0 && r < 2147483648.0) ? s32(r) : s32(0x80000000);
}
inline s32 cvt_f2i_trunc(float v) {
    double r = std::trunc(v);
    return (r >= -2147483648.0 && r < 2147483648.0) ? s32(r) : s32(0x80000000);
}

// --- MMX (64-bit packed integer) -----------------------------------------------
namespace mmx {
template <typename L, typename F>
inline u64 map2(u64 a, u64 b, F f) {
    constexpr unsigned bits = sizeof(L) * 8, n = 64 / bits;
    using U = std::make_unsigned_t<L>;
    u64 r = 0;
    for (unsigned i = 0; i < n; ++i) {
        const L x = L(U(a >> (i * bits))), y = L(U(b >> (i * bits)));
        r |= u64(U(f(x, y))) << (i * bits);
    }
    return r;
}
template <typename L, typename W>
inline L sat(W v) {
    return L(std::clamp<W>(v, W(std::numeric_limits<L>::min()), W(std::numeric_limits<L>::max())));
}
inline u64 shift_count(u64 c, unsigned bits) { return c >= bits ? bits : c; }
template <typename L>
inline u64 shl(u64 a, u64 c) {
    if (c >= sizeof(L) * 8) return 0;
    return map2<L>(a, 0, [c](L x, L) { return L(std::make_unsigned_t<L>(x) << c); });
}
template <typename L>
inline u64 shr(u64 a, u64 c) {
    if (c >= sizeof(L) * 8) return 0;
    return map2<L>(a, 0, [c](L x, L) { return L(std::make_unsigned_t<L>(x) >> c); });
}
template <typename L>  // L signed
inline u64 sar(u64 a, u64 c) {
    c = shift_count(c, sizeof(L) * 8 - 1);
    return map2<L>(a, 0, [c](L x, L) { return L(x >> c); });
}
template <typename S, typename D>  // pack signed S lanes of a then b into saturated D
inline u64 pack(u64 a, u64 b) {
    constexpr unsigned sb = sizeof(S) * 8, n = 64 / sb, db = sizeof(D) * 8;
    using SU = std::make_unsigned_t<S>;
    using DU = std::make_unsigned_t<D>;
    u64 r = 0;
    for (unsigned i = 0; i < 2 * n; ++i) {
        const u64 src = i < n ? a : b;
        const S v = S(SU(src >> ((i % n) * sb)));
        r |= u64(DU(sat<D, s64>(s64(v)))) << (i * db);
    }
    return r;
}
template <typename L>  // interleave low (hi=false) or high halves
inline u64 unpack(u64 a, u64 b, bool hi) {
    constexpr unsigned bits = sizeof(L) * 8, n = 32 / bits;
    const u64 m = bits == 64 ? ~0ull : (1ull << bits) - 1;
    u64 r = 0;
    for (unsigned i = 0; i < n; ++i) {
        const unsigned src = (hi ? n : 0) + i;
        r |= ((a >> (src * bits)) & m) << (2 * i * bits);
        r |= ((b >> (src * bits)) & m) << ((2 * i + 1) * bits);
    }
    return r;
}
[[gnu::always_inline]] inline u64 pmaddwd(u64 a, u64 b) {
    u64 r = 0;
    for (int i = 0; i < 2; ++i) {
        const s32 s = s32(s16(a >> (32 * i))) * s16(b >> (32 * i)) + s32(s16(a >> (32 * i + 16))) * s16(b >> (32 * i + 16));
        r |= u64(u32(s)) << (32 * i);
    }
    return r;
}
[[gnu::always_inline]] inline u64 psadbw(u64 a, u64 b) {
    u32 s = 0;
    for (int i = 0; i < 8; ++i) s += u32(std::abs(int(u8(a >> (8 * i))) - int(u8(b >> (8 * i)))));
    return s;
}
inline u64 pshufw(u64 a, u8 imm) {
    u64 r = 0;
    for (int i = 0; i < 4; ++i) r |= ((a >> (16 * ((imm >> (2 * i)) & 3))) & 0xFFFF) << (16 * i);
    return r;
}
[[gnu::always_inline]] inline u32 pmovmskb(u64 a) { return u32(_mm_movemask_epi8(_mm_cvtsi64_si128(s64(a)))) & 0xFF; }

// Named MMX operations on SSE2 (each maps to one or two host instructions; the
// lifter emits these). Forced inline: lifted functions can be huge, and GCC stops
// inlining into very large functions otherwise.
#define VC3_MMX [[gnu::always_inline]] inline
VC3_MMX __m128i X(u64 v) { return _mm_cvtsi64_si128(s64(v)); }
VC3_MMX u64 U(__m128i v) { return u64(_mm_cvtsi128_si64(v)); }
VC3_MMX __m128i XX(u64 a, u64 b) { return _mm_unpacklo_epi64(X(a), X(b)); }  // a low, b high
#define VC3_MMX_BIN(name, intr) VC3_MMX u64 name(u64 a, u64 b) { return U(intr(X(a), X(b))); }
VC3_MMX_BIN(paddb, _mm_add_epi8) VC3_MMX_BIN(paddw, _mm_add_epi16) VC3_MMX_BIN(paddd, _mm_add_epi32)
VC3_MMX_BIN(psubb, _mm_sub_epi8) VC3_MMX_BIN(psubw, _mm_sub_epi16) VC3_MMX_BIN(psubd, _mm_sub_epi32)
VC3_MMX_BIN(paddsb, _mm_adds_epi8) VC3_MMX_BIN(paddsw, _mm_adds_epi16)
VC3_MMX_BIN(psubsb, _mm_subs_epi8) VC3_MMX_BIN(psubsw, _mm_subs_epi16)
VC3_MMX_BIN(paddusb, _mm_adds_epu8) VC3_MMX_BIN(paddusw, _mm_adds_epu16)
VC3_MMX_BIN(psubusb, _mm_subs_epu8) VC3_MMX_BIN(psubusw, _mm_subs_epu16)
VC3_MMX_BIN(pmullw, _mm_mullo_epi16) VC3_MMX_BIN(pmulhw, _mm_mulhi_epi16) VC3_MMX_BIN(pmulhuw, _mm_mulhi_epu16)
VC3_MMX_BIN(pcmpeqb, _mm_cmpeq_epi8) VC3_MMX_BIN(pcmpeqw, _mm_cmpeq_epi16) VC3_MMX_BIN(pcmpeqd, _mm_cmpeq_epi32)
VC3_MMX_BIN(pcmpgtb, _mm_cmpgt_epi8) VC3_MMX_BIN(pcmpgtw, _mm_cmpgt_epi16) VC3_MMX_BIN(pcmpgtd, _mm_cmpgt_epi32)
VC3_MMX_BIN(pavgb, _mm_avg_epu8) VC3_MMX_BIN(pavgw, _mm_avg_epu16)
VC3_MMX_BIN(pminub, _mm_min_epu8) VC3_MMX_BIN(pmaxub, _mm_max_epu8)
VC3_MMX_BIN(pminsw, _mm_min_epi16) VC3_MMX_BIN(pmaxsw, _mm_max_epi16)
VC3_MMX_BIN(pmaddwd_x, _mm_madd_epi16) VC3_MMX_BIN(psadbw_x, _mm_sad_epu8)
// Shifts take the full 64-bit count; SSE2 shifts by register have the same rules
// (logical: >= width gives 0; arithmetic: saturates to width - 1).
VC3_MMX_BIN(psllw, _mm_sll_epi16) VC3_MMX_BIN(pslld, _mm_sll_epi32)
VC3_MMX_BIN(psrlw, _mm_srl_epi16) VC3_MMX_BIN(psrld, _mm_srl_epi32)
VC3_MMX_BIN(psraw, _mm_sra_epi16) VC3_MMX_BIN(psrad, _mm_sra_epi32)
#undef VC3_MMX_BIN
VC3_MMX u64 packsswb(u64 a, u64 b) { const __m128i v = XX(a, b); return U(_mm_packs_epi16(v, v)); }
VC3_MMX u64 packssdw(u64 a, u64 b) { const __m128i v = XX(a, b); return U(_mm_packs_epi32(v, v)); }
VC3_MMX u64 packuswb(u64 a, u64 b) { const __m128i v = XX(a, b); return U(_mm_packus_epi16(v, v)); }
// Interleaving the full 64-bit inputs gives the low result in the low half and
// the high result in the high half.
VC3_MMX u64 punpcklbw(u64 a, u64 b) { return U(_mm_unpacklo_epi8(X(a), X(b))); }
VC3_MMX u64 punpckhbw(u64 a, u64 b) { return U(_mm_srli_si128(_mm_unpacklo_epi8(X(a), X(b)), 8)); }
VC3_MMX u64 punpcklwd(u64 a, u64 b) { return U(_mm_unpacklo_epi16(X(a), X(b))); }
VC3_MMX u64 punpckhwd(u64 a, u64 b) { return U(_mm_srli_si128(_mm_unpacklo_epi16(X(a), X(b)), 8)); }
VC3_MMX u64 punpckldq(u64 a, u64 b) { return (a & 0xFFFFFFFFull) | (b << 32); }
VC3_MMX u64 punpckhdq(u64 a, u64 b) { return (a >> 32) | (b & 0xFFFFFFFF00000000ull); }
#undef VC3_MMX

inline u64 cvtps2pi(const Cpu& c, float x0, float x1, bool trunc) {
    const s32 a = trunc ? cvt_f2i_trunc(x0) : cvt_f2i_mxcsr(c, x0);
    const s32 b = trunc ? cvt_f2i_trunc(x1) : cvt_f2i_mxcsr(c, x1);
    return u64(u32(a)) | u64(u32(b)) << 32;
}
}  // namespace mmx

// --- misc ---------------------------------------------------------------------
inline u32 bswap32(u32 v) { return __builtin_bswap32(v); }

void rdtsc(Cpu& c);
void debug_service(Cpu& c);  // int 2Dh
u32 port_in(Cpu& c, u32 port, unsigned size);
void port_out(Cpu& c, u32 port, u32 value, unsigned size);
void cpuid(Cpu& c);
[[noreturn]] void unimplemented_insn(Cpu& c, u32 addr, const char* text);
[[noreturn]] void guest_trap(Cpu& c, u32 addr, const char* what);
void jump_indirect(Cpu& c, u32 target);  // tail-jump to a function (return address already on stack)

}  // namespace vc3
