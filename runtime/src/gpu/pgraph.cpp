#include "pgraph.h"
#include "pgraph_internal.h"
#include "gl_backend.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <set>
#include <vector>

#include "vc3/helpers.h"
#include "vc3/memory.h"

namespace vc3::gpu::pgraph {

namespace detail {

Host g_host;

// VC3_FPS profiling counters (ns / counts since last report)
std::atomic<u64> g_ns_state{0}, g_ns_assemble{0}, g_ns_xform{0}, g_xf_gpu{0};
std::atomic<u64> g_ns_clear{0}, g_ns_vertex{0}, g_ns_raster{0}, g_draws{0}, g_pixels{0}, g_xf_verts{0}, g_xf_ff{0};
// Feature usage per draw (VC3_GPU_STATS=1), to find what the renderer is missing.
std::map<std::string, u64> g_feature_use;
struct ScopedNs {
    std::atomic<u64>& acc;
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    ~ScopedNs() { acc += u64(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count()); }
};
const u32* K = nullptr;  // kelvin state
inline u32 k(u32 method) { return K[method / 4]; }
inline float kf(u32 method) { float f; std::memcpy(&f, &K[method / 4], 4); return f; }

u8* ram(u32 phys) { return host_ptr(phys & (mem::kRamSize - 1)); }

void warn_once(const char* what, u32 v) {
    static std::mutex m;
    std::lock_guard lk(m);
    static std::set<std::pair<const char*, u32>> seen;
    if (seen.insert({what, v}).second) std::fprintf(stderr, "[pgraph] unsupported %s %#x\n", what, v);
}

// ================================================================ vertex program
struct Program {
    u32 code[136][4] = {};
    Vec4 c[192] = {};
    u32 load_slot = 0, load_word = 0;
    u32 const_slot = 0, const_word = 0;
} g_vp;

constexpr int kOutPos = 0, kOutD0 = 3, kOutD1 = 4, kOutFog = 5, kOutB0 = 7, kOutB1 = 8, kOutT0 = 9;

inline u32 field(const u32* t, int word, int bit, int len) { return (t[word] >> bit) & ((1u << len) - 1); }

inline float hw_mul(float a, float b) { return (a == 0.0f || b == 0.0f) ? 0.0f : a * b; }

struct VpState {
    Vec4 v[16];
    Vec4 r[12];
    Vec4 o[16];
    int a0 = 0;
};

Vec4 vp_read(const u32* t, VpState& s, int mux, int rreg, int neg_bit_word, int neg_bit, int swz_word, int swz_bit) {
    Vec4 src{};
    switch (mux) {
        case 1: src = rreg == 12 ? s.o[kOutPos] : s.r[rreg < 12 ? rreg : 0]; break;  // R (R12 mirrors oPos)
        case 2: src = s.v[field(t, 1, 9, 4)]; break;                                       // V
        case 3: {                                                                          // C
            int idx = int(field(t, 1, 13, 8));
            if (field(t, 3, 1, 1)) idx += s.a0;
            src = (idx >= 0 && idx < 192) ? g_vp.c[idx] : Vec4{};
            break;
        }
        default: break;
    }
    Vec4 out;
    for (int i = 0; i < 4; ++i) out[size_t(i)] = src[field(t, swz_word, swz_bit + 2 * (3 - i), 2)];
    if (field(t, neg_bit_word, neg_bit, 1))
        for (auto& x : out) x = -x;
    return out;
}

void vp_write(Vec4& dst, const Vec4& val, u32 mask) {  // mask bit3 = x ... bit0 = w
    for (int i = 0; i < 4; ++i)
        if (mask & (8u >> i)) dst[size_t(i)] = val[size_t(i)];
}

void run_program(VpState& s, u32 start) {
    for (u32 pc = start; pc < 136; ++pc) {
        const u32* t = g_vp.code[pc];
        const u32 ilu = field(t, 1, 25, 3), mac = field(t, 1, 21, 4);
        if (ilu || mac) {
            const Vec4 a = vp_read(t, s, int(field(t, 2, 26, 2)), int(field(t, 2, 28, 4)), 1, 8, 1, 0);
            const Vec4 b = vp_read(t, s, int(field(t, 2, 11, 2)), int(field(t, 2, 13, 4)), 2, 25, 2, 17);
            const int c_reg = int((field(t, 2, 0, 2) << 2) | field(t, 3, 30, 2));
            const Vec4 cc = vp_read(t, s, int(field(t, 3, 28, 2)), c_reg, 2, 10, 2, 2);

            Vec4 mres{}, ires{};
            switch (mac) {
                case 1: mres = a; break;                                                   // MOV
                case 2: for (int i = 0; i < 4; ++i) mres[i] = hw_mul(a[i], b[i]); break;   // MUL
                case 3: for (int i = 0; i < 4; ++i) mres[i] = a[i] + cc[i]; break;         // ADD (A + C)
                case 4: for (int i = 0; i < 4; ++i) mres[i] = hw_mul(a[i], b[i]) + cc[i]; break;  // MAD
                case 5: mres.fill(hw_mul(a[0], b[0]) + hw_mul(a[1], b[1]) + hw_mul(a[2], b[2])); break;  // DP3
                case 6: mres.fill(hw_mul(a[0], b[0]) + hw_mul(a[1], b[1]) + hw_mul(a[2], b[2]) + b[3]); break;  // DPH
                case 7: mres.fill(hw_mul(a[0], b[0]) + hw_mul(a[1], b[1]) + hw_mul(a[2], b[2]) + hw_mul(a[3], b[3])); break;  // DP4
                case 8: mres = {1.0f, hw_mul(a[1], b[1]), a[2], b[3]}; break;              // DST
                case 9: for (int i = 0; i < 4; ++i) mres[i] = std::min(a[i], b[i]); break; // MIN
                case 10: for (int i = 0; i < 4; ++i) mres[i] = std::max(a[i], b[i]); break;  // MAX
                case 11: for (int i = 0; i < 4; ++i) mres[i] = a[i] < b[i] ? 1.0f : 0.0f; break;  // SLT
                case 12: for (int i = 0; i < 4; ++i) mres[i] = a[i] >= b[i] ? 1.0f : 0.0f; break;  // SGE
                case 13: break;                                                            // ARL (below)
                default: break;
            }
            const float x = cc[0];
            switch (ilu) {
                case 1: ires = cc; break;  // MOV
                case 2: ires.fill(1.0f / x); break;  // RCP
                case 3: {                            // RCC
                    float r = 1.0f / x;
                    if (std::fabs(r) < 5.42101e-20f) r = std::copysign(5.42101e-20f, r);
                    if (std::fabs(r) > 1.884467e+19f) r = std::copysign(1.884467e+19f, r);
                    ires.fill(r);
                    break;
                }
                case 4: ires.fill(x == 0.0f ? INFINITY : (std::isinf(x) ? 0.0f : 1.0f / std::sqrt(std::fabs(x)))); break;  // RSQ
                case 5: ires = {std::exp2(std::floor(x)), x - std::floor(x), std::exp2(x), 1.0f}; break;  // EXP
                case 6: {  // LOG
                    const float m = std::fabs(x);
                    if (m == 0.0f) ires = {-INFINITY, 1.0f, -INFINITY, 1.0f};
                    else {
                        const float e = std::floor(std::log2(m));
                        ires = {e, m / std::exp2(e), std::log2(m), 1.0f};
                    }
                    break;
                }
                case 7: {  // LIT
                    const float sx = std::max(cc[0], 0.0f), sy = std::max(cc[1], 0.0f);
                    const float sw = std::clamp(cc[3], -(128.0f - 1.0f / 256), 128.0f - 1.0f / 256);
                    ires = {1.0f, sx, sx > 0.0f ? std::exp2(sw * std::log2(sy)) : 0.0f, 1.0f};
                    break;
                }
                default: break;
            }

            const u32 out_r = field(t, 3, 20, 4), mac_mask = field(t, 3, 24, 4), ilu_mask = field(t, 3, 16, 4);
            const u32 o_mask = field(t, 3, 12, 4), orb = field(t, 3, 11, 1), out_addr = field(t, 3, 3, 8);
            const u32 out_mux = field(t, 3, 2, 1);  // 0 MAC, 1 ILU
            auto write_out = [&](const Vec4& val) {
                if (!o_mask) return;
                if (orb == 0) {  // writable constant
                    if (out_addr < 192) vp_write(g_vp.c[out_addr], val, o_mask);
                } else {
                    vp_write(s.o[out_addr & 0xF], val, o_mask);
                }
            };
            auto write_r = [&](u32 reg, const Vec4& val, u32 mask) {
                if (!mask) return;
                if (reg == 12) vp_write(s.o[kOutPos], val, mask);
                else if (reg < 12) vp_write(s.r[reg], val, mask);
            };
            if (mac) {
                if (mac == 13) s.a0 = int(std::floor(a[0] + 0.001f));
                else {
                    write_r(out_r, mres, (ilu && out_r == 1) ? 0 : mac_mask);
                    if (out_mux == 0) write_out(mres);
                }
            }
            if (ilu) {
                write_r(mac ? 1 : out_r, ires, ilu_mask);
                if (out_mux == 1) write_out(ires);
            }
        }
        if (field(t, 3, 0, 1)) break;  // FINAL
    }
}

// ================================================================ vertex fetch

Vec4 g_current[16];  // inline attribute values (SET_VERTEX_DATA*)
std::vector<u32> g_inline;
std::vector<u32> g_indices;
std::vector<std::pair<u32, u32>> g_ranges;  // DRAW_ARRAYS (start, count)
u32 g_prim = 0;

int attr_dwords(u32 fmt) {
    const u32 type = fmt & 0xF, size = (fmt >> 4) & 0xF;
    if (!size) return 0;
    switch (type) {
        case 0: case 4: return 1;                    // UB: 4 bytes
        case 1: case 5: return int((size * 2 + 3) / 4);  // shorts
        case 6: return 1;                            // CMP
        default: return int(size);                   // float
    }
}

Vec4 decode_attr(u32 fmt, const u8* p) {
    const u32 type = fmt & 0xF, size = (fmt >> 4) & 0xF;
    Vec4 v{0, 0, 0, 1};
    const u32 n = size == 7 ? 3 : std::min<u32>(size, 4);
    switch (type) {
        case 0:  // UB_D3D (BGRA)
            v = {p[2] / 255.0f, p[1] / 255.0f, p[0] / 255.0f, p[3] / 255.0f};
            if (n < 4) v[3] = 1.0f;
            break;
        case 4:  // UB_OGL (RGBA)
            for (u32 i = 0; i < n; ++i) v[i] = p[i] / 255.0f;
            break;
        case 1:  // S1: normalised shorts
            for (u32 i = 0; i < n; ++i) { s16 x; std::memcpy(&x, p + 2 * i, 2); v[i] = std::max(x / 32767.0f, -1.0f); }
            break;
        case 5:  // S32K: integer shorts
            for (u32 i = 0; i < n; ++i) { s16 x; std::memcpy(&x, p + 2 * i, 2); v[i] = float(x); }
            break;
        case 6: {  // CMP: 11:11:10 signed normalised
            u32 x; std::memcpy(&x, p, 4);
            v[0] = float(s32(x << 21) >> 21) / 1023.0f;
            v[1] = float(s32(x << 10) >> 21) / 1023.0f;
            v[2] = float(s32(x) >> 22) / 511.0f;
            break;
        }
        default:  // F
            for (u32 i = 0; i < n; ++i) std::memcpy(&v[i], p + 4 * i, 4);
            break;
    }
    return v;
}

// ================================================================ fixed-function T&L
// The NV2A's fixed-function vertex pipeline (Kelvin methods), evaluated per vertex
// on the CPU: skinning with up to 4 model-view matrices, normals, texgen, texture
// matrices, lighting (8 infinite/local/spot lights, colour material) and fog.
// State is snapshotted once per draw by prepare_vertex_state().
FfState g_ff;

Vec4 k4(u32 m) { return {kf(m), kf(m + 4), kf(m + 8), kf(m + 12)}; }
Vec4 k3(u32 m) { return {kf(m), kf(m + 4), kf(m + 8), 0.0f}; }
Mat4 kmat(u32 m) { return {k4(m), k4(m + 16), k4(m + 32), k4(m + 48)}; }
Vec4 mul(const Vec4& v, const Mat4& m) {
    Vec4 r;
    for (int j = 0; j < 4; ++j) r[j] = v[0] * m[j][0] + v[1] * m[j][1] + v[2] * m[j][2] + v[3] * m[j][3];
    return r;
}
float dot3(const Vec4& a, const Vec4& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
float dot4(const Vec4& a, const Vec4& b) { return dot3(a, b) + a[3] * b[3]; }
Vec4 norm3(Vec4 v) {
    const float l = std::sqrt(dot3(v, v));
    if (l > 0) for (int i = 0; i < 3; ++i) v[i] /= l;
    return v;
}

void prepare_vertex_state() {
    FfState& f = g_ff;
    f.program = (k(0x1E94) & 3) == 2;
    f.fog = k(0x2A4) & 1;
    f.fog_mode = k(0x29C);
    f.foggen = k(0x2A0);
    f.fog_p0 = kf(0x9C0);
    f.fog_p1 = kf(0x9C4);
    f.fog_plane = k4(0x9D0);
    f.vp_off = k4(0xA20);
    f.two_sided = k(0x17C4) & 1;
    if (f.program) return;
    f.has_back_diffuse = (k(0x1760 + 4 * 7) >> 4) & 0xF;
    f.has_back_specular = (k(0x1760 + 4 * 8) >> 4) & 0xF;
    static const int kCount[7] = {0, 2, 2, 3, 3, 4, 4};
    const u32 skin = std::min<u32>(k(0x328), 6);
    f.skin_count = kCount[skin];
    f.skin_mix = skin & 1;  // 2G/3G/4G: last weight is 1 - sum(others)
    f.normalize = k(0x3A4) & 1;
    for (int i = 0; i < 4; ++i) {
        f.mv[i] = kmat(0x480 + 64 * i);
        f.imv[i] = kmat(0x580 + 64 * i);
        f.texmat[i] = kmat(0x6C0 + 64 * i);
        f.texmat_en[i] = k(0x420 + 4 * i) & 1;
        for (int c = 0; c < 4; ++c) {
            f.texgen[i][c] = k(0x3C0 + 16 * i + 4 * c);
            f.texplane[i][c] = k4(0x840 + 64 * i + 16 * c);
        }
    }
    f.composite = kmat(0x680);
    f.infinite_viewer = k(0x9CC) == 1;
    f.lighting = k(0x314) & 1;
    f.specular_enable = k(0x3B8) & 1;
    const u32 lc = k(0x294);
    f.separate_specular = lc & 1;
    f.local_eye = lc & (1u << 16);
    if (!f.lighting) return;
    const u32 cm = k(0x298);
    f.emission_src = cm & 3;
    f.ambient_src = (cm >> 2) & 3;
    f.diffuse_src = (cm >> 4) & 3;
    f.specular_src = (cm >> 6) & 3;
    f.scene_ambient = k3(0xA10);
    f.mat_emission = k3(0x3A8);
    f.mat_alpha = kf(0x3B4);
    f.eye_pos = k4(0xA50);
    f.spec_params = k3(0x9E0);
    if (f.spec_params[0] == 0 && f.spec_params[1] == 0 && f.spec_params[2] == 0)
        f.spec_params = {0.0f, -0.494592f, 1.494592f, 0};  // power-up default
    f.back_scene_ambient = k3(0x17A0);
    f.back_alpha = kf(0x17AC);
    f.back_emission = k3(0x17B0);
    f.back_spec_params = k3(0x1E28);
    if (f.back_spec_params[0] == 0 && f.back_spec_params[1] == 0 && f.back_spec_params[2] == 0)
        f.back_spec_params = f.spec_params;
    f.nlights = 0;
    const u32 mask = k(0x3BC);
    for (int i = 0; i < 8; ++i) {
        const int type = int((mask >> (2 * i)) & 3);
        if (!type) continue;
        const u32 b = 0x1000 + 128 * u32(i);
        Light& l = f.lights[f.nlights++];
        l.type = type;
        l.amb = k3(b + 0x00);
        l.dif = k3(b + 0x0C);
        l.spc = k3(b + 0x18);
        l.range = kf(b + 0x24);
        l.half = k3(b + 0x28);
        l.dir = k3(b + 0x34);
        l.falloff = k3(b + 0x40);
        l.spot_dir = k4(b + 0x4C);
        l.pos = k3(b + 0x5C);
        l.atten = k3(b + 0x68);
        const u32 bb = 0xC00 + 64 * u32(i);
        l.bamb = k3(bb + 0x00);
        l.bdif = k3(bb + 0x0C);
        l.bspc = k3(bb + 0x18);
    }
}

// Fog factor from a fog distance, per the Kelvin fog mode (clamped later).
float fog_factor(float d) {
    const FfState& f = g_ff;
    if (std::isinf(d)) return 1.0f;
    float r;
    switch (f.fog_mode) {
        case 0x2601: case 0x804: r = f.fog_p0 + d * f.fog_p1 - 1.0f; break;                        // linear
        case 0x800: case 0x802: r = f.fog_p0 + std::exp2(d * f.fog_p1 * 16.0f) - 1.5f; break;       // exp
        case 0x801: case 0x803: r = f.fog_p0 + std::exp2(-d * d * f.fog_p1 * f.fog_p1 * 32.0f) - 1.5f; break;
        default: r = 1.0f; break;
    }
    if (f.fog_mode == 0x802 || f.fog_mode == 0x803 || f.fog_mode == 0x804) r = std::fabs(r);
    return std::isnan(r) ? 1.0f : r;
}

void fixed_function(VpState& s) {
    const FfState& f = g_ff;
    const Vec4& pos = s.v[0];
    const Vec4& wt = s.v[1];
    const Vec4 nrm{s.v[2][0], s.v[2][1], s.v[2][2], 0.0f};
    const Vec4& diffuse = s.v[3];
    const Vec4& specular = s.v[4];

    // Skinning: eye-space position and normal.
    Vec4 epos{}, enrm{};
    if (f.skin_count == 0) {
        epos = mul(pos, f.mv[0]);
        enrm = mul(nrm, f.imv[0]);
    } else {
        float rest = 1.0f;
        for (int i = 0; i < f.skin_count; ++i) {
            float w;
            if (f.skin_mix && i == f.skin_count - 1) w = rest;
            else { w = wt[i]; rest -= w; }
            const Vec4 p = mul(pos, f.mv[i]), n = mul(nrm, f.imv[i]);
            for (int c = 0; c < 4; ++c) { epos[c] += p[c] * w; enrm[c] += n[c] * w; }
        }
    }
    enrm[3] = 0;
    if (f.normalize) enrm = norm3(enrm);

    // Texture coordinates: texgen per channel, then the optional texture matrix.
    for (int i = 0; i < 4; ++i) {
        Vec4 t = s.v[9 + i];
        bool refl_done = false;
        Vec4 refl{};
        for (int c = 0; c < 4; ++c) {
            switch (f.texgen[i][c]) {
                case 0: break;
                case 0x2400: t[c] = dot4(f.texplane[i][c], epos); break;  // eye linear
                case 0x2401: t[c] = dot4(f.texplane[i][c], pos); break;   // object linear
                case 0x2402:                                              // sphere map
                case 0x8512: {                                            // reflection map
                    if (!refl_done) {
                        const Vec4 u = f.infinite_viewer ? Vec4{0, 0, 1, 0} : norm3(epos);
                        const float d2 = 2.0f * dot3(enrm, u);
                        for (int q = 0; q < 3; ++q) refl[q] = u[q] - d2 * enrm[q];
                        refl_done = true;
                    }
                    if (f.texgen[i][c] == 0x8512) { if (c < 3) t[c] = refl[c]; break; }
                    const float m = 2.0f * std::sqrt(refl[0] * refl[0] + refl[1] * refl[1] + (refl[2] + 1) * (refl[2] + 1));
                    if (c < 2) t[c] = m > 0 ? refl[c] / m + 0.5f : 0.5f;
                    break;
                }
                case 0x8511: if (c < 3) t[c] = enrm[c]; break;  // normal map
                default: warn_once("texgen mode", f.texgen[i][c]); break;
            }
        }
        if (f.texmat_en[i]) t = mul(t, f.texmat[i]);
        s.o[kOutT0 + i] = t;
    }

    // Lighting, for the front side and (two-sided lighting) the back side, which
    // uses its own light colours, material, scene ambient and the negated normal.
    auto light_side = [&](bool back, Vec4& d0, Vec4& d1) {
        d0 = diffuse;
        d1 = specular;
        if (f.lighting) {
            const Vec4 n = back ? Vec4{-enrm[0], -enrm[1], -enrm[2], 0} : enrm;
            const Vec4& scene_ambient = back ? f.back_scene_ambient : f.scene_ambient;
            const Vec4& emission = back ? f.back_emission : f.mat_emission;
            const Vec4& sp = back ? f.back_spec_params : f.spec_params;
            auto src = [&](u32 which) -> Vec4 { return which == 1 ? diffuse : which == 2 ? specular : Vec4{1, 1, 1, 1}; };
            const float alpha = f.diffuse_src == 0 ? (back ? f.back_alpha : f.mat_alpha)
                                                   : f.diffuse_src == 2 ? specular[3] : diffuse[3];
            Vec4 c0{}, c1{0, 0, 0, specular[3]};
            if (f.emission_src == 0 || f.ambient_src == 0) {
                c0 = scene_ambient;
                if (f.emission_src != 0) for (int q = 0; q < 3; ++q) c0[q] += src(f.emission_src)[q] * emission[q];
                else if (f.ambient_src != 0) for (int q = 0; q < 3; ++q) c0[q] += src(f.ambient_src)[q] * emission[q];
            } else {
                const Vec4 e = src(f.emission_src), a = src(f.ambient_src);
                for (int q = 0; q < 3; ++q) c0[q] = e[q] + a[q] * emission[q];
            }
            Vec4 vpeye{0, 0, 0, 0};
            if (f.local_eye) {
                const float ew = f.eye_pos[3] != 0 ? f.eye_pos[3] : 1.0f, pw = epos[3] != 0 ? epos[3] : 1.0f;
                for (int q = 0; q < 3; ++q) vpeye[q] = f.eye_pos[q] / ew - epos[q] / pw;
                vpeye = norm3(vpeye);
            }
            for (int li = 0; li < f.nlights; ++li) {
                const Light& l = f.lights[li];
                float att = 1.0f, ndl, ndh;
                if (l.type == 1) {
                    const Vec4 dir = norm3(l.dir);
                    ndl = std::max(0.0f, dot3(n, dir));
                    if (f.local_eye) {
                        Vec4 h{dir[0] + vpeye[0], dir[1] + vpeye[1], dir[2] + vpeye[2], 0};
                        ndh = std::max(0.0f, dot3(n, norm3(h)));
                    } else {
                        ndh = std::max(0.0f, dot3(n, l.half));
                    }
                } else {
                    const float pw = epos[3] != 0 ? epos[3] : 1.0f;
                    Vec4 vp{l.pos[0] - epos[0] / pw, l.pos[1] - epos[1] / pw, l.pos[2] - epos[2] / pw, 0};
                    const float dist = std::sqrt(dot3(vp, vp));
                    if (dist > l.range) continue;
                    vp = norm3(vp);
                    att = 1.0f / (l.atten[0] + l.atten[1] * dist + l.atten[2] * dist * dist);
                    Vec4 h{vp[0] + vpeye[0], vp[1] + vpeye[1], vp[2] + vpeye[2], 0};
                    ndl = std::max(0.0f, dot3(n, vp));
                    ndh = std::max(0.0f, dot3(n, norm3(h)));
                    if (l.type == 3) {  // spot: rational falloff (s + L) / (M s + N)
                        const float spt = dot3(l.spot_dir, vp) + l.spot_dir[3];
                        if (spt <= 0.0f) att = 0.0f;
                        else if (spt < 1.0f) {
                            float factor = 0.0f;
                            if (spt > -l.falloff[0]) {
                                const float den = l.falloff[1] * spt + l.falloff[2];
                                if (den == 0.0f) factor = 1.0f;
                                else if (den > 0.0f) factor = std::clamp((spt + l.falloff[0]) / den, 0.0f, 1.0f);
                            }
                            att *= factor;
                        }
                    }
                }
                float pf = 0.0f;  // specular power as (x + L) / (M x + N)
                if (ndl > 0.0f && ndh > -sp[0]) {
                    const float den = sp[1] * ndh + sp[2];
                    if (den == 0.0f) pf = 10000.0f;
                    else if (den > 0.0f) pf = std::clamp((ndh + sp[0]) / den, 0.0f, 10000.0f);
                }
                const Vec4& lamb = back ? l.bamb : l.amb;
                const Vec4& ldif = back ? l.bdif : l.dif;
                const Vec4& lspc = back ? l.bspc : l.spc;
                const Vec4 as = src(f.ambient_src), ds = src(f.diffuse_src), ss = src(f.specular_src);
                for (int q = 0; q < 3; ++q) {
                    c0[q] += as[q] * lamb[q] * att + ds[q] * ldif[q] * att * ndl;
                    c1[q] += ss[q] * lspc[q] * att * pf;
                }
            }
            c0[3] = alpha;
            d0 = c0;
            d1 = c1;
        } else if (back) {  // unlit: back colour attributes if the vertex has them
            if (f.has_back_diffuse) d0 = s.v[7];
            if (f.has_back_specular) d1 = s.v[8];
        }
        if (!f.specular_enable) {
            d1 = {0, 0, 0, 1};
        } else if (!f.separate_specular) {
            if (f.lighting) for (int q = 0; q < 3; ++q) d0[q] += d1[q];
            d1 = back && !f.lighting && f.has_back_specular ? s.v[8] : specular;
        }
    };
    Vec4 d0, d1;
    light_side(false, d0, d1);
    s.o[kOutD0] = d0;
    s.o[kOutD1] = d1;
    if (f.two_sided) {
        light_side(true, d0, d1);
        s.o[kOutB0] = d0;
        s.o[kOutB1] = d1;
    }

    // Fog distance.
    float fogd = 0.0f;
    if (f.fog) {
        switch (f.foggen) {
            case 0: fogd = std::clamp(specular[3], 0.0f, 1.0f); break;
            case 1: fogd = std::sqrt(dot3(epos, epos)); break;
            case 2: case 3:
                fogd = dot3(f.fog_plane, epos) + f.fog_plane[3];
                if (f.foggen == 3) fogd = std::fabs(fogd);
                break;
            case 6: fogd = s.v[5][0]; break;
            default: break;
        }
    }
    s.o[kOutFog] = {fogd, fogd, fogd, fogd};

    // Position: skinned vertices are already in eye space and the composite
    // matrix then holds only projection + viewport.
    s.o[kOutPos] = mul(f.skin_count ? epos : pos, f.composite);
}

Vertex transform(VpState& s) {
    for (auto& o : s.o) o = {0, 0, 0, 1};
    s.o[kOutD0] = s.o[kOutD1] = {1, 1, 1, 1};
    const float kUnset = std::numeric_limits<float>::quiet_NaN();
    s.o[kOutB0][0] = s.o[kOutB1][0] = kUnset;  // back colours default to the front ones
    const bool program = g_ff.program;
    if (program) {
        for (auto& r : s.r) r = {0, 0, 0, 0};
        s.a0 = 0;
        run_program(s, k(0x1EA0));
    } else {
        fixed_function(s);
    }
    Vertex out;
    Vec4 p = s.o[kOutPos];
    float w = p[3];
    if (std::fabs(w) < 5.42101e-20f) w = std::copysign(5.42101e-20f, w == 0 ? 1.0f : w);
    if (!program) {  // fixed function leaves clip space: divide and offset
        p[0] = p[0] / w + g_ff.vp_off[0];
        p[1] = p[1] / w + g_ff.vp_off[1];
        p[2] = p[2] / w;
    }
    out.x = p[0];
    out.y = p[1];
    out.z = p[2];
    out.w = w;
    for (int i = 0; i < 4; ++i) {
        out.d0[i] = std::clamp(s.o[kOutD0][i], 0.0f, 1.0f);
        out.d1[i] = std::clamp(s.o[kOutD1][i], 0.0f, 1.0f);
    }
    for (int i = 0; i < 4; ++i) out.t[i] = s.o[kOutT0 + i];
    if (g_ff.two_sided) {
        for (int i = 0; i < 4; ++i) {
            out.b0[i] = std::isnan(s.o[kOutB0][0]) ? out.d0[i] : std::clamp(s.o[kOutB0][i], 0.0f, 1.0f);
            out.b1[i] = std::isnan(s.o[kOutB1][0]) ? out.d1[i] : std::clamp(s.o[kOutB1][i], 0.0f, 1.0f);
        }
    }
    out.fog = g_ff.fog ? fog_factor(s.o[kOutFog][0]) : 1.0f;
    return out;
}

// Fetch vertex `index` from the vertex arrays.
// Vertex array layout, resolved once per draw (DMA objects -> RAM addresses).
struct ArrayAttr {
    u32 fmt = 0, base = 0, stride = 0;
    bool enabled = false;
};
ArrayAttr g_arrays[16];

void prepare_arrays() {
    for (int a = 0; a < 16; ++a) {
        ArrayAttr& at = g_arrays[a];
        at.fmt = k(0x1760 + 4 * a);
        at.enabled = (at.fmt >> 4) & 0xF;
        if (!at.enabled) continue;
        const u32 off = k(0x1720 + 4 * a);
        const u32 dma = (off & 0x80000000u) ? k(0x01A0) : k(0x019C);
        at.base = g_host.dma_base(dma) + (off & 0x7FFFFFFF);
        at.stride = at.fmt >> 8;
    }
}

void DrawState::own_target_bounds(const Vertex* v, size_t n) {
    fb_bounds = false;
    if (!n) return;
    float x0 = 1e30f, y0 = 1e30f, x1 = -1e30f, y1 = -1e30f;
    auto add = [&](float x, float y) {
        x0 = std::min(x0, x); x1 = std::max(x1, x);
        y0 = std::min(y0, y); y1 = std::max(y1, y);
    };
    for (size_t k2 = 0; k2 < n; ++k2) {
        const Vertex& p = v[k2];
        if (!(p.w > 0.0f) || !std::isfinite(p.x) || !std::isfinite(p.y)) return;  // behind the eye: unknown
        add(p.x, p.y);  // where it draws
        for (int i = 0; i < tex_used && i < 4; ++i) {  // where it reads its own target
            if (!tex[i].enabled || tex[i].cube || tex[i].base != surf.color_base) continue;
            const u32 mode = (stage_modes >> (5 * i)) & 0x1F;
            float s = p.t[i][0], t = p.t[i][1];
            if (mode == 1) {  // projective
                if (!(p.t[i][3] > 0.0f)) return;
                s /= p.t[i][3];
                t /= p.t[i][3];
            }
            if (!tex[i].linear) {  // normalised coordinates
                s *= float(tex[i].w);
                t *= float(tex[i].h);
            }
            if (!std::isfinite(s) || !std::isfinite(t)) return;
            add(s, t);
        }
    }
    // An all-zero result is a vertex program the CPU couldn't run: unknown.
    if (x1 - x0 < 1.0f && y1 - y0 < 1.0f) return;
    // Margin for per-pixel offsets (bump-mapped refraction) and filtering.
    constexpr float kMargin = 48.0f;
    fbx0 = x0 - kMargin;
    fby0 = y0 - kMargin;
    fbx1 = x1 + kMargin;
    fby1 = y1 + kMargin;
    fb_bounds = true;
}

Vertex fetch_array_vertex(u32 index) {
    VpState s;
    for (int a = 0; a < 16; ++a) {
        const ArrayAttr& at = g_arrays[a];
        s.v[a] = at.enabled ? decode_attr(at.fmt, ram(at.base + index * at.stride)) : g_current[a];
    }
    return transform(s);
}

// ================================================================ surfaces

u32 swizzle_offset(u32 x, u32 y, u32 log_w, u32 log_h) {
    u32 off = 0, bit = 0;
    for (u32 i = 0; i < std::max(log_w, log_h); ++i) {
        if (i < log_w) off |= ((x >> i) & 1) << bit++;
        if (i < log_h) off |= ((y >> i) & 1) << bit++;
    }
    return off;
}

Surface current_surface() {
    Surface s;
    const u32 fmt = k(0x208);
    s.color_fmt = fmt & 0xF;
    s.zeta_fmt = (fmt >> 4) & 0xF;
    s.swizzled = ((fmt >> 8) & 0xF) == 2;
    const u32 aa = (fmt >> 12) & 0xF;
    s.scale_x = aa >= 1 ? 2 : 1;
    s.scale_y = aa >= 2 ? 2 : 1;
    s.sw_log_w = (fmt >> 16) & 0xFF;
    s.sw_log_h = (fmt >> 24) & 0xFF;
    s.width = int((k(0x200) >> 16) * u32(s.scale_x));
    s.height = int((k(0x204) >> 16) * u32(s.scale_y));
    s.color_pitch = k(0x20C) & 0xFFFF;
    s.zeta_pitch = k(0x20C) >> 16;
    s.color_base = g_host.dma_base(k(0x194)) + k(0x210);
    s.zeta_base = g_host.dma_base(k(0x198)) + k(0x214);
    // No depth buffer bound (null/empty DMA context): never touch zeta memory.
    // D3D binds "no depth buffer" as offset 0 (physical page 0 is never a surface).
    if (!k(0x198) || !g_host.dma_limit(k(0x198)) || (s.zeta_base & (mem::kRamSize - 1)) == 0) s.zeta_fmt = 0;
    if (!k(0x194) || !g_host.dma_limit(k(0x194)) || (s.color_base & (mem::kRamSize - 1)) == 0) s.width = s.height = 0;
    return s;
}

u8* color_ptr(const Surface& s, int x, int y) {
    if (s.swizzled) return ram(s.color_base + swizzle_offset(u32(x), u32(y), s.sw_log_w, s.sw_log_h) * u32(s.color_bpp()));
    return ram(s.color_base + u32(y) * s.color_pitch + u32(x) * u32(s.color_bpp()));
}
u8* zeta_ptr(const Surface& s, int x, int y) {
    const u32 bpp = s.zeta_fmt == 1 ? 2 : 4;
    if (s.swizzled) return ram(s.zeta_base + swizzle_offset(u32(x), u32(y), s.sw_log_w, s.sw_log_h) * bpp);
    return ram(s.zeta_base + u32(y) * s.zeta_pitch + u32(x) * bpp);
}

Vec4 read_color(const Surface& s, int x, int y) {
    const u8* p = color_ptr(s, x, y);
    if (s.color_bpp() == 4) return {p[2] / 255.0f, p[1] / 255.0f, p[0] / 255.0f, p[3] / 255.0f};
    u16 v;
    std::memcpy(&v, p, 2);
    if (s.color_fmt == 3) return {((v >> 11) & 31) / 31.0f, ((v >> 5) & 63) / 63.0f, (v & 31) / 31.0f, 1.0f};
    return {((v >> 10) & 31) / 31.0f, ((v >> 5) & 31) / 31.0f, (v & 31) / 31.0f, 1.0f};
}

void write_color(const Surface& s, int x, int y, const Vec4& c, u32 mask /*argb byte enables*/) {
    u8* p = color_ptr(s, x, y);
    auto q = [](float f, int m) { return u32(std::lround(std::clamp(f, 0.0f, 1.0f) * float(m))); };
    if (s.color_bpp() == 4) {
        if (mask & 1) p[0] = u8(q(c[2], 255));
        if (mask & 2) p[1] = u8(q(c[1], 255));
        if (mask & 4) p[2] = u8(q(c[0], 255));
        if (mask & 8) p[3] = s.color_fmt == 8 ? u8(q(c[3], 255)) : 0xFF;
    } else if (s.color_fmt == 3) {
        const u16 v = u16(q(c[0], 31) << 11 | q(c[1], 63) << 5 | q(c[2], 31));
        std::memcpy(p, &v, 2);
    } else {
        const u16 v = u16(0x8000 | q(c[0], 31) << 10 | q(c[1], 31) << 5 | q(c[2], 31));
        std::memcpy(p, &v, 2);
    }
}

float max_depth(const Surface& s) { return s.zeta_fmt == 1 ? 65535.0f : 16777215.0f; }

u32 read_depth(const Surface& s, int x, int y) {
    const u8* p = zeta_ptr(s, x, y);
    if (s.zeta_fmt == 1) { u16 v; std::memcpy(&v, p, 2); return v; }
    u32 v; std::memcpy(&v, p, 4);
    return v >> 8;
}
void write_depth(const Surface& s, int x, int y, u32 z) {
    u8* p = zeta_ptr(s, x, y);
    if (s.zeta_fmt == 1) { const u16 v = u16(z); std::memcpy(p, &v, 2); return; }
    u32 v; std::memcpy(&v, p, 4);
    v = (v & 0xFF) | (z << 8);
    std::memcpy(p, &v, 4);
}

void clear_surface(u32 flags) {
    ScopedNs prof{g_ns_clear};
    const Surface s = current_surface();
    if (glb::active()) {
        glb::clear(s, flags, k(0x1D98), k(0x1D9C), k(0x1D90), k(0x1D8C));
        return;
    }
    static const bool tr = std::getenv("VC3_TRACE_GPU") != nullptr;
    if (tr)
        std::fprintf(stderr, "[pgraph] clear %02x rect %08x %08x color %08x+%u fmt %x zeta %08x+%u fmt %x (dma %05x lim %x) size %dx%d aa %dx%d swz %d\n",
                     flags, k(0x1D98), k(0x1D9C), s.color_base, s.color_pitch, s.color_fmt, s.zeta_base, s.zeta_pitch,
                     s.zeta_fmt, k(0x198), g_host.dma_limit(k(0x198)), s.width, s.height, s.scale_x, s.scale_y, s.swizzled);
    const int x0 = int(k(0x1D98) & 0xFFF), x1 = int((k(0x1D98) >> 16) & 0xFFF);
    const int y0 = int(k(0x1D9C) & 0xFFF), y1 = int((k(0x1D9C) >> 16) & 0xFFF);
    const int xmax = std::min(x1, s.width - 1), ymax = std::min(y1, s.height - 1);
    const u32 cc = k(0x1D90);
    const u32 zs = k(0x1D8C);
    if (!s.swizzled && s.color_bpp() == 4 && (flags & 0xF0) == 0xF0 && x0 <= xmax) {
        for (int y = y0; y <= ymax; ++y) std::fill_n(reinterpret_cast<u32*>(color_ptr(s, x0, y)), xmax - x0 + 1, cc);
        flags &= ~0xF0u;
    }
    if (!s.swizzled && s.zeta_fmt == 2 && (flags & 3) == 3 && x0 <= xmax) {
        for (int y = y0; y <= ymax; ++y) std::fill_n(reinterpret_cast<u32*>(zeta_ptr(s, x0, y)), xmax - x0 + 1, zs);
        flags &= ~3u;
    }
    if (!(flags & 0xF3)) return;
    for (int y = y0; y <= ymax; ++y)
        for (int x = x0; x <= xmax; ++x) {
            if (flags & 0xF0) {
                u8* p = color_ptr(s, x, y);
                if (s.color_bpp() == 4) {
                    for (int b = 0; b < 4; ++b)
                        if (flags & (0x10u << b)) p[b] = u8(cc >> (8 * b));  // B G R A order matches flag bits
                } else {
                    const Vec4 c{((cc >> 16) & 0xFF) / 255.0f, ((cc >> 8) & 0xFF) / 255.0f, (cc & 0xFF) / 255.0f, 1};
                    write_color(s, x, y, c, 0xF);
                }
            }
            if ((flags & 3) && s.zeta_fmt) {
                u8* p = zeta_ptr(s, x, y);
                if (s.zeta_fmt == 1) { const u16 v = u16(zs); std::memcpy(p, &v, 2); }
                else {
                    u32 v; std::memcpy(&v, p, 4);
                    if (flags & 1) v = (v & 0xFF) | (zs & 0xFFFFFF00u);
                    if (flags & 2) v = (v & 0xFFFFFF00u) | (zs & 0xFF);
                    std::memcpy(p, &v, 4);
                }
            }
        }
}

// ================================================================ textures

bool format_linear(u32 f) {
    return (f >= 0x10 && f <= 0x13) || f == 0x17 || (f >= 0x1B && f <= 0x20) || f == 0x24 || f == 0x25 ||
           (f >= 0x2E && f <= 0x31) || f == 0x35 || f >= 0x3F;
}

Texture texture_state(int i) {
    const u32 o = 0x40u * u32(i);
    Texture t;
    t.enabled = (k(0x1B0C + o) >> 30) & 1;
    if (!t.enabled) return t;
    const u32 f = k(0x1B04 + o);
    const u32 dma = (f & 3) == 2 ? k(0x0188) : k(0x0184);
    t.base = g_host.dma_base(dma) + k(0x1B00 + o);
    t.fmt = (f >> 8) & 0xFF;
    t.linear = format_linear(t.fmt);
    t.cube = f & 4;
    t.levels = std::max(1u, (f >> 16) & 0xF);
    if (t.linear) {
        t.w = k(0x1B1C + o) >> 16;
        t.h = k(0x1B1C + o) & 0xFFFF;
        t.pitch = k(0x1B10 + o) >> 16;
    } else {
        t.log_w = (f >> 20) & 0xF;
        t.log_h = (f >> 24) & 0xF;
        t.w = 1u << t.log_w;
        t.h = 1u << t.log_h;
    }
    t.w = std::max(t.w, 1u);
    t.h = std::max(t.h, 1u);
    t.addr_u = k(0x1B08 + o) & 0xF;
    t.addr_v = (k(0x1B08 + o) >> 8) & 0xF;
    const u32 filt = k(0x1B14 + o);
    t.filter_linear = ((filt >> 24) & 0xF) != 1;  // mag filter: 1 = nearest
    t.palette = k(0x1B20 + o);
    t.border = k(0x1B24 + o);
    // Filter 0x1B14: bits 0-12 LOD bias (signed, 1/256 level), 16-21 min filter, 24-27 mag.
    // Control 0x1B0C: bits 6-17 max LOD clamp, 18-29 min LOD clamp (1/256 level).
    t.min_filter = (filt >> 16) & 0x3F;
    t.lod_bias = float(int32_t(filt << 19) >> 19) / 256.0f;
    const u32 ctl = k(0x1B0C + o);
    t.max_lod = float((ctl >> 6) & 0xFFF) / 256.0f;
    t.min_lod = float((ctl >> 18) & 0xFFF) / 256.0f;
    return t;
}


u32 rgb565(u16 v) {
    const u32 r = (v >> 11) & 31, g = (v >> 5) & 63, b = v & 31;
    return 0xFF000000u | ((r << 3 | r >> 2) << 16) | ((g << 2 | g >> 4) << 8) | (b << 3 | b >> 2);
}

u32 dxt_texel(const Texture& t, u32 x, u32 y) {
    const u32 bx = x / 4, by = y / 4, blocks_w = std::max(1u, (t.w + 3) / 4);
    const u32 block_size = t.fmt == 0x0C ? 8 : 16;
    const u8* b = ram(t.base + (by * blocks_w + bx) * block_size);
    const u8* cb = t.fmt == 0x0C ? b : b + 8;
    u16 c0, c1;
    std::memcpy(&c0, cb, 2);
    std::memcpy(&c1, cb + 2, 2);
    u32 idx; std::memcpy(&idx, cb + 4, 4);
    const u32 sel = (idx >> (2 * ((y & 3) * 4 + (x & 3)))) & 3;
    const u32 a = rgb565(c0), bb = rgb565(c1);
    auto mix = [](u32 p, u32 q, int wp, int wq, int d) {
        u32 r = 0;
        for (int s = 0; s < 24; s += 8) r |= (((((p >> s) & 0xFF) * u32(wp)) + (((q >> s) & 0xFF) * u32(wq))) / u32(d)) << s;
        return r;
    };
    u32 col;
    if (sel == 0) col = a & 0xFFFFFF;
    else if (sel == 1) col = bb & 0xFFFFFF;
    else if (c0 > c1 || t.fmt != 0x0C) col = sel == 2 ? mix(a, bb, 2, 1, 3) : mix(a, bb, 1, 2, 3);
    else col = sel == 2 ? mix(a, bb, 1, 1, 2) : 0;
    u32 alpha = 0xFF;
    if (t.fmt == 0x0C) alpha = (c0 <= c1 && sel == 3) ? 0 : 0xFF;
    else if (t.fmt == 0x0E) {  // DXT2/3: explicit 4-bit alpha
        u64 al; std::memcpy(&al, b, 8);
        alpha = u32((al >> (4 * ((y & 3) * 4 + (x & 3)))) & 0xF) * 17;
    } else {  // DXT4/5: interpolated alpha
        const u32 a0 = b[0], a1 = b[1];
        u64 bits = 0; std::memcpy(&bits, b + 2, 6);
        const u32 ai = u32((bits >> (3 * ((y & 3) * 4 + (x & 3)))) & 7);
        if (ai == 0) alpha = a0;
        else if (ai == 1) alpha = a1;
        else if (a0 > a1) alpha = ((8 - ai) * a0 + (ai - 1) * a1) / 7;
        else if (ai == 6) alpha = 0;
        else if (ai == 7) alpha = 255;
        else alpha = ((6 - ai) * a0 + (ai - 1) * a1) / 5;
    }
    return (alpha << 24) | col;
}

u32 fetch_texel(const Texture& t, u32 x, u32 y) {
    if (t.fmt == 0x0C || t.fmt == 0x0E || t.fmt == 0x0F) return dxt_texel(t, x, y);
    int bpp;
    switch (t.fmt) {
        case 0x00: case 0x01: case 0x0B: case 0x13: case 0x19: case 0x1B: case 0x1F: bpp = 1; break;
        case 0x02: case 0x03: case 0x04: case 0x05: case 0x10: case 0x11: case 0x17: case 0x1A: case 0x1C: case 0x1D:
        case 0x20: case 0x27: case 0x28: case 0x29: case 0x2C: case 0x30: case 0x31: case 0x32: case 0x35:
            bpp = 2; break;
        default: bpp = 4; break;
    }
    const u32 off = t.linear ? y * t.pitch + x * u32(bpp) : swizzle_offset(x, y, t.log_w, t.log_h) * u32(bpp);
    const u8* p = ram(t.base + off);
    u32 v = 0;
    std::memcpy(&v, p, size_t(bpp));
    auto x5 = [](u32 c) { return (c << 3) | (c >> 2); };
    auto x4 = [](u32 c) { return c * 17; };
    switch (t.fmt) {
        case 0x06: case 0x12: return v;                                    // A8R8G8B8
        case 0x07: case 0x1E: return v | 0xFF000000u;                      // X8R8G8B8
        case 0x3A: case 0x3F: return (v & 0xFF00FF00u) | ((v & 0xFF) << 16) | ((v >> 16) & 0xFF);  // A8B8G8R8
        case 0x3B: case 0x40: return __builtin_bswap32(v);                 // B8G8R8A8
        case 0x3C: case 0x41: return (v >> 8) | (v << 24);                 // R8G8B8A8
        case 0x05: case 0x11: return rgb565(u16(v));
        case 0x02: case 0x10:
            return (v & 0x8000 ? 0xFF000000u : 0) | x5((v >> 10) & 31) << 16 | x5((v >> 5) & 31) << 8 | x5(v & 31);
        case 0x03: case 0x1C: return 0xFF000000u | x5((v >> 10) & 31) << 16 | x5((v >> 5) & 31) << 8 | x5(v & 31);
        case 0x04: case 0x1D: return x4(v >> 12) << 24 | x4((v >> 8) & 15) << 16 | x4((v >> 4) & 15) << 8 | x4(v & 15);
        case 0x00: case 0x13: return 0xFF000000u | v * 0x010101u;          // Y8 (luminance)
        case 0x01: case 0x1B: return v * 0x01010101u;                      // AY8
        case 0x19: case 0x1F: return v << 24;                              // A8
        case 0x1A: case 0x20: return ((v >> 8) << 24) | (v & 0xFF) * 0x010101u;  // A8Y8
        case 0x0B: {                                                       // I8 palette
            const u32 pal = g_host.dma_base((t.palette & 3) == 2 ? k(0x0188) : k(0x0184)) + (t.palette & ~0x3Fu);
            return rd<u32>((pal + v * 4) & (mem::kRamSize - 1));
        }
        case 0x28: case 0x17: return 0xFF000000u | ((v >> 8) & 0xFF) << 8 | (v & 0xFF);  // G8B8
        default:
            warn_once("texture format", t.fmt);
            return 0xFFFF00FFu;
    }
}

int wrap_coord(int c, int size, u32 mode) {
    switch (mode) {
        case 1: c %= size; return c < 0 ? c + size : c;                  // wrap
        case 2: {                                                          // mirror
            int p = c % (2 * size);
            if (p < 0) p += 2 * size;
            return p < size ? p : 2 * size - 1 - p;
        }
        case 4: return (c < 0 || c >= size) ? -1 : c;                     // border
        default: return std::clamp(c, 0, size - 1);                       // clamp
    }
}

Vec4 sample(const Texture& t, float s, float tt) {
    // Swizzled textures use normalised coordinates, linear ("rect") ones use texels.
    float u = t.linear ? s : s * float(t.w);
    float v = t.linear ? tt : tt * float(t.h);
    // Fast path: linear 32-bit texture (full-screen copies, UI): direct row access.
    const bool fast32 = t.linear && (t.fmt == 0x12 || t.fmt == 0x1E);
    const u8* base = fast32 ? ram(t.base) : nullptr;
    const u32 alpha_or = t.fmt == 0x1E ? 0xFF000000u : 0;
    auto texel = [&](int x, int y) -> Vec4 {
        x = wrap_coord(x, int(t.w), t.addr_u);
        y = wrap_coord(y, int(t.h), t.addr_v);
        if (x < 0 || y < 0) return argb(t.border);
        if (fast32) {
            u32 v;
            std::memcpy(&v, base + u32(y) * t.pitch + u32(x) * 4, 4);
            return argb(v | alpha_or);
        }
        return argb(fetch_texel(t, u32(x), u32(y)));
    };
    if (!t.filter_linear) return texel(int(std::floor(u)), int(std::floor(v)));
    u -= 0.5f;
    v -= 0.5f;
    const int x0 = int(std::floor(u)), y0 = int(std::floor(v));
    const float fx = u - float(x0), fy = v - float(y0);
    const Vec4 a = texel(x0, y0), b = texel(x0 + 1, y0), c = texel(x0, y0 + 1), d = texel(x0 + 1, y0 + 1);
    Vec4 r;
    for (int i = 0; i < 4; ++i) r[i] = (a[i] * (1 - fx) + b[i] * fx) * (1 - fy) + (c[i] * (1 - fx) + d[i] * fx) * fy;
    return r;
}

// ================================================================ register combiners

float map_input(float x, u32 mapping) {
    switch (mapping) {
        case 0: return std::max(0.0f, x);
        case 1: return 1.0f - std::clamp(x, 0.0f, 1.0f);
        case 2: return 2.0f * std::max(0.0f, x) - 1.0f;
        case 3: return 1.0f - 2.0f * std::max(0.0f, x);
        case 4: return std::max(0.0f, x) - 0.5f;
        case 5: return 0.5f - std::max(0.0f, x);
        case 6: return x;
        default: return -x;
    }
}

struct CombinerRegs {
    Rgba r[16];  // indexed by register number
};

// Input byte: bits 0-3 register, bit 4 alpha replicate (blue for alpha portion), bits 5-7 mapping.
std::array<float, 3> get_rgb(const CombinerRegs& cr, u32 in) {
    const Rgba& v = cr.r[in & 0xF];
    const u32 map = (in >> 5) & 7;
    if (in & 0x10) { const float a = map_input(v.a, map); return {a, a, a}; }
    return {map_input(v.r, map), map_input(v.g, map), map_input(v.b, map)};
}
float get_alpha(const CombinerRegs& cr, u32 in) {
    const Rgba& v = cr.r[in & 0xF];
    return map_input((in & 0x10) ? v.a : v.b, (in >> 5) & 7);
}

float out_map(float x, u32 m) {
    switch (m) {
        case 0x08: return x - 0.5f;
        case 0x10: return x * 2;
        case 0x18: return (x - 0.5f) * 2;
        case 0x20: return x * 4;
        case 0x30: return x * 0.5f;
        default: return x;
    }
}


Rgba to_rgba(u32 v) { const Vec4 c = argb(v); return {c[0], c[1], c[2], c[3]}; }

CombinerSetup make_combiner_setup() {
    CombinerSetup cs;
    const u32 ctl = k(0x1E60);
    cs.stages = std::min(ctl & 0xFF, 8u);
    cs.mux_msb = (ctl >> 8) & 1;
    const bool unique_c0 = (ctl >> 12) & 1, unique_c1 = (ctl >> 16) & 1;
    for (u32 st = 0; st < cs.stages; ++st) {
        auto& x = cs.st[st];
        x.ci = k(0xAC0 + 4 * st);
        x.ai = k(0x260 + 4 * st);
        x.co = k(0x1E40 + 4 * st);
        x.ao = k(0xAA0 + 4 * st);
        x.c0 = to_rgba(k(0xA60 + 4 * (unique_c0 ? st : 0)));
        x.c1 = to_rgba(k(0xA80 + 4 * (unique_c1 ? st : 0)));
    }
    cs.f0 = k(0x288);
    cs.f1 = k(0x28C);
    cs.final = cs.f0 || cs.f1;
    cs.fc0 = to_rgba(k(0x1E20));
    cs.fc1 = to_rgba(k(0x1E24));
    return cs;
}

Vec4 run_combiners(CombinerRegs& cr, const CombinerSetup& cs) {
    const bool mux_msb = cs.mux_msb;
    for (u32 st = 0; st < cs.stages; ++st) {
        cr.r[1] = cs.st[st].c0;
        cr.r[2] = cs.st[st].c1;
        const u32 ci = cs.st[st].ci, ai = cs.st[st].ai;
        const u32 co = cs.st[st].co, ao = cs.st[st].ao;
        // RGB portion
        const auto A = get_rgb(cr, ci >> 24), B = get_rgb(cr, (ci >> 16) & 0xFF), C = get_rgb(cr, (ci >> 8) & 0xFF),
                   D = get_rgb(cr, ci & 0xFF);
        const u32 cflags = co >> 12;
        std::array<float, 3> ab, cd, sum;
        if (cflags & 2) { const float d = A[0] * B[0] + A[1] * B[1] + A[2] * B[2]; ab = {d, d, d}; }
        else ab = {A[0] * B[0], A[1] * B[1], A[2] * B[2]};
        if (cflags & 1) { const float d = C[0] * D[0] + C[1] * D[1] + C[2] * D[2]; cd = {d, d, d}; }
        else cd = {C[0] * D[0], C[1] * D[1], C[2] * D[2]};
        const float r0a = cr.r[12].a;
        const bool mux_sel = mux_msb ? r0a >= 0.5f : (u32(std::lround(r0a * 255)) & 1);
        for (int i = 0; i < 3; ++i) sum[i] = (cflags & 4) ? (mux_sel ? cd[i] : ab[i]) : ab[i] + cd[i];
        // Alpha portion
        const float aA = get_alpha(cr, ai >> 24), aB = get_alpha(cr, (ai >> 16) & 0xFF), aC = get_alpha(cr, (ai >> 8) & 0xFF),
                    aD = get_alpha(cr, ai & 0xFF);
        const u32 aflags = ao >> 12;
        const float aab = aA * aB, acd = aC * aD;
        const float asum = (aflags & 4) ? (mux_sel ? acd : aab) : aab + acd;
        // Write back (all outputs computed from the inputs above)
        auto clampf = [](float x) { return std::clamp(x, -1.0f, 1.0f); };
        const u32 cm = cflags & 0x38, am = aflags & 0x38;
        auto wr_rgb = [&](u32 reg, const std::array<float, 3>& v, bool blue_to_alpha) {
            if (!reg) return;
            Rgba& d = cr.r[reg];
            d.r = clampf(out_map(v[0], cm));
            d.g = clampf(out_map(v[1], cm));
            d.b = clampf(out_map(v[2], cm));
            if (blue_to_alpha) d.a = d.b;
        };
        auto wr_a = [&](u32 reg, float v) {
            if (reg) cr.r[reg].a = clampf(out_map(v, am));
        };
        wr_rgb(co & 0xF, cd, cflags & 0x40);
        wr_rgb((co >> 4) & 0xF, ab, cflags & 0x80);
        wr_rgb((co >> 8) & 0xF, sum, false);
        if (!(cflags & 0x40)) wr_a(ao & 0xF, acd);
        if (!(cflags & 0x80)) wr_a((ao >> 4) & 0xF, aab);
        wr_a((ao >> 8) & 0xF, asum);
    }

    const u32 f0 = cs.f0, f1 = cs.f1;
    if (!cs.final) {
        const Rgba& r0 = cr.r[12];
        return {r0.r, r0.g, r0.b, r0.a};
    }
    // Final combiner: rgb = A*B + (1-A)*C + D, alpha = G. Extra inputs: E*F (15), V1+R0 (14).
    cr.r[1] = cs.fc0;
    cr.r[2] = cs.fc1;
    const u32 flags = f1 & 0xFF;
    const auto E = get_rgb(cr, f1 >> 24), F = get_rgb(cr, (f1 >> 16) & 0xFF);
    cr.r[15] = {E[0] * F[0], E[1] * F[1], E[2] * F[2], 0};
    {
        Rgba v1 = cr.r[5], r0 = cr.r[12];
        if (flags & 0x40) { v1.r = 1 - v1.r; v1.g = 1 - v1.g; v1.b = 1 - v1.b; }
        if (flags & 0x20) { r0.r = 1 - r0.r; r0.g = 1 - r0.g; r0.b = 1 - r0.b; }
        Rgba s{v1.r + r0.r, v1.g + r0.g, v1.b + r0.b, 0};
        if (flags & 0x80) { s.r = std::min(s.r, 1.0f); s.g = std::min(s.g, 1.0f); s.b = std::min(s.b, 1.0f); }
        cr.r[14] = s;
    }
    const auto A = get_rgb(cr, f0 >> 24), B = get_rgb(cr, (f0 >> 16) & 0xFF), C = get_rgb(cr, (f0 >> 8) & 0xFF),
               D = get_rgb(cr, f0 & 0xFF);
    const float G = get_alpha(cr, (f1 >> 8) & 0xFF);
    Vec4 out;
    for (int i = 0; i < 3; ++i) out[i] = std::clamp(A[i] * B[i] + (1 - A[i]) * C[i] + D[i], 0.0f, 1.0f);
    out[3] = std::clamp(G, 0.0f, 1.0f);
    return out;
}

// ================================================================ fragment pipeline
bool compare(u32 func, float a, float b) {
    switch (func & 7) {
        case 0: return false;
        case 1: return a < b;
        case 2: return a == b;
        case 3: return a <= b;
        case 4: return a > b;
        case 5: return a != b;
        case 6: return a >= b;
        default: return true;
    }
}

float blend_factor(u32 f, const Vec4& src, const Vec4& dst, const Vec4& cst, int i) {
    switch (f) {
        case 0x0000: return 0;
        case 0x0001: return 1;
        case 0x0300: return src[i];
        case 0x0301: return 1 - src[i];
        case 0x0302: return src[3];
        case 0x0303: return 1 - src[3];
        case 0x0304: return dst[3];
        case 0x0305: return 1 - dst[3];
        case 0x0306: return dst[i];
        case 0x0307: return 1 - dst[i];
        case 0x0308: return i == 3 ? 1.0f : std::min(src[3], 1 - dst[3]);
        case 0x8001: return cst[i];
        case 0x8002: return 1 - cst[i];
        case 0x8003: return cst[3];
        case 0x8004: return 1 - cst[3];
        default: warn_once("blend factor", f); return 1;
    }
}


DrawState make_draw_state() {
    DrawState d;
    d.surf = current_surface();
    for (int i = 0; i < 4; ++i) d.tex[i] = texture_state(i);
    d.stage_modes = k(0x1E70);
    d.dot_mapping = k(0x1E74) & 0xFFF;
    d.other_stage_input = k(0x1E78) & 0xFFFF000;
    for (int i = 1; i < 4; ++i) {  // bump env: method order m00, m01, m11, m10
        const u32 b = 0x1B28 + 64u * u32(i);
        d.bump_mat[i][0] = kf(b);
        d.bump_mat[i][1] = kf(b + 4);
        d.bump_mat[i][2] = kf(b + 12);
        d.bump_mat[i][3] = kf(b + 8);
        d.bump_scale[i] = kf(0x1B38 + 64u * u32(i));
        d.bump_offset[i] = kf(0x1B3C + 64u * u32(i));
    }
    d.comb = make_combiner_setup();
    for (int i = 0; i < 4; ++i)
        if ((d.stage_modes >> (5 * i)) & 0x1F) d.tex_used = i + 1;
    d.alpha_test = k(0x300) & 1;
    d.blend = k(0x304) & 1;
    d.depth_test = (k(0x30C) & 1) && d.surf.zeta_fmt;
    d.depth_write = (k(0x35C) & 1) && d.surf.zeta_fmt;
    d.alpha_func = k(0x33C) & 0xF;
    d.alpha_ref = float(k(0x340) & 0xFF) / 255.0f;
    d.depth_func = k(0x354) & 0xF;
    const u32 cm = k(0x358);
    d.color_mask_bits = ((cm & 0xFF) ? 1 : 0) | ((cm >> 8 & 0xFF) ? 2 : 0) | ((cm >> 16 & 0xFF) ? 4 : 0) | ((cm >> 24) ? 8 : 0);
    d.sfactor = k(0x344);
    d.dfactor = k(0x348);
    d.equation = k(0x350);
    d.blend_color = argb(k(0x34C));
    d.fog = k(0x2A4) & 1;
    d.fog_color = [&] { const u32 c = k(0x2A8); return Vec4{(c & 0xFF) / 255.0f, ((c >> 8) & 0xFF) / 255.0f, ((c >> 16) & 0xFF) / 255.0f, (c >> 24) / 255.0f}; }();
    // Window clip 0 (pixels, inclusive) intersected with the surface.
    d.clip_x0 = 0;
    d.clip_y0 = 0;
    d.clip_x1 = d.surf.width - 1;
    d.clip_y1 = d.surf.height - 1;
    d.cull = k(0x308) & 1;
    d.cull_face = k(0x39C);
    d.front_ccw = k(0x3A0) == 0x901;
    d.point_size = std::max(1.0f, float(k(0x43C) & 0x1FF) / 8.0f);
    d.point_sprite = g_prim == 1 && (k(0x31C) & 1);
    d.stencil_test = (k(0x32C) & 1) && d.surf.zeta_fmt == 2;
    d.stencil_write_mask = k(0x360) & 0xFF;
    d.stencil_func = 0x200 + (k(0x364) & 7);
    d.stencil_ref = k(0x368) & 0xFF;
    d.stencil_func_mask = k(0x36C) & 0xFF;
    d.stencil_fail = k(0x370);
    d.stencil_zfail = k(0x374);
    d.stencil_zpass = k(0x378);
    // Polygon offset enables per fill mode: point 0x330, line 0x334, fill 0x338; scale factor 0x384,
    // bias 0x388 (depth-buffer units). The game uses it for decals (lit windows, billboards).
    {
        const bool lines = g_prim == 2 || g_prim == 3 || g_prim == 4;
        const u32 en = g_prim == 1 ? k(0x330) : lines ? k(0x334) : k(0x338);
        d.poly_offset = (en & 1) && d.depth_test;
        d.poly_offset_scale = kf(0x384);
        d.poly_offset_bias = kf(0x388);
    }
    return d;
}

void shade_pixel(const DrawState& d, int x, int y, float z, const Vertex& in) {
    // Depth test
    u32 zi = u32(std::clamp(z, 0.0f, max_depth(d.surf)));
    if (d.depth_test) {
        const u32 cur = read_depth(d.surf, x, y);
        if (!compare(d.depth_func, float(zi), float(cur))) return;
    }
    // Texture stages
    CombinerRegs cr{};
    for (int i = 0; i < d.tex_used; ++i) {
        const u32 mode = (d.stage_modes >> (5 * i)) & 0x1F;
        Vec4 c{0, 0, 0, 0};
        if (mode == 4) c = {std::clamp(in.t[i][0], 0.0f, 1.0f), std::clamp(in.t[i][1], 0.0f, 1.0f), std::clamp(in.t[i][2], 0.0f, 1.0f), std::clamp(in.t[i][3], 0.0f, 1.0f)};
        else if (mode && d.tex[i].enabled) {
            if (mode != 1 && mode != 2) warn_once("texture shader mode", mode);
            const float q = mode == 1 && in.t[i][3] != 0 ? in.t[i][3] : 1.0f;
            c = sample(d.tex[i], in.t[i][0] / q, in.t[i][1] / q);
        }
        cr.r[8 + i] = {c[0], c[1], c[2], c[3]};
    }
    cr.r[4] = {in.d0[0], in.d0[1], in.d0[2], in.d0[3]};
    cr.r[5] = {in.d1[0], in.d1[1], in.d1[2], in.d1[3]};
    const float fogf = d.fog ? std::clamp(in.fog, 0.0f, 1.0f) : 1.0f;
    cr.r[3] = {d.fog_color[0], d.fog_color[1], d.fog_color[2], fogf};
    cr.r[12].a = cr.r[8].a;  // spare0 alpha starts as texture 0 alpha
    Vec4 src = run_combiners(cr, d.comb);
    // Alpha test
    if (d.alpha_test && !compare(d.alpha_func, std::round(src[3] * 255) / 255, d.alpha_ref)) return;
    // Blend
    if (d.blend) {
        const Vec4 dst = read_color(d.surf, x, y);
        Vec4 out;
        for (int i = 0; i < 4; ++i) {
            const float s = src[i] * blend_factor(d.sfactor, src, dst, d.blend_color, i);
            const float t = dst[i] * blend_factor(d.dfactor, src, dst, d.blend_color, i);
            switch (d.equation) {
                case 0x800A: out[i] = s - t; break;
                case 0x800B: out[i] = t - s; break;
                case 0x8007: out[i] = std::min(src[i], dst[i]); break;
                case 0x8008: out[i] = std::max(src[i], dst[i]); break;
                default: out[i] = s + t; break;
            }
        }
        src = out;
    }
    write_color(d.surf, x, y, src, d.color_mask_bits);
    if (d.depth_write) write_depth(d.surf, x, y, zi);
}

// ================================================================ rasteriser
// Minimal fork/join pool for row bands.
class RowPool {
public:
    RowPool() {
        const unsigned n = std::max(1u, std::thread::hardware_concurrency());
        for (unsigned i = 0; i + 1 < n; ++i) workers_.emplace_back([this] { loop(); });
    }
    void run(int y0, int y1, const std::function<void(int, int)>& fn) {
        const int bands = int(workers_.size()) + 1;
        const int per = std::max(1, (y1 - y0 + bands) / bands);
        {
            std::lock_guard lk(m_);
            fn_ = &fn;
            next_ = y0;
            end_ = y1;
            step_ = per;
            pending_ = int(workers_.size());
            ++gen_;
        }
        cv_.notify_all();
        work();
        std::unique_lock lk(m_);
        done_cv_.wait(lk, [&] { return pending_ == 0; });
    }

private:
    void work() {
        for (;;) {
            int a;
            {
                std::lock_guard lk(m_);
                if (next_ > end_) return;
                a = next_;
                next_ += step_;
            }
            (*fn_)(a, std::min(a + step_ - 1, end_));
        }
    }
    void loop() {
        u64 seen = 0;
        for (;;) {
            {
                std::unique_lock lk(m_);
                cv_.wait(lk, [&] { return gen_ != seen; });
                seen = gen_;
            }
            work();
            std::lock_guard lk(m_);
            if (--pending_ == 0) done_cv_.notify_all();
        }
    }
    std::vector<std::thread> workers_;
    std::mutex m_;
    std::condition_variable cv_, done_cv_;
    const std::function<void(int, int)>* fn_ = nullptr;
    int next_ = 0, end_ = 0, step_ = 1, pending_ = 0;
    u64 gen_ = 0;
};

// Vertex transforms: a few helpers taking 256-vertex chunks from an atomic
// counter. Small draws run inline; waking every core for a few hundred
// vertices (what the row pool does) costs more than the work.
class VertexPool {
public:
    static constexpr int kChunk = 256;
    VertexPool() {
        const int n = std::clamp(int(std::thread::hardware_concurrency()) - 1, 1, 7);
        for (int i = 0; i < n; ++i) workers_.emplace_back([this] { loop(); });
    }
    void run(int count, const std::function<void(int, int)>& fn) {
        const int chunks = (count + kChunk - 1) / kChunk;
        if (chunks <= 1) { fn(0, count - 1); return; }
        {
            std::lock_guard lk(m_);
            fn_ = &fn;
            count_ = count;
            chunks_ = chunks;
            next_.store(0);
            done_.store(0);
            ++gen_;
        }
        cv_.notify_all();
        work();
        std::unique_lock lk(m_);
        done_cv_.wait(lk, [&] { return done_.load() == chunks_; });
    }

private:
    void work() {
        for (;;) {
            const int c = next_.fetch_add(1);
            if (c >= chunks_) return;
            const int a = c * kChunk;
            (*fn_)(a, std::min(a + kChunk, count_) - 1);
            if (done_.fetch_add(1) + 1 == chunks_) {
                std::lock_guard lk(m_);
                done_cv_.notify_all();
            }
        }
    }
    void loop() {
        u64 seen = 0;
        for (;;) {
            {
                std::unique_lock lk(m_);
                cv_.wait(lk, [&] { return gen_ != seen; });
                seen = gen_;
            }
            work();
        }
    }
    std::vector<std::thread> workers_;
    std::mutex m_;
    std::condition_variable cv_, done_cv_;
    const std::function<void(int, int)>* fn_ = nullptr;
    int count_ = 0, chunks_ = 0;
    std::atomic<int> next_{0}, done_{0};
    u64 gen_ = 0;
};

void parallel_vertices(int count, const std::function<void(int, int)>& fn) {
    static VertexPool pool;
    pool.run(count, fn);
}

void parallel_rows(int y0, int y1, const std::function<void(int, int)>& fn) {
    static RowPool pool;
    pool.run(y0, y1, fn);
}

bool back_facing(const Vertex& a, const Vertex& b, const Vertex& c);
Vertex with_back_colours(Vertex v);

void draw_triangle(const DrawState& d, const Vertex& a0, const Vertex& b0, const Vertex& c0) {
    if (a0.w <= 0 || b0.w <= 0 || c0.w <= 0) return;  // TODO: near-plane clipping
    const bool back = g_ff.two_sided && back_facing(a0, b0, c0);
    const Vertex a = back ? with_back_colours(a0) : a0, b = back ? with_back_colours(b0) : b0,
                 c = back ? with_back_colours(c0) : c0;
    const float area = (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y);
    if (area == 0 || !std::isfinite(area)) return;
    if (k(0x308) & 1) {  // culling
        const bool ccw = area < 0;  // screen y grows downwards
        const bool front = (k(0x3A0) == 0x901) ? ccw : !ccw;
        const u32 face = k(0x39C);
        if ((face == 0x404 && front) || (face == 0x405 && !front) || face == 0x408) return;
    }
    const int x0 = std::max(d.clip_x0, int(std::floor(std::min({a.x, b.x, c.x}))));
    const int x1 = std::min(d.clip_x1, int(std::ceil(std::max({a.x, b.x, c.x}))));
    const int y0 = std::max(d.clip_y0, int(std::floor(std::min({a.y, b.y, c.y}))));
    const int y1 = std::min(d.clip_y1, int(std::ceil(std::max({a.y, b.y, c.y}))));
    if (x0 > x1 || y0 > y1) return;
    const float ia = 1 / a.w, ib = 1 / b.w, ic = 1 / c.w;
    auto edge = [](float ax, float ay, float bx, float by, float px, float py) { return (bx - ax) * (py - ay) - (by - ay) * (px - ax); };
    const bool flat = (k(0x37C) & 0xFFFF) == 0x1D00;  // GL_FLAT
    const int ntex = d.tex_used;
    // Edge functions are linear in x: w(x) = w(x0) + dw * (x - x0). Per row we
    // solve for the span where all three are >= 0 instead of testing every pixel.
    const float sgn = area < 0 ? -1.0f : 1.0f;
    const float inv_area = 1.0f / std::fabs(area);
    const float dw0 = -(c.y - b.y) * sgn, dw1 = -(a.y - c.y) * sgn, dw2 = -(b.y - a.y) * sgn;
    auto rows = [&](int ya, int yb) {
    for (int y = ya; y <= yb; ++y) {
        const float py = float(y) + 0.5f;
        const float bx = float(x0) + 0.5f;
        const float e0 = edge(b.x, b.y, c.x, c.y, bx, py) * sgn, e1 = edge(c.x, c.y, a.x, a.y, bx, py) * sgn,
                    e2 = edge(a.x, a.y, b.x, b.y, bx, py) * sgn;
        float lo = 0, hi = float(x1 - x0);
        auto clip = [&](float e, float dw) {  // keep t with e + dw * t >= 0
            if (dw > 0) lo = std::max(lo, std::ceil(-e / dw));
            else if (dw < 0) hi = std::min(hi, std::floor(-e / dw));
            else if (e < 0) hi = -1;
        };
        clip(e0, dw0);
        clip(e1, dw1);
        clip(e2, dw2);
        if (lo > hi) continue;
        for (int x = x0 + int(lo); x <= x0 + int(hi); ++x) {
            const float t = float(x - x0);
            const float w0 = e0 + dw0 * t, w1 = e1 + dw1 * t, w2 = e2 + dw2 * t;
            if (w0 < 0 || w1 < 0 || w2 < 0) continue;  // float edge cases at span ends
            const float l0 = w0 * inv_area, l1 = w1 * inv_area, l2 = w2 * inv_area;
            const float z = l0 * a.z + l1 * b.z + l2 * c.z;
            const float pw = l0 * ia + l1 * ib + l2 * ic;
            const float p0 = l0 * ia / pw, p1 = l1 * ib / pw, p2 = l2 * ic / pw;
            Vertex v;
            for (int i = 0; i < 4; ++i) {
                v.d0[i] = flat ? c.d0[i] : p0 * a.d0[i] + p1 * b.d0[i] + p2 * c.d0[i];
                v.d1[i] = flat ? c.d1[i] : p0 * a.d1[i] + p1 * b.d1[i] + p2 * c.d1[i];
                for (int t = 0; t < ntex; ++t) v.t[t][i] = p0 * a.t[t][i] + p1 * b.t[t][i] + p2 * c.t[t][i];
            }
            v.fog = p0 * a.fog + p1 * b.fog + p2 * c.fog;
            shade_pixel(d, x, y, z, v);
        }
    }
    };
    // Big triangles are split into row bands across worker threads. Pixels of one
    // triangle never overlap, so this preserves ordering/blending semantics.
    g_pixels += u64((x1 - x0 + 1) * (y1 - y0 + 1));
    static const long par_min = std::getenv("VC3_PAR_MIN") ? std::atol(std::getenv("VC3_PAR_MIN")) : 8192;
    if ((x1 - x0 + 1) * (y1 - y0 + 1) >= par_min) parallel_rows(y0, y1, rows);
    else rows(y0, y1);
}

// CPU-side face culling in screen space (y down). Skipped when a vertex is behind
// the eye (w <= 0): its projected position is unreliable and GL clips it anyway.
bool culled(const Vertex& a, const Vertex& b, const Vertex& c) {
    if (!(k(0x308) & 1) || a.w <= 0 || b.w <= 0 || c.w <= 0 || glb::no_cull()) return false;
    const float area = (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y);
    const bool ccw = area < 0;
    const bool front = (k(0x3A0) == 0x901) ? ccw : !ccw;
    const u32 face = k(0x39C);
    return (face == 0x404 && front) || (face == 0x405 && !front) || face == 0x408;
}

void count_features() {
    static const bool on = std::getenv("VC3_GPU_STATS") != nullptr;
    if (!on) return;
    auto& f = g_feature_use;
    const bool program = (k(0x1E94) & 3) == 2;
    ++f[program ? "vertex program" : "fixed function"];
    if (!program) {
        if (k(0x314) & 1) ++f["ff lighting"];
        if (k(0x328)) ++f["ff skinning"];
        for (int i = 0; i < 4; ++i) {
            if (k(0x420 + 4 * i) & 1) ++f["ff texture matrix"];
            for (int c = 0; c < 4; ++c)
                if (k(0x3C0 + 16 * i + 4 * c)) ++f["ff texgen"];
        }
    }
    if (k(0x32C) & 1) ++f["stencil test"];
    if (k(0x2A4) & 1) ++f["fog"];
    if (k(0x17C4) & 1) ++f["two-sided lighting"];
    for (int i = 0; i < 4; ++i) {
        const u32 mode = (k(0x1E70) >> (5 * i)) & 0x1F;
        if (!mode || !(k(0x1B0C + 64 * i) >> 30)) continue;  // stage off / texture disabled
        char b2[40];
        std::snprintf(b2, sizeof b2, "tex shader mode %u", mode);
        ++f[b2];
        if (k(0x1B04 + 64 * i) & 4) ++f["cube map"];
    }
    if ((k(0x37C) & 0xFFFF) == 0x1D00) ++f["flat shading"];
    char b[32];
    std::snprintf(b, sizeof b, "prim %u", g_prim);
    ++f[b];
}

// Two-sided lighting: a triangle facing away from the viewer is shaded with the
// back colours. Facing follows the same rule as culling (screen y grows down).
bool back_facing(const Vertex& a, const Vertex& b, const Vertex& c) {
    const float area = (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y);
    const bool ccw = area < 0;
    const bool front = (k(0x3A0) == 0x901) ? ccw : !ccw;
    return !front;
}
Vertex with_back_colours(Vertex v) {
    v.d0 = v.b0;
    v.d1 = v.b1;
    return v;
}

void draw_primitives_gl(DrawState d, const std::vector<Vertex>& v) {
    // Points and lines go to GL as submitted.
    switch (g_prim) {
        case 1: d.gl_prim = 0x0000; glb::draw(d, v); return;  // GL_POINTS
        case 2: d.gl_prim = 0x0001; glb::draw(d, v); return;  // GL_LINES
        case 3: d.gl_prim = 0x0002; glb::draw(d, v); return;  // GL_LINE_LOOP
        case 4: d.gl_prim = 0x0003; glb::draw(d, v); return;  // GL_LINE_STRIP
        default: break;
    }
    std::vector<Vertex> tris;
    tris.reserve(v.size() * 2);
    const bool two_sided = g_ff.two_sided;
    auto tri = [&](size_t a, size_t b, size_t c) {
        if (culled(v[a], v[b], v[c])) return;
        if (two_sided && back_facing(v[a], v[b], v[c])) {
            tris.push_back(with_back_colours(v[a]));
            tris.push_back(with_back_colours(v[b]));
            tris.push_back(with_back_colours(v[c]));
            return;
        }
        tris.push_back(v[a]);
        tris.push_back(v[b]);
        tris.push_back(v[c]);
    };
    const size_t n = v.size();
    switch (g_prim) {
        case 5: for (size_t i = 0; i + 2 < n; i += 3) tri(i, i + 1, i + 2); break;
        case 6: for (size_t i = 0; i + 2 < n; ++i) { if (i & 1) tri(i + 1, i, i + 2); else tri(i, i + 1, i + 2); } break;
        case 7: case 10: for (size_t i = 1; i + 1 < n; ++i) tri(0, i, i + 1); break;
        case 8: for (size_t i = 0; i + 3 < n; i += 4) { tri(i, i + 1, i + 2); tri(i, i + 2, i + 3); } break;
        case 9: for (size_t i = 0; i + 3 < n; i += 2) { tri(i, i + 1, i + 3); tri(i, i + 3, i + 2); } break;
        default: warn_once("primitive type", g_prim); return;
    }
    d.gl_prim = 0x0004;
    glb::draw(d, tris);
}

void draw_primitives(const std::vector<Vertex>& v) {
    if (v.empty()) return;
    ScopedNs prof{g_ns_raster};
    ++g_draws;
    count_features();
    const auto t_state = std::chrono::steady_clock::now();
    const DrawState d = make_draw_state();
    g_ns_state += u64(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t_state).count());
    if (!d.surf.width || !d.surf.height) return;
    if (glb::active()) {
        draw_primitives_gl(d, v);
        return;
    }
    const size_t n = v.size();
    switch (g_prim) {
        case 5: for (size_t i = 0; i + 2 < n; i += 3) draw_triangle(d, v[i], v[i + 1], v[i + 2]); break;  // triangles
        case 6:  // triangle strip
            for (size_t i = 0; i + 2 < n; ++i) {
                if (i & 1) draw_triangle(d, v[i + 1], v[i], v[i + 2]);
                else draw_triangle(d, v[i], v[i + 1], v[i + 2]);
            }
            break;
        case 7: case 10: for (size_t i = 1; i + 1 < n; ++i) draw_triangle(d, v[0], v[i], v[i + 1]); break;  // fan / polygon
        case 8:  // quads
            for (size_t i = 0; i + 3 < n; i += 4) {
                draw_triangle(d, v[i], v[i + 1], v[i + 2]);
                draw_triangle(d, v[i], v[i + 2], v[i + 3]);
            }
            break;
        case 9:  // quad strip
            for (size_t i = 0; i + 3 < n; i += 2) {
                draw_triangle(d, v[i], v[i + 1], v[i + 3]);
                draw_triangle(d, v[i], v[i + 3], v[i + 2]);
            }
            break;
        default: warn_once("primitive type (points/lines)", g_prim); break;
    }
}

// GPU vertex path for a draw from vertex arrays (fixed function or vertex program). Returns false if the
// draw has to take the CPU path (unsupported formats, sparse/huge index range).
bool try_draw_ff_gpu() {
    static const bool cpu_only = std::getenv("VC3_CPU_VERTEX") != nullptr;
    if (cpu_only || !glb::active() || !g_inline.empty()) return false;
    if (g_indices.empty() && g_ranges.empty()) return false;
    GpuArrays arr;
    for (int a = 0; a < 16; ++a) {
        const ArrayAttr& at = g_arrays[a];
        auto& o = arr.attr[a];
        o.constant = g_current[a];
        if (!at.enabled) continue;
        const u32 type = at.fmt & 0xF, size = (at.fmt >> 4) & 0xF;
        if (type == 0 && size != 4) return false;  // partial D3D colours: CPU path
        if (type == 3 || type > 6) return false;
        o.enabled = true;
        o.fmt = at.fmt;
        o.base = at.base & (mem::kRamSize - 1);
        o.stride = at.stride;
    }
    // Absolute vertex order, then the index range it covers.
    static std::vector<u32> order;
    order.clear();
    order.insert(order.end(), g_indices.begin(), g_indices.end());
    for (auto [start, count] : g_ranges)
        for (u32 i = 0; i < count; ++i) order.push_back(start + i);
    if (order.empty()) return true;
    u32 lo = order[0], hi = order[0];
    for (u32 i : order) { lo = std::min(lo, i); hi = std::max(hi, i); }
    if (hi - lo > (1u << 20)) return false;
    arr.min_index = lo;
    arr.max_index = hi;

    // Primitive assembly into GL index lists (same winding rules as the CPU path).
    static std::vector<u32> idx;
    idx.clear();
    const size_t n = order.size();
    auto v = [&](size_t i) { return order[i] - lo; };
    auto tri = [&](size_t a, size_t b, size_t c) { idx.push_back(v(a)); idx.push_back(v(b)); idx.push_back(v(c)); };
    u32 gl_prim = 0x0004;
    switch (g_prim) {
        case 1: gl_prim = 0x0000; for (size_t i = 0; i < n; ++i) idx.push_back(v(i)); break;
        case 2: gl_prim = 0x0001; for (size_t i = 0; i < n; ++i) idx.push_back(v(i)); break;
        case 3: gl_prim = 0x0002; for (size_t i = 0; i < n; ++i) idx.push_back(v(i)); break;
        case 4: gl_prim = 0x0003; for (size_t i = 0; i < n; ++i) idx.push_back(v(i)); break;
        case 5: for (size_t i = 0; i + 2 < n; i += 3) tri(i, i + 1, i + 2); break;
        case 6: for (size_t i = 0; i + 2 < n; ++i) { if (i & 1) tri(i + 1, i, i + 2); else tri(i, i + 1, i + 2); } break;
        case 7: case 10: for (size_t i = 1; i + 1 < n; ++i) tri(0, i, i + 1); break;
        case 8: for (size_t i = 0; i + 3 < n; i += 4) { tri(i, i + 1, i + 2); tri(i, i + 2, i + 3); } break;
        case 9: for (size_t i = 0; i + 3 < n; i += 2) { tri(i, i + 1, i + 3); tri(i, i + 3, i + 2); } break;
        default: return false;
    }

    ScopedNs prof{g_ns_raster};
    DrawState d = make_draw_state();
    if (!d.surf.width || !d.surf.height) return true;
    d.gl_prim = gl_prim;
    d.gpu_two_sided = g_ff.two_sided && gl_prim == 0x0004;
    // Widescreen: a small 2D draw covering the whole surface (fade, background,
    // post-processing copy) is stretched to the full width; the CPU transforms its few
    // vertices to find out. Everything else keeps its shape.
    static std::vector<std::array<float, 8>> hud_bounds;
    static std::vector<u32> hud_corners;
    hud_bounds.clear();
    hud_corners.clear();
    size_t hud_vertices = 0;
    // (Bigger draws too if they read a widened target: screen distortion effects.)
    d.big_mesh = hi - lo >= 64;
    if (glb::widescreen() && (hi - lo < 64 || (hi - lo < 4096 && d.tex_used && d.tex[0].enabled && glb::samples_wide_target(d.tex[0])))) {
        float x0 = 1e30f, x1 = -1e30f, y0 = 1e30f, y1 = -1e30f;
        bool flat = true;
        static std::vector<Vertex> vs;
        vs.clear();
        // Bigger draws here only matter if flat (a full-screen layer reading a widened target;
        // their bounds decide stretching): the first vertex off w = 1 shows a 3D draw (glass
        // shards, distortion meshes), and the rest needn't be transformed -- on the CPU, through
        // the vertex program interpreter, thousands per draw (frame drops in 16:9 otherwise).
        const bool big = hi - lo >= 64;
        for (u32 i = lo; i <= hi; ++i) {
            vs.push_back(fetch_array_vertex(i));
            const Vertex& vx = vs.back();
            flat = flat && std::fabs(vx.w - 1.0f) < 1e-4f;
            x0 = std::min(x0, vx.x); x1 = std::max(x1, vx.x);
            y0 = std::min(y0, vx.y); y1 = std::max(y1, vx.y);
            if (big && !flat) break;
        }
        d.classify_wide(flat, x0, x1, y0, y1);
        // HUD pieces: triangle bounds for the backend's per-element decision.
        if (flat && gl_prim == 0x0004 && d.surf.width == 640 && d.surf.height == 480) {  // HUD: screen-size targets
            hud_vertices = vs.size();
            for (size_t t = 0; t + 2 < idx.size(); t += 3) {
                const Vertex &a = vs[idx[t]], &b = vs[idx[t + 1]], &c = vs[idx[t + 2]];
                hud_bounds.push_back({std::min({a.x, b.x, c.x}), std::min({a.y, b.y, c.y}), std::max({a.x, b.x, c.x}),
                                      std::max({a.y, b.y, c.y}), std::min({a.t[0][0], b.t[0][0], c.t[0][0]}),
                                      std::min({a.t[0][1], b.t[0][1], c.t[0][1]}), std::max({a.t[0][0], b.t[0][0], c.t[0][0]}),
                                      std::max({a.t[0][1], b.t[0][1], c.t[0][1]})});
                hud_corners.insert(hud_corners.end(), {idx[t], idx[t + 1], idx[t + 2]});
            }
        }
    }
    glb::set_hud_triangles(hud_bounds, hud_corners, hud_vertices);
    // A capture: every draw's screen bounds (vertices in front of the eye) for draws.txt.
    if (glb::tracing_now() && hi - lo < 65536 && (d.bx0 == 0 && d.bx1 == 0)) {
        float x0 = 1e30f, x1 = -1e30f, y0 = 1e30f, y1 = -1e30f;
        for (u32 i = lo; i <= hi; ++i) {
            const Vertex vx = fetch_array_vertex(i);
            if (!(vx.w > 0)) continue;
            x0 = std::min(x0, vx.x); x1 = std::max(x1, vx.x);
            y0 = std::min(y0, vx.y); y1 = std::max(y1, vx.y);
        }
        if (x0 <= x1) { d.bx0 = x0; d.bx1 = x1; d.by0 = y0; d.by1 = y1; }
    }
    if (d.samples_own_target() && hi - lo < 8192) {
        static std::vector<Vertex> own;
        own.clear();
        for (u32 i = lo; i <= hi; ++i) own.push_back(fetch_array_vertex(i));
        d.own_target_bounds(own.data(), own.size());
    }
    if (g_ff.program) {
        const glb::VpInfo vp{g_vp.code, k(0x1EA0), g_vp.c};
        if (!glb::draw_vp(d, g_ff, arr, idx, vp)) return false;
    } else if (!glb::draw_ff(d, g_ff, arr, idx)) {
        return false;
    }
    ++g_draws;
    count_features();
    g_xf_gpu += u64(hi - lo + 1);
    return true;
}

void flush_draw() {
    std::vector<Vertex> verts;
    {
    ScopedNs prof{g_ns_vertex};
    prepare_vertex_state();
    prepare_arrays();
    }
    if (try_draw_ff_gpu()) {
        g_inline.clear();
        g_indices.clear();
        g_ranges.clear();
        return;
    }
    {
    ScopedNs prof{g_ns_vertex};
    if (!g_inline.empty()) {
        // Interleaved attributes in attribute order, each enabled one packed as dwords.
        int stride = 0;
        int offsets[16];
        for (int a = 0; a < 16; ++a) {
            offsets[a] = stride;
            stride += attr_dwords(k(0x1760 + 4 * a));
        }
        if (stride) {
            for (size_t base = 0; base + size_t(stride) <= g_inline.size(); base += size_t(stride)) {
                VpState s;
                for (int a = 0; a < 16; ++a) {
                    const u32 fmt = k(0x1760 + 4 * a);
                    s.v[a] = attr_dwords(fmt) ? decode_attr(fmt, reinterpret_cast<const u8*>(&g_inline[base + size_t(offsets[a])]))
                                              : g_current[a];
                }
                verts.push_back(transform(s));
            }
        }
    }
    // Array vertices: transform each distinct index once (indexed meshes reference
    // most vertices several times), in parallel for big draws, then expand.
    if (!g_indices.empty() || !g_ranges.empty()) {
        static std::vector<u32> slot_of;  // index -> slot + 1 (0 = not seen in this draw)
        static std::vector<u32> unique;
        static std::vector<Vertex> xf;
        unique.clear();
        auto slot = [&](u32 i) -> u32 {
            if (i >= slot_of.size()) slot_of.resize(size_t(i) + 1 + 4096, 0);
            u32& sl = slot_of[i];
            if (!sl) {
                unique.push_back(i);
                sl = u32(unique.size());
            }
            return sl - 1;
        };
        std::vector<u32> order;
        order.reserve(g_indices.size() + 64);
        for (u32 i : g_indices) order.push_back(slot(i));
        for (auto [start, count] : g_ranges)
            for (u32 i = 0; i < count; ++i) order.push_back(slot(start + i));
        xf.resize(unique.size());
        const int n = int(unique.size());
        g_xf_verts += u64(n);
        if (!g_ff.program) g_xf_ff += u64(n);
        auto work = [&](int a, int b) { for (int i = a; i <= b; ++i) xf[size_t(i)] = fetch_array_vertex(unique[size_t(i)]); };
        const auto t_x = std::chrono::steady_clock::now();
        if (n) parallel_vertices(n, work);
        g_ns_xform += u64(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t_x).count());
        for (u32 i : unique) slot_of[i] = 0;
        verts.reserve(verts.size() + order.size());
        for (u32 o : order) verts.push_back(xf[o]);
    }
    }
    draw_primitives(verts);
    g_inline.clear();
    g_indices.clear();
    g_ranges.clear();
}

}  // namespace detail

using namespace detail;

std::string profile_report() {
    if (!g_feature_use.empty()) {
        std::string r = "[pgraph] draws by feature:";
        for (const auto& [name, n] : g_feature_use) r += " " + name + "=" + std::to_string(n);
        std::fprintf(stderr, "%s\n", r.c_str());
    }
    char b[260];
    std::snprintf(b, sizeof b, "gpu-vertex %llu, xform %.0f ms, drawstate %.0f ms, clear %.0f ms, vertex %.0f ms (%llu xf, %llu ff), raster %.0f ms, %llu draws, %.1f Mpix",
                  (unsigned long long)g_xf_gpu.exchange(0), g_ns_xform.exchange(0) / 1e6, g_ns_state.exchange(0) / 1e6, g_ns_clear.exchange(0) / 1e6, g_ns_vertex.exchange(0) / 1e6,
                  (unsigned long long)g_xf_verts.exchange(0), (unsigned long long)g_xf_ff.exchange(0),
                  g_ns_raster.exchange(0) / 1e6, (unsigned long long)g_draws.exchange(0), g_pixels.exchange(0) / 1e6);
    return b;
}

void init(const Host& host) {
    g_host = host;
    K = host.kelvin;
    for (auto& c : g_current) c = {0, 0, 0, 1};
    g_current[3] = g_current[4] = {1, 1, 1, 1};
}

void method(u32 m, u32 data) {
    static const bool disabled = std::getenv("VC3_NO_RENDER") != nullptr;  // debugging aid
    if (disabled) return;
    // The GL context must live on the GPU thread, which is the only caller of method().
    static const bool use_gl = [] {
        const char* r = std::getenv("VC3_RENDERER");
        if (r && std::string(r) == "soft") return false;
        return glb::init();
    }();
    (void)use_gl;
    if (m == 0x130 && glb::active()) glb::flush();  // FLIP_STALL: end of frame
    if (m >= 0x0B00 && m < 0x0B80) {  // SET_TRANSFORM_PROGRAM: 4 dwords per slot
        if (g_vp.load_slot < 136) g_vp.code[g_vp.load_slot][g_vp.load_word] = data;
        if (++g_vp.load_word == 4) { g_vp.load_word = 0; ++g_vp.load_slot; }
        return;
    }
    // The viewport scale/offset registers live in the transform engine's context, which shares
    // the constant memory: they are vertex-program constants 58 and 59 (c[-38], c[-37] to D3D,
    // whose vertex-shader epilogue maps positions to the screen with them). Programs that
    // read them (the hard stage's sewer water) collapsed to a point without this.
    if ((m >= 0x0A20 && m < 0x0A30) || (m >= 0x0AF0 && m < 0x0B00))
        std::memcpy(&g_vp.c[m >= 0x0AF0 ? 58 : 59][(m & 0xF) / 4], &data, 4);
    if (m >= 0x0B80 && m < 0x0C00) {  // SET_TRANSFORM_CONSTANT: 4 floats per constant
        if (g_vp.const_slot < 192) std::memcpy(&g_vp.c[g_vp.const_slot][g_vp.const_word], &data, 4);
        if (++g_vp.const_word == 4) { g_vp.const_word = 0; ++g_vp.const_slot; }
        return;
    }
    if (m >= 0x1880 && m < 0x1A00) {  // SET_VERTEX_DATA2F_M / 2S / 4UB / 4S_M (current attribute values)
        float f; std::memcpy(&f, &data, 4);
        if (m < 0x1900) { const u32 a = (m - 0x1880) / 8, c = ((m - 0x1880) / 4) & 1; g_current[a][c] = f; if (c == 1) { g_current[a][2] = 0; g_current[a][3] = 1; } }
        else if (m < 0x1940) { const u32 a = (m - 0x1900) / 4; g_current[a] = {float(s16(data)), float(s16(data >> 16)), 0, 1}; }
        else if (m < 0x1980) { const u32 a = (m - 0x1940) / 4; g_current[a] = {(data & 0xFF) / 255.0f, ((data >> 8) & 0xFF) / 255.0f, ((data >> 16) & 0xFF) / 255.0f, (data >> 24) / 255.0f}; }
        else { const u32 a = (m - 0x1980) / 8, c = ((m - 0x1980) / 4) & 1; g_current[a][c * 2] = float(s16(data)); g_current[a][c * 2 + 1] = float(s16(data >> 16)); }
        return;
    }
    if (m >= 0x1A00 && m < 0x1B00) {  // SET_VERTEX_DATA4F_M
        const u32 a = (m - 0x1A00) / 16, c = ((m - 0x1A00) / 4) & 3;
        std::memcpy(&g_current[a][c], &data, 4);
        return;
    }
    switch (m) {
        case 0x1E9C: g_vp.load_slot = data; g_vp.load_word = 0; break;    // SET_TRANSFORM_PROGRAM_LOAD
        case 0x1EA4: g_vp.const_slot = data; g_vp.const_word = 0; break;  // SET_TRANSFORM_CONSTANT_LOAD
        case 0x17FC:  // SET_BEGIN_END
            if (data == 0) flush_draw();
            else { g_prim = data; g_inline.clear(); g_indices.clear(); g_ranges.clear(); }
            break;
        case 0x1818: g_inline.push_back(data); break;  // INLINE_ARRAY
        case 0x1800: g_indices.push_back(data & 0xFFFF); g_indices.push_back(data >> 16); break;  // ARRAY_ELEMENT16
        case 0x1808: g_indices.push_back(data); break;                                        // ARRAY_ELEMENT32
        case 0x1810: g_ranges.push_back({data & 0xFFFFFF, (data >> 24) + 1}); break;          // DRAW_ARRAYS
        case 0x1D94: clear_surface(data); break;                                              // CLEAR_SURFACE
        default: break;
    }
}

}  // namespace vc3::gpu::pgraph
