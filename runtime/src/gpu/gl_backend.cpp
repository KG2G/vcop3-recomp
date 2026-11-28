#include "gl_backend.h"
#include "gpu/device.h"
#include "gpu/gl_device.h"
#include "gpu/smaa.h"
#include "platform.h"
#ifdef VC3_HAVE_VULKAN
#include "gpu/vk_present.h"
#endif
#include "io/jvs.h"
#include "io/ohci.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

namespace vc3::gpu::pgraph::glb {

using namespace detail;

namespace {

bool g_active = false;
dev::Device* g_dev = nullptr;  // the GPU thread's graphics device
// Rendered frames reach the window straight from GPU memory (display slots); otherwise
// the window shows guest RAM.
bool g_shared = false;

// ================================================================ helpers
// Time spent per backend activity (VC3_FPS report from flush()).
struct Prof { const char* name; std::chrono::nanoseconds t{}; u64 n = 0; };
// Surface transfer counts by reason (VC3_FPS report).
u64 g_n_fb_copy = 0, g_fb_copy_px = 0, g_n_dl_flip = 0, g_n_dl_sync = 0, g_n_dl_reshape = 0, g_n_up_new = 0, g_n_up_changed = 0;
Prof g_prof_prog{"program"}, g_prof_texbind{"tex-bind"}, g_prof_state{"uniforms+state"}, g_prof_vbo{"vbo+draw"};
Prof g_prof_tex_hash{"tex-hash"}, g_prof_tex_upload{"tex-decode+upload"}, g_prof_surf_up{"surf-upload"},
    g_prof_surf_down{"surf-download"}, g_prof_draw{"draw-submit"}, g_prof_rt_hash{"rt-hash"};
struct ProfScope {
    Prof& p;
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    explicit ProfScope(Prof& pr) : p(pr) {}
    ~ProfScope() { p.t += std::chrono::steady_clock::now() - t0; ++p.n; }
};

u64 hash_bytes(const u8* p, size_t n) {
    // Four independent lanes so the multiplies overlap (a single chain is latency-bound).
    u64 h0 = 0x9E3779B97F4A7C15ull ^ n, h1 = 0xC2B2AE3D27D4EB4Full, h2 = 0x165667B19E3779F9ull, h3 = 0x27D4EB2F165667C5ull;
    size_t i = 0;
    auto mix = [](u64 h, u64 v) { h = (h ^ v) * 0xFF51AFD7ED558CCDull; return h ^ (h >> 29); };
    for (; i + 32 <= n; i += 32) {
        u64 v[4];
        std::memcpy(v, p + i, 32);
        h0 = mix(h0, v[0]);
        h1 = mix(h1, v[1]);
        h2 = mix(h2, v[2]);
        h3 = mix(h3, v[3]);
    }
    u64 h = mix(mix(mix(h0, h1), h2), h3);
    for (; i < n; ++i) h = (h ^ p[i]) * 0x100000001B3ull;
    return h;
}

// Cheap change detector: one 8-byte word every 256 bytes. Only used to notice the CPU
// overwriting a GPU-resident surface wholesale (a movie frame, a software blit).
u64 sparse_hash(const u8* p, size_t n) {
    u64 h = 0x9E3779B97F4A7C15ull ^ n;
    for (size_t i = 0; i + 8 <= n; i += 256) {
        u64 v;
        std::memcpy(&v, p + i, 8);
        h = (h ^ v) * 0xFF51AFD7ED558CCDull;
        h ^= h >> 29;
    }
    return h;
}

// ================================================================ surfaces
struct ColorSurface {
    Surface desc;  // color part
    dev::Tex tex;
    int gw = 0, gh = 0;  // GPU texture size (scaled; wider for widescreen)
    bool wide = false;
    bool gl_dirty = false;
    u64 ram_hash = 0;
    u64 ram_sparse = 0;         // sparse_hash of RAM at the same moment as ram_hash
    u64 checked_epoch = ~0ull;  // RAM compared against ram_hash this frame
    bool sampled = false;       // used directly as a texture (render-to-texture)
    // Asynchronous write-back in flight (device read id), not yet in RAM.
    u32 read_id = 0;
    bool pending = false;
    u32 bytes() const {
        return desc.swizzled ? u32(desc.width * desc.height * desc.color_bpp()) : desc.color_pitch * u32(desc.height);
    }
};
struct ZetaSurface {
    u32 base;
    int w, h;    // GPU size
    dev::Tex tex;
};

std::vector<ColorSurface> g_color;
// VC3_GL_FULL_SYNC=1: write every GL-rendered surface back to RAM at each flip
// (otherwise render targets that are only ever sampled as textures stay on the GPU).
const bool g_full_sync = std::getenv("VC3_GL_FULL_SYNC") != nullptr;
// Internal resolution scale (VC3_RES_SCALE=N, 1-8): GL copies of guest surfaces are
// N times the guest size in each direction. Guest RAM always holds the native-size
// image: uploads are scaled up and write-backs scaled down at that boundary.
// Widescreen (VC3_WIDESCREEN=auto (the display's shape), 16:9, 16:10, 21:9, or a width/height ratio): screen-size
// surfaces (640x480) get extra width; the 4:3 picture keeps its shape in the middle
// (so the crosshair still lines up with what the game hit-tests) and 3D extends into
// the sides. g_wide_k is the 4:3 picture's share of the width (1: off).
// These are read by configure() at startup (after the launcher has set the options).
double g_wide_k = 1.0;
bool displayed(u32 base);
// Screen-size surfaces: the displayed buffers and the offscreen ones scenes are
// rendered into before a full-screen copy. Draws keep their shape in the middle, and
// sampling one into anything but a full-screen draw reads that middle part, so
// smaller uses (preview boxes) come out as on the original.
bool wide_surface(const Surface& s) {
    return g_wide_k < 1.0 && s.width == 640 && s.height == 480;
}
float g_cur_wide = 1.0f;  // uWide for draws into the bound target
int g_cur_gw = 0, g_cur_gh = 0, g_cur_native_w = 0;  // its GPU size and guest width
dev::Tex g_cur_color_tex;  // texture of the bound colour target

int g_scale = 1;
// Addresses the display has scanned out (a few, for double/triple buffering).
constexpr int kScanouts = 8;
std::atomic<u32> g_scanouts[kScanouts];
std::atomic<int> g_scanout_next{0};
bool displayed(u32 base) {
    for (const auto& a : g_scanouts)
        if (a.load(std::memory_order_relaxed) == base) return true;
    return false;
}
// ---------------------------------------------------------------- display slots
struct Slot {
    dev::Tex tex;
    int w = 0, h = 0;
    dev::Sync read_fence = 0;  // the window's copy of this slot (wait before overwriting)
};
constexpr int kSlots = 4;
Slot g_slots[kSlots];
int g_slot_next = 0;
std::mutex g_display_m;  // guards g_published, read fences
std::map<u32, DisplayFrame> g_published;

u64 g_epoch = 0;
u64 g_3d_epoch = ~0ull;  // last frame with a 3D draw (any target)
std::atomic<bool> g_no_cull{false};   // F8 (developer)
std::atomic<bool> g_no_atest{false};  // F7 (developer)
std::atomic<bool> g_no_depth{false};  // F6 (developer)
std::map<u32, u64> g_smaa_done;   // colour target -> frame it was last anti-aliased in (SMAA)
std::map<u32, bool> g_smaa_dirty;  // colour target -> scene drawn since its last pass (SMAA)
std::vector<ZetaSurface> g_zeta;

u8* color_addr(const Surface& s, int x, int y) {
    if (s.swizzled) return ram(s.color_base + swizzle_offset(u32(x), u32(y), s.sw_log_w, s.sw_log_h) * u32(s.color_bpp()));
    return ram(s.color_base + u32(y) * s.color_pitch + u32(x) * u32(s.color_bpp()));
}

// Native-size scratch textures for scaling at the RAM boundary (by size).
std::map<std::pair<int, int>, dev::Tex> g_staging;
dev::Tex staging(int w, int h) {
    dev::Tex& t = g_staging[{w, h}];
    if (!t) t = g_dev->create_texture(w, h, 1, dev::Fmt::BGRA8);
    return t;
}

void upload_color(ColorSurface& cs) {
    ProfScope prof(g_prof_surf_up);
    const Surface& s = cs.desc;
    std::vector<u32> px(size_t(s.width) * size_t(s.height));
    for (int y = 0; y < s.height; ++y)
        for (int x = 0; x < s.width; ++x) {
            const u8* p = color_addr(s, x, y);
            u32 v;
            if (s.color_bpp() == 4) {
                std::memcpy(&v, p, 4);
                if (s.color_fmt != 8) v |= 0xFF000000u;
            } else {
                u16 c;
                std::memcpy(&c, p, 2);
                if (s.color_fmt == 3) {
                    const u32 r = (c >> 11) & 31, g = (c >> 5) & 63, b = c & 31;
                    v = 0xFF000000u | (r << 3 | r >> 2) << 16 | (g << 2 | g >> 4) << 8 | (b << 3 | b >> 2);
                } else {
                    const u32 r = (c >> 10) & 31, g = (c >> 5) & 31, b = c & 31;
                    v = 0xFF000000u | (r << 3 | r >> 2) << 16 | (g << 3 | g >> 2) << 8 | (b << 3 | b >> 2);
                }
            }
            px[size_t(y) * size_t(s.width) + size_t(x)] = v;
        }
    const size_t bytes = px.size() * 4;
    if (cs.gw == s.width && cs.gh == s.height) {
        g_dev->upload(cs.tex, 0, 0, 0, 0, s.width, s.height, px.data(), bytes);
    } else {  // upload at native size, then scale up (nearest: same look as the guest image)
        const dev::Tex st = staging(s.width, s.height);
        g_dev->upload(st, 0, 0, 0, 0, s.width, s.height, px.data(), bytes);
        // Widescreen: the guest image is the middle 4:3 part (where draws put it too); the
        // sides keep what the GPU drew there.
        const int off = (cs.gw - s.width * g_scale) / 2;
        g_dev->blit(st, 0, 0, s.width, s.height, cs.tex, off, 0, off + s.width * g_scale, cs.gh, false);
    }
    cs.ram_hash = hash_bytes(ram(s.color_base), cs.bytes());
    cs.ram_sparse = sparse_hash(ram(s.color_base), cs.bytes());
}

std::atomic<u64> g_ram_writes{0};
std::mutex g_ram_write_m;

// Store GPU pixels (BGRA8, row-major) of `cs` into guest RAM in the surface's format.
void write_ram(ColorSurface& cs, const u32* px) {
    std::lock_guard write_lock(g_ram_write_m);
    const Surface& s = cs.desc;
    {
        // Guard: a write-back must never land on the USB driver's descriptors. (Windows + OpenGL crashed
        // 4 times in the XDK USB driver following a descriptor whose next pointer read 0xFFFFFFFx --
        // what a white pixel looks like. If this triggers, the log says which surface did it.)
        const u32 lo = s.color_base & (mem::kRamSize - 1), hi = lo + u32(cs.bytes());
        if (io::ohci::overlaps_descriptors(lo, hi)) {
            static int logged = 0;
            if (logged++ < 10) {
                u32 dlo, dhi;
                io::ohci::descriptor_range(dlo, dhi);
                std::fprintf(stderr, "[gl] write-back of surface %08x (%dx%d, pitch %u, format %u%s, %zu bytes) overlaps USB "
                                     "descriptors %08x-%08x: not written\n",
                             s.color_base, s.width, s.height, s.color_pitch, s.color_fmt, s.swizzled ? ", swizzled" : "",
                             size_t(cs.bytes()), dlo, dhi);
            }
            return;
        }
    }
    if (!s.swizzled && s.color_bpp() == 4) {
        for (int y = 0; y < s.height; ++y) {
            u8* row = ram(s.color_base + u32(y) * s.color_pitch);
            const u32* src = &px[size_t(y) * size_t(s.width)];
            if (s.color_fmt == 8) {
                std::memcpy(row, src, size_t(s.width) * 4);
            } else {  // X8R8G8B8: hardware leaves the X byte as written; keep it opaque
                u32* dst = reinterpret_cast<u32*>(row);  // guest RAM rows are 4-byte aligned here
                for (int x = 0; x < s.width; ++x) dst[x] = src[x] | 0xFF000000u;
            }
        }
    } else {
        for (int y = 0; y < s.height; ++y)
            for (int x = 0; x < s.width; ++x) {
                const u32 v = px[size_t(y) * size_t(s.width) + size_t(x)];
                u8* p = color_addr(s, x, y);
                if (s.color_bpp() == 4) {
                    const u32 w = s.color_fmt == 8 ? v : (v | 0xFF000000u);
                    std::memcpy(p, &w, 4);
                } else if (s.color_fmt == 3) {
                    const u16 c = u16(((v >> 19) & 31) << 11 | ((v >> 10) & 63) << 5 | ((v >> 3) & 31));
                    std::memcpy(p, &c, 2);
                } else {
                    const u16 c = u16(0x8000 | ((v >> 19) & 31) << 10 | ((v >> 11) & 31) << 5 | ((v >> 3) & 31));
                    std::memcpy(p, &c, 2);
                }
            }
    }
    cs.ram_hash = hash_bytes(ram(s.color_base), cs.bytes());
    cs.ram_sparse = sparse_hash(ram(s.color_base), cs.bytes());
    g_ram_writes.fetch_add(1, std::memory_order_release);
    if (g_shared) {
        std::lock_guard lk(g_display_m);
        if (auto it = g_published.find(s.color_base & (mem::kRamSize - 1)); it != g_published.end() && !it->second.ram_sparse)
            it->second.ram_sparse = cs.ram_sparse ? cs.ram_sparse : 1;
    }
}

// Complete an asynchronous write-back. Returns false if not ready (non-blocking).
bool finish_readback(ColorSurface& cs, bool block) {
    if (!cs.pending) return true;
    const void* px = g_dev->read_poll(cs.read_id, block);
    if (!px) return false;
    {
        ProfScope prof(g_prof_surf_down);
        write_ram(cs, static_cast<const u32*>(px));
    }
    g_dev->read_release(cs.read_id);
    cs.read_id = 0;
    cs.pending = false;
    return true;
}

// Start writing a GL-modified surface back to RAM without waiting for the GPU.
// The texture to read guest-size pixels from: the surface itself, or (scaled) a
// native-size copy filtered down from it.
dev::Tex native_image(const ColorSurface& cs) {
    const int w = cs.desc.width, h = cs.desc.height;
    if (cs.gw == w && cs.gh == h) return cs.tex;
    const dev::Tex st = staging(w, h);
    // Widescreen surfaces: guest RAM gets the middle 4:3 part (what the game itself
    // would have drawn; consistent with uploads).
    const int off = (cs.gw - w * g_scale) / 2;
    g_dev->blit(cs.tex, off, 0, off + w * g_scale, cs.gh, st, 0, 0, w, h, true);
    return st;
}

void start_readback(ColorSurface& cs) {
    finish_readback(cs, true);  // at most one in flight per surface
    cs.read_id = g_dev->read_start(native_image(cs), cs.desc.width, cs.desc.height);
    cs.gl_dirty = false;
    cs.pending = true;
}

// Synchronous write-back (something needs the RAM copy now).
void download_color(ColorSurface& cs) {
    finish_readback(cs, true);
    if (!cs.gl_dirty) return;
    ProfScope prof(g_prof_surf_down);
    std::vector<u32> px(size_t(cs.desc.width) * size_t(cs.desc.height));
    g_dev->read(native_image(cs), cs.desc.width, cs.desc.height, px.data());
    write_ram(cs, px.data());
    cs.gl_dirty = false;
}

// A surface GL rendered (not yet written back) whose memory the CPU has since
// overwritten -- e.g. a render target from an earlier scene now reused by the game
// as a CPU-written texture (Sofdec movie frames). The CPU data wins: reload it into
// the GL copy. Checked once per frame per surface. Returns true if reloaded.
bool reload_if_cpu_wrote(ColorSurface& cs) {
    if (cs.checked_epoch == g_epoch || !cs.gl_dirty) return false;  // clean: bind_color checks those
    if (!finish_readback(cs, false)) return false;  // RAM still being written back: try next time
    cs.checked_epoch = g_epoch;
    bool changed;
    { ProfScope prof(g_prof_rt_hash); changed = sparse_hash(ram(cs.desc.color_base), cs.bytes()) != cs.ram_sparse; }
    if (!changed) return false;
    ++g_n_up_changed;
    upload_color(cs);
    cs.gl_dirty = false;
    return true;
}

void poll_readbacks() {
    for (auto& cs : g_color)
        if (cs.pending) finish_readback(cs, false);
}

// Find (or create) the GL copy of the colour surface `s`, refreshed from RAM if
// the CPU changed it since GL last synchronised.
// `any_layout`: a clear addresses pixels, not memory layout, so it may use an
// existing copy that differs only in swizzling/pitch (VC3 clears its swizzled
// render targets through a linear surface description every frame).
ColorSurface& bind_color(const Surface& s, bool any_layout = false) {
    if (any_layout)
        for (auto& cs : g_color) {
            const Surface& d = cs.desc;
            if (d.color_base == s.color_base && d.width == s.width && d.height == s.height && d.color_fmt == s.color_fmt)
                return cs;
        }
    for (auto it = g_color.begin(); it != g_color.end(); ++it) {
        const Surface& d = it->desc;
        if (d.color_base != s.color_base) continue;
        if (d.width == s.width && d.height == s.height && d.color_pitch == s.color_pitch && d.color_fmt == s.color_fmt &&
            d.swizzled == s.swizzled) {
            if (it->wide != wide_surface(s)) {  // became displayed (or not): rebuild at the new width
                finish_readback(*it, true);
                if (it->gl_dirty) download_color(*it);
                g_dev->destroy_texture(it->tex);
                g_color.erase(it);
                break;
            }
            // CPU writes to a render target are detected once per frame.
            if (it->gl_dirty) reload_if_cpu_wrote(*it);
            if (!it->gl_dirty && it->checked_epoch != g_epoch) {
                it->checked_epoch = g_epoch;
                finish_readback(*it, true);
                bool changed;
                { ProfScope prof(g_prof_rt_hash); changed = hash_bytes(ram(s.color_base), it->bytes()) != it->ram_hash; }
                if (changed) { ++g_n_up_changed; upload_color(*it); }
            }
            return *it;
        }
        finish_readback(*it, true);
        if (it->gl_dirty) { ++g_n_dl_reshape; download_color(*it); }  // same address, new shape: retire the old copy
        g_dev->destroy_texture(it->tex);
        g_color.erase(it);
        break;
    }
    ColorSurface cs;
    cs.desc = s;
    cs.wide = wide_surface(s);
    cs.gw = cs.wide ? int(std::lround(s.width * g_scale / g_wide_k / 2.0)) * 2 : s.width * g_scale;
    cs.gh = s.height * g_scale;
    cs.tex = g_dev->create_texture(cs.gw, cs.gh, 1, dev::Fmt::BGRA8);
    ++g_n_up_new;
    upload_color(cs);
    g_color.push_back(cs);
    return g_color.back();
}

dev::Tex bind_zeta(const Surface& s, int gw, int gh) {
    if (!s.zeta_fmt) return {};
    for (auto& z : g_zeta)
        if (z.base == s.zeta_base && z.w == gw && z.h == gh) return z.tex;
    ZetaSurface z{s.zeta_base, gw, gh, {}};
    z.tex = g_dev->create_texture(gw, gh, 1, dev::Fmt::D24S8);
    g_zeta.push_back(z);
    return z.tex;
}


// Bind the render target for `s`; returns the colour surface (nullptr if none).
ColorSurface* bind_target(const Surface& s, bool any_layout = false) {
    if (s.width <= 0 || s.height <= 0) return nullptr;
    ColorSurface& cs = bind_color(s, any_layout);
    const dev::Tex z = bind_zeta(s, cs.gw, cs.gh);
    g_dev->set_target(cs.tex, z, cs.gw, cs.gh);
    g_cur_wide = cs.wide ? float(g_wide_k) : 1.0f;
    g_cur_gw = cs.gw;
    g_cur_color_tex = cs.tex;
    g_cur_gh = cs.gh;
    g_cur_native_w = s.width;
    return &cs;
}

// Write back any GL-modified surface overlapping [base, base+size).
void sync_range(u32 base, u32 size) {
    for (auto& cs : g_color)
        if ((cs.gl_dirty || cs.pending) && base < cs.desc.color_base + cs.bytes() && cs.desc.color_base < base + size) {
            if (cs.gl_dirty) ++g_n_dl_sync;
            download_color(cs);
        }
}

// ================================================================ textures
struct CachedTexture {
    dev::Tex tex;
    u64 hash = 0;
    u64 epoch = ~0ull;
    bool compressed = false;  // S3TC storage (DXT uploaded as is)
};
std::map<std::array<u32, 7>, CachedTexture> g_textures;

u32 level_bytes(const Texture& t, u32 w, u32 h) {
    switch (t.fmt) {
        case 0x0C: return std::max(1u, (w + 3) / 4) * std::max(1u, (h + 3) / 4) * 8;
        case 0x0E: case 0x0F: return std::max(1u, (w + 3) / 4) * std::max(1u, (h + 3) / 4) * 16;
        default: break;
    }
    if (t.linear) return t.pitch * h;
    u32 bpp = 4;
    switch (t.fmt) {
        case 0x00: case 0x01: case 0x0B: case 0x19: bpp = 1; break;
        case 0x02: case 0x03: case 0x04: case 0x05: case 0x1A: case 0x27: case 0x28: case 0x29: case 0x2C: case 0x32: bpp = 2; break;
        default: break;
    }
    return w * h * bpp;
}

// Distance between cube faces: one face with its whole mip chain, 128-byte aligned.
u32 cube_face_stride(const Texture& t) {
    u32 sum = 0;
    for (u32 l = 0; l < t.levels; ++l) sum += level_bytes(t, std::max(1u, t.w >> l), std::max(1u, t.h >> l));
    return (sum + 127) & ~127u;
}

// Mip levels to upload: the game's chain (stored after level 0, each level its own swizzled
// image), capped at the full chain; rect (linear) textures have none.
u32 gl_levels(const Texture& t) {
    if (t.linear) return 1;
    u32 full = 1;
    while ((std::max(t.w, t.h) >> full) != 0) ++full;
    return std::min(t.levels, full);
}

// One level's dimensions and its offset inside a face.
Texture mip_level(const Texture& t, u32 l, u32& offset) {
    offset = 0;
    for (u32 i = 0; i < l; ++i) offset += level_bytes(t, std::max(1u, t.w >> i), std::max(1u, t.h >> i));
    Texture m = t;
    m.w = std::max(1u, t.w >> l);
    m.h = std::max(1u, t.h >> l);
    m.log_w = t.log_w > l ? t.log_w - l : 0;
    m.log_h = t.log_h > l ? t.log_h - l : 0;
    return m;
}

u32 texture_bytes(const Texture& t) {
    if (t.cube) return cube_face_stride(t) * 6;
    u32 off = 0;
    mip_level(t, gl_levels(t), off);
    return off;
}

// Render-to-texture: a texture that is exactly a surface the GPU rendered (same address,
// size, layout and pixel size) is sampled from the GPU copy, with no RAM round trip.
// *alpha_one: sample alpha as 1 (the surface or texture format has none).
dev::Tex alias_render_target(const Texture& t, float* wide_k, bool* alpha_one) {
    if (t.cube) return {};
    bool argb32 = false, rgb565 = false, x8 = false;
    switch (t.fmt) {
        case 0x06: case 0x12: argb32 = true; break;              // A8R8G8B8 (swizzled / linear)
        case 0x07: case 0x1E: argb32 = true; x8 = true; break;   // X8R8G8B8
        case 0x05: case 0x11: rgb565 = true; break;              // R5G6B5
        default: return {};
    }
    for (auto& cs : g_color) {
        const Surface& d = cs.desc;
        if (!(cs.gl_dirty || cs.pending) || d.color_base != t.base || u32(d.width) != t.w || u32(d.height) != t.h) continue;
        if (d.swizzled == t.linear) continue;
        if (argb32 && d.color_bpp() != 4) continue;
        if (rgb565 && d.color_fmt != 3) continue;
        if (reload_if_cpu_wrote(cs)) return {};  // RAM is current: sample it as a texture
        cs.sampled = true;
        // X8 surfaces/textures read with opaque alpha, as the RAM copy would have.
        if (alpha_one) *alpha_one = x8 || rgb565 || d.color_fmt != 8;
        if (wide_k) *wide_k = cs.wide ? float(g_wide_k) : 1.0f;
        return cs.tex;
    }
    return {};
}

// Block-compressed formats for the Xbox DXT formats (BGRA8: none, decode on the CPU).
dev::Fmt bc_format(u32 fmt) {
    switch (fmt) {
        case 0x0C: return dev::Fmt::BC1;  // DXT1 (1-bit alpha)
        case 0x0E: return dev::Fmt::BC2;  // DXT2/3
        case 0x0F: return dev::Fmt::BC3;  // DXT4/5
        default: return dev::Fmt::BGRA8;
    }
}

// Decode one face of a texture into BGRA8 (A8R8G8B8 as a little-endian u32).
// 32-bit formats take fast paths (row copies; swizzled ones via per-axis offset
// tables); the rest go texel by texel.
void decode_texture(const Texture& t, u32* px) {
    const bool argb = t.fmt == 0x06 || t.fmt == 0x12, xrgb = t.fmt == 0x07 || t.fmt == 0x1E;
    if (argb || xrgb) {
        const u32 alpha = xrgb ? 0xFF000000u : 0;
        if (t.linear) {
            for (u32 y = 0; y < t.h; ++y) {
                const u8* row = ram(t.base + y * t.pitch);
                u32* out = px + size_t(y) * t.w;
                std::memcpy(out, row, size_t(t.w) * 4);
                if (alpha) for (u32 x = 0; x < t.w; ++x) out[x] |= alpha;
            }
        } else {
            // Swizzled offset = (bits of x) | (bits of y), interleaved: tabulate each axis.
            std::vector<u32> ox(t.w), oy(t.h);
            for (u32 x = 0; x < t.w; ++x) ox[x] = swizzle_offset(x, 0, t.log_w, t.log_h) * 4;
            for (u32 y = 0; y < t.h; ++y) oy[y] = swizzle_offset(0, y, t.log_w, t.log_h) * 4;
            const u8* base = ram(t.base);
            for (u32 y = 0; y < t.h; ++y) {
                u32* out = px + size_t(y) * t.w;
                const u8* rowb = base + oy[y];
                for (u32 x = 0; x < t.w; ++x) {
                    u32 v;
                    std::memcpy(&v, rowb + ox[x], 4);
                    out[x] = v | alpha;
                }
            }
        }
        return;
    }
    for (u32 y = 0; y < t.h; ++y)
        for (u32 x = 0; x < t.w; ++x) px[size_t(y) * t.w + x] = fetch_texel(t, x, y);
}

dev::Tex texture_for(const Texture& t, float* wide_k = nullptr, u64* content = nullptr, bool* alpha_one = nullptr) {
    if (wide_k) *wide_k = 1.0f;
    if (content) *content = 0;
    if (alpha_one) *alpha_one = false;
    if (const dev::Tex rt = alias_render_target(t, wide_k, alpha_one)) return rt;
    const std::array<u32, 7> key{t.base, t.fmt, t.w, t.h, t.pitch, (t.linear ? 1u : 0u) | (t.cube ? 2u : 0u) | (t.levels << 4),
                                 t.fmt == 0x0B ? t.palette : 0u};
    CachedTexture& c = g_textures[key];
    struct Out { CachedTexture& c; u64* content; ~Out() { if (content) *content = c.hash; } } out{c, content};
    if (c.tex && c.epoch == g_epoch) return c.tex;
    const u32 bytes = texture_bytes(t);
    sync_range(t.base, bytes);  // render-to-texture: make RAM current first
    u64 h;
    {
        ProfScope prof(g_prof_tex_hash);
        h = hash_bytes(ram(t.base), bytes);
        if (t.fmt == 0x0B) h ^= hash_bytes(ram(t.palette & ~0x3Fu), 1024) * 31;
    }
    c.epoch = g_epoch;
    if (c.tex && c.hash == h) return c.tex;
    c.hash = h;
    ProfScope prof_up(g_prof_tex_upload);
    const u32 levels = gl_levels(t);
    const dev::Fmt bc = g_dev->supports_bc() ? bc_format(t.fmt) : dev::Fmt::BGRA8;
    if (c.tex && c.compressed != (bc != dev::Fmt::BGRA8)) {
        g_dev->destroy_texture(c.tex);
        c.tex = {};
    }
    if (!c.tex) {
        c.tex = g_dev->create_texture(int(t.w), int(t.h), int(levels), bc, t.cube);
        c.compressed = bc != dev::Fmt::BGRA8;
    }
    const int faces = t.cube ? 6 : 1;
    const u32 stride = t.cube ? cube_face_stride(t) : 0;
    // DXT: the GPU decodes it natively; upload the blocks as they are.
    if (c.compressed) {
        for (int face = 0; face < faces; ++face)
            for (u32 l = 0; l < levels; ++l) {
                u32 off;
                const Texture m = mip_level(t, l, off);
                const u8* src = ram(t.base + u32(face) * stride + off);
                g_dev->upload(c.tex, int(l), face, 0, 0, int(m.w), int(m.h), src, level_bytes(t, m.w, m.h));
            }
        return c.tex;
    }
    std::vector<u32> px(size_t(t.w) * t.h);
    // VC3_DUMP_CUBES=1: write each cube map upload to work/cubes/ as a strip of its 6 faces.
    static const bool dump_cubes = std::getenv("VC3_DUMP_CUBES") != nullptr;
    std::vector<u32> strip;
    if (t.cube && dump_cubes) strip.resize(size_t(t.w) * 6 * t.h);
    for (int face = 0; face < faces; ++face) {  // +X -X +Y -Y +Z -Z, as in GL
        Texture ft = t;
        ft.base = t.base + u32(face) * stride;
        for (u32 l = 1; l < levels; ++l) {  // smaller levels first; level 0 below reuses px
            u32 off;
            Texture m = mip_level(ft, l, off);
            m.base = ft.base + off;
            decode_texture(m, px.data());
            g_dev->upload(c.tex, int(l), face, 0, 0, int(m.w), int(m.h), px.data(), size_t(m.w) * m.h * 4);
        }
        decode_texture(ft, px.data());
        if (!strip.empty())
            for (u32 y = 0; y < t.h; ++y)
                std::memcpy(&strip[size_t(y) * t.w * 6 + size_t(face) * t.w], &px[size_t(y) * t.w], t.w * 4);
        g_dev->upload(c.tex, 0, face, 0, 0, int(t.w), int(t.h), px.data(), px.size() * 4);
    }
    if (!strip.empty()) {
        static int n = 0;
        std::filesystem::create_directories("work/cubes");
        char name[96];
        std::snprintf(name, sizeof name, "work/cubes/cube_%03d_%08x_fmt%02x_%ux%u.ppm", n++, t.base, t.fmt, t.w, t.h);
        if (std::FILE* f = std::fopen(name, "wb")) {
            std::fprintf(f, "P6\n%u %u\n255\n", t.w * 6, t.h);
            for (u32 v : strip) { const u8 rgb[3] = {u8(v >> 16), u8(v >> 8), u8(v)}; std::fwrite(rgb, 1, 3, f); }
            std::fclose(f);
        }
    }
    return c.tex;
}

u8 dev_wrap(u32 mode) {
    switch (mode) {
        case 1: return dev::kRepeat;
        case 2: return dev::kMirror;
        case 4: return dev::kClampBorder;
        default: return dev::kClampEdge;
    }
}

// ================================================================ shaders
// Resources shared by every generated program (same bindings in the Vulkan backend):
// binding 0 the per-draw block below, 1 the vertex stage's block (fixed-function state or
// vertex-program constants), 3 the widescreen HUD sides, 4-7 2D textures, 8-11 cube maps.
// Varyings have fixed locations so any vertex shader pairs with any fragment shader.
const char* const kDrawBlock = R"(layout(std140, binding = 0) uniform Draw {
    vec4 uC0[8], uC1[8];    // combiner constants per stage
    vec4 uFC0, uFC1, uFogColor;
    vec4 uBumpMat[4];
    vec4 uTexScale[4];      // .xy: texture coordinate scale (rect textures use texels)
    vec4 uTexWide;          // per stage: a widened target's 4:3 share of its width (1: normal)
    vec4 uBumpScale, uBumpOffset;
    vec4 uClipX;            // .xy: the 4:3 part, framebuffer x range
    float uAlphaRef, uFogEnable, uPointSize;
    // Widescreen: uWide is the 4:3 picture's share of the target's width (1: off): the
    // draw keeps its shape in the middle. -1: a full-screen 2D draw, stretched to the full
    // width (vFull tells texture lookups to read whole widened targets).
    float uWide;
    float uWideShift;       // 1: HUD pieces move out (per-vertex side in hudSide[])
    int uHudBase;           // hudSide index = vertex index - uHudBase
    float uPointScale;      // render scale for point sizes a vertex program writes (oPts)
};
)";
const char* const kVsCommon = R"(layout(std430, binding = 3) readonly buffer HudSides { float hudSide[]; };
layout(location = 0) out vec4 vD0;
layout(location = 1) out vec4 vD1;
layout(location = 2) out vec4 vT0;
layout(location = 3) out vec4 vT1;
layout(location = 4) out vec4 vT2;
layout(location = 5) out vec4 vT3;
layout(location = 6) out vec4 vB0;
layout(location = 7) out vec4 vB1;
layout(location = 8) out float vFog;
layout(location = 9) out float vFull;
layout(location = 10) out float vClip;  // 1: keep inside the 4:3 part (hudSide 2: shot sparks batched with HUD pieces)
vec4 widen(vec4 c) {
    vFull = uWide < 0.0 ? 1.0 : 0.0;
    vClip = 0.0;
    if (uWide > 0.0 && uWide < 1.0) {
        float hs = uWideShift > 0.5 ? hudSide[VERTEX_ID - uHudBase] : 0.0;  // VERTEX_ID: defined by the device
        if (hs > 1.5) { vClip = 1.0; hs = 0.0; }
        c.x = c.x * uWide + hs * (1.0 - uWide) * c.w;
    }
    return c;
}
)";
const std::string kVertexShader = std::string("#version 450\n") + R"(layout(location = 0) in vec4 aPos;
layout(location = 1) in vec4 aD0;
layout(location = 2) in vec4 aD1;
layout(location = 3) in vec4 aT0;
layout(location = 4) in vec4 aT1;
layout(location = 5) in vec4 aT2;
layout(location = 6) in vec4 aT3;
layout(location = 7) in float aFog;
)" + kDrawBlock + kVsCommon + R"(void main() {
    gl_Position = widen(aPos);
    gl_PointSize = uPointSize;
    vB0 = aD0; vB1 = aD1;
    vD0 = aD0; vD1 = aD1; vT0 = aT0; vT1 = aT1; vT2 = aT2; vT3 = aT3; vFog = aFog;
}
)";

// Combiner input (register byte) as GLSL: rgb (vec3) or alpha (float).
std::string input_rgb(u32 in) {
    const std::string reg = "R[" + std::to_string(in & 0xF) + "]";
    const std::string v = (in & 0x10) ? "vec3(" + reg + ".a)" : reg + ".rgb";
    switch ((in >> 5) & 7) {
        case 0: return "max(" + v + ", 0.0)";
        case 1: return "(1.0 - clamp(" + v + ", 0.0, 1.0))";
        case 2: return "(2.0 * max(" + v + ", 0.0) - 1.0)";
        case 3: return "(1.0 - 2.0 * max(" + v + ", 0.0))";
        case 4: return "(max(" + v + ", 0.0) - 0.5)";
        case 5: return "(0.5 - max(" + v + ", 0.0))";
        case 6: return v;
        default: return "(-" + v + ")";
    }
}
std::string input_alpha(u32 in) {
    const std::string reg = "R[" + std::to_string(in & 0xF) + "]";
    const std::string v = (in & 0x10) ? reg + ".a" : reg + ".b";
    switch ((in >> 5) & 7) {
        case 0: return "max(" + v + ", 0.0)";
        case 1: return "(1.0 - clamp(" + v + ", 0.0, 1.0))";
        case 2: return "(2.0 * max(" + v + ", 0.0) - 1.0)";
        case 3: return "(1.0 - 2.0 * max(" + v + ", 0.0))";
        case 4: return "(max(" + v + ", 0.0) - 0.5)";
        case 5: return "(0.5 - max(" + v + ", 0.0))";
        case 6: return v;
        default: return "(-" + v + ")";
    }
}
std::string out_map(const std::string& x, u32 m) {
    switch (m) {
        case 0x08: return "(" + x + " - 0.5)";
        case 0x10: return "(" + x + " * 2.0)";
        case 0x18: return "((" + x + " - 0.5) * 2.0)";
        case 0x20: return "(" + x + " * 4.0)";
        case 0x30: return "(" + x + " * 0.5)";
        default: return x;
    }
}

struct ShaderKey {
    std::vector<u32> words;
    bool operator<(const ShaderKey& o) const { return words < o.words; }
    bool operator==(const ShaderKey& o) const { return words == o.words; }
};
struct ShaderKeyHash {
    size_t operator()(const ShaderKey& k) const {
        u64 h = 0x9E3779B97F4A7C15ull;
        for (u32 w : k.words) h = (h ^ w) * 0x100000001B3ull;
        return size_t(h ^ (h >> 29));
    }
};

ShaderKey make_key(const DrawState& d) {
    ShaderKey k;
    for (int i = 0; i < 4; ++i)
        k.words.push_back(((d.stage_modes >> (5 * i)) & 0x1F) | ((i < d.tex_used && d.tex[i].enabled) ? 0x100u : 0u) |
                          ((i < d.tex_used && d.tex[i].enabled && d.tex[i].cube) ? 0x200u : 0u));
    k.words.push_back(d.comb.stages | (d.comb.mux_msb ? 0x100u : 0u) | (d.alpha_test ? 0x1000u | (d.alpha_func << 16) : 0u));
    for (u32 s = 0; s < d.comb.stages; ++s) {
        k.words.push_back(d.comb.st[s].ci);
        k.words.push_back(d.comb.st[s].ai);
        k.words.push_back(d.comb.st[s].co);
        k.words.push_back(d.comb.st[s].ao);
    }
    k.words.push_back(d.comb.final ? d.comb.f0 : 0);
    k.words.push_back(d.comb.final ? d.comb.f1 : 0);
    k.words.push_back(d.comb.final ? 1 : 0);
    k.words.push_back(d.point_sprite ? 1 : 0);
    k.words.push_back(d.dot_mapping | d.other_stage_input);
    k.words.push_back(d.gpu_two_sided ? 1 : 0);
    return k;
}

std::string fragment_source(const ShaderKey& key, const DrawState& d) {
    std::string s = std::string("#version 450\n") + kDrawBlock + R"(layout(location = 0) in vec4 vD0;
layout(location = 1) in vec4 vD1;
layout(location = 2) in vec4 vT0;
layout(location = 3) in vec4 vT1;
layout(location = 4) in vec4 vT2;
layout(location = 5) in vec4 vT3;
layout(location = 6) in vec4 vB0;
layout(location = 7) in vec4 vB1;
layout(location = 8) in float vFog;
layout(location = 9) in float vFull;
layout(location = 10) in float vClip;
layout(binding = 4) uniform sampler2D uTex0;
layout(binding = 5) uniform sampler2D uTex1;
layout(binding = 6) uniform sampler2D uTex2;
layout(binding = 7) uniform sampler2D uTex3;
layout(binding = 8) uniform samplerCube uCube0;
layout(binding = 9) uniform samplerCube uCube1;
layout(binding = 10) uniform samplerCube uCube2;
layout(binding = 11) uniform samplerCube uCube3;
// Widescreen: a widened render target sampled by anything but full-screen 2D is read
// from its middle 4:3 part (uTexWide: that part's share of the width, 1 = normal).
vec2 wuv(vec2 uv, int i) {
    float k = uTexWide[i];
    if (k < 1.0 && vFull < 0.5) uv.x = 0.5 + (uv.x - 0.5) * k;
    return uv;
}
// Signed texel components stored as unsigned bytes (texture shader helpers).
float sign1(float x) { return (x * 255.0 - 128.0) / 127.0; }
float sign2(float x) { x *= 255.0; return x >= 128.0 ? (x - 255.5) / 127.5 : (x + 0.5) / 127.5; }
float sign3(float x) { x *= 255.0; return x >= 128.0 ? (x - 256.0) / 127.0 : x / 127.0; }
layout(location = 0) out vec4 fragColor;
void main() {
    if (vClip > 0.5 && (gl_FragCoord.x < uClipX.x || gl_FragCoord.x > uClipX.y)) discard;
    vec4 R[16];
    for (int i = 0; i < 16; ++i) R[i] = vec4(0.0);
    vec4 T[4] = vec4[4](vT0, vT1, vT2, vT3);
    float dots[4] = float[4](0.0, 0.0, 0.0, 0.0);
)";
    if (d.point_sprite) s += "    T[3] = vec4(gl_PointCoord, 1.0, 1.0);\n";
    for (int i = 0; i < 4; ++i) {
        const u32 mode = key.words[size_t(i)] & 0x1F;
        const bool has_tex = key.words[size_t(i)] & 0x100;
        const std::string I = std::to_string(i);
        // Texture-shader stages that read another stage's texel.
        const u32 dm = i >= 1 ? (d.dot_mapping >> (4 * (i - 1))) & 0xF : 0;  // stage i's dot mapping
        const int in = i == 1 ? 0 : i == 2 ? int((d.other_stage_input >> 16) & 0xF) : int((d.other_stage_input >> 20) & 0xF);
        const std::string RI = "R[" + std::to_string(8 + i) + "]", RIN = "R[" + std::to_string(8 + in) + "]";
        auto dotmap = [&](const std::string& c) -> std::string {
            switch (dm) {
                case 1: return "vec3(sign1(" + c + ".r), sign1(" + c + ".g), sign1(" + c + ".b))";
                case 2: return "vec3(sign2(" + c + ".r), sign2(" + c + ".g), sign2(" + c + ".b))";
                case 3: return "vec3(sign3(" + c + ".r), sign3(" + c + ".g), sign3(" + c + ".b))";
                default: return c + ".rgb";  // 0: 0..1 (HILO modes approximated the same way)
            }
        };
        if (i >= 1 && (mode == 6 || mode == 7)) {  // BUMPENVMAP(_LUM): offset by the transformed (du, dv)
            s += "    { vec4 src = " + RIN + "; vec2 dsdt = mat2(uBumpMat[" + I + "].xy, uBumpMat[" + I +
                 "].zw) * vec2(sign3(src.b), sign3(src.g));\n";
            if (has_tex)
                s += "      " + RI + " = texture(uTex" + I + ", wuv((T[" + I + "].xy + dsdt) * uTexScale[" + I + "].xy, " + I + "));\n";
            if (mode == 7) s += "      " + RI + " *= uBumpScale[" + I + "] * src.r + uBumpOffset[" + I + "];\n";
            s += "    }\n";
            continue;
        }
        if (i >= 1 && (mode == 0x11 || mode == 0x0A || mode == 0x0C)) {  // DOT_PRODUCT / DOT_ZW / DOT_RFLCT_SPEC
            s += "    dots[" + I + "] = dot(T[" + I + "].xyz, " + dotmap(RIN) + ");\n";
            s += "    " + RI + " = vec4(0.0);\n";
            if (mode == 0x0C && i == 3 && has_tex) {  // reflect the eye vector about the bumped normal
                s += "    { vec3 n = vec3(dots[1], dots[2], dots[3]); vec3 e = vec3(T[1].w, T[2].w, T[3].w);\n"
                     "      vec3 rv = 2.0 * n * dot(n, e) / max(dot(n, n), 1e-20) - e;\n";
                s += (key.words[size_t(i)] & 0x200) ? "      " + RI + " = texture(uCube3, rv); }\n"
                                                    : "      " + RI + " = texture(uTex3, rv.xy * 0.5 + 0.5); }\n";
            }
            continue;
        }
        if (mode == 4) s += "    R[" + std::to_string(8 + i) + "] = clamp(T[" + I + "], 0.0, 1.0);\n";
        else if (mode && has_tex && (key.words[size_t(i)] & 0x200))  // cube map: direction lookup
            s += "    R[" + std::to_string(8 + i) + "] = texture(uCube" + I + ", T[" + I + "].xyz);\n";
        else if (mode && has_tex) {
            const std::string q = mode == 1 ? "(T[" + I + "].w != 0.0 ? T[" + I + "].w : 1.0)" : "1.0";
            s += "    R[" + std::to_string(8 + i) + "] = texture(uTex" + I + ", wuv(T[" + I + "].xy / " + q + " * uTexScale[" + I + "].xy, " + I + "));\n";
        }
    }
    s += R"(    R[3] = vec4(uFogColor.rgb, uFogEnable > 0.5 ? clamp(vFog, 0.0, 1.0) : 1.0);
    R[4] = vD0;
    R[5] = vD1;
)";
    if (d.gpu_two_sided) s += "    if (!gl_FrontFacing) { R[4] = vB0; R[5] = vB1; }\n";
    s += R"(
    R[12].a = R[8].a;
)";
    for (u32 st = 0; st < d.comb.stages; ++st) {
        const auto& c = d.comb.st[st];
        const std::string S = std::to_string(st);
        const u32 cf = c.co >> 12, af = c.ao >> 12;
        s += "    {\n        R[1] = uC0[" + S + "]; R[2] = uC1[" + S + "];\n";
        s += "        vec3 A = " + input_rgb(c.ci >> 24) + ", B = " + input_rgb((c.ci >> 16) & 0xFF) + ", C = " +
             input_rgb((c.ci >> 8) & 0xFF) + ", D = " + input_rgb(c.ci & 0xFF) + ";\n";
        s += std::string("        vec3 ab = ") + ((cf & 2) ? "vec3(dot(A, B))" : "A * B") + ";\n";
        s += std::string("        vec3 cd = ") + ((cf & 1) ? "vec3(dot(C, D))" : "C * D") + ";\n";
        s += std::string("        bool mux = ") +
             (d.comb.mux_msb ? "R[12].a >= 0.5" : "(int(floor(R[12].a * 255.0 + 0.5)) & 1) != 0") + ";\n";
        s += std::string("        vec3 sum = ") + ((cf & 4) ? "(mux ? cd : ab)" : "ab + cd") + ";\n";
        s += "        float aA = " + input_alpha(c.ai >> 24) + ", aB = " + input_alpha((c.ai >> 16) & 0xFF) + ", aC = " +
             input_alpha((c.ai >> 8) & 0xFF) + ", aD = " + input_alpha(c.ai & 0xFF) + ";\n";
        s += std::string("        float aab = aA * aB, acd = aC * aD, asum = ") + ((af & 4) ? "(mux ? acd : aab)" : "aab + acd") + ";\n";
        auto wr_rgb = [&](u32 reg, const char* v, bool b2a) {
            if (!reg) return;
            const std::string R = "R[" + std::to_string(reg) + "]";
            s += "        " + R + ".rgb = clamp(" + out_map(v, cf & 0x38) + ", -1.0, 1.0);\n";
            if (b2a) s += "        " + R + ".a = " + R + ".b;\n";
        };
        auto wr_a = [&](u32 reg, const char* v) {
            if (reg) s += "        R[" + std::to_string(reg) + "].a = clamp(" + out_map(v, af & 0x38) + ", -1.0, 1.0);\n";
        };
        wr_rgb(c.co & 0xF, "cd", cf & 0x40);
        wr_rgb((c.co >> 4) & 0xF, "ab", cf & 0x80);
        wr_rgb((c.co >> 8) & 0xF, "sum", false);
        if (!(cf & 0x40)) wr_a(c.ao & 0xF, "acd");
        if (!(cf & 0x80)) wr_a((c.ao >> 4) & 0xF, "aab");
        wr_a((c.ao >> 8) & 0xF, "asum");
        s += "    }\n";
    }
    if (d.comb.final) {
        const u32 f0 = d.comb.f0, f1 = d.comb.f1, flags = f1 & 0xFF;
        s += "    R[1] = uFC0; R[2] = uFC1;\n";
        s += "    R[15] = vec4(" + input_rgb(f1 >> 24) + " * " + input_rgb((f1 >> 16) & 0xFF) + ", 0.0);\n";
        s += std::string("    { vec3 v1 = ") + ((flags & 0x40) ? "1.0 - R[5].rgb" : "R[5].rgb") + ", r0 = " +
             ((flags & 0x20) ? "1.0 - R[12].rgb" : "R[12].rgb") + "; vec3 su = v1 + r0; " +
             ((flags & 0x80) ? "su = min(su, vec3(1.0)); " : "") + "R[14] = vec4(su, 0.0); }\n";
        s += "    vec3 fA = " + input_rgb(f0 >> 24) + ", fB = " + input_rgb((f0 >> 16) & 0xFF) + ", fC = " +
             input_rgb((f0 >> 8) & 0xFF) + ", fD = " + input_rgb(f0 & 0xFF) + ";\n";
        s += "    vec4 outc = vec4(clamp(fA * fB + (1.0 - fA) * fC + fD, 0.0, 1.0), clamp(" + input_alpha((f1 >> 8) & 0xFF) +
             ", 0.0, 1.0));\n";
    } else {
        s += "    vec4 outc = R[12];\n";
    }
    if (d.alpha_test) {
        static const char* ops[] = {"false", "a < uAlphaRef", "a == uAlphaRef", "a <= uAlphaRef",
                                    "a > uAlphaRef", "a != uAlphaRef", "a >= uAlphaRef", "true"};
        s += "    { float a = floor(clamp(outc.a, 0.0, 1.0) * 255.0 + 0.5) / 255.0; if (!(" +
             std::string(ops[d.alpha_func & 7]) + ")) discard; }\n";
    }
    s += "    fragColor = clamp(outc, 0.0, 1.0);\n}\n";
    return s;
}

// Programs by structural state (the device caches them by source too).
std::unordered_map<ShaderKey, dev::Program, ShaderKeyHash> g_programs;

dev::Program program_for(const DrawState& d) {
    const ShaderKey key = make_key(d);
    dev::Program& p = g_programs[key];
    if (!p) p = g_dev->program(kVertexShader, fragment_source(key, d));
    return p;
}

// The per-draw uniform block (kDrawBlock, std140).
struct DrawUbo {
    float c0[8][4], c1[8][4];
    float fc0[4], fc1[4], fog_color[4];
    float bump_mat[4][4];
    float tex_scale[4][4];
    float tex_wide[4];
    float bump_scale[4], bump_offset[4];
    float clip_x[4];
    float alpha_ref, fog_enable, point_size, wide;
    float wide_shift;
    int32_t hud_base;
    float point_scale;
    float pad;
};
static_assert(sizeof(DrawUbo) == 528, "std140 layout of Draw");
DrawUbo g_draw_ubo;  // filled by setup_draw; hud_base completed by the draw

struct GpuVertex {
    float pos[4], d0[4], d1[4], t[4][4], fog;
};


}  // namespace

bool init() {
    const char* r = std::getenv("VC3_RENDERER");
    if (r && std::string(r) == "vulkan") {
        g_dev = dev::create_vulkan_device();
        if (!g_dev) std::fprintf(stderr, "[gpu] Vulkan unavailable: using OpenGL\n");
    }
    if (!g_dev) {
        if (!dev::gl::create_context()) {
            std::fprintf(stderr, "[gl] no OpenGL 4.5 context available\n");
            return false;
        }
        g_dev = dev::create_gl_device();
        g_shared = dev::gl::shared();
    } else {
#ifdef VC3_HAVE_VULKAN
        g_shared = vkp::active();  // the window presents device textures directly
#endif
    }
    g_active = true;
    return true;
}

bool active() { return g_active; }

void clear(const Surface& s, u32 flags, u32 rect_h, u32 rect_v, u32 color, u32 zs) {
    poll_readbacks();
    ColorSurface* cs = bind_target(s, true);
    if (!cs) return;
    const int x0 = int(rect_h & 0xFFF), x1 = int((rect_h >> 16) & 0xFFF);
    const int y0 = int(rect_v & 0xFFF), y1 = int((rect_v >> 16) & 0xFFF);
    int sx, sw;
    if (!cs->wide) {
        sx = x0 * g_scale;
        sw = (x1 - x0 + 1) * g_scale;
    } else if (x0 <= 0 && x1 >= s.width - 1) {  // full width: the whole wide surface
        sx = 0;
        sw = cs->gw;
    } else {  // part of the 4:3 picture, which sits in the middle
        const int off = (cs->gw - s.width * g_scale) / 2;
        sx = off + x0 * g_scale;
        sw = (x1 - x0 + 1) * g_scale;
    }
    const bool do_color = flags & 0xF0;
    const bool has_alpha = s.color_fmt == 8;  // no alpha channel: alpha stays 1 (see setup_draw)
    const u8 mask = u8((flags & 0x40 ? 1 : 0) | (flags & 0x20 ? 2 : 0) | (flags & 0x10 ? 4 : 0) |
                       (has_alpha ? (flags & 0x80 ? 8 : 0) : 8));
    const float rgba[4] = {((color >> 16) & 0xFF) / 255.0f, ((color >> 8) & 0xFF) / 255.0f, (color & 0xFF) / 255.0f,
                           has_alpha ? (color >> 24) / 255.0f : 1.0f};
    const bool do_depth = (flags & 1) && s.zeta_fmt;
    const bool do_stencil = (flags & 2) && s.zeta_fmt == 2;
    const float z = s.zeta_fmt == 1 ? float((zs & 0xFFFF) / 65535.0) : float((zs >> 8) / 16777215.0);
    if (do_color) cs->gl_dirty = true;
    g_dev->clear(sx, y0 * g_scale, sw, (y1 - y0 + 1) * g_scale, do_color, mask, rgba, do_depth, z, do_stencil, u8(zs & 0xFF));
}

// ---------------------------------------------------------------- widescreen HUD
// HUD panels move out to the edges of a widescreen picture (keeping their shape);
// everything else -- the crosshair, enemy markers, score popups, shootable menu items --
// stays where the game put it, so aim and what it shows keep matching. Elements
// drawn with the HUD's textures move out to the edge on their side of the screen
// (player 1 left, player 2 right). VC3_WIDE_HUD=0 keeps the HUD in the 4:3 part.
bool g_wide_hud = true;

// The gameplay HUD's textures, by content (hash_bytes of the texture data: game data,
// the same every run). Only these move: shootable menu items (name entry), banners,
// popups and the crosshair use other textures and stay where the game hit-tests them.
constexpr u64 kHudTextures[] = {
    0x015263d3d78cc80aull,  // panel atlas (badges, bars, ammo gauge, weapon panel)
    0xf80c121fe47f2e37ull,  // level / lives digits
    0x036c40758a50879bull,  // score digits
    0xcde6b4b6c95f6c73ull,  // ammo digits
    0x2f811f12202e0fabull,  // ammo gauge bullets
};
bool hud_texture(u64 content) {
    for (u64 h : kHudTextures)
        if (h == content) return true;
    return false;
}

// HUD pieces are decided per connected element across the whole frame: triangles drawn
// with HUD textures side by side in a row at a similar height (letters of a text, often
// separate draws) form one element, which moves only if all of it lies
// in a single zone, so text crossing a zone border stays whole. Elements are grouped at
// the end of each frame and the result is used for the next (HUD and text are static);
// a piece not seen in the last frame uses its own zone test meanwhile.
struct TriBox { float x0, y0, x1, y1, u0 = 0, v0 = 0, u1 = 0, v1 = 0; };
int zone_side(float x0, float y0, float x1, float y1);
// Centred HUD areas (box, last frame seen): see close_hud_frame.
std::vector<std::pair<TriBox, u64>> g_hud_center;
bool in_centre_area(const TriBox& b) {
    for (const auto& [c, ep] : g_hud_center)
        if (ep + 2 >= g_epoch && b.x0 >= c.x0 - 8 && b.x1 <= c.x1 + 8 && b.y0 >= c.y0 - 4 && b.y1 <= c.y1 + 4) return true;
    return false;
}
using BoxKey = std::array<int, 4>;
BoxKey box_key(const TriBox& b) { return {int(std::lround(b.x0)), int(std::lround(b.y0)), int(std::lround(b.x1)), int(std::lround(b.y1))}; }
std::vector<TriBox> g_hud_frame_tris;   // this frame's HUD triangles
std::map<BoxKey, int> g_hud_prev_side;  // decision per triangle position (remembered)
u64 g_hud_frame = ~0ull;

struct Steady { u64 last = 0; int count = 0; };
std::map<BoxKey, Steady> g_hud_steady;  // element box -> consecutive frames seen
std::map<BoxKey, u64> g_hud_areas;      // established HUD element boxes -> last frame seen steady
// An element belongs to a HUD area if it overlaps one and has about the same size as
// the steady element that established it (the ammo gauge sliding during a reload keeps
// its size; a shot spark landing on a panel does not).
bool in_hud_area(const TriBox& b, bool size_match = true) {
    constexpr float pad = 8;
    const float w = b.x1 - b.x0, h = b.y1 - b.y0;
    for (const auto& [k, last] : g_hud_areas) {
        if (!(b.x0 < float(k[2]) + pad && b.x1 > float(k[0]) - pad && b.y0 < float(k[3]) + pad && b.y1 > float(k[1]) - pad)) continue;
        if (!size_match) return true;
        const float kw = float(k[2] - k[0]), kh = float(k[3] - k[1]);
        if (w >= 0.6f * kw - 2 && w <= 1.6f * kw + 2 && h >= 0.6f * kh - 2 && h <= 1.6f * kh + 2) return true;
    }
    return false;
}

// Shot sparks are small; panel parts that animate (the magazine sliding in a reload) are big.
bool small_piece(const TriBox& b) { return b.x1 - b.x0 <= 40.0f && b.y1 - b.y0 <= 40.0f; }
// The bullet sprite of the ammo gauge is also the blue shot spark. Bullets are unrotated
// 40x30 boxes on the gauge's slanted line (x 41 at y 296, one px left per 8 px down;
// player 2 mirrored), sliding straight up at x 41 during a reload; the spark spins and lands anywhere.
bool gauge_bullet(const TriBox& b) {
    const float w = b.x1 - b.x0, h = b.y1 - b.y0;
    if (w < 37 || w > 43 || h < 27 || h > 33 || b.y0 < 250 || b.y0 > 640) return false;
    const float line = 41.0f - std::max(0.0f, b.y0 - 296.0f) / 8.0f;
    // resting on the line / sliding in at x 41 / falling out with the magazine (x ~36, below the gauge)
    auto on = [&](float x) { return std::fabs(x - line) <= 4 || (x >= 25.0f && x <= 50.0f); };
    return on(b.x0) || on(640.0f - b.x1);
}
// Same place and size as a steady HUD element (centre within 12 px, size within 40%).
bool at_steady_spot(const TriBox& b) {
    const float cx = (b.x0 + b.x1) / 2, cy = (b.y0 + b.y1) / 2, w = b.x1 - b.x0, h = b.y1 - b.y0;
    for (const auto& [k, last] : g_hud_areas) {
        const float kx = float(k[0] + k[2]) / 2, ky = float(k[1] + k[3]) / 2, kw = float(k[2] - k[0]), kh = float(k[3] - k[1]);
        if (std::fabs(cx - kx) <= 12 && std::fabs(cy - ky) <= 12 && w >= 0.6f * kw && w <= 1.4f * kw + 2 && h >= 0.6f * kh && h <= 1.4f * kh + 2)
            return true;
    }
    return false;
}
bool inside_steady_box(const TriBox& b) {
    constexpr float pad = 12;  // the magazine's bullet slots stick out of the gauge frame a little
    for (const auto& [k, last] : g_hud_areas)
        if (b.x0 >= float(k[0]) - pad && b.x1 <= float(k[2]) + pad && b.y0 >= float(k[1]) - pad && b.y1 <= float(k[3]) + pad) return true;
    return false;
}

void close_hud_frame() {
    std::vector<TriBox>& tris = g_hud_frame_tris;
    const size_t n = tris.size();
    std::vector<int> parent(n);
    for (size_t i = 0; i < n; ++i) parent[i] = int(i);
    auto find = [&](int i) { while (parent[size_t(i)] != i) i = parent[size_t(i)] = parent[size_t(parent[size_t(i)])]; return i; };
    constexpr float gap = 16.0f;  // letter and word gaps
    for (size_t i = 0; i < n; ++i)
        for (size_t j = i + 1; j < n; ++j) {
            const TriBox &a = tris[i], &b = tris[j];
            // Side by side (small gap), overlapping rows, similar height: letters of a line,
            // not a thin bar beside a tall badge.
            // Two wide panels are separate elements even when they nearly touch: the boss's
            // life bar frame (centred) beside the score panel (top left) dragged the panel
            // into the middle, while its digits still moved out.
            const float ha = a.y1 - a.y0, hb = b.y1 - b.y0;
            const bool two_panels = a.x1 - a.x0 > 60.0f && b.x1 - b.x0 > 60.0f;
            if (!two_panels && a.x0 <= b.x1 + gap && b.x0 <= a.x1 + gap && a.y0 < b.y1 && b.y0 < a.y1 &&
                std::max(ha, hb) <= 2.0f * std::max(1.0f, std::min(ha, hb)))
                parent[size_t(find(int(i)))] = find(int(j));
        }
    std::vector<TriBox> box(n, TriBox{1e30f, 1e30f, -1e30f, -1e30f});
    for (size_t i = 0; i < n; ++i) {
        TriBox& b = box[size_t(find(int(i)))];
        b.x0 = std::min(b.x0, tris[i].x0); b.y0 = std::min(b.y0, tris[i].y0);
        b.x1 = std::max(b.x1, tris[i].x1); b.y1 = std::max(b.y1, tris[i].y1);
    }
    // Only steady elements are HUD: one that stays at the same place for 8 frames
    // establishes a HUD area (remembered for 10 s); elements overlapping such an area move
    // too (the ammo gauge sliding during a reload). Short-lived things drawn from the HUD
    // atlas -- the shot effect where a bullet lands near the screen edge -- stay put.
    std::vector<int> elem_side(n, 0);
    for (size_t i = 0; i < n; ++i) {
        if (find(int(i)) != int(i)) continue;
        const TriBox& b = box[i];
        int side = zone_side(b.x0, b.y0, b.x1, b.y1);
        // A wide element that stays in the middle of a HUD band (the life bar's frame) marks
        // a centred area: what's drawn inside it stays put too (the bar's fill, whose centre
        // moves left as it drains).
        if (!side && b.x1 - b.x0 >= 100.0f && b.x1 - b.x0 <= 400.0f && (b.y0 >= 360.0f || b.y1 <= 72.0f)) {
            bool known = false;
            for (auto& [cb, ep] : g_hud_center)
                if (std::fabs(cb.x0 - b.x0) < 2 && std::fabs(cb.x1 - b.x1) < 2 && std::fabs(cb.y0 - b.y0) < 2 && std::fabs(cb.y1 - b.y1) < 2) {
                    ep = g_epoch;
                    known = true;
                }
            if (!known && g_hud_center.size() < 16) g_hud_center.push_back({b, g_epoch});
        }
        if (side && in_centre_area(b)) side = 0;
        if (side) {
            Steady& st = g_hud_steady[box_key(b)];
            st.count = st.last + 1 == g_epoch ? st.count + 1 : 1;
            st.last = g_epoch;
            if (st.count >= 8) g_hud_areas[box_key(b)] = g_epoch;
        }
        elem_side[i] = side;
    }
    for (auto it = g_hud_areas.begin(); it != g_hud_areas.end();)  // forget areas unseen for 10 s
        it = it->second + 600 < g_epoch ? g_hud_areas.erase(it) : std::next(it);
    std::erase_if(g_hud_center, [](const auto& c) { return c.second + 600 < g_epoch; });
    if (g_hud_steady.size() > 20000) g_hud_steady.clear();
    // Decisions are remembered by position (blinking text keeps its decision when it
    // comes back); every frame's grouping updates them.
    if (g_hud_prev_side.size() > 50000) g_hud_prev_side.clear();
    for (size_t i = 0; i < n; ++i) g_hud_prev_side[box_key(tris[i])] = elem_side[size_t(find(int(i)))];
    tris.clear();
}

// Which edge a whole HUD element goes to: by its centre, with a dead zone in the middle
// (the health bar and centred texts stay put). Elements keep moving consistently while
// they animate (the ammo gauge sliding during a reload).
int zone_side(float x0, float y0, float x1, float y1) {
    if (x1 - x0 > 400.0f) return 0;  // spans most of the screen: not a side panel
    // HUD panels sit against a screen edge or in the top/bottom bands; anything else drawn
    // with the HUD atlas (shot effects, the communication window) stays where it is.
    // Under a top corner panel (down to y 112: the kill points and the shot names such as
    // JUSTICE SHOT appear there) counts as that corner: decided by the band or by touching the
    // edge, those texts flipped between moving and staying as they changed (jumping about).
    const bool under_corner = y1 <= 112.0f && (x0 < 200.0f || x1 > 440.0f);
    if (!(x0 < 50.0f || x1 > 590.0f || y1 <= 72.0f || y0 >= 360.0f || under_corner)) return 0;  // gauge bullets start at x 41; comm window at 58
    const float cx = (std::max(x0, 0.0f) + std::min(x1, 640.0f)) / 2;
    return cx < 320.0f - 48.0f ? -1 : cx > 320.0f + 48.0f ? 1 : 0;
}

// Per-vertex sides for the draw being set up (GPU path: from pgraph; CPU path: here),
// read by the vertex shader from an SSBO at gl_VertexID - uHudBase.
u64 g_draw_tex0_hash = 0;  // content hash of stage 0's texture for the draw being set up
// Per-vertex sides for the draw being set up, read by the vertex shader from an SSBO at
// gl_VertexID - uHudBase.
std::vector<float> g_hud_sides;
std::vector<TriBox> g_draw_tris;    // this draw's triangles (640x480 bounds)
std::vector<u32> g_draw_corners;    // their corner vertices (3 each)
size_t g_draw_vertices = 0;
int32_t g_hud_base = 0;

// Whether this draw's HUD pieces may move: not stretched full-screen draws, nor
// anything that follows the aim (the crosshair).
// The HUD only moves during gameplay -- frames after one that drew the score panel's
// frame (an atlas piece in a top corner). Menus reuse the HUD textures for things that
// must stay put (the mission-select cursor follows the aim), and cutscene text (drawn
// with the score font) is better left where it is.
constexpr u64 kHudAtlas = 0x015263d3d78cc80aull;
constexpr u64 kAmmoSheet = 0x2f811f12202e0fabull;  // ammo gauge bullets (also the communication window's decorations)
u64 g_score_seen_epoch = ~0ull;  // last frame the gameplay HUD's score panel was drawn

// The HUD atlas also holds the shot sparks (frames in u 0.7..1 x v 0.375..0.5 and
// v 0.59..0.67 -- game data, found by logging parts that turn up all over the screen; identified by the atlas's content hash). Spark
// pieces never move, wherever they land; every other atlas piece belongs to the HUD
// (panels, the magazine's bullets sliding in, the red flash of the top bullet).
bool spark_part(const TriBox& b) {
    return b.u0 >= 0.70f && ((b.v0 >= 0.37f && b.v1 <= 0.51f) ||   // blue spark, 2 frames
                             (b.v0 >= 0.58f && b.v1 <= 0.68f));    // second spark animation, 3 frames
}

struct RecentSpark { float x, y; u64 epoch; };
std::vector<RecentSpark> g_recent_sparks;
RecentSpark g_mag_fall{0, 0, 0};  // where/when the magazine last started falling out

bool hud_enabled(const DrawState& d) {
    // VC3_TRACE_HUD_TEX=1 (testing): each texture drawn flat in 2D during gameplay, with where (up to 12
    // places per texture) -- to find HUD pieces missing from kHudTextures.
    if (static const bool tr = std::getenv("VC3_TRACE_HUD_TEX") != nullptr;
        tr && !d.full_screen_2d && d.flat2d && !g_draw_tris.empty() && g_score_seen_epoch != ~0ull && g_score_seen_epoch + 1 >= g_epoch) {
        // =top: every distinct (texture, place) in the top band, once each (a boss's bar and name).
        static const bool top = !std::strcmp(std::getenv("VC3_TRACE_HUD_TEX"), "top");
        static std::map<u64, int> seen;
        static std::set<std::pair<u64, BoxKey>> seen_top;
        TriBox bb = g_draw_tris[0];
        for (const TriBox& b : g_draw_tris) {
            bb.x0 = std::min(bb.x0, b.x0); bb.y0 = std::min(bb.y0, b.y0);
            bb.x1 = std::max(bb.x1, b.x1); bb.y1 = std::max(bb.y1, b.y1);
        }
        const bool log_it = top ? bb.y1 < 140.0f && seen_top.size() < 5000 && seen_top.insert({g_draw_tex0_hash, box_key(bb)}).second
                                : seen[g_draw_tex0_hash]++ < 12;
        if (log_it) {
            std::fprintf(stderr, "[hudtex] %016llx %s frame %llu box %.0f,%.0f-%.0f,%.0f (%zu tris)\n",
                         (unsigned long long)g_draw_tex0_hash, hud_texture(g_draw_tex0_hash) ? "HUD  " : "other",
                         (unsigned long long)g_epoch, bb.x0, bb.y0, bb.x1, bb.y1, g_draw_tris.size());
        }
    }
    if (!g_wide_hud || d.full_screen_2d || !d.flat2d || g_draw_tris.empty() || !hud_texture(g_draw_tex0_hash)) return false;
    if (g_draw_tex0_hash == kHudAtlas)
        for (const TriBox& b : g_draw_tris)  // the score panel frame: ~190 px wide, in a top corner
            if (b.y0 < 16 && b.y1 < 70 && b.x1 - b.x0 > 170 && b.x1 - b.x0 < 220 && (b.x0 < 16 || b.x1 > 624)) g_score_seen_epoch = g_epoch;
    if (g_score_seen_epoch == ~0ull || g_score_seen_epoch + 1 < g_epoch) return false;  // not in gameplay
    if (g_hud_frame != g_epoch) {
        close_hud_frame();
        g_hud_frame = g_epoch;
    }
    // Default: stay and be clipped to the 4:3 part like other 2D (2). Only pieces moving out to an
    // edge (-1 / 1) and sparks of a shot into a side area (0) may be drawn in the side areas: a piece
    // left at 0 was drawn there unclipped -- the communication window's parts the game parks just off
    // its 4:3 screen showed in the side area (with widescreen shooting).
    g_hud_sides.assign(g_draw_vertices, 2.0f);
    bool any = false;
    const bool atlas_draw = g_draw_tex0_hash == kHudAtlas;
    for (size_t t = 0; t < g_draw_tris.size(); ++t) {
        // Sparks don't take part in grouping: one landing beside a
        // panel would drag the panel's element toward the middle.
        // The bullet sprite (u 0.73-0.97, v 0.83-0.89, normal/red frames) doubles as the
        // spinning blue spark: only pieces on the gauge line are bullets.
        const TriBox& tb = g_draw_tris[t];
        const bool flash = tb.u0 >= 0.72f && tb.v0 >= 0.82f && tb.v1 <= 0.90f;
        // Seen last frame: its element's decision. New (a changed digit or bullet count):
        // its own zone test, so steady panels don't flicker when a part changes.
        auto it = g_hud_prev_side.find(box_key(g_draw_tris[t]));
        // Sparks are spawned at a gun's aim point (several HUD textures carry spark frames): a
        // piece appearing there that isn't an established HUD piece is a spark.
        bool at_aim = false;
        // The magazine body (atlas u 0-0.2, v 0.7-1) falls out during a reload: never a spark.
        const bool magazine = atlas_draw && tb.u1 <= 0.2f && tb.v0 >= 0.70f;
        const float mcx = (tb.x0 + tb.x1) / 2, mcy = (tb.y0 + tb.y1) / 2;
        if (magazine && it == g_hud_prev_side.end()) g_mag_fall = {mcx, mcy, g_epoch};  // falling out
        // Bullets tumbling out with it (the bullet sprite, rotated) belong to the magazine too.
        const bool mag_bullet = flash && g_mag_fall.epoch + 90 >= g_epoch && std::fabs(mcx - g_mag_fall.x) <= 260 &&
                                std::fabs(mcy - g_mag_fall.y) <= 260;
        if (it == g_hud_prev_side.end() && !gauge_bullet(tb) && !magazine && !mag_bullet)
            for (int gun = 0; gun < 2 && !at_aim; ++gun) {
                float ax, ay;
                vc3::io::aim_point(gun, ax, ay);
                at_aim = ax >= tb.x0 - 24 && ax <= tb.x1 + 24 && ay >= tb.y0 - 24 && ay <= tb.y1 + 24;
            }
        // ...and its later animation frames stay sparks after the aim has moved on.
        const float cx = (tb.x0 + tb.x1) / 2, cy = (tb.y0 + tb.y1) / 2;
        if (!at_aim && it == g_hud_prev_side.end() && !gauge_bullet(tb) && !magazine && !mag_bullet)
            for (const auto& sp : g_recent_sparks)
                if (sp.epoch + 30 >= g_epoch && std::fabs(cx - sp.x) <= 48 && std::fabs(cy - sp.y) <= 48) { at_aim = true; break; }
        const bool spark = at_aim || (atlas_draw && spark_part(tb));  // the bullet sprite spark is caught at the aim
        if (spark) {
            if (g_recent_sparks.size() >= 32) g_recent_sparks.erase(g_recent_sparks.begin());
            g_recent_sparks.push_back({cx, cy, g_epoch});
        }
        if (!spark) g_hud_frame_tris.push_back(g_draw_tris[t]);
        const TriBox& b = g_draw_tris[t];
        // New pieces: digits and gauge parts join the panel they overlap; pieces from the
        // atlas (which also holds the shot spark) must match a panel's size too.
        int side = it != g_hud_prev_side.end() ? it->second : zone_side(b.x0, b.y0, b.x1, b.y1);
        if (side && in_centre_area(b)) side = 0;  // inside a centred element (the life bar's fill)
        if ((magazine || mag_bullet) && it == g_hud_prev_side.end()) side = g_mag_fall.x < 320.0f ? -1 : 1;  // falls with its panel
        // The atlas also holds the shot spark: only learned panel parts move.
        if (spark) {  // stays where it landed, clipped to the 4:3 part like other 2D -- unless it
                      // landed in a widescreen side area (a shot there: widescreen shooting)
            // Only a spark at a gun aimed into a side area: other atlas pieces counted as sparks (the
            // communication window's, parked off-screen while it animates) stay clipped to the 4:3 part.
            const float sx = (b.x0 + b.x1) / 2, sy = (b.y0 + b.y1) / 2;
            bool side_aim = false;
            if (sx < 0.0f || sx > 640.0f)
                for (int gun = 0; gun < 2 && !side_aim; ++gun) {
                    float ax, ay;
                    vc3::io::aim_point(gun, ax, ay);
                    side_aim = (ax < 0.0f || ax > 640.0f) && std::fabs(ax - sx) <= 64.0f && std::fabs(ay - sy) <= 64.0f;
                }
            const float s = side_aim ? 0.0f : 2.0f;
            for (int k = 0; k < 3; ++k) g_hud_sides[g_draw_corners[3 * t + size_t(k)]] = s;
            any = true;
            continue;
        }
        // The ammo-gauge sheet's HUD use is the gauge (low on the left / right); in the upper half its
        // pieces are the communication window's (header bracket, stripes), which sits in the top-left band
        // like the score panel: those were taken for HUD and moved to the edge. They stay (clipped).
        if (side && g_draw_tex0_hash == kAmmoSheet && b.y1 < 240.0f) side = 0;
        if (!side) continue;
        any = true;
        for (int k = 0; k < 3; ++k) g_hud_sides[g_draw_corners[3 * t + size_t(k)]] = float(side);
    }
    return any;
}

// Textures, uniforms and pipeline state shared by both vertex paths.
bool rendered_texture(const Texture& t);
bool had_3d_this_frame(u32 base);
bool shown_outside_gameplay(u32 base, const DrawState& d);
extern std::map<u32, u64> g_3d_drawn;
dev::Tex g_fb_copy;
int g_fb_copy_w = 0, g_fb_copy_h = 0;
// Copy of the bound target for a draw that samples it: only the part the draw can read
// when that is known (d.fb_bounds), else all of it.
dev::Tex feedback_copy(dev::Tex src, const DrawState& d) {
    if (!g_fb_copy || g_fb_copy_w != g_cur_gw || g_fb_copy_h != g_cur_gh) {
        if (g_fb_copy) g_dev->destroy_texture(g_fb_copy);
        g_fb_copy = g_dev->create_texture(g_cur_gw, g_cur_gh, 1, dev::Fmt::BGRA8);
        g_fb_copy_w = g_cur_gw;
        g_fb_copy_h = g_cur_gh;
    }
    int x0 = 0, y0 = 0, x1 = g_cur_gw, y1 = g_cur_gh;
    if (d.fb_bounds) {
        // Guest pixels -> target texels (widescreen: the 4:3 picture sits in the middle).
        // Distortion sprites spread across the whole width (see feedback_sprites) read
        // the widened target with that same mapping: the copy must cover what they read,
        // or near the sides they sample stale texels (hard-edged smears).
        const bool spread = g_cur_wide < 1.0f && d.blend && !d.full_screen_2d;
        const float off = spread ? 0.0f : float(g_cur_gw - g_cur_native_w * g_scale) / 2;
        const float sx = spread ? float(g_cur_gw) / float(g_cur_native_w) : float(g_scale);
        x0 = std::clamp(int(std::floor(off + d.fbx0 * sx)), 0, g_cur_gw);
        x1 = std::clamp(int(std::ceil(off + d.fbx1 * sx)), 0, g_cur_gw);
        y0 = std::clamp(int(std::floor(d.fby0 * float(g_scale))), 0, g_cur_gh);
        y1 = std::clamp(int(std::ceil(d.fby1 * float(g_scale))), 0, g_cur_gh);
        if (x1 <= x0 || y1 <= y0) return g_fb_copy;
    }
    ++g_n_fb_copy;
    g_fb_copy_px += u64(x1 - x0) * u64(y1 - y0);
    g_dev->copy_region(src, g_fb_copy, x0, y0, x1 - x0, y1 - y0);
    return g_fb_copy;
}

dev::DrawCall g_dc;  // the draw being set up
extern const DrawState* g_trace_d;
extern const char* g_trace_path;

// NV2A (GL-valued) enums -> device enums.
u8 stencil_op(u32 op) {
    switch (op) {
        case 0x0000: return dev::kZero;
        case 0x1E01: return dev::kReplace;
        case 0x1E02: return dev::kIncr;
        case 0x1E03: return dev::kDecr;
        case 0x150A: return dev::kInvert;
        case 0x8507: return dev::kIncrWrap;
        case 0x8508: return dev::kDecrWrap;
        default: return dev::kKeep;
    }
}
u8 blend_factor(u32 f) {
    switch (f) {
        case 0x0000: return dev::kZeroF;
        case 0x0001: return dev::kOne;
        case 0x0300: return dev::kSrcColor;
        case 0x0301: return dev::kOneMinusSrcColor;
        case 0x0302: return dev::kSrcAlpha;
        case 0x0303: return dev::kOneMinusSrcAlpha;
        case 0x0304: return dev::kDstAlpha;
        case 0x0305: return dev::kOneMinusDstAlpha;
        case 0x0306: return dev::kDstColor;
        case 0x0307: return dev::kOneMinusDstColor;
        case 0x0308: return dev::kSrcAlphaSaturate;
        case 0x8001: return dev::kConstColor;
        case 0x8002: return dev::kOneMinusConstColor;
        case 0x8003: return dev::kConstAlpha;
        case 0x8004: return dev::kOneMinusConstAlpha;
        default: return dev::kOne;
    }
}
u8 blend_eq(u32 e) {
    switch (e) {
        case 0x800A: return dev::kSub;
        case 0x800B: return dev::kRevSub;
        case 0x8007: return dev::kMin;
        case 0x8008: return dev::kMax;
        default: return dev::kAdd;  // FUNC_ADD 0x8006 (and anything unsupported)
    }
}
dev::Topology topology(u32 gl_prim) {
    switch (gl_prim) {
        case 0x0000: return dev::kPoints;
        case 0x0001: return dev::kLines;
        case 0x0002: return dev::kLineLoop;
        case 0x0003: return dev::kLineStrip;
        case 0x0005: return dev::kTriStrip;
        case 0x0006: return dev::kTriFan;
        default: return dev::kTriangles;
    }
}

void setup_draw(const DrawState& d, dev::Program prog) {
    g_trace_d = &d;
    std::optional<ProfScope> ps;
    g_dc = dev::DrawCall{};
    g_dc.prog = prog;
    dev::RasterState& rs = g_dc.rs;
    ps.emplace(g_prof_texbind);

    // Textures (bound before drawing: may write back surfaces used as textures)
    float scale[8], tex_wide[4] = {1, 1, 1, 1};
    g_draw_tex0_hash = 0;
    for (int i = 0; i < 4; ++i) {
        scale[2 * i] = scale[2 * i + 1] = 1.0f;
        const u32 mode = (d.stage_modes >> (5 * i)) & 0x1F;
        if (i >= d.tex_used || !mode || mode == 4 || !d.tex[i].enabled) continue;
        const Texture& t = d.tex[i];
        bool alpha_one = false;
        dev::Tex tex = texture_for(t, &tex_wide[i], i == 0 ? &g_draw_tex0_hash : nullptr, &alpha_one);
        // Sampling the target being drawn into (water distortion re-reads the picture): GPUs
        // leave that undefined (tile-shaped garbage on NVIDIA). Sample a copy of it instead,
        // as the hardware reads what was there before the draw.
        if (tex && tex == g_cur_color_tex) tex = feedback_copy(tex, d);
        const int slot = t.cube ? 4 + i : i;  // shader bindings 4-7 / 8-11
        g_dc.tex[slot] = tex;
        dev::SamplerDesc& sd = g_dc.samp[slot];
        // Minification as the game asks (trilinear for most 3D textures): without the mip
        // chain, detailed textures (city windows) shimmer and flicker at a distance.
        // Textures with a single level (render targets, rect textures) sample level 0.
        // NV2A min filters: 1 nearest, 2 linear, 3-6 nearest/linear x mip nearest/linear.
        const u32 mf = t.min_filter < 7 ? t.min_filter : 2;
        sd.min = (mf == 1 || mf == 3 || mf == 5) ? dev::kNearest : dev::kLinear;
        sd.mip = mf == 3 || mf == 4 ? dev::kMipNearest : mf == 5 || mf == 6 ? dev::kMipLinear : dev::kMipNone;
        sd.mag = t.filter_linear ? dev::kLinear : dev::kNearest;
        sd.lod_bias = t.lod_bias;
        sd.min_lod = t.min_lod;
        sd.max_lod = t.max_lod;
        sd.wrap_u = dev_wrap(t.addr_u);
        sd.wrap_v = dev_wrap(t.addr_v);
        sd.alpha_one = alpha_one;
        const Vec4 b = argb(t.border);
        std::memcpy(sd.border, b.data(), sizeof sd.border);
        if (t.linear) {  // rect textures use texel coordinates
            scale[2 * i] = 1.0f / float(t.w);
            scale[2 * i + 1] = 1.0f / float(t.h);
        }
    }

    ps.emplace(g_prof_state);
    // Uniforms (uploaded by the draw, once it knows uHudBase)
    DrawUbo& u = g_draw_ubo;
    u = DrawUbo{};
    for (u32 s = 0; s < 8; ++s) {
        const Rgba a = s < d.comb.stages ? d.comb.st[s].c0 : Rgba{}, b = s < d.comb.stages ? d.comb.st[s].c1 : Rgba{};
        u.c0[s][0] = a.r; u.c0[s][1] = a.g; u.c0[s][2] = a.b; u.c0[s][3] = a.a;
        u.c1[s][0] = b.r; u.c1[s][1] = b.g; u.c1[s][2] = b.b; u.c1[s][3] = b.a;
    }
    for (int i = 0; i < 4; ++i) { u.tex_scale[i][0] = scale[2 * i]; u.tex_scale[i][1] = scale[2 * i + 1]; }
    // Full-screen 2D stretches if it is a solid fill, copies a widened render target (the
    // game's scene) or is blended; an opaque full-screen picture (menu background) keeps
    // its shape so what the game draws over it stays aligned.
    const bool copies_wide = tex_wide[0] < 1.0f || tex_wide[1] < 1.0f || tex_wide[2] < 1.0f || tex_wide[3] < 1.0f;
    // Blended full-screen layers with small textures are effects (glows, the crack
    // overlay) and stretch; full-screen pictures (menu backgrounds 512x512, movie frames
    // 640x480) keep their shape so what is drawn over them stays aligned.
    const bool effect_layer = d.blend && d.tex[0].w <= 256 && d.tex[0].h <= 256;
    // Screen transitions drawn from an offscreen buffer (the game-over TV static, which
    // narrows from full width to a band): full height, centred, blended -- stretched about
    // the centre so the whole widescreen takes part. Movie frames (640x480) are excluded.
    if (!d.flat2d && d.gl_prim == 0x0004) g_3d_drawn[d.surf.color_base] = g_epoch;
    // SMAA: the scene (anything but overlays) changed this target since it was last
    // anti-aliased; 3D from the vertex paths (not the CPU path's screen-space vertices).
    if (!d.overlay2d) g_smaa_dirty[d.surf.color_base] = true;
    if (d.depth_test && d.gl_prim == 0x0004 && std::strcmp(g_trace_path, "cpu") != 0) g_3d_epoch = g_epoch;
    const bool centred_transition = d.flat2d && d.blend && d.tex_used && d.tex[0].linear && rendered_texture(d.tex[0]) &&
                                    !(d.tex[0].w == 640 && d.tex[0].h == 480) && d.by0 <= 2.0f &&
                                    d.by1 >= float(d.surf.height) - 2.0f && d.bx1 - d.bx0 > float(d.surf.width) / 2 &&
                                    std::fabs(d.bx0 + d.bx1 - float(d.surf.width)) < 4.0f &&
                                    // The same buffer carries the mission-load map, a 3D scene rendered
                                    // into it (4:3 content, keeps its shape); the static is flat noise.
                                    !had_3d_this_frame(d.tex[0].base) && !shown_outside_gameplay(d.tex[0].base, d);
    const bool stretch = (d.full_screen_2d && (!d.textured() || copies_wide || effect_layer)) || centred_transition;
    // A full-screen copy of a widened target reads all of it, whatever the destination
    // (blur and glow passes copy the scene into smaller buffers).
    // A mesh drawing the widened scene back over (nearly) the whole screen is a screen
    // distortion effect (the glass shattering when you're hit): it spans the whole width
    // too. Smaller ones (preview boxes) keep their shape.
    // Bounds come from the CPU's vertex evaluation, which can't do every vertex program
    // (it returns zeros): a big mesh without usable bounds is taken as covering the screen.
    const bool no_bounds = d.bx1 - d.bx0 < 1.0f && d.by1 - d.by0 < 1.0f;
    const bool screen_effect = copies_wide && !d.full_screen_2d &&
                               ((d.bx1 - d.bx0 >= float(d.surf.width) - 40.0f && d.by1 - d.by0 >= float(d.surf.height) - 40.0f) ||
                                (d.big_mesh && no_bounds));
    // Distortion sprites that sample the picture they draw into (the heat haze at the GALE
    // fight: ~190-pixel squares offset by a small noise texture, scattered over the 640-wide
    // view): spread across the whole width with the same mapping as their reads of the
    // widened picture, so the haze covers the widescreen sides too instead of only the
    // 4:3 middle (with a seam where it ended).
    bool reads_itself = false;
    for (int i = 0; i < 4; ++i)
        reads_itself |= tex_wide[i] < 1.0f && d.tex[i].base == d.surf.color_base;
    const bool feedback_sprites = g_cur_wide < 1.0f && reads_itself && d.blend && !d.full_screen_2d;
    const bool full_read = (stretch && (g_cur_wide < 1.0f || copies_wide)) || screen_effect || feedback_sprites;
    u.wide = full_read ? -1.0f : g_cur_wide;
    const bool hud = g_cur_wide < 1.0f && !stretch && d.surf.width == 640 && d.surf.height == 480 && hud_enabled(d);
    // 2D that the game keeps off-screen in 4:3 (pieces of an animation waiting to slide
    // in) would show in the widescreen side areas: clip flat draws to the 4:3 part, as
    // the original screen does. Not for stretched layers or HUD pieces moving out.
    // Shot effects (sparks, glows, streaks) are spawned where the gun is aimed; at the 4:3
    // edge they would spill into the widescreen sides. Small blended 3D effects drawn over a
    // gun's aim point are clipped to the 4:3 part like 2D.
    // A gun aimed into a widescreen side area shoots there (widescreen shooting): what's drawn at
    // that aim point -- sparks (2D), glows and streaks (3D) -- is shown there, not clipped.
    bool shot_effect = false, side_shot = false;
    if (g_cur_wide < 1.0f && d.bx1 > d.bx0 && d.bx1 - d.bx0 < 260.0f && d.by1 - d.by0 < 260.0f && d.surf.width == 640) {
        for (int gun = 0; gun < 2; ++gun) {
            float ax, ay;
            vc3::io::aim_point(gun, ax, ay);
            const bool at_aim = ax >= d.bx0 - 16 && ax <= d.bx1 + 16 && ay >= d.by0 - 16 && ay <= d.by1 + 16;
            if (!at_aim) continue;
            if (ax < 0.0f || ax > 640.0f) side_shot = true;
            else if (d.blend && !d.flat2d) shot_effect = true;
        }
    }
    if (static const bool tc = std::getenv("VC3_TRACE_CLIP") != nullptr; tc && g_cur_wide < 1.0f && d.bx0 < 0.0f && d.by0 < 140.0f && g_epoch >= 1000 && d.bx1 - d.bx0 < 600.0f) {
        static int n = 0;
        if (n++ < 300) {
            float a0x, a0y, a1x, a1y;
            vc3::io::aim_point(0, a0x, a0y);
            vc3::io::aim_point(1, a1x, a1y);
            std::fprintf(stderr, "[clip] epoch %llu flat %d blend %d box %.0f,%.0f-%.0f,%.0f full_read %d hud %d side_shot %d shot_effect %d stretch %d tex %016llx aims %.0f,%.0f %.0f,%.0f\n",
                         (unsigned long long)g_epoch, d.flat2d, d.blend, d.bx0, d.by0, d.bx1, d.by1, full_read, hud, side_shot, shot_effect, stretch,
                         (unsigned long long)g_draw_tex0_hash, a0x, a0y, a1x, a1y);
        }
    }
    if (g_cur_wide < 1.0f && ((d.flat2d && !full_read && !hud) || shot_effect) && !side_shot) {
        rs.scissor = true;
        rs.sx = (g_cur_gw - g_cur_native_w * g_scale) / 2;
        rs.sy = 0;
        rs.sw = g_cur_native_w * g_scale;
        rs.sh = g_cur_gh;
    }
    u.wide_shift = hud ? 1.0f : 0.0f;
    u.hud_base = g_hud_base;
    if (hud) {
        const size_t hb = std::max<size_t>(g_hud_sides.size(), 1) * sizeof(float);
        u8* hp = g_dev->stream(hb, 256, g_dc.hud);
        std::memset(hp, 0, hb);
        std::memcpy(hp, g_hud_sides.data(), g_hud_sides.size() * sizeof(float));
        const float off = float(g_cur_gw - g_cur_native_w * g_scale) / 2;
        u.clip_x[0] = off;
        u.clip_x[1] = off + float(g_cur_native_w * g_scale);
    }
    std::memcpy(u.tex_wide, tex_wide, sizeof u.tex_wide);
    u.fc0[0] = d.comb.fc0.r; u.fc0[1] = d.comb.fc0.g; u.fc0[2] = d.comb.fc0.b; u.fc0[3] = d.comb.fc0.a;
    u.fc1[0] = d.comb.fc1.r; u.fc1[1] = d.comb.fc1.g; u.fc1[2] = d.comb.fc1.b; u.fc1[3] = d.comb.fc1.a;
    std::memcpy(u.fog_color, d.fog_color.data(), sizeof u.fog_color);
    u.alpha_ref = d.alpha_ref;
    if (g_no_atest) {  // F7: every alpha test passes (functions 1 less ... 7 always, GL order)
        const u32 f = d.alpha_func & 7;
        if (f == 4 || f == 6) u.alpha_ref = -1.0f;      // greater / gequal
        else if (f == 1 || f == 3) u.alpha_ref = 2.0f;  // less / lequal
    }
    u.fog_enable = d.fog ? 1.0f : 0.0f;
    u.point_size = d.point_size * float(d.surf.scale_x) * float(g_scale);
    u.point_scale = float(d.surf.scale_x) * float(g_scale);
    std::memcpy(u.bump_mat, &d.bump_mat[0][0], sizeof u.bump_mat);
    std::memcpy(u.bump_scale, d.bump_scale, sizeof u.bump_scale);
    std::memcpy(u.bump_offset, d.bump_offset, sizeof u.bump_offset);

    // Fixed-function state (NV2A enum values are the GL ones)
    rs.depth_test = d.depth_test && !g_no_depth;  // F6 (developer): every draw passes
    rs.depth_func = u8(d.depth_func & 7);
    rs.depth_write = d.depth_write;
    if (d.poly_offset) {
        // NV2A bias is in units of the surface's depth format; the GPU buffer is always 24-bit.
        rs.poly_offset = true;
        rs.poly_offset_scale = d.poly_offset_scale;
        rs.poly_offset_units = d.poly_offset_bias * (d.surf.zeta_fmt == 1 ? 256.0f : 1.0f);
    }
    if (d.stencil_test) {  // NV2A stencil enums are the GL ones (KEEP 0x1E00, ZERO 0, ...)
        rs.stencil_test = true;
        rs.stencil_func = u8(d.stencil_func & 7);
        rs.stencil_ref = u8(d.stencil_ref);
        rs.stencil_read_mask = u8(d.stencil_func_mask);
        rs.stencil_write_mask = u8(d.stencil_write_mask);
        rs.stencil_fail = stencil_op(d.stencil_fail);
        rs.stencil_zfail = stencil_op(d.stencil_zfail);
        rs.stencil_zpass = stencil_op(d.stencil_zpass);
    }
    if (d.blend) {
        rs.blend = true;
        rs.blend_src = blend_factor(d.sfactor);
        rs.blend_dst = blend_factor(d.dfactor);
        rs.blend_eq = blend_eq(d.equation);
        std::memcpy(rs.blend_color, d.blend_color.data(), sizeof rs.blend_color);
    }
    // Colour formats without alpha (X8R8G8B8, R5G6B5...): the hardware has no alpha to
    // write, and destination alpha reads as 1. Keep the GPU copy's alpha at 1 so blends
    // using DST_ALPHA behave the same (they cut wall-tile-shaped holes otherwise).
    const bool has_alpha = d.surf.color_fmt == 8;
    rs.color_mask = u8((d.color_mask_bits & 4 ? 1 : 0) | (d.color_mask_bits & 2 ? 2 : 0) | (d.color_mask_bits & 1 ? 4 : 0) |
                       ((d.color_mask_bits & 8) && has_alpha ? 8 : 0));
    rs.topology = topology(d.gl_prim);
}

// Upload the per-draw uniform block and hand the draw to the device.
// VC3_DRAW_TRACE=N: during frame N (renderer epoch, ~ guest flip), save the colour target
// after every draw to work/drawtrace/DDDD_<target>.ppm and describe each draw in
// work/drawtrace/draws.txt -- for finding what a missing effect does.
// request_draw_trace(dir) (the F9 capture) does the same for the next frame, into dir.
const DrawState* g_trace_d = nullptr;
const char* g_trace_path = "";  // "cpu", "ff", "vp"
std::mutex g_trace_m;
std::string g_trace_req_dir;  // a capture's folder, waiting for the next frame (guarded)
std::string g_trace_dir;          // the trace being written
u64 g_trace_epoch = ~0ull;        // ...during this frame
int g_trace_n = 0;
std::FILE* g_trace_log = nullptr;
// A requested trace starts with the next whole frame (GPU thread).
void trace_take_request() {
    std::lock_guard lk(g_trace_m);
    if (!g_trace_req_dir.empty() && g_trace_epoch != g_epoch + 1 && (g_trace_epoch == ~0ull || g_epoch > g_trace_epoch)) {
        if (g_trace_log) std::fclose(g_trace_log);
        g_trace_log = nullptr;
        g_trace_dir = g_trace_req_dir;
        g_trace_epoch = g_epoch + 1;
        g_trace_req_dir.clear();
        g_trace_n = 0;
    }
}
void trace_draw() {
    static const long at = std::getenv("VC3_DRAW_TRACE") ? std::atol(std::getenv("VC3_DRAW_TRACE")) : -1;
    std::string& dir = g_trace_dir;
    int& n = g_trace_n;
    std::FILE*& log = g_trace_log;
    trace_take_request();
    const bool requested = g_epoch == g_trace_epoch;
    if (!g_trace_d || (!requested && (at < 0 || long(g_epoch) != at))) return;
    if (!requested && dir.empty()) dir = "work/drawtrace";
    if (!log) {
        std::filesystem::create_directories(dir);
        log = std::fopen((dir + "/draws.txt").c_str(), "w");
        if (!log) return;
    }
    const DrawState& d = *g_trace_d;
    std::fprintf(log, "%04d %s target %08x %dx%d prim %u count %u blend %d %x/%x eq %x atest %d f%u ref %.2f depth t%d f%u w%d "
                      "stencil %d mask %x comb %u cull %d face %x ccw %d bounds %.0f,%.0f-%.0f,%.0f",
                 n, g_trace_path, d.surf.color_base, d.surf.width, d.surf.height, d.gl_prim, g_dc.count, int(d.blend), d.sfactor,
                 d.dfactor, d.equation, int(d.alpha_test), d.alpha_func, d.alpha_ref, int(d.depth_test), d.depth_func,
                 int(d.depth_write), int(d.stencil_test), d.color_mask_bits, d.comb.stages, int(d.cull), d.cull_face, int(d.front_ccw), d.bx0,
                 d.by0, d.bx1, d.by1);
    for (int i = 0; i < d.tex_used && i < 4; ++i)
        if (d.tex[i].enabled)
            std::fprintf(log, " t%d[%08x fmt%02x %ux%u mode%u]", i, d.tex[i].base, d.tex[i].fmt, d.tex[i].w, d.tex[i].h,
                         (d.stage_modes >> (5 * i)) & 0x1F);
    // Widescreen HUD: how many of this draw's vertices moved left / right / stayed clipped.
    if (g_draw_ubo.wide_shift > 0.5f && !g_hud_sides.empty()) {
        int l = 0, r = 0, c = 0;
        for (float sd : g_hud_sides) l += sd < -0.5f, r += sd > 0.5f && sd < 1.5f, c += sd > 1.5f;
        std::fprintf(log, " hud L%d R%d clip%d", l, r, c);
    }
    std::fprintf(log, "\n");
    std::fflush(log);
    const int w = g_cur_gw, h = g_cur_gh;
    std::vector<u32> px(size_t(w) * size_t(h));
    g_dev->read(g_cur_color_tex, w, h, px.data());
    const int step = std::max(1, w / 640);
    char name[96];
    std::snprintf(name, sizeof name, "/%04d_%08x.ppm", n, d.surf.color_base);
    if (std::FILE* f = std::fopen((dir + name).c_str(), "wb")) {
        std::fprintf(f, "P6\n%d %d\n255\n", w / step, h / step);
        for (int y = 0; y + step <= h; y += step)
            for (int x = 0; x + step <= w; x += step) {
                const u32 v = px[size_t(y) * size_t(w) + size_t(x)];
                const u8 rgb[3] = {u8(v >> 16), u8(v >> 8), u8(v)};
                std::fwrite(rgb, 1, 3, f);
            }
        std::fclose(f);
    }
    ++n;
}

bool smaa_enabled() { return smaa::enabled(); }

bool toggle_no_alpha_test() { return g_no_atest = !g_no_atest; }
bool toggle_no_depth_test() { return g_no_depth = !g_no_depth; }
bool debug_no_depth() { return g_no_depth; }
bool debug_no_alpha() { return g_no_atest; }
bool no_cull() { return g_no_cull; }
bool toggle_no_cull() { return g_no_cull = !g_no_cull; }

bool tracing_now() {
    static const long at = std::getenv("VC3_DRAW_TRACE") ? std::atol(std::getenv("VC3_DRAW_TRACE")) : -1;
    trace_take_request();
    return g_epoch == g_trace_epoch || (at >= 0 && long(g_epoch) == at);
}

void request_draw_trace(const std::string& dir) {
    std::lock_guard lk(g_trace_m);
    g_trace_req_dir = dir;
}

void submit_draw() {
    u8* p = g_dev->stream(sizeof(DrawUbo), g_dev->ubo_align(), g_dc.ubo0);
    std::memcpy(p, &g_draw_ubo, sizeof(DrawUbo));
    g_dev->draw(g_dc);
    trace_draw();
}

// SMAA (VC3_SMAA) goes on the 3D picture before the 2D overlays (HUD, text), which stay
// sharp: when an overlay is drawn into a displayed buffer whose scene changed since its last
// pass, in a frame with 3D (the scene may be rendered elsewhere and copied in, as in
// widescreen), the buffer is anti-aliased in place. The last run of overlays (the HUD) is
// always after such a pass; an overlay between 3D draws may get smoothed too. A buffer whose
// scene changed after its last pass (or that had none) is anti-aliased when published, if
// the frame has 3D at all (movies, menus and transitions are left alone).
dev::Tex g_smaa_tmp;
int g_smaa_tmp_w = 0, g_smaa_tmp_h = 0;
// The final-stage cheat (hooks.cpp) skips the way the game normally gets there, and the
// chosen mission's briefing map -- a 512x512 picture (rendered by the game) laid over the whole
// screen each frame, which the game removes when that mission starts -- stays up through the final
// stage's intro. Until it stops being drawn (or 20 s), that draw is skipped.
std::atomic<bool> g_hide_map{false};
u64 g_hide_map_seen = 0, g_hide_map_from = 0;
bool skip_stale_map(const DrawState& d, const std::vector<Vertex>& tris) {
    if (!g_hide_map.load()) return false;
    if (!g_hide_map_from) g_hide_map_from = g_epoch;
    if (g_epoch > g_hide_map_from + 1200 || (g_hide_map_seen && g_epoch > g_hide_map_seen + 30)) {
        g_hide_map = false;
        g_hide_map_from = g_hide_map_seen = 0;
        return false;
    }
    if (tris.size() > 6 || !d.tex_used || !d.tex[0].enabled || d.tex[0].w != 512 || d.tex[0].h != 512 ||
        !displayed(d.surf.color_base & (mem::kRamSize - 1)))
        return false;
    float x0 = 1e30f, x1 = -1e30f, y0 = 1e30f, y1 = -1e30f;
    for (const Vertex& v : tris) {
        x0 = std::min(x0, v.x); x1 = std::max(x1, v.x);
        y0 = std::min(y0, v.y); y1 = std::max(y1, v.y);
    }
    if (x0 > 2 || y0 > 2 || x1 < float(d.surf.width) - 2 || y1 < float(d.surf.height) - 2) return false;
    g_hide_map_seen = g_epoch;
    return true;
}
void hide_stale_map() { g_hide_map = true; }


void smaa_before_overlay(const DrawState& d, ColorSurface*& cs) {
    if (!smaa::enabled() || !d.overlay2d || g_3d_epoch != g_epoch || !g_smaa_dirty[d.surf.color_base]) return;
    if (!displayed(d.surf.color_base & (mem::kRamSize - 1))) return;  // not a picture that's shown
    u64& done = g_smaa_done[d.surf.color_base];
    if (!g_smaa_tmp || g_smaa_tmp_w != cs->gw || g_smaa_tmp_h != cs->gh) {
        if (g_smaa_tmp) g_dev->destroy_texture(g_smaa_tmp);
        g_smaa_tmp = g_dev->create_texture(cs->gw, cs->gh, 1, dev::Fmt::BGRA8);
        g_smaa_tmp_w = cs->gw;
        g_smaa_tmp_h = cs->gh;
    }
    if (smaa::apply(*g_dev, cs->tex, g_smaa_tmp, cs->gw, cs->gh)) {
        g_dev->copy(g_smaa_tmp, cs->tex, cs->gw, cs->gh);
        done = g_epoch;
        g_smaa_dirty[d.surf.color_base] = false;
    }
    cs = bind_target(d.surf);  // the passes drew elsewhere
}

bool samples_wide_target(const Texture& t);
void draw(const DrawState& d_in, const std::vector<Vertex>& tris) {
    if (tris.empty() || skip_stale_map(d_in, tris)) return;
    DrawState d = d_in;
    d.big_mesh = tris.size() >= 96;
    // Small draws (any target), and draws reading a widened target (screen effects).
    const bool small = tris.size() <= 384;
    if ((g_wide_k < 1.0 && (small || (d.tex_used && d.tex[0].enabled && samples_wide_target(d.tex[0])))) ||
        (smaa::enabled() && small)) {
        float x0 = 1e30f, x1 = -1e30f, y0 = 1e30f, y1 = -1e30f;
        bool flat = true;
        for (const Vertex& v : tris) {
            flat = flat && std::fabs(v.w - 1.0f) < 1e-4f;
            x0 = std::min(x0, v.x); x1 = std::max(x1, v.x);
            y0 = std::min(y0, v.y); y1 = std::max(y1, v.y);
        }
        if (g_wide_k < 1.0) d.classify_wide(flat, x0, x1, y0, y1);
        if (small) d.set_overlay2d(flat, x0, x1, y0, y1);
    }
    if (d.samples_own_target()) d.own_target_bounds(tris.data(), tris.size());
    g_draw_tris.clear();
    g_draw_corners.clear();
    g_draw_vertices = 0;
    if (g_wide_hud && d.flat2d && d.gl_prim == 0x0004 && d.surf.width == 640 && d.surf.height == 480) {
        for (size_t t = 0; t + 2 < tris.size(); t += 3) {
            const Vertex &a = tris[t], &b = tris[t + 1], &c = tris[t + 2];
            g_draw_tris.push_back({std::min({a.x, b.x, c.x}), std::min({a.y, b.y, c.y}), std::max({a.x, b.x, c.x}), std::max({a.y, b.y, c.y}),
                                   std::min({a.t[0][0], b.t[0][0], c.t[0][0]}), std::min({a.t[0][1], b.t[0][1], c.t[0][1]}),
                                   std::max({a.t[0][0], b.t[0][0], c.t[0][0]}), std::max({a.t[0][1], b.t[0][1], c.t[0][1]})});
            g_draw_corners.insert(g_draw_corners.end(), {u32(t), u32(t + 1), u32(t + 2)});
        }
        g_draw_vertices = tris.size();
    }
    poll_readbacks();
    ProfScope prof(g_prof_draw);
    ColorSurface* cs = bind_target(d.surf);
    if (!cs) return;
    smaa_before_overlay(d, cs);
    if (!cs) return;
    std::optional<ProfScope> ps;
    ps.emplace(g_prof_prog);
    g_trace_path = "cpu";
    setup_draw(d, program_for(d));  // culled on the CPU: no face culling here

    ps.emplace(g_prof_vbo);
    // Vertices: screen space -> clip space (keeping w for perspective-correct interpolation)
    const float W = float(d.surf.width), H = float(d.surf.height);
    const float zmax = d.surf.zeta_fmt == 1 ? 65535.0f : 16777215.0f;
    dev::Slice vs;
    GpuVertex* vb = reinterpret_cast<GpuVertex*>(g_dev->stream(tris.size() * sizeof(GpuVertex), 16, vs));
    for (size_t i = 0; i < tris.size(); ++i) {
        const Vertex& v = tris[i];
        GpuVertex& g = vb[i];
        const float w = v.w;
        g.pos[0] = (2.0f * v.x / W - 1.0f) * w;
        g.pos[1] = (2.0f * v.y / H - 1.0f) * w;
        g.pos[2] = std::clamp(v.z / zmax, 0.0f, 1.0f) * w;
        g.pos[3] = w;
        std::memcpy(g.d0, v.d0.data(), 16);
        std::memcpy(g.d1, v.d1.data(), 16);
        for (int t = 0; t < 4; ++t) std::memcpy(g.t[t], v.t[t].data(), 16);
        g.fog = v.fog;
    }
    g_dc.nbind = 1;
    g_dc.vbuf[0] = vs;
    g_dc.stride[0] = sizeof(GpuVertex);
    auto attr = [](u8 loc, u8 fmt, size_t off) {
        dev::VertexAttr& a = g_dc.attr[g_dc.nattr++];
        a.location = loc;
        a.binding = 0;
        a.fmt = fmt;
        a.offset = u32(off);
    };
    attr(0, dev::kF32x4, offsetof(GpuVertex, pos));
    attr(1, dev::kF32x4, offsetof(GpuVertex, d0));
    attr(2, dev::kF32x4, offsetof(GpuVertex, d1));
    for (int t = 0; t < 4; ++t) attr(u8(3 + t), dev::kF32x4, offsetof(GpuVertex, t) + sizeof(float) * 4 * size_t(t));
    attr(7, dev::kF32x1, offsetof(GpuVertex, fog));
    g_dc.count = u32(tris.size());
    g_draw_ubo.hud_base = 0;  // vertex ids start at 0 in this draw's own slice
    submit_draw();
    if (d.color_mask_bits) cs->gl_dirty = true;
}

namespace {
// ================================================================ GPU vertex path
// Fixed-function draws run their vertex processing in a generated vertex shader on
// the guest's own vertex arrays (uploaded as-is), instead of transforming every
// vertex on the CPU. The shader mirrors fixed_function()/transform() in pgraph.cpp.



// Uniform block (std140): matrices uploaded row-per-column so `v * M` is the NV2A's
// out[j] = dot(v, row j).
struct FfUbo {
    float mv[4][16], imv[4][16], comp[16], texm[4][16];
    float plane[16][4];
    float g[12][4];   // 0 scene ambient, 1 material emission, 2 specular params, 3 eye position, 11 (light count, types)
                      // 4 back scene ambient, 5 back emission, 6 back specular params,
                      // 7 (material alpha, back alpha, fog p0, fog p1), 8 fog plane,
                      // 9 viewport offset, 10 (W, H, zmax, 0)
    float L[8 * 13][4];  // per light: amb, dif, spc, bamb, bdif, bspc, half, dir, falloff, spot dir, pos, atten, range
};

void put4(float* dst, const Vec4& v) { std::memcpy(dst, v.data(), 16); }
void put_mat(float* dst, const Mat4& m) { for (int r = 0; r < 4; ++r) std::memcpy(dst + 4 * r, m[r].data(), 16); }

// Structural state of the generated vertex shader.
std::vector<u32> ff_vs_key(const FfState& f, u32 cmp_mask) {
    std::vector<u32> k;
    k.push_back(u32(f.skin_count) | (f.skin_mix ? 0x10u : 0) | (f.normalize ? 0x20u : 0) | (f.lighting ? 0x40u : 0) |
                (f.local_eye ? 0x80u : 0) | (f.specular_enable ? 0x100u : 0) | (f.separate_specular ? 0x200u : 0) |
                (f.two_sided ? 0x400u : 0) | (f.has_back_diffuse ? 0x800u : 0) | (f.has_back_specular ? 0x1000u : 0) |
                (f.infinite_viewer ? 0x2000u : 0) | (f.fog ? 0x4000u : 0));
    k.push_back(f.emission_src | f.ambient_src << 2 | f.diffuse_src << 4 | f.specular_src << 6 | f.foggen << 8 | cmp_mask << 16);
    k.push_back(f.fog_mode);
    // Light count and types are uniforms (g[11]): the shader loops over the lights, so the
    // ever-changing set of dynamic lights (gunfire, explosions) doesn't make new shaders.
    k.push_back(0);
    for (int i = 0; i < 4; ++i) {
        u32 w = f.texmat_en[i] ? 1u : 0u;
        for (int c = 0; c < 4; ++c) w |= (f.texgen[i][c] & 0xFFFF) << 0;  // folded below
        k.push_back(w);
        for (int c = 0; c < 4; ++c) k.push_back(f.texgen[i][c]);
    }
    return k;
}

std::string ff_vertex_source(const FfState& f, u32 cmp_mask) {
    std::string s = "#version 450\n";
    static const int kUsed[] = {0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12};
    for (int a : kUsed) {
        const std::string A = std::to_string(a);
        if (cmp_mask & (1u << a)) s += "layout(location = " + A + ") in uint ic" + A + ";\n";
        else s += "layout(location = " + A + ") in vec4 iv" + A + ";\n";
    }
    s += R"(layout(std140, binding = 1) uniform FF {
    mat4 mv[4]; mat4 imv[4]; mat4 comp; mat4 texm[4]; vec4 plane[16]; vec4 g[12]; vec4 L[104];
};
)" + std::string(kDrawBlock) + kVsCommon + R"(vec4 cmpv(uint x) {
    return vec4(float(int(x << 21u) >> 21) / 1023.0, float(int(x << 10u) >> 21) / 1023.0, float(int(x) >> 22) / 511.0, 1.0);
}
vec3 nrm3(vec3 v) { float l = length(v); return l > 0.0 ? v / l : v; }
)";
    auto A = [&](int a) -> std::string {
        const std::string n = std::to_string(a);
        return (cmp_mask & (1u << a)) ? "cmpv(ic" + n + ")" : "iv" + n;
    };
    // Lighting for one side, as GLSL: emits code computing c0/c1 (vec4) from n, epos.
    auto light_code = [&](bool back) -> std::string {
        const std::string B = back ? "true" : "false";
        auto src = [&](u32 which) -> std::string { return which == 1 ? "diffuse" : which == 2 ? "specular" : "vec4(1.0)"; };
        std::string c;
        const std::string n = back ? "(-en.xyz)" : "en.xyz";
        const std::string sceneAmb = back ? "g[4]" : "g[0]", emission = back ? "g[5]" : "g[1]", sp = back ? "g[6]" : "g[2]";
        const std::string alpha = f.diffuse_src == 0 ? (back ? "g[7].y" : "g[7].x") : f.diffuse_src == 2 ? "specular.a" : "diffuse.a";
        c += "    { vec3 n = " + n + "; vec4 c0 = vec4(0.0), c1 = vec4(0.0, 0.0, 0.0, specular.a);\n";
        if (f.emission_src == 0 || f.ambient_src == 0) {
            c += "      c0 = " + sceneAmb + ";\n";
            if (f.emission_src != 0) c += "      c0.rgb += " + src(f.emission_src) + ".rgb * " + emission + ".rgb;\n";
            else if (f.ambient_src != 0) c += "      c0.rgb += " + src(f.ambient_src) + ".rgb * " + emission + ".rgb;\n";
        } else {
            c += "      c0.rgb = " + src(f.emission_src) + ".rgb + " + src(f.ambient_src) + ".rgb * " + emission + ".rgb;\n";
        }
        // Lights: count in g[11].x, types (2 bits each: 1 directional, 2 point, 3 spot) in
        // g[11].y; light i's 13 vec4s start at L[13 * i].
        c += "      int nl = int(g[11].x), lt = int(g[11].y);\n"
             "      for (int li = 0; li < nl; ++li) {\n"
             "        int t = (lt >> (2 * li)) & 3, b = 13 * li;\n"
             "        float att = 1.0, ndl = 0.0, ndh = 0.0; bool lit = true;\n"
             "        if (t == 1) {\n"
             "          vec3 dir = L[b + 7].xyz; ndl = max(0.0, dot(n, dir));\n";
        if (f.local_eye) c += "          ndh = max(0.0, dot(n, nrm3(dir + vpeye)));\n";
        else c += "          ndh = max(0.0, dot(n, L[b + 6].xyz));\n";
        c += "        } else {\n"
             "          vec3 vp = L[b + 10].xyz - epos3; float dist = length(vp);\n"
             "          if (dist <= L[b + 12].x) {\n"
             "            vp = nrm3(vp);\n"
             "            att = 1.0 / (L[b + 11].x + L[b + 11].y * dist + L[b + 11].z * dist * dist);\n"
             "            ndl = max(0.0, dot(n, vp)); ndh = max(0.0, dot(n, nrm3(vp + vpeye)));\n"
             "            if (t == 3) {\n"
             "              float spt = dot(L[b + 9].xyz, vp) + L[b + 9].w;\n"
             "              if (spt <= 0.0) att = 0.0;\n"
             "              else if (spt < 1.0) { float fac = 0.0; vec3 fo = L[b + 8].xyz;\n"
             "                if (spt > -fo.x) { float den = fo.y * spt + fo.z;\n"
             "                  if (den == 0.0) fac = 1.0; else if (den > 0.0) fac = clamp((spt + fo.x) / den, 0.0, 1.0); }\n"
             "                att *= fac; }\n"
             "            }\n"
             "          } else lit = false;\n"
             "        }\n"
             "        if (lit) {\n"
             "        float pf = 0.0;\n"
             "        if (ndl > 0.0 && ndh > -" + sp + ".x) { float den = " + sp + ".y * ndh + " + sp + ".z;\n"
             "          if (den == 0.0) pf = 10000.0; else if (den > 0.0) pf = clamp((ndh + " + sp + ".x) / den, 0.0, 10000.0); }\n";
        const std::string amb = back ? "L[b + 3]" : "L[b + 0]", dif = back ? "L[b + 4]" : "L[b + 1]", spc = back ? "L[b + 5]" : "L[b + 2]";
        c += "        c0.rgb += " + src(f.ambient_src) + ".rgb * " + amb + ".rgb * att + " + src(f.diffuse_src) + ".rgb * " + dif +
             ".rgb * att * ndl;\n"
             "        c1.rgb += " + src(f.specular_src) + ".rgb * " + spc + ".rgb * att * pf;\n"
             "        }\n"
             "      }\n";
        c += "      c0.a = " + alpha + ";\n";
        const std::string d0 = back ? "b0" : "d0", d1 = back ? "b1" : "d1";
        c += "      " + d0 + " = c0; " + d1 + " = c1; }\n";
        (void)B;
        return c;
    };

    s += "void main() {\n";
    s += "    vec4 pos = " + A(0) + ", wt = " + A(1) + ", diffuse = " + A(3) + ", specular = " + A(4) + ";\n";
    s += "    vec4 nrm = vec4(" + A(2) + ".xyz, 0.0);\n";
    if (f.skin_count == 0) {
        s += "    vec4 epos = pos * mv[0]; vec4 en = nrm * imv[0];\n";
    } else {
        s += "    vec4 epos = vec4(0.0), en = vec4(0.0); float rest = 1.0, sw;\n";
        for (int i = 0; i < f.skin_count; ++i) {
            const std::string I = std::to_string(i);
            if (f.skin_mix && i == f.skin_count - 1) s += "    sw = rest;\n";
            else s += "    sw = wt[" + I + "]; rest -= sw;\n";
            s += "    epos += (pos * mv[" + I + "]) * sw; en += (nrm * imv[" + I + "]) * sw;\n";
        }
    }
    s += "    en.w = 0.0;\n";
    if (f.normalize) s += "    en.xyz = nrm3(en.xyz);\n";
    // texture coordinates
    for (int i = 0; i < 4; ++i) {
        const std::string I = std::to_string(i);
        s += "    { vec4 t = " + A(9 + i) + ";\n";
        bool refl = false;
        for (int c = 0; c < 4; ++c) refl |= f.texgen[i][c] == 0x2402 || f.texgen[i][c] == 0x8512;
        if (refl)
            s += std::string("      vec3 u = ") + (f.infinite_viewer ? "vec3(0.0, 0.0, 1.0)" : "nrm3(epos.xyz)") +
                 "; vec3 r = u - 2.0 * dot(en.xyz, u) * en.xyz;\n"
                 "      float m = 2.0 * sqrt(r.x * r.x + r.y * r.y + (r.z + 1.0) * (r.z + 1.0));\n";
        for (int c = 0; c < 4; ++c) {
            const std::string C = std::to_string(c), P = "plane[" + std::to_string(i * 4 + c) + "]";
            const char* comp = "xyzw";
            const std::string tc = std::string("t.") + comp[c];
            switch (f.texgen[i][c]) {
                case 0: break;
                case 0x2400: s += "      " + tc + " = dot(" + P + ", epos);\n"; break;
                case 0x2401: s += "      " + tc + " = dot(" + P + ", pos);\n"; break;
                case 0x2402: if (c < 2) s += "      " + tc + " = m > 0.0 ? r." + comp[c] + " / m + 0.5 : 0.5;\n"; break;
                case 0x8512: if (c < 3) s += "      " + tc + " = r." + comp[c] + ";\n"; break;
                case 0x8511: if (c < 3) s += "      " + tc + " = en." + comp[c] + ";\n"; break;
                default: break;
            }
        }
        if (f.texmat_en[i]) s += "      t = t * texm[" + I + "];\n";
        s += "      vT" + I + " = t; }\n";
    }
    // lighting
    s += "    vec4 d0 = diffuse, d1 = specular, b0 = diffuse, b1 = specular;\n";
    if (f.lighting) {
        s += "    vec3 vpeye = vec3(0.0);\n";
        if (f.local_eye)
            s += "    { float ew = g[3].w != 0.0 ? g[3].w : 1.0, pw = epos.w != 0.0 ? epos.w : 1.0;\n"
                 "      vpeye = nrm3(g[3].xyz / ew - epos.xyz / pw); }\n";
        s += "    vec3 epos3 = epos.xyz / (epos.w != 0.0 ? epos.w : 1.0);\n";
        s += light_code(false);
        if (f.two_sided) s += light_code(true);
    } else if (f.two_sided) {
        if (f.has_back_diffuse) s += "    b0 = " + A(7) + ";\n";
        if (f.has_back_specular) s += "    b1 = " + A(8) + ";\n";
    }
    auto spec_fix = [&](const std::string& c0, const std::string& c1, bool back) {
        if (!f.specular_enable) {
            s += "    " + c1 + " = vec4(0.0, 0.0, 0.0, 1.0);\n";
        } else if (!f.separate_specular) {
            if (f.lighting) s += "    " + c0 + ".rgb += " + c1 + ".rgb;\n";
            s += "    " + c1 + " = " + ((back && !f.lighting && f.has_back_specular) ? A(8) : std::string("specular")) + ";\n";
        }
    };
    spec_fix("d0", "d1", false);
    if (f.two_sided) spec_fix("b0", "b1", true);
    s += "    vD0 = clamp(d0, 0.0, 1.0); vD1 = clamp(d1, 0.0, 1.0);\n";
    s += f.two_sided ? "    vB0 = clamp(b0, 0.0, 1.0); vB1 = clamp(b1, 0.0, 1.0);\n" : "    vB0 = vD0; vB1 = vD1;\n";
    // fog
    if (f.fog) {
        std::string dist;
        switch (f.foggen) {
            case 0: dist = "clamp(specular.a, 0.0, 1.0)"; break;
            case 1: dist = "length(epos.xyz)"; break;
            case 2: dist = "(dot(g[8].xyz, epos.xyz) + g[8].w)"; break;
            case 3: dist = "abs(dot(g[8].xyz, epos.xyz) + g[8].w)"; break;
            case 6: dist = A(5) + ".x"; break;
            default: dist = "0.0"; break;
        }
        s += "    { float fd = " + dist + "; float r;\n";
        switch (f.fog_mode) {
            case 0x2601: case 0x804: s += "      r = g[7].z + fd * g[7].w - 1.0;\n"; break;
            case 0x800: case 0x802: s += "      r = g[7].z + exp2(fd * g[7].w * 16.0) - 1.5;\n"; break;
            case 0x801: case 0x803: s += "      r = g[7].z + exp2(-fd * fd * g[7].w * g[7].w * 32.0) - 1.5;\n"; break;
            default: s += "      r = 1.0;\n"; break;
        }
        if (f.fog_mode == 0x802 || f.fog_mode == 0x803 || f.fog_mode == 0x804) s += "      r = abs(r);\n";
        s += "      vFog = (isinf(fd) || isnan(r)) ? 1.0 : r; }\n";
    } else {
        s += "    vFog = 1.0;\n";
    }
    // position: screen space as the NV2A computes it, then GL clip space
    s += std::string("    vec4 p = ") + (f.skin_count ? "epos" : "pos") + " * comp;\n";
    s += R"(    float w = p.w;
    if (abs(w) < 5.42101e-20) w = w < 0.0 ? -5.42101e-20 : 5.42101e-20;
    float sx = p.x / w + g[9].x, sy = p.y / w + g[9].y, sz = p.z / w;
    gl_Position = widen(vec4((2.0 * sx / g[10].x - 1.0) * w, (2.0 * sy / g[10].y - 1.0) * w, clamp(sz / g[10].z, 0.0, 1.0) * w, w));
    gl_PointSize = uPointSize;
}
)";
    return s;
}

std::map<std::vector<u32>, std::string> g_ff_vs;  // vertex shader sources by key

dev::Program program_for_ff(const DrawState& d, const FfState& f, u32 cmp_mask) {
    const std::vector<u32> vk = ff_vs_key(f, cmp_mask);
    ShaderKey key = make_key(d);
    key.words.push_back(0xFFFFFFFFu);  // separates the vertex shader key
    key.words.insert(key.words.end(), vk.begin(), vk.end());
    dev::Program& p = g_programs[key];
    if (p) return p;
    std::string& vs = g_ff_vs[vk];
    if (vs.empty()) vs = ff_vertex_source(f, cmp_mask);
    p = g_dev->program(vs, fragment_source(make_key(d), d));
    return p;
}

}  // namespace

// Shared GPU-vertex draw: uploads the arrays (attributes in `used`), sets culling,
// binds the program from `program` and the uniform block written by `fill_ubo`.
bool draw_gpu(const DrawState& d, const GpuArrays& arr, const std::vector<u32>& idx, const int* used, int nused,
              size_t ubo_size, const std::function<dev::Program(u32 cmp_mask)>& program,
              const std::function<void(u8* dst)>& fill_ubo) {
    if (idx.empty()) return true;
    // Upload sizes first: bail out (CPU path) before touching any GL state.
    struct Group { u32 stride, lo, hi; size_t off; };
    Group groups[16];
    int ngroups = 0;
    int group_of[16];
    u32 cmp_mask = 0;
    auto attr_bytes = [](u32 fmt) -> u32 {
        const u32 type = fmt & 0xF, size = (fmt >> 4) & 0xF, n = size == 7 ? 3 : std::min<u32>(size, 4);
        switch (type) {
            case 0: case 4: return 4;
            case 1: case 5: return 2 * n;
            case 6: return 4;
            default: return 4 * n;
        }
    };
    size_t total = 0;
    for (int a = 0; a < 16; ++a) group_of[a] = -1;
    for (int ui = 0; ui < nused; ++ui) {
        const int a = used[ui];
        const auto& at = arr.attr[a];
        if (!at.enabled) continue;
        if ((at.fmt & 0xF) == 6) cmp_mask |= 1u << a;
        const u32 lo = at.base + arr.min_index * at.stride;
        const u32 hi = at.base + arr.max_index * at.stride + attr_bytes(at.fmt);
        int gi = -1;
        for (int g2 = 0; g2 < ngroups; ++g2)
            if (groups[g2].stride == at.stride && lo + 2047 > groups[g2].lo && lo < groups[g2].lo + 2047) { gi = g2; break; }
        if (gi < 0) { gi = ngroups++; groups[gi] = {at.stride, lo, hi, 0}; }
        groups[gi].lo = std::min(groups[gi].lo, lo);
        groups[gi].hi = std::max(groups[gi].hi, hi);
        group_of[a] = gi;
    }
    for (int g2 = 0; g2 < ngroups; ++g2) total += groups[g2].hi - groups[g2].lo + 16;
    total += idx.size() * 4 + ubo_size + 512;
    if (total > (8u << 20)) return false;
    const dev::Program prog = program(cmp_mask);
    if (!prog) return false;

    poll_readbacks();
    ProfScope prof(g_prof_draw);
    ColorSurface* cs = bind_target(d.surf);
    if (!cs) return true;
    smaa_before_overlay(d, cs);
    if (!cs) return true;
    std::optional<ProfScope> ps;
    setup_draw(d, prog);
    ps.emplace(g_prof_vbo);

    // Face culling on the GPU (NV2A meaning; the device maps it to its own orientation).
    g_dc.rs.front_ccw = d.front_ccw;
    if (d.cull && !g_no_cull) g_dc.rs.cull = d.cull_face == 0x404 ? dev::kCullFront : d.cull_face == 0x405 ? dev::kCullBack : dev::kCullBoth;

    // Vertex data: one upload per group of attributes sharing a stride and region.
    for (int g2 = 0; g2 < ngroups; ++g2) {
        Group& gr = groups[g2];
        const size_t n = gr.hi - gr.lo;
        std::memcpy(g_dev->stream(n, 16, g_dc.vbuf[g2]), ram(gr.lo), n);
        g_dc.stride[g2] = gr.stride;
    }
    g_dc.nbind = u8(ngroups);
    for (int ui = 0; ui < nused; ++ui) {
        const int a = used[ui];
        const auto& at = arr.attr[a];
        dev::VertexAttr& va = g_dc.attr[g_dc.nattr++];
        va.location = u8(a);
        if (!at.enabled) {
            va.constant = true;
            std::memcpy(va.value, at.constant.data(), sizeof va.value);
            continue;
        }
        const Group& gr = groups[group_of[a]];
        va.binding = u8(group_of[a]);
        va.offset = at.base + arr.min_index * at.stride - gr.lo;
        const u32 type = at.fmt & 0xF, size = (at.fmt >> 4) & 0xF;
        const u8 n = u8(size == 7 ? 3 : std::min<u32>(size, 4));
        switch (type) {
            case 0: va.fmt = dev::kBGRA8; break;
            case 4: va.fmt = u8(dev::kUN8x1 + n - 1); break;
            case 1: va.fmt = u8(dev::kSN16x1 + n - 1); break;
            case 5: va.fmt = u8(dev::kS16x1 + n - 1); break;
            case 6: va.fmt = dev::kU32x1; break;
            default: va.fmt = u8(dev::kF32x1 + n - 1); break;
        }
    }

    // Uniforms
    fill_ubo(g_dev->stream(ubo_size, g_dev->ubo_align(), g_dc.ubo1));

    // Indices
    std::memcpy(g_dev->stream(idx.size() * 4, 4, g_dc.index), idx.data(), idx.size() * 4);
    g_dc.count = u32(idx.size());
    submit_draw();
    if (d.color_mask_bits) cs->gl_dirty = true;
    return true;
}

bool draw_ff(const DrawState& d, const FfState& f, const GpuArrays& arr, const std::vector<u32>& idx) {
    g_trace_path = "ff";
    static const int kUsed[] = {0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12};
    return draw_gpu(
        d, arr, idx, kUsed, int(std::size(kUsed)), sizeof(FfUbo),
        [&](u32 cmp) { return program_for_ff(d, f, cmp); },
        [&](u8* dst) {
            FfUbo u{};
            for (int i = 0; i < 4; ++i) {
                put_mat(u.mv[i], f.mv[i]);
                put_mat(u.imv[i], f.imv[i]);
                put_mat(u.texm[i], f.texmat[i]);
                for (int c = 0; c < 4; ++c) put4(u.plane[i * 4 + c], f.texplane[i][c]);
            }
            put_mat(u.comp, f.composite);
            put4(u.g[0], f.scene_ambient);
            put4(u.g[1], f.mat_emission);
            put4(u.g[2], f.spec_params);
            put4(u.g[3], f.eye_pos);
            put4(u.g[4], f.back_scene_ambient);
            put4(u.g[5], f.back_emission);
            put4(u.g[6], f.back_spec_params);
            u.g[7][0] = f.mat_alpha; u.g[7][1] = f.back_alpha; u.g[7][2] = f.fog_p0; u.g[7][3] = f.fog_p1;
            put4(u.g[8], f.fog_plane);
            put4(u.g[9], f.vp_off);
            u.g[10][0] = float(d.surf.width); u.g[10][1] = float(d.surf.height);
            u.g[10][2] = d.surf.zeta_fmt == 1 ? 65535.0f : 16777215.0f;
            u32 lt = 0;
            for (int i = 0; i < f.nlights; ++i) lt |= u32(f.lights[i].type) << (2 * i);
            u.g[11][0] = float(f.nlights);
            u.g[11][1] = float(lt);
            for (int i = 0; i < f.nlights; ++i) {
                const Light& l = f.lights[i];
                float(*L)[4] = &u.L[i * 13];
                put4(L[0], l.amb); put4(L[1], l.dif); put4(L[2], l.spc);
                put4(L[3], l.bamb); put4(L[4], l.bdif); put4(L[5], l.bspc);
                put4(L[6], l.half);
                Vec4 dn = l.dir;
                const float len = std::sqrt(dn[0] * dn[0] + dn[1] * dn[1] + dn[2] * dn[2]);
                if (len > 0) for (int q = 0; q < 3; ++q) dn[q] /= len;
                put4(L[7], dn);
                put4(L[8], l.falloff); put4(L[9], l.spot_dir); put4(L[10], l.pos); put4(L[11], l.atten);
                L[12][0] = l.range;
            }
            std::memcpy(dst, &u, sizeof u);
        });
}

// ---------------------------------------------------------------- vertex programs
// NV2A vertex microcode -> GLSL, mirroring run_program() in pgraph.cpp.
namespace {
inline u32 vfield(const u32* t, int word, int bit, int len) { return (t[word] >> bit) & ((1u << len) - 1); }

// Returns an empty string if the program can't run on the GPU (writes constants).
std::string vp_vertex_source(const u32 (*code)[4], u32 start, u32 cmp_mask, const FfState& f) {
    std::string s = "#version 450\n";
    for (int a = 0; a < 16; ++a) {
        const std::string A = std::to_string(a);
        if (cmp_mask & (1u << a)) s += "layout(location = " + A + ") in uint ic" + A + ";\n";
        else s += "layout(location = " + A + ") in vec4 iv" + A + ";\n";
    }
    s += R"(layout(std140, binding = 1) uniform VP { vec4 c[192]; vec4 g[4]; };
)" + std::string(kDrawBlock) + kVsCommon + R"(vec4 cmpv(uint x) {
    return vec4(float(int(x << 21u) >> 21) / 1023.0, float(int(x << 10u) >> 21) / 1023.0, float(int(x) >> 22) / 511.0, 1.0);
}
float hm(float a, float b) { return (a == 0.0 || b == 0.0) ? 0.0 : a * b; }
vec4 hm4(vec4 a, vec4 b) { return vec4(hm(a.x, b.x), hm(a.y, b.y), hm(a.z, b.z), hm(a.w, b.w)); }
vec4 C(int i) { return (i >= 0 && i < 192) ? c[i] : vec4(0.0); }
const float INF = uintBitsToFloat(0x7F800000u);
void main() {
)";
    for (int a = 0; a < 16; ++a) {
        const std::string A = std::to_string(a);
        s += "    vec4 v" + A + " = " + ((cmp_mask & (1u << a)) ? "cmpv(ic" + A + ")" : "iv" + A) + ";\n";
    }
    s += "    vec4 R0 = vec4(0.0), R1 = R0, R2 = R0, R3 = R0, R4 = R0, R5 = R0, R6 = R0, R7 = R0, R8 = R0, R9 = R0, R10 = R0, R11 = R0;\n";
    s += "    vec4 o0 = vec4(0.0, 0.0, 0.0, 1.0), o1 = o0, o2 = o0, o3 = vec4(1.0), o4 = vec4(1.0), o5 = o0, o6 = o0, o7 = o0,\n"
         "         o8 = o0, o9 = o0, o10 = o0, o11 = o0, o12 = o0, o13 = o0, o14 = o0, o15 = o0;\n";
    s += "    int a0 = 0;\n";
    bool writes_b0 = false, writes_b1 = false, writes_pts = false;
    auto read = [&](const u32* t, int mux, int rreg, int negw, int negb, int sww, int swb) -> std::string {
        std::string src;
        switch (mux) {
            case 1: src = rreg == 12 ? "o0" : "R" + std::to_string(rreg < 12 ? rreg : 0); break;
            case 2: src = "v" + std::to_string(vfield(t, 1, 9, 4)); break;
            case 3: {
                const int idx = int(vfield(t, 1, 13, 8));
                if (vfield(t, 3, 1, 1)) src = "C(a0 + " + std::to_string(idx) + ")";
                else src = idx < 192 ? "c[" + std::to_string(idx) + "]" : "vec4(0.0)";
                break;
            }
            default: src = "vec4(0.0)"; break;
        }
        std::string sw = ".";
        for (int i = 0; i < 4; ++i) sw += "xyzw"[vfield(t, sww, swb + 2 * (3 - i), 2)];
        return std::string(vfield(t, negw, negb, 1) ? "-" : "") + "(" + src + ")" + sw;
    };
    auto mask = [](u32 m) {
        std::string r;
        for (int i = 0; i < 4; ++i)
            if (m & (8u >> i)) r += "xyzw"[i];
        return r;
    };
    for (u32 pc = start; pc < 136; ++pc) {
        const u32* t = code[pc];
        const u32 ilu = vfield(t, 1, 25, 3), mac = vfield(t, 1, 21, 4);
        if (ilu || mac) {
            s += "    {\n";
            s += "        vec4 A = " + read(t, int(vfield(t, 2, 26, 2)), int(vfield(t, 2, 28, 4)), 1, 8, 1, 0) + ";\n";
            s += "        vec4 B = " + read(t, int(vfield(t, 2, 11, 2)), int(vfield(t, 2, 13, 4)), 2, 25, 2, 17) + ";\n";
            const int c_reg = int((vfield(t, 2, 0, 2) << 2) | vfield(t, 3, 30, 2));
            s += "        vec4 Cc = " + read(t, int(vfield(t, 3, 28, 2)), c_reg, 2, 10, 2, 2) + ";\n";
            s += "        vec4 m = vec4(0.0), l = vec4(0.0); float x = Cc.x;\n";
            switch (mac) {
                case 1: s += "        m = A;\n"; break;
                case 2: s += "        m = hm4(A, B);\n"; break;
                case 3: s += "        m = A + Cc;\n"; break;
                case 4: s += "        m = hm4(A, B) + Cc;\n"; break;
                case 5: s += "        m = vec4(hm(A.x, B.x) + hm(A.y, B.y) + hm(A.z, B.z));\n"; break;
                case 6: s += "        m = vec4(hm(A.x, B.x) + hm(A.y, B.y) + hm(A.z, B.z) + B.w);\n"; break;
                case 7: s += "        m = vec4(hm(A.x, B.x) + hm(A.y, B.y) + hm(A.z, B.z) + hm(A.w, B.w));\n"; break;
                case 8: s += "        m = vec4(1.0, hm(A.y, B.y), A.z, B.w);\n"; break;
                case 9: s += "        m = min(A, B);\n"; break;
                case 10: s += "        m = max(A, B);\n"; break;
                case 11: s += "        m = vec4(lessThan(A, B));\n"; break;
                case 12: s += "        m = vec4(greaterThanEqual(A, B));\n"; break;
                default: break;
            }
            switch (ilu) {
                case 1: s += "        l = Cc;\n"; break;
                case 2: s += "        l = vec4(1.0 / x);\n"; break;
                case 3: s += "        { float r = 1.0 / x; if (abs(r) < 5.42101e-20) r = r < 0.0 ? -5.42101e-20 : 5.42101e-20;\n"
                             "          if (abs(r) > 1.884467e+19) r = r < 0.0 ? -1.884467e+19 : 1.884467e+19; l = vec4(r); }\n"; break;
                case 4: s += "        l = vec4(x == 0.0 ? INF : (isinf(x) ? 0.0 : 1.0 / sqrt(abs(x))));\n"; break;
                case 5: s += "        l = vec4(exp2(floor(x)), x - floor(x), exp2(x), 1.0);\n"; break;
                case 6: s += "        { float mg = abs(x); if (mg == 0.0) l = vec4(-INF, 1.0, -INF, 1.0);\n"
                             "          else { float e = floor(log2(mg)); l = vec4(e, mg / exp2(e), log2(mg), 1.0); } }\n"; break;
                case 7: s += "        { float sx = max(Cc.x, 0.0), sy = max(Cc.y, 0.0), sw = clamp(Cc.w, -(128.0 - 1.0 / 256.0), 128.0 - 1.0 / 256.0);\n"
                             "          l = vec4(1.0, sx, sx > 0.0 ? exp2(sw * log2(sy)) : 0.0, 1.0); }\n"; break;
                default: break;
            }
            const u32 out_r = vfield(t, 3, 20, 4), mac_mask = vfield(t, 3, 24, 4), ilu_mask = vfield(t, 3, 16, 4);
            const u32 o_mask = vfield(t, 3, 12, 4), orb = vfield(t, 3, 11, 1), out_addr = vfield(t, 3, 3, 8);
            const u32 out_mux = vfield(t, 3, 2, 1);
            auto write_out = [&](const char* val) -> bool {
                if (!o_mask) return true;
                if (orb == 0) return false;  // writes a constant: CPU path
                const u32 o = out_addr & 0xF;
                if (o == 7) writes_b0 = true;
                if (o == 6) writes_pts = true;
                if (o == 8) writes_b1 = true;
                const std::string mk = mask(o_mask);
                s += "        o" + std::to_string(o) + "." + mk + " = " + val + "." + mk + ";\n";
                return true;
            };
            auto write_r = [&](u32 reg, const char* val, u32 m) {
                if (!m) return;
                const std::string mk = mask(m);
                if (reg == 12) s += "        o0." + mk + " = " + val + "." + mk + ";\n";
                else if (reg < 12) s += "        R" + std::to_string(reg) + "." + mk + " = " + val + "." + mk + ";\n";
            };
            // All reads above happen before these writes (paired MAC+ILU semantics).
            if (mac) {
                if (mac == 13) s += "        a0 = int(floor(A.x + 0.001));\n";
                else {
                    write_r(out_r, "m", (ilu && out_r == 1) ? 0 : mac_mask);
                    if (out_mux == 0 && !write_out("m")) return "";
                }
            }
            if (ilu) {
                write_r(mac ? 1 : out_r, "l", ilu_mask);
                if (out_mux == 1 && !write_out("l")) return "";
            }
            s += "    }\n";
        }
        if (vfield(t, 3, 0, 1)) break;
    }
    // Epilogue (as transform() for programs): o0 is in screen space already.
    s += R"(    vec4 p = o0;
    float w = p.w;
    if (abs(w) < 5.42101e-20) w = w < 0.0 ? -5.42101e-20 : 5.42101e-20;
    gl_Position = widen(vec4((2.0 * p.x / g[0].x - 1.0) * w, (2.0 * p.y / g[0].y - 1.0) * w, clamp(p.z / g[0].z, 0.0, 1.0) * w, w));
)";
    // Point size: the program's oPts.x (screen pixels: bubbles, sparks) if it writes one.
    s += writes_pts ? "    gl_PointSize = max(o6.x, 1.0) * uPointScale;\n" : "    gl_PointSize = uPointSize;\n";
    s += R"(    vD0 = clamp(o3, 0.0, 1.0); vD1 = clamp(o4, 0.0, 1.0);
    vT0 = o9; vT1 = o10; vT2 = o11; vT3 = o12;
)";
    s += writes_b0 ? "    vB0 = clamp(o7, 0.0, 1.0);\n" : "    vB0 = vD0;\n";
    s += writes_b1 ? "    vB1 = clamp(o8, 0.0, 1.0);\n" : "    vB1 = vD1;\n";
    if (f.fog) {
        s += "    { float fd = o5.x; float r;\n";
        switch (f.fog_mode) {
            case 0x2601: case 0x804: s += "      r = g[1].x + fd * g[1].y - 1.0;\n"; break;
            case 0x800: case 0x802: s += "      r = g[1].x + exp2(fd * g[1].y * 16.0) - 1.5;\n"; break;
            case 0x801: case 0x803: s += "      r = g[1].x + exp2(-fd * fd * g[1].y * g[1].y * 32.0) - 1.5;\n"; break;
            default: s += "      r = 1.0;\n"; break;
        }
        if (f.fog_mode == 0x802 || f.fog_mode == 0x803 || f.fog_mode == 0x804) s += "      r = abs(r);\n";
        s += "      vFog = (isinf(fd) || isnan(r)) ? 1.0 : r; }\n";
    } else {
        s += "    vFog = 1.0;\n";
    }
    s += "}\n";
    return s;
}

struct VpShader { std::string src; bool tried = false; };  // empty src: not translatable
std::unordered_map<ShaderKey, VpShader, ShaderKeyHash> g_vp_vs;

dev::Program program_for_vp(const DrawState& d, const FfState& f, const VpInfo& vp, u32 cmp_mask) {
    ShaderKey vk;
    vk.words.push_back(vp.start);
    vk.words.push_back(cmp_mask);
    vk.words.push_back(f.fog ? (0x10000u | f.fog_mode) : 0u);
    for (u32 pc = vp.start; pc < 136; ++pc) {
        for (int w = 0; w < 4; ++w) vk.words.push_back(vp.code[pc][w]);
        if (vp.code[pc][3] & 1) break;
    }
    VpShader& sh = g_vp_vs[vk];
    if (!sh.tried) {
        sh.tried = true;
        sh.src = vp_vertex_source(vp.code, vp.start, cmp_mask, f);
    }
    if (sh.src.empty()) return {};
    ShaderKey key = make_key(d);
    key.words.push_back(0xFFFFFFFEu);
    key.words.insert(key.words.end(), vk.words.begin(), vk.words.end());
    dev::Program& p = g_programs[key];
    if (!p) p = g_dev->program(sh.src, fragment_source(make_key(d), d));
    return p;
}
}  // namespace

bool draw_vp(const DrawState& d, const FfState& f, const GpuArrays& arr, const std::vector<u32>& idx, const VpInfo& vp) {
    g_trace_path = "vp";
    static const int kAll[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    struct VpUbo { float c[192][4]; float g[4][4]; };
    return draw_gpu(
        d, arr, idx, kAll, 16, sizeof(VpUbo), [&](u32 cmp) { return program_for_vp(d, f, vp, cmp); },
        [&](u8* dst) {
            VpUbo u{};
            for (int i = 0; i < 192; ++i) std::memcpy(u.c[i], vp.consts[i].data(), 16);
            u.g[0][0] = float(d.surf.width);
            u.g[0][1] = float(d.surf.height);
            u.g[0][2] = d.surf.zeta_fmt == 1 ? 65535.0f : 16777215.0f;
            u.g[1][0] = f.fog_p0;
            u.g[1][1] = f.fog_p1;
            std::memcpy(dst, &u, sizeof u);
        });
}

namespace {
}  // namespace

// Copy a displayed surface into the next display slot and publish it for the window.
void publish(const ColorSurface& cs) {
    Slot& sl = g_slots[g_slot_next];
    const int slot = g_slot_next;
    g_slot_next = (g_slot_next + 1) % kSlots;
    const int w = cs.gw, h = cs.gh;
    {
        std::lock_guard lk(g_display_m);
        if (sl.read_fence) {  // the window may still be copying the previous frame in it
            g_dev->gpu_wait(sl.read_fence);
            sl.read_fence = 0;
        }
        // The slot's previous frame is gone after this: forget any record of it.
        for (auto it = g_published.begin(); it != g_published.end();) {
            if (it->second.slot == slot) {
                g_dev->release(it->second.fence);
                it = g_published.erase(it);
            } else {
                ++it;
            }
        }
        // Never resize a slot the window might hold: make a new one.
        if (!sl.tex || sl.w != w || sl.h != h) {
            if (sl.tex) g_dev->destroy_texture(sl.tex);
            sl.tex = g_dev->create_texture(w, h, 1, dev::Fmt::BGRA8);
            sl.w = w;
            sl.h = h;
        }
    }
    // The picture as shown: copied, anti-aliased here (VC3_SMAA) unless that was done before
    // its overlays (smaa_before_overlay).
    const auto aa = g_smaa_done.find(cs.desc.color_base);
    const bool aa_done = aa != g_smaa_done.end() && aa->second == g_epoch && !g_smaa_dirty[cs.desc.color_base];
    const bool scene_3d = g_3d_epoch == g_epoch;  // movies, menus, transitions: left alone
    if (aa_done || !scene_3d || !smaa::apply(*g_dev, cs.tex, sl.tex, w, h)) g_dev->copy(cs.tex, sl.tex, w, h);
    g_smaa_dirty[cs.desc.color_base] = false;
    const dev::Sync fence = g_dev->fence();  // visible to the window's context/queue
    std::lock_guard lk(g_display_m);
    DisplayFrame& f = g_published[cs.desc.color_base & (mem::kRamSize - 1)];
    g_dev->release(f.fence);  // superseded before the window used it
    f = DisplayFrame{sl.tex, w, h, cs.desc.width, cs.desc.height, slot, fence, cs.bytes(), 0};
}

void flush() {
    static const bool report = std::getenv("VC3_FPS") != nullptr;
    static int frames = 0;
    if (report && ++frames % 300 == 0) {
        std::string r = "[gl] per 300 frames:";
        for (Prof* p : {&g_prof_draw, &g_prof_prog, &g_prof_texbind, &g_prof_state, &g_prof_vbo, &g_prof_tex_hash, &g_prof_tex_upload, &g_prof_rt_hash, &g_prof_surf_up, &g_prof_surf_down}) {
            char b[96];
            std::snprintf(b, sizeof b, " %s %.0f ms (%llu)", p->name, p->t.count() / 1e6, (unsigned long long)p->n);
            r += b;
            p->t = {};
            p->n = 0;
        }
        std::fprintf(stderr, "%s | feedback copies %llu (%.1f Mpx)", r.c_str(), (unsigned long long)g_n_fb_copy,
                     double(g_fb_copy_px) / 1e6);
        g_n_fb_copy = g_fb_copy_px = 0;
        std::fprintf(stderr, " | dl flip %llu sync %llu reshape %llu | up new %llu changed %llu | surfaces %zu\n",
                     (unsigned long long)g_n_dl_flip, (unsigned long long)g_n_dl_sync, (unsigned long long)g_n_dl_reshape,
                     (unsigned long long)g_n_up_new, (unsigned long long)g_n_up_changed, g_color.size());
        g_n_dl_flip = g_n_dl_sync = g_n_dl_reshape = g_n_up_new = g_n_up_changed = 0;
    }
    if (g_shared)
        for (auto& cs : g_color)
            if (cs.gl_dirty && displayed(cs.desc.color_base & (mem::kRamSize - 1))) {
                // The previous frame's write-back must land before this frame is published:
                // the record's RAM fingerprint is taken from the next write-back, and a late
                // one from the previous frame would look like the CPU overwriting this one.
                finish_readback(cs, true);
                publish(cs);
            }
    for (auto& cs : g_color)
        if (cs.gl_dirty && (g_full_sync || displayed(cs.desc.color_base & (mem::kRamSize - 1)))) {
            ++g_n_dl_flip;
            if (g_full_sync) download_color(cs);
            else start_readback(cs);
        }
    g_dev->flush();  // end of the guest frame: submit it
    ++g_epoch;
}

void poll() {
    if (g_active) poll_readbacks();
}

void set_share(void* dpy, void* cfg, void* ctx) { dev::gl::set_share(dpy, cfg, ctx); }

void configure() {
    if (const char* e = std::getenv("VC3_WIDESCREEN"); e && *e && std::strcmp(e, "0")) {
        double a = 16.0 / 9.0;
        if (!std::strcmp(e, "auto")) {  // match the display's shape (4:3 or narrower: the original)
            SDL_InitSubSystem(SDL_INIT_VIDEO);
            const SDL_DisplayMode* m = SDL_GetDesktopDisplayMode(SDL_GetPrimaryDisplay());
            a = m && m->h > 0 ? double(m->w) / double(m->h) : 4.0 / 3.0;
            std::fprintf(stderr, "[gpu] aspect ratio: %.3f (display %dx%d)\n", std::max(a, 4.0 / 3.0), m ? m->w : 0, m ? m->h : 0);
        } else if (const char* c = std::strchr(e, ':')) {
            a = std::atof(e) / std::max(0.1, std::atof(c + 1));
        } else if (std::atof(e) > 1.34) {
            a = std::atof(e);
        }
        g_wide_k = std::clamp((4.0 / 3.0) / a, 0.3, 1.0);
    }
    if (const char* e = std::getenv("VC3_RES_SCALE")) {
        if (!std::strcmp(e, "auto")) {  // match the display: the scale nearest its height
            SDL_InitSubSystem(SDL_INIT_VIDEO);
            const SDL_DisplayMode* m = SDL_GetDesktopDisplayMode(SDL_GetPrimaryDisplay());
            g_scale = m ? std::clamp(int(std::lround(double(m->h) / 480.0)), 1, 8) : 1;
            std::fprintf(stderr, "[gpu] resolution: %dx (display %dx%d)\n", g_scale, m ? m->w : 0, m ? m->h : 0);
        } else {
            const int n = std::atoi(e);
            g_scale = n >= 1 && n <= 8 ? n : 1;
        }
    }
    g_wide_hud = !(std::getenv("VC3_WIDE_HUD") && !std::strcmp(std::getenv("VC3_WIDE_HUD"), "0"));
}
bool shared() { return g_shared; }
bool widescreen() { return g_wide_k < 1.0; }
double widescreen_factor() { return g_wide_k; }
// The texture is a surface the GPU rendered (an offscreen render target), not a picture.
// 3D (non-flat) draws into each colour target: last frame one happened.
std::map<u32, u64> g_3d_drawn;  // colour base -> last epoch a 3D draw went into it
bool had_3d_this_frame(u32 base) {
    auto it = g_3d_drawn.find(base);
    return it != g_3d_drawn.end() && it->second + 600 >= g_epoch;  // 3D drawn into it in the last 10 s
}
// The mission maps and the game-over static come out of the same buffer, drawn the same
// way. A map is always shown for a while outside gameplay first (the HARD map then
// crossfades into the stage intro, which already draws the HUD); the static only ever
// appears straight after gameplay. So a buffer shown full-screen outside gameplay in the
// last 30 s holds a map and keeps its shape.
std::map<u32, u64> g_shown_outside;
bool shown_outside_gameplay(u32 base, const DrawState& d) {
    const bool gameplay = g_score_seen_epoch != ~0ull && g_score_seen_epoch + 180 >= g_epoch;
    if (!gameplay && d.bx0 <= 2.0f) g_shown_outside[base] = g_epoch;
    auto it = g_shown_outside.find(base);
    return !gameplay || (it != g_shown_outside.end() && it->second + 1800 >= g_epoch);
}
bool rendered_texture(const Texture& t) {
    for (const auto& cs : g_color)
        if ((cs.gl_dirty || cs.pending || cs.sampled) && cs.desc.color_base == t.base && u32(cs.desc.width) == t.w && u32(cs.desc.height) == t.h)
            return true;
    return false;
}
bool samples_wide_target(const Texture& t) {
    if (g_wide_k >= 1.0 || t.cube) return false;
    for (const auto& cs : g_color)
        if (cs.wide && (cs.gl_dirty || cs.pending) && cs.desc.color_base == t.base && u32(cs.desc.width) == t.w &&
            u32(cs.desc.height) == t.h)
            return true;
    return false;
}
void set_hud_triangles(const std::vector<std::array<float, 8>>& bounds, const std::vector<u32>& corners, size_t vertices) {
    g_draw_tris.clear();
    for (const auto& b : bounds) g_draw_tris.push_back({b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]});
    g_draw_corners = corners;
    g_draw_vertices = vertices;
    g_hud_base = 0;
}

bool display_frame(u32 base, DisplayFrame& out) {
    std::lock_guard lk(g_display_m);
    auto it = g_published.find(base & (mem::kRamSize - 1));
    if (it == g_published.end()) return false;
    out = it->second;
    it->second.fence = 0;  // the caller owns (waits on and releases) it now
    return true;
}

void display_done(int slot, dev::Sync fence) {
    std::lock_guard lk(g_display_m);
    if (slot < 0 || slot >= kSlots) { g_dev->release(fence); return; }
    g_dev->release(g_slots[slot].read_fence);
    g_slots[slot].read_fence = fence;
}

u64 sparse_ram_hash(u32 base, u32 bytes) { return sparse_hash(ram(base & (mem::kRamSize - 1)), bytes); }

void shader_warmup_progress(int& done, int& total) { done = dev::shader_store::warm_done.load(); total = dev::shader_store::warm_total.load(); }

void shader_stats(u64& n, u64& ns) { n = dev::shader_store::compile_n.load(); ns = dev::shader_store::compile_ns.load(); }

void set_wide_hud(bool on) { g_wide_hud = on; }
u64 ram_writes() { return g_ram_writes.load(std::memory_order_acquire); }
std::mutex& ram_write_mutex() { return g_ram_write_m; }

void note_scanout(u32 phys) {
    phys &= mem::kRamSize - 1;
    if (!phys || displayed(phys)) return;
    g_scanouts[g_scanout_next.fetch_add(1) % kScanouts].store(phys, std::memory_order_relaxed);
}

}  // namespace vc3::gpu::pgraph::glb
