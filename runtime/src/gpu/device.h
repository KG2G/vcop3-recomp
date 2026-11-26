// Graphics device layer: the few operations the NV2A renderer (renderer.cpp) needs from a
// graphics API, implemented for OpenGL 4.5 (gl_device.cpp) and Vulkan 1.3 (vk_device.cpp).
//
// The renderer keeps all emulation logic (surface tracking, write-backs, shader
// generation, widescreen rules) and describes each draw completely in a DrawCall; a
// backend applies it (GL: state calls, Vulkan: a cached pipeline plus dynamic state).
//
// Conventions shared by both backends:
//  - Textures hold BGRA8 texels (upload data is little-endian A8R8G8B8 words), or DXT
//    blocks, or depth/stencil. Row 0 is the guest's top line (memory order).
//  - Clip space: x, y in -1..1 with y = -1 at row 0, depth 0..1.
//  - Generated GLSL (#version 450) uses: uniform block binding 0 (per draw), binding 1
//    (vertex stage), storage buffer binding 3 (HUD sides), samplers 4-7 (2D) and 8-11
//    (cube); varyings at fixed locations; the vertex index is VERTEX_ID (the device
//    defines it). The Vulkan backend compiles the same source.
#pragma once
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <thread>

struct SDL_Window;

namespace vc3::gpu::dev {

// A shader met for the first time builds in the background (startup builds every known
// one, see shader_seed). A draw that needs one still building waits until it is ready,
// so nothing ever pops in; that can cost one frame once, the first time a shader not in
// the cache shows up. VC3_SHADER_WAIT_MS=N caps the wait per guest frame (testing).
class ShaderWait {
public:
    void new_frame() { left_ns_ = budget_ns(); }
    // Waits while ready() is false and budget is left; returns ready().
    template <class Ready>
    bool wait(Ready&& ready) {
        if (ready()) return true;
        if (left_ns_ <= 0) return false;
        const auto t0 = std::chrono::steady_clock::now();
        bool ok = false;
        while (!(ok = ready())) {
            const long long spent = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - t0).count();
            if (spent >= left_ns_) break;
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        left_ns_ -= std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
        return ok;
    }

private:
    static long long budget_ns() {
        static const long long ns = [] {
            const char* e = std::getenv("VC3_SHADER_WAIT_MS");
            return e ? (long long)(std::atof(e) * 1e6) : (1LL << 62);  // default: no cap
        }();
        return ns;
    }
    long long left_ns_ = budget_ns();
};


using u8 = uint8_t;
using u32 = uint32_t;
using u64 = uint64_t;

enum class Fmt : u8 { BGRA8, D24S8, BC1, BC2, BC3 };

// Handles are small ids owned by the device (0: none).
struct Tex {
    u32 id = 0;
    explicit operator bool() const { return id != 0; }
    friend bool operator==(Tex a, Tex b) { return a.id == b.id; }
    friend bool operator!=(Tex a, Tex b) { return a.id != b.id; }
    friend bool operator<(Tex a, Tex b) { return a.id < b.id; }
};
struct Program {
    u32 id = 0;
    explicit operator bool() const { return id != 0; }
};
// A GPU-side completion point (0: none). Both backends can wait for one on the CPU or
// order later GPU work after it.
using Sync = u64;

// Streaming memory for one draw (vertices, indices, uniform blocks): valid until the
// GPU has used it; the device recycles it behind fences.
struct Slice {
    u32 buf = 0;
    size_t off = 0, size = 0;
    explicit operator bool() const { return size != 0; }
};

enum Filter : u8 { kNearest = 0, kLinear = 1 };
enum MipMode : u8 { kMipNone = 0, kMipNearest = 1, kMipLinear = 2 };
enum Wrap : u8 { kRepeat = 0, kMirror = 1, kClampEdge = 2, kClampBorder = 3 };
struct SamplerDesc {
    u8 min = kLinear, mag = kLinear, mip = kMipNone;
    u8 wrap_u = kClampEdge, wrap_v = kClampEdge;
    bool alpha_one = false;  // sample alpha as 1 (formats without alpha)
    float lod_bias = 0, min_lod = 0, max_lod = 1000;
    float border[4] = {0, 0, 0, 0};
};

// Compare functions in GL order (NV2A uses the same): never, less, equal, lequal,
// greater, notequal, gequal, always.
// Stencil ops: keep, zero, replace, incr (clamp), decr (clamp), invert, incr_wrap, decr_wrap.
enum StencilOp : u8 { kKeep, kZero, kReplace, kIncr, kDecr, kInvert, kIncrWrap, kDecrWrap };
// Blend factors (GL/NV2A order of meaning).
enum BlendFactor : u8 {
    kZeroF, kOne, kSrcColor, kOneMinusSrcColor, kSrcAlpha, kOneMinusSrcAlpha, kDstAlpha, kOneMinusDstAlpha,
    kDstColor, kOneMinusDstColor, kSrcAlphaSaturate, kConstColor, kOneMinusConstColor, kConstAlpha, kOneMinusConstAlpha
};
enum BlendEq : u8 { kAdd, kSub, kRevSub, kMin, kMax };
enum Topology : u8 { kPoints, kLines, kLineLoop, kLineStrip, kTriangles, kTriStrip, kTriFan };
enum Cull : u8 { kCullNone, kCullFront, kCullBack, kCullBoth };

struct RasterState {
    Topology topology = kTriangles;
    bool depth_test = false, depth_write = false;
    u8 depth_func = 7;
    bool stencil_test = false;
    u8 stencil_func = 7, stencil_ref = 0, stencil_read_mask = 0xFF, stencil_write_mask = 0xFF;
    u8 stencil_fail = kKeep, stencil_zfail = kKeep, stencil_zpass = kKeep;
    bool blend = false;
    u8 blend_src = kOne, blend_dst = kZeroF, blend_eq = kAdd;
    float blend_color[4] = {0, 0, 0, 0};
    u8 color_mask = 0xF;  // bit 0 R, 1 G, 2 B, 3 A
    bool poly_offset = false;
    float poly_offset_scale = 0, poly_offset_units = 0;  // units of a 24-bit depth buffer
    u8 cull = kCullNone;
    bool front_ccw = true;  // as the NV2A means it: counter-clockwise on screen (row 0 at the top)
    bool scissor = false;
    int sx = 0, sy = 0, sw = 0, sh = 0;  // pixels, row 0 = top
};

// Vertex attribute formats.
enum AttrFmt : u8 {
    kF32x1, kF32x2, kF32x3, kF32x4,
    kUN8x1, kUN8x2, kUN8x3, kUN8x4,  // unsigned bytes normalised
    kBGRA8,                          // D3DCOLOR: bytes B, G, R, A -> (r, g, b, a) normalised
    kSN16x1, kSN16x2, kSN16x3, kSN16x4,  // signed shorts normalised
    kS16x1, kS16x2, kS16x3, kS16x4,      // signed shorts as floats
    kU32x1,                          // integer attribute (uint in the shader)
};
struct VertexAttr {
    u8 location = 0;
    u8 binding = 0;     // index into DrawCall::vbuf
    u8 fmt = kF32x4;
    bool constant = false;  // not fetched: the shader sees `value` for every vertex
    u32 offset = 0;         // within the binding's slice
    float value[4] = {0, 0, 0, 1};
};

struct DrawCall {
    Program prog;
    RasterState rs;
    u8 nattr = 0, nbind = 0;
    VertexAttr attr[16];
    Slice vbuf[16];
    u32 stride[16] = {};
    Slice ubo0, ubo1, hud;  // uniform blocks at bindings 0 and 1, HUD sides storage (binding 3)
    Tex tex[8];             // bindings 4-11: 2D textures 0-3, cube maps 0-3
    SamplerDesc samp[8];
    Slice index;            // u32 indices; none: vertices first..first+count
    u32 first = 0, count = 0;
};

// Requested anisotropic filtering (VC3_ANISOTROPIC: 1 = off, 2, 4, 8, 16).
float anisotropy_setting();

class Device {
public:
    virtual ~Device() = default;
    virtual const char* name() const = 0;
    virtual bool supports_bc() const = 0;  // DXT textures can be uploaded as is

    // ---- textures
    virtual Tex create_texture(int w, int h, int levels, Fmt fmt, bool cube = false) = 0;
    virtual void destroy_texture(Tex t) = 0;
    // Texels (BGRA8 words, or DXT blocks: `bytes` of them) into one level/face.
    virtual void upload(Tex t, int level, int face, int x, int y, int w, int h, const void* data, size_t bytes) = 0;
    // Whole level 0 of `src` into `dst` (same size and format).
    virtual void copy(Tex src, Tex dst, int w, int h) = 0;
    // A rectangle of level 0 to the same place in `dst`.
    virtual void copy_region(Tex src, Tex dst, int x, int y, int w, int h) = 0;
    // Scaled copy between colour textures (rectangles in texels, x1/y1 exclusive).
    virtual void blit(Tex src, int sx0, int sy0, int sx1, int sy1, Tex dst, int dx0, int dy0, int dx1, int dy1,
                      bool linear) = 0;

    // ---- read-back of colour textures (BGRA8 rows, tightly packed)
    virtual void read(Tex t, int w, int h, void* dst) = 0;  // waits
    // Asynchronous: start, then poll until ready; the data stays valid until release.
    virtual u32 read_start(Tex t, int w, int h) = 0;
    virtual const void* read_poll(u32 id, bool wait) = 0;  // nullptr: not ready yet
    virtual void read_release(u32 id) = 0;

    // ---- rendering
    // Draws and clears go to this target until the next call (depth may be none).
    virtual void set_target(Tex color, Tex depth, int w, int h) = 0;
    virtual void clear(int x, int y, int w, int h, bool color, u8 color_mask, const float rgba[4], bool depth, float z,
                       bool stencil, u8 s) = 0;
    // GLSL vertex + fragment source -> program (cached by the caller).
    virtual Program program(const std::string& vs, const std::string& fs) = 0;
    // Streaming memory for draws; nullptr if `bytes` is too large (callers fall back).
    virtual u8* stream(size_t bytes, size_t align, Slice& out) = 0;
    virtual size_t ubo_align() const = 0;
    // false: nothing drawn (its shader or pipeline is still being built in the background).
    virtual bool draw(const DrawCall& dc) = 0;

    // ---- synchronisation and frames
    virtual Sync fence() = 0;                 // completion of everything submitted so far
    virtual bool wait(Sync s, u64 timeout_ns) = 0;  // CPU wait; true if reached (then released)
    virtual void gpu_wait(Sync s) = 0;        // later GPU work waits for `s` (then released)
    virtual void release(Sync s) = 0;         // no longer needed
    virtual void flush() = 0;                 // submit pending work (end of a guest frame)
};

// A solid rectangle drawn over the window's picture (crosshairs): window pixels, top-left origin.
struct OverlayRect {
    int x, y, w, h;
    u8 r, g, b;
};

// The GPU thread's device (created on that thread by the renderer). nullptr until then.
Device* device();
Device* create_gl_device();      // current GL context required
Device* create_vulkan_device();  // nullptr if Vulkan is unavailable

}  // namespace vc3::gpu::dev
