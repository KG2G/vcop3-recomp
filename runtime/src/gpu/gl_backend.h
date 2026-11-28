// OpenGL 4.5 backend for PGRAPH.
//
// Vertex processing stays on the CPU (pgraph.cpp: vertex programs, fixed-function T&L, culling,
// primitive assembly); this backend rasterises the resulting triangles with a
// fragment shader generated from the register-combiner state, into GL copies of
// the guest surfaces. Guest RAM stays authoritative: GL results are written back
// at every flip and whenever a texture read overlaps a surface GL has modified.
#pragma once
#include <string>
#include <array>
#include <mutex>
#include <vector>

#include "gpu/device.h"
#include "gpu/pgraph_internal.h"

namespace vc3::gpu::pgraph::glb {

// Reads the display options (VC3_WIDESCREEN, VC3_RES_SCALE, VC3_WIDE_HUD); main calls it
// once the launcher has set them, before anything renders.
void configure();
bool init();      // creates an offscreen EGL context on the calling (GPU) thread
// Windowed mode: the window's EGL display/config/context, set before init(). The GPU
// thread's context then shares objects with the window's, which shows rendered
// frames straight from GPU memory (see display_frame).
void set_share(void* egl_display, void* egl_config, void* egl_context);
bool shared();
// Widescreen is on (screen-size 2D draws then need classifying: DrawState::full_screen_2d).
bool widescreen();
// The 4:3 picture's share of the displayed width (1: widescreen off).
double widescreen_factor();
// Widescreen: whether texture `t` is read from a widened render target (pgraph then
// computes screen bounds for bigger draws too, to recognise screen distortion effects).
bool samples_wide_target(const detail::Texture& t);
// Widescreen HUD: the next GPU-path draw's triangles as 640x480 screen bounds
// {x0, y0, x1, y1} with their corner vertices (3 per triangle, relative to the draw's
// lowest index) and its vertex count; empty: no per-triangle data.
// bounds: {x0, y0, x1, y1, u0, v0, u1, v1} (screen box, stage-0 texture coordinate box).
void set_hud_triangles(const std::vector<std::array<float, 8>>& bounds, const std::vector<u32>& corners, size_t vertices);

// A frame published for display: at every flip, each displayed surface GL rendered is
// copied into a display slot (never freed or resized behind the reader's back).
struct DisplayFrame {
    dev::Tex tex;              // device texture (shared with the window), BGRA8, w x h (render resolution)
    int w = 0, h = 0;          // texture size
    int native_w = 0, native_h = 0;
    int slot = -1;
    dev::Sync fence = 0;       // wait for it before reading (GL: a GLsync)
    u32 bytes = 0;             // guest bytes of the surface (for ram_matches)
    u64 ram_sparse = 0;        // sparse hash of guest RAM once the write-back landed (0: not yet)
};
// Latest frame published for scan-out address `base` (window thread).
bool display_frame(u32 base, DisplayFrame& out);
// The window thread finished copying slot `slot`; `fence` marks that point.
void display_done(int slot, dev::Sync fence);
// Same sparse hash the backend records (to see whether the CPU overwrote a frame).
u64 sparse_ram_hash(u32 base, u32 bytes);
bool active();

void clear(const detail::Surface& s, u32 flags, u32 rect_h, u32 rect_v, u32 color, u32 zeta_stencil);
// `tris` holds 3 vertices per triangle, already culled.
void draw(const detail::DrawState& d, const std::vector<detail::Vertex>& tris);
// GPU vertex path for fixed-function draws: raw guest vertex arrays, an index list
// relative to `arrays.min_index` (for d.gl_prim) and the fixed-function state, which
// a generated vertex shader evaluates. Returns false if the draw is too large (the
// caller then uses the CPU path).
bool draw_ff(const detail::DrawState& d, const detail::FfState& ff, const detail::GpuArrays& arrays,
             const std::vector<u32>& indices);
// Same for draws using a vertex program (NV2A microcode translated to GLSL).
struct VpInfo {
    const u32 (*code)[4];
    u32 start;
    const detail::Vec4* consts;  // 192 transform constants
};
bool draw_vp(const detail::DrawState& d, const detail::FfState& ff, const detail::GpuArrays& arrays,
             const std::vector<u32>& indices, const VpInfo& vp);
// End of frame: write GL-modified surfaces back to guest RAM. Only surfaces that
// have been displayed are written back eagerly; others stay on the GPU until
// something reads their memory in a way that can't use the GL copy.
void flush();
// Trace every draw of the next frame into `dir` (images + draws.txt; the F9 capture).
void request_draw_trace(const std::string& dir);
// A trace is being written this frame: draws then also find their screen bounds.
bool tracing_now();
// SMAA is on (VC3_SMAA): draws then also find out whether they are 2D overlays.
bool smaa_enabled();
// The final-stage cheat: hide the chosen mission's briefing map, which otherwise stays over
// the final stage's intro.
void hide_stale_map();
// Developer switch (F8): all face culling off, to tell missing geometry culled by mistake
// from geometry never drawn. Returns the new state.
bool no_cull();
bool toggle_no_cull();
// Developer switch (F7): every alpha test passes, to tell a surface discarded by its alpha.
bool toggle_no_alpha_test();
// Developer switch (F6): depth test off (painter's order), to tell a surface hidden by a wrong
// depth from one never drawn.
bool toggle_no_depth_test();
// HUD in widescreen (VC3_WIDE_HUD) while playing (the pause menu).
void set_wide_hud(bool on);
bool debug_no_depth();
bool debug_no_alpha();
// Complete finished asynchronous write-backs (GPU thread, when otherwise idle).
void poll();
// The display start address was programmed (any thread).
void note_scanout(u32 phys);
// Surface write-backs into guest RAM: a counter (the picture can have changed) and
// the lock held while one is written (the window's upload takes it so it never
// reads a half-written frame).
u64 ram_writes();
std::mutex& ram_write_mutex();
// Cumulative shader compiles+links and the time they took (any thread).
void shader_stats(u64& n, u64& ns);
// Startup shader compilation (programs known from earlier runs): done / total (any thread).
void shader_warmup_progress(int& done, int& total);

}  // namespace vc3::gpu::pgraph::glb
