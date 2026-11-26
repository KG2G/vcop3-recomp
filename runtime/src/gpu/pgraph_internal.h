// Internal PGRAPH types shared by the state decoder (pgraph.cpp), the software
// rasteriser and the OpenGL backend (gl_backend.cpp).
#pragma once
#include <array>
#include <cstdint>
#include <cstring>

#include "vc3/helpers.h"
#include "vc3/memory.h"

namespace vc3::gpu::pgraph::detail {

using Vec4 = std::array<float, 4>;

struct Vertex {
    float x, y, z, w;   // screen space, w = clip w
    Vec4 d0, d1, t[4];
    Vec4 b0, b1;        // back-side colours (two-sided lighting)
    float fog;
};

struct Surface {
    u32 color_base = 0, zeta_base = 0, color_pitch = 0, zeta_pitch = 0;
    u32 color_fmt = 0, zeta_fmt = 0;
    int width = 0, height = 0, scale_x = 1, scale_y = 1;
    bool swizzled = false;
    u32 sw_log_w = 0, sw_log_h = 0;
    int color_bpp() const { return (color_fmt == 3 || color_fmt == 1 || color_fmt == 2) ? 2 : (color_fmt == 9 ? 1 : 4); }
};

struct Texture {
    bool enabled = false;
    u32 base = 0, fmt = 0, w = 1, h = 1, pitch = 0, log_w = 0, log_h = 0;
    bool linear = false;
    u32 addr_u = 1, addr_v = 1;
    bool filter_linear = false;
    u32 palette = 0;
    u32 border = 0;
    bool cube = false;  // six faces, each with all its mip levels, 128-byte aligned
    u32 levels = 1;
    // Minification: NV2A filter (1 nearest, 2 linear, 3-6 the GL mipmap modes in order
    // NEAREST_MIPMAP_NEAREST, LINEAR_MIPMAP_NEAREST, NEAREST_MIPMAP_LINEAR, LINEAR_MIPMAP_LINEAR),
    // LOD bias and the LOD clamps, in mip levels.
    u32 min_filter = 2;
    float lod_bias = 0, min_lod = 0, max_lod = 15;
};

struct Rgba {
    float r, g, b, a;
};

struct CombinerSetup {
    u32 stages = 0;
    bool mux_msb = false;
    struct Stage {
        u32 ci, ai, co, ao;
        Rgba c0, c1;
    } st[8];
    bool final = false;
    u32 f0 = 0, f1 = 0;
    Rgba fc0, fc1;
};

struct DrawState {
    Surface surf;
    CombinerSetup comb;
    int tex_used = 0;  // highest texture stage in use + 1
    Texture tex[4];
    u32 stage_modes;
    bool alpha_test, blend, depth_test, depth_write;
    u32 alpha_func, depth_func, color_mask_bits;
    float alpha_ref;
    u32 sfactor, dfactor, equation;
    Vec4 blend_color;
    bool fog;
    Vec4 fog_color;
    int clip_x0, clip_y0, clip_x1, clip_y1;
    // Primitive: GL_TRIANGLES for assembled triangles, or GL_POINTS / GL_LINES /
    // GL_LINE_LOOP / GL_LINE_STRIP with the vertices as submitted.
    u32 gl_prim = 0x0004;
    float point_size = 1.0f;
    bool point_sprite = false;  // points with POINT_SMOOTH: T3 = gl_PointCoord
    // Texture-shader stage inputs: dot-product mapping per stage (0x1E74) and the
    // stage each of stages 2/3 reads (0x1E78); bump-env matrix/luminance per stage.
    u32 dot_mapping = 0, other_stage_input = 0;
    float bump_mat[4][4] = {};  // m00, m01, m10, m11
    float bump_scale[4] = {}, bump_offset[4] = {};
    // Face culling / front face (the GPU vertex path culls in GL; the CPU path culls itself).
    bool cull = false, front_ccw = true;
    u32 cull_face = 0x405;
    bool gpu_two_sided = false;  // fragment shader picks back colours by facing
    bool full_screen_2d = false; // 2D draw covering the whole surface (widescreen: stretched)
    bool big_mesh = false;       // 64+ vertices
    // Textured with real image content (tiny textures, like the 8x8 fill behind letterbox
    // bars, count as solid colour).
    bool textured() const {
        for (int i = 0; i < tex_used && i < 4; ++i) {
            const u32 mode = (stage_modes >> (5 * i)) & 0x1F;
            if (mode && mode != 4 && tex[i].enabled && (tex[i].w > 16 || tex[i].h > 16)) return true;
        }
        return false;
    }
    // Widescreen: stretch a flat (2D) draw spanning the surface's width if it covers the
    // full height too (fades, backgrounds, copies) or is a solid fill (letterbox bars).
    // Textured bands (title bars with lettering) keep their shape.
    // Screen-space bounds of a small flat draw (widescreen HUD placement).
    bool flat2d = false;
    // A 2D overlay (HUD, text): a small draw of screen-space vertices (the CPU vertex path;
    // w = 1) not covering the whole target -- where SMAA stops (set with SMAA on).
    bool overlay2d = false;
    void set_overlay2d(bool flat, float x0, float x1, float y0, float y1) {
        const bool full = x0 <= 2.0f && x1 >= float(surf.width) - 2.0f && y0 <= 2.0f && y1 >= float(surf.height) - 2.0f;
        overlay2d = flat && !full;
    }
    float bx0 = 0, bx1 = 0, by0 = 0, by1 = 0;
    void classify_wide(bool flat, float x0, float x1, float y0, float y1) {
        flat2d = flat;
        bx0 = x0; bx1 = x1; by0 = y0; by1 = y1;  // (bounds kept for non-flat draws too)
        // 2 px of slack: some layers are offset by a pixel (pixel-centre conventions).
        const bool full_w = x0 <= 2.0f && x1 >= float(surf.width) - 2.0f;
        const bool full_h = y0 <= 2.0f && y1 >= float(surf.height) - 2.0f;
        full_screen_2d = flat && full_w && (full_h || !textured());
    }
    bool stencil_test = false;
    u32 stencil_func = 0x207, stencil_ref = 0, stencil_func_mask = 0xFF, stencil_write_mask = 0xFF;
    u32 stencil_fail = 0x1E00, stencil_zfail = 0x1E00, stencil_zpass = 0x1E00;
    // Polygon offset (depth bias): enabled for this primitive type, slope scale, bias in depth-buffer units.
    bool poly_offset = false;
    float poly_offset_scale = 0, poly_offset_bias = 0;
    // Draws that sample their own colour target (the sewer water's refraction): the part of
    // the target they can read, in guest pixels, so only that is copied first (a full copy
    // per draw is costly at high resolutions). fb_bounds false: unknown, copy it all.
    bool fb_bounds = false;
    float fbx0 = 0, fby0 = 0, fbx1 = 0, fby1 = 0;
    bool samples_own_target() const {
        for (int i = 0; i < tex_used && i < 4; ++i)
            if (tex[i].enabled && !tex[i].cube && tex[i].base == surf.color_base) return true;
        return false;
    }
    // From the draw's vertices (screen positions and texture coordinates).
    void own_target_bounds(const Vertex* v, size_t n);
};

// ---- fixed-function vertex state (snapshotted per draw; see pgraph.cpp)
using Mat4 = std::array<Vec4, 4>;  // rows as sent by the methods: out[j] = dot(v, row j)

struct Light {
    int type;  // 1 infinite, 2 local, 3 spot
    Vec4 amb, dif, spc, half, dir, falloff, spot_dir, pos, atten;
    Vec4 bamb, bdif, bspc;  // back-side colours (two-sided lighting)
    float range;
};

struct FfState {
    bool program;
    int skin_count = 0;
    bool skin_mix = false;
    bool normalize;
    Mat4 mv[4], imv[4], composite, texmat[4];
    bool texmat_en[4];
    u32 texgen[4][4];
    Vec4 texplane[4][4];
    bool infinite_viewer;
    bool lighting;
    u32 emission_src, ambient_src, diffuse_src, specular_src;
    Vec4 scene_ambient, mat_emission, spec_params, eye_pos;
    float mat_alpha;
    bool local_eye, separate_specular, specular_enable;
    int nlights = 0;
    Light lights[8];
    bool two_sided = false, has_back_diffuse = false, has_back_specular = false;
    Vec4 back_scene_ambient, back_emission, back_spec_params;
    float back_alpha = 1.0f;
    bool fog;
    u32 fog_mode, foggen;
    float fog_p0, fog_p1;
    Vec4 fog_plane;
    Vec4 vp_off;
};

// Vertex arrays of a draw, for the GPU vertex path: the guest's own buffers are
// uploaded as they are and decoded by GL (packed CMP normals in the shader).
struct GpuArrays {
    struct Attr {
        bool enabled = false;
        u32 fmt = 0, base = 0, stride = 0;  // base: physical address of element 0
        Vec4 constant{0, 0, 0, 1};          // value when not an array
    } attr[16];
    u32 min_index = 0, max_index = 0;
};

constexpr float kInv255 = 1.0f / 255.0f;
inline Vec4 argb(u32 v) {
    return {float((v >> 16) & 0xFF) * kInv255, float((v >> 8) & 0xFF) * kInv255, float(v & 0xFF) * kInv255,
            float(v >> 24) * kInv255};
}

// Defined in pgraph.cpp
u8* ram(u32 phys);
u32 swizzle_offset(u32 x, u32 y, u32 log_w, u32 log_h);
u32 fetch_texel(const Texture& t, u32 x, u32 y);
void warn_once(const char* what, u32 v);

}  // namespace vc3::gpu::pgraph::detail
