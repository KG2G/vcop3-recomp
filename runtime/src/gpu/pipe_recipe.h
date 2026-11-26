#pragma once
// Vulkan pipeline recipes: the draw state a pipeline is built from, keyed so they can be
// stored (work/shadercache/v4/vk-recipes.bin) and built up front in later runs. Shared
// with the OpenGL device, which records them too: whoever plays on OpenGL still collects
// the pipelines Vulkan players (and the shader seed) need.

#include <cstddef>
#include <cstring>
#include <string>

#include "gpu/device.h"

namespace vc3::gpu::dev {

struct PipeKey {
    u32 prog;
    u8 topology, blend, src, dst, eq, mask, depth, nattr, nbind, pad[3];
    u8 attr[16][3];  // location, binding, format
    u32 offset[16];
};
struct Recipe {
    u64 prog_key;  // program_key() of its sources (program ids differ between runs)
    PipeKey key;   // with prog = 0
};
static_assert(sizeof(Recipe) == 136, "recipe layout is stored on disk (and in tools/make_shader_seed.py)");

inline u64 recipe_fnv(const void* p, size_t n, u64 h = 0xCBF29CE484222325ull) {
    const u8* b = static_cast<const u8*>(p);
    for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 0x100000001B3ull;
    return h;
}
// A program's key from its stored (backend-neutral) sources.
inline u64 program_key(const std::string& vs, const std::string& fs) {
    return recipe_fnv(fs.data(), fs.size(), recipe_fnv(vs.data(), vs.size()));
}
// The VkPrimitiveTopology value for a draw's topology (line loops are drawn as strips).
inline u8 recipe_topology(u8 t) {
    switch (t) {
        case kPoints: return 0;     // POINT_LIST
        case kLines: return 1;      // LINE_LIST
        case kLineLoop:
        case kLineStrip: return 2;  // LINE_STRIP
        case kTriStrip: return 4;   // TRIANGLE_STRIP
        case kTriFan: return 5;     // TRIANGLE_FAN
        default: return 3;          // TRIANGLE_LIST
    }
}
// The key of a draw on a GPU that fetches every vertex format: constant attributes get a
// stride-0 binding each after the draw's own (the Vulkan device converts formats it can't
// fetch on top of this, see for_this_device()).
inline PipeKey canonical_pipe_key(const DrawCall& dc, bool depth) {
    PipeKey k{};
    k.prog = dc.prog.id;
    k.topology = recipe_topology(dc.rs.topology);
    k.blend = dc.rs.blend;
    if (dc.rs.blend) {
        k.src = dc.rs.blend_src;
        k.dst = dc.rs.blend_dst;
        k.eq = dc.rs.blend_eq;
    }
    k.mask = dc.rs.color_mask & 0xF;
    k.depth = depth;
    k.nattr = dc.nattr;
    u32 nb = dc.nbind;
    for (int i = 0; i < dc.nattr && i < 16; ++i) {
        const VertexAttr& a = dc.attr[i];
        k.attr[i][0] = a.location;
        k.attr[i][1] = a.constant ? u8(nb++) : a.binding;
        k.attr[i][2] = a.constant ? u8(kF32x4) : a.fmt;
        k.offset[i] = a.constant ? 0 : a.offset;
    }
    k.nbind = u8(nb);
    return k;
}

}  // namespace vc3::gpu::dev
