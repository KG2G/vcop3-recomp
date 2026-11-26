// PGRAPH Kelvin (NV097) renderer.
//
// Consumes Kelvin methods (state captured in a method-indexed array) and
// renders into the guest surfaces in RAM, so scan-out sees the result with no
// copies. Current backend is a software rasteriser:
//   vertex programs (interpreted) / basic fixed-function transform,
//   inline / indexed / array vertex fetch, all primitive types,
//   perspective-correct triangles, textures (swizzled/linear, 16/32-bit,
//   DXT1/3/5, paletted), register combiners + final combiner,
//   alpha test, depth test, blending, colour mask, clears.
#pragma once
#include <cstdint>
#include <functional>
#include <string>

namespace vc3::gpu::pgraph {

struct Host {
    const uint32_t* kelvin;                    // method state, index = method / 4
    std::function<uint32_t(uint32_t)> dma_base;  // DMA object instance -> physical base
    std::function<uint32_t(uint32_t)> dma_limit; // DMA object instance -> limit (0 for none)
};

void init(const Host& host);
void method(uint32_t m, uint32_t data);  // called after kelvin[m/4] = data
std::string profile_report();            // time spent since the last call (VC3_FPS)

}  // namespace vc3::gpu::pgraph
