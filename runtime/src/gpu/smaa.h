// SMAA (Enhanced Subpixel Morphological Antialiasing, Jimenez et al.) on the finished
// picture: luma edge detection, blending weights, neighbourhood blending -- the reference
// shader (SMAA.hlsl, GLSL mode) run through the device layer, so both renderers have it.
// VC3_SMAA=low|medium|high|ultra (the reference presets); off or unset: not used.
#pragma once
#include "gpu/device.h"

namespace vc3::gpu::smaa {

bool enabled();
// Anti-aliases `src` (w x h, BGRA8) into `dst`. false: nothing written (the passes' shaders
// are still being built in the background) -- the caller copies instead.
bool apply(dev::Device& d, dev::Tex src, dev::Tex dst, int w, int h);

}  // namespace vc3::gpu::smaa
