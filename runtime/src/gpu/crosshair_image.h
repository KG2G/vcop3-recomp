// Custom crosshair images (VC3_CROSSHAIR_IMAGE_P1 / _P2: an image file per player; PNG, BMP, TGA,
// JPG, GIF). An image larger than the crosshair's size is scaled down to fit it (keeping its shape);
// a smaller one is drawn pixel for pixel. Prepared for drawing without blending (the Vulkan window
// path only copies pixels): fully transparent parts are left out; partly transparent ones are
// blended over black (a dark edge, like the built-in crosshair's outline).
#pragma once
#include <cstdint>
#include <vector>

namespace vc3::gpu::crosshair {

struct Image {
    int w = 0, h = 0;
    std::vector<uint32_t> px;  // 0xAARRGGBB, alpha 0 or 255
    uint64_t version = 0;      // changes whenever px does
};

// Player p's image fitted into `size` x `size` pixels, or nullptr (none set, or it couldn't be read).
const Image* get(int player, int size);
// Whether `path` is an image this can read, and its size (the launcher shows it).
bool probe(const char* path, int& w, int& h);

}  // namespace vc3::gpu::crosshair
