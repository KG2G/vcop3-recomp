// Vulkan window presentation (window thread): the counterpart of the GL path in nv2a.cpp.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "gpu/device.h"

struct SDL_Window;

namespace vc3::gpu::vkp {

bool init(SDL_Window* window);  // vk::init(window) first; false: use OpenGL instead
bool active();
// Blocking (vsync) presents or not (newest picture wins).
void set_vsync(bool block);
// The next present shows device texture `t` (w x h), copied once `ready` (a device Sync)
// is reached. Returns the Sync marking the end of that copy (the slot may be reused).
dev::Sync take_texture(dev::Tex t, int w, int h, dev::Sync ready);
// ...or guest RAM pixels (BGRA, w x h).
void take_ram(const void* bgra, int w, int h);
// Draws the current picture into the window rectangle (pixels) and presents.
void present(float x, float y, float pw, float ph, const std::vector<dev::OverlayRect>& overlay = {});
// The next present also saves the window to `path` (PPM; the F9 capture).
void request_shot(const std::string& path);
// The current picture (BGRA rows) for screenshots.
bool read_picture(std::vector<uint8_t>& bgra, int& w, int& h);
void shutdown();

}  // namespace vc3::gpu::vkp
