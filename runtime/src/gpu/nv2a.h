// NV2A (GeForce3-derived) GPU.
//
// D3D8 is linked statically into the game and drives the hardware directly:
// it builds push buffers in RAM, kicks them through the channel DMA_PUT
// register and reads back progress through semaphores. This module emulates
// the hardware side (LLE, like xemu/MAME):
//   - register blocks at 0xFD000000 (via the MMIO trap layer)
//   - PRAMIN (0xFD700000) as a direct alias of the top 1 MiB of RAM
//   - vblank interrupts from PCRTC (IRQ 3)
//   - the DMA pusher/puller on a GPU thread, decoding methods for bound objects
// Rendering of PGRAPH methods is built on top of pgraph state captured here.
#pragma once
#include <string>
#include <SDL3/SDL.h>

#include <cstdint>

namespace vc3::gpu {

void init_hardware();  // registers MMIO, maps PRAMIN, starts vblank + GPU threads
void set_scanout(uint32_t addr);  // PCRTC_START, as programmed by AvSetDisplayMode

// The window should be created for Vulkan (VC3_RENDERER=vulkan and built with it).
bool wants_vulkan_window();
bool init(SDL_Window* window);
// Show the newest frame in the window if there is one; returns false if nothing changed.
bool present();
// Save the window as the next present shows it (PPM; the F9 capture).
void request_window_shot(const std::string& path);
// Map window coordinates in a mouse event to the 640x480 picture (letterboxed).
void to_screen_coords(SDL_Event& e);
// A point given as a fraction of the window (0..1 each way: where a lightgun points) ->
// the game's 640x480 screen coordinates (outside 0..640 in widescreen side areas).
void window_to_game(float nx, float ny, float& gx, float& gy);
// Toggle fullscreen; prefers a display mode whose refresh is a multiple of 60 Hz.
void toggle_fullscreen();
// Settings changed while playing (the pause menu): fullscreen (including the borderless kind),
// VSync, the crosshairs.
bool fullscreen_on();
void set_vsync(bool on);
void set_crosshair(bool on, float size);

void shutdown();

}  // namespace vc3::gpu
