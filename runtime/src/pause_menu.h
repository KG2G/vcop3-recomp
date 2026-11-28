// The pause menu (Resume / Options / Exit to desktop): opened by the pause key or button (setting
// "pause"; Esc and the gamepad's guide button by default). Drawn with Dear ImGui into a picture on
// the CPU and shown over the frozen game like the custom crosshairs, so it works the same with
// both renderers. Options change live and are saved to settings.ini.
#pragma once
#include <SDL3/SDL.h>

#include "gpu/device.h"

namespace vc3::pause_menu {

// Window events: true if the menu took it (opening or closing it, or input for it while open).
bool handle_event(SDL_Window* win, const SDL_Event& e);
// Once per window-loop pass: lays out and draws the menu while paused.
void frame(SDL_Window* win);
// The menu's picture to draw over the game (nullptr when not paused). Window pixels.
const vc3::gpu::dev::OverlayImage* image();
// "Exit to desktop" was chosen.
bool exit_requested();
// Changes are saved to settings.ini (not for runs without settings, e.g. --no-settings).
void set_saving(bool on);
// (testing) Open the menu (on its options page) / close it, as the pause key would.
void test_open(bool open, bool options = false);

}  // namespace vc3::pause_menu
