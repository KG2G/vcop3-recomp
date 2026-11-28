// Cabinet controls as seen by the JVS I/O board, fed by io/input.cpp (configurable
// bindings) or an input script. TEST/SERVICE (F2/F3) are the filter-board buttons (base
// board PINSA bits 7/6).
#pragma once
#include <SDL3/SDL.h>

#include <cstdint>
#include <string>

namespace vc3::io {

struct GunState {
    uint16_t x = 0x8000, y = 0x8000;  // analog position, 0..0xFFFF across the screen
    bool trigger = false, offscreen = false, pedal = false, change = false, start = false;
};

struct CabinetState {
    GunState gun[2];
    bool coin[2] = {}, test = false, service = false;
};

// Where gun 0/1 is aimed, in 640x480 screen pixels.
void aim_point(int gun, float& x, float& y);
// Widescreen shooting: how far (game pixels) a gun may aim beyond each side of the 4:3 screen -- the
// widescreen side areas (0 without widescreen, or with VC3_WIDE_AIM=0). The aim point (aim_point)
// keeps the real place; the game is told about it through a hook (hooks.cpp, wide_aim).
float aim_extent();
void set_wide_aim(bool on);  // (the pause menu)

// Scripted input for testing (VC3_INPUT_SCRIPT=file). Lines: "<vblank> <action> [args]"
//   coin1|coin2|start1|start2|test|service|pedal1|change1|trigger1|trigger2|offscreen1 [down|up]  (no arg = 4-frame press)
//   aim1|aim2 <x 0..1> <y 0..1>
// Called once per vblank from the video timing thread.
void script_tick(uint64_t vblank);
CabinetState state();  // thread-safe snapshot (buttons: what the game is being shown)
// The game polls the controls (one I/O board message).
void begin_poll();
unsigned long long poll_count();  // polls so far (for the log)

// Live input (io/input.cpp). Aim in 640x480 screen pixels (off-screen positions allowed).
void set_player(int p, float x, float y, bool offscreen);
// A button's level now and how often it was pressed since the last call (a press already let
// go still shows for the game's next poll).
enum Button { kTrigger, kPedal, kChange, kStart, kCoin };
// press_event_ns: when the first of those presses happened, now_ns: now (SDL_GetTicksNS
// clock; 0 = unknown), for the timing in the log.
void set_button(int p, Button b, int presses, bool down, long long press_event_ns = 0, long long now_ns = 0);
std::string timing_report();  // input timing since the last call, for the log
void set_test_service(bool test, bool service);
// TEST went down within the last `seconds` (the game reboots for its test menu after TEST).
bool test_pressed_within(double seconds);
bool scripted();  // an input script is running: live input stays off

}  // namespace vc3::io
