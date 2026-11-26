// Cabinet controls as seen by the JVS I/O board, fed by io/input.cpp (configurable
// bindings) or an input script. TEST/SERVICE (F2/F3) are the filter-board buttons (base
// board PINSA bits 7/6).
#pragma once
#include <SDL3/SDL.h>

#include <cstdint>

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

// Scripted input for testing (VC3_INPUT_SCRIPT=file). Lines: "<vblank> <action> [args]"
//   coin1|coin2|start1|start2|test|service|pedal1|change1|trigger1|trigger2|offscreen1 [down|up]  (no arg = 4-frame press)
//   aim1|aim2 <x 0..1> <y 0..1>
// Called once per vblank from the video timing thread.
void script_tick(uint64_t vblank);
CabinetState state();  // thread-safe snapshot

// Live input (io/input.cpp). Aim in 640x480 screen pixels (off-screen positions allowed).
void set_player(int p, float x, float y, bool trigger, bool offscreen, bool pedal, bool change, bool start);
void set_coin(int slot, bool down);
void set_test_service(bool test, bool service);
bool scripted();  // an input script is running: live input stays off

}  // namespace vc3::io
