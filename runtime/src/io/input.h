// Host input -> cabinet controls, configurable per player (settings / launcher).
//
// Each player has an aim device and bindings for the cabinet's buttons:
//   aim:      "mouse"     the system mouse (cursor)
//             "raw:NAME"  one specific mouse, lightgun or Wiimote, read on its own, so two
//                         mice/guns don't share a cursor (Linux: evdev, needs read access to
//                         /dev/input; Windows: Raw Input for mice / guns, HID for Wiimotes)
//             "pad"       the player's gamepad, right stick (left stick: "pad_left")
//             "none"
//   pad:      which gamepad belongs to the player: "1", "2", ... (connection order)
//   bindings: comma-separated inputs per action (trigger, reload, pedal, change, start, coin):
//             "key:NAME"   keyboard key (SDL key name, e.g. "Space", "1")
//             "mouse:BTN"  the player's mouse / lightgun button: left, right, middle, x1, x2
//             "pad:BTN"    the player's gamepad: a, b, x, y, back, start, leftshoulder,
//                          rightshoulder, lefttrigger, righttrigger, leftstick, rightstick,
//                          dpup, dpdown, dpleft, dpright
//   Reload points the gun off screen (the cabinet's screen sensor; the game reloads from it).
// The controller API (Windows: XInput / DirectInput / Raw Input) is an SDL setting applied
// before gamepads are opened (input_api()).
#pragma once
#include <SDL3/SDL.h>

#include <string>
#include <vector>

namespace vc3::io::input {

// SDL hints for the chosen controller API; call before SDL_Init(SDL_INIT_GAMEPAD).
void apply_api_hints();
// Reads the bindings (settings / environment) and opens devices. Call after SDL video init.
void init();
// Window events (mouse, keyboard, gamepad hot-plug).
void handle_event(const SDL_Event& e);
// Once per main-loop iteration: gamepad sticks, raw devices -> cabinet state.
void update();
// The pause menu's key / button (setting "pause"): this event presses it.
bool is_pause_event(const SDL_Event& e);
// Forget held keys and buttons (events missed while the pause menu had them).
void release_held();

// Player p's aim (game pixels) and whether they have a working aim device (crosshair).
// The longest gap between input updates (window loop) since the last call, for the log.
double take_loop_gap_ms();
bool crosshair(int p, float& x, float& y);

// Devices for the launcher.
struct Device {
    std::string id;     // value for an aim setting ("mouse", "raw:NAME", "pad", ...)
    std::string label;  // shown to the player
};
std::vector<Device> aim_devices();
// Raw mice / lightguns found, and whether /dev/input could be read (Linux).
std::vector<std::string> raw_device_names(bool* access_ok = nullptr);

// Default bindings per player and action (for the launcher's reset).
std::string default_binding(int player, const char* action);
extern const char* const kActions[6];        // trigger, reload, pedal, change, start, coin
extern const char* const kActionLabels[6];

}  // namespace vc3::io::input
