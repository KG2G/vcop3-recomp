#include "jvs.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <atomic>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace vc3::io {

static std::mutex g_m;
static CabinetState g_state;

// Screen position (0..1 across the 640x480 frame) -> raw 16-bit gun channel.
//
// The game converts raw readings to centred "gun units" as
//   ux = floor(raw_x * 640 / 65536) - 320,   uy = 240 - floor(raw_y * 480 / 65536)
// and then maps units to the screen per axis, piecewise-linearly through the
// five GUN ADJUSTMENT points (LEFT/CENTER/RIGHT for x, TOP/CENTER/BOTTOM for y)
// whose targets are the dots at pixels 15.5 / 319.5 / 623.5 and 15.5 / 239.5 /
// 463.5. The factory DEFAULT calibration is skewed (a real gun's sensor), so a
// plain linear mapping lands up to ~60px off. We invert the default table
// instead, so the cursor sits under the mouse with DEFAULT calibration (and a
// fresh calibration done with the mouse reproduces those defaults).
// VC3_GUN_LINEAR=1 switches to the plain linear mapping (ux = px - 320), for
// use with a calibration recorded under it.
namespace {
struct Axis { float px[3]; float unit[3]; };
constexpr Axis kDefX{{15.5f, 319.5f, 623.5f}, {-318.f, -10.f, 242.f}};
constexpr Axis kDefY{{15.5f, 239.5f, 463.5f}, {233.f, 15.f, -204.f}};

float to_unit(const Axis& a, float px) {
    const int i = px < a.px[1] ? 0 : 1;  // segment, extrapolated past the outer dots
    const float t = (px - a.px[i]) / (a.px[i + 1] - a.px[i]);
    return a.unit[i] + t * (a.unit[i + 1] - a.unit[i]);
}

uint16_t to_raw(float v) { return uint16_t(std::clamp(v, 0.0f, 65535.0f)); }
}  // namespace

// Last aim point of each gun on the 640x480 screen (the renderer keeps shot effects there
// inside the 4:3 part in widescreen).
static std::atomic<float> g_aim_px[2][2] = {{320.0f, 240.0f}, {320.0f, 240.0f}};
static void set_aim(GunState& g, float nx, float ny) {
    nx = std::clamp(nx, 0.0f, 1.0f);
    ny = std::clamp(ny, 0.0f, 1.0f);
    const int gi = &g == &g_state.gun[1] ? 1 : 0;
    g_aim_px[gi][0] = nx * 640.0f;
    g_aim_px[gi][1] = ny * 480.0f;
    static const bool linear = std::getenv("VC3_GUN_LINEAR") != nullptr;
    if (linear) {
        g.x = to_raw(nx * 65535.0f);
        g.y = to_raw(ny * 65535.0f);
        return;
    }
    const float ux = to_unit(kDefX, nx * 640.0f), uy = to_unit(kDefY, ny * 480.0f);
    g.x = to_raw((ux + 320.0f + 0.5f) * 65536.0f / 640.0f);  // centre of the unit's raw bucket
    g.y = to_raw((240.0f - uy + 0.5f) * 65536.0f / 480.0f);
}

// Both guns start aimed at the screen centre.
static const bool g_aim_init = [] {
    for (auto& g : g_state.gun) set_aim(g, 0.5f, 0.5f);
    return true;
}();

CabinetState state() {
    std::lock_guard lk(g_m);
    return g_state;
}


namespace {
struct ScriptEvent {
    uint64_t at;
    std::string action;
    float a = 0, b = 0;
    int press = -1;  // -1 tap, 1 down, 0 up
};
std::vector<ScriptEvent> load_script() {
    std::vector<ScriptEvent> ev;
    const char* path = std::getenv("VC3_INPUT_SCRIPT");
    if (!path) return ev;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        ScriptEvent e;
        std::string arg;
        ss >> e.at >> e.action;
        if (e.action.rfind("aim", 0) == 0) ss >> e.a >> e.b;
        else if (ss >> arg) e.press = arg == "down" ? 1 : 0;
        if (e.press == -1 && e.action.rfind("aim", 0) != 0) {  // tap: expand into down + up
            ScriptEvent up = e;
            e.press = 1;
            up.press = 0;
            up.at += 4;
            ev.push_back(e);
            ev.push_back(up);
            continue;
        }
        ev.push_back(e);
    }
    std::fprintf(stderr, "[input] loaded %zu scripted events from %s\n", ev.size(), path);
    return ev;
}
}  // namespace

void script_tick(uint64_t vblank) {
    static std::vector<ScriptEvent> events = load_script();
    if (events.empty()) return;
    std::lock_guard lk(g_m);
    // "offscreen" like a real gun (and the live reload): the screen sensor 3 vblanks ahead of
    // the trigger when pressing, the trigger let go 3 vblanks before the sensor comes back.
    static uint64_t trig_at[2] = {0, 0}, back_at[2] = {0, 0};
    for (int p = 0; p < 2; ++p) {
        if (trig_at[p] == vblank) g_state.gun[p].trigger = true;
        if (back_at[p] == vblank) g_state.gun[p].offscreen = false;
    }
    for (const auto& e : events) {
        if (e.at != vblank) continue;
        const bool down = e.press == 1;
        const std::string& a = e.action;
        const int p = a.back() == '2' ? 1 : 0;
        auto& g = g_state.gun[p];
        if (a.rfind("aim", 0) == 0) set_aim(g, e.a, e.b);
        else if (a.rfind("coin", 0) == 0) g_state.coin[p] = down;
        else if (a.rfind("start", 0) == 0) g.start = down;
        else if (a.rfind("trigger", 0) == 0) g.trigger = down;
        else if (a.rfind("away", 0) == 0) g.offscreen = down;  // aimed off screen, trigger untouched
        else if (a.rfind("offscreen", 0) == 0) {
            if (down) {
                g.offscreen = true;
                trig_at[p] = vblank + 3;
            } else {
                g.trigger = false;
                trig_at[p] = 0;
                back_at[p] = vblank + 3;
            }
        }
        else if (a.rfind("pedal", 0) == 0) g.pedal = down;
        else if (a.rfind("change", 0) == 0) g.change = down;
        else if (a == "test") g_state.test = down;
        else if (a == "service") g_state.service = down;
        std::fprintf(stderr, "[input] vblank %llu: %s %s\n", (unsigned long long)vblank, a.c_str(),
                     a.rfind("aim", 0) == 0 ? "" : (down ? "down" : "up"));
    }
}

void set_player(int p, float x, float y, bool trigger, bool offscreen, bool pedal, bool change, bool start) {
    std::lock_guard lk(g_m);
    GunState& g = g_state.gun[p & 1];
    set_aim(g, x / 640.0f, y / 480.0f);
    g.trigger = trigger;  // (reload: input.cpp sequences the trigger after the screen sensor)
    g.offscreen = offscreen;
    g.pedal = pedal;
    g.change = change;
    g.start = start;
}
void set_coin(int slot, bool down) {
    std::lock_guard lk(g_m);
    g_state.coin[slot & 1] = down;
}
void set_test_service(bool test, bool service) {
    std::lock_guard lk(g_m);
    g_state.test = test;
    g_state.service = service;
}
bool scripted() { return std::getenv("VC3_INPUT_SCRIPT") != nullptr; }

void aim_point(int gun, float& x, float& y) {
    x = g_aim_px[gun & 1][0].load(std::memory_order_relaxed);
    y = g_aim_px[gun & 1][1].load(std::memory_order_relaxed);
}

}  // namespace vc3::io
