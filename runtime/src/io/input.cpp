#include "io/input.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
extern "C" __declspec(dllimport) BOOLEAN __stdcall HidD_GetProductString(HANDLE, PVOID, ULONG);  // hid.dll
#endif
#ifdef __linux__
#include <dirent.h>
#include <fcntl.h>
#include <linux/input.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <unistd.h>
#endif

#include "gpu/nv2a.h"
#include "io/jvs.h"
#include "settings.h"

namespace vc3::io::input {

const char* const kActions[6] = {"trigger", "reload", "pedal", "change", "start", "coin"};
const char* const kActionLabels[6] = {"Trigger", "Reload", "Pedal (E.S. slow motion)", "Change weapon", "Start",
                                      "Insert coin"};

std::string default_binding(int player, const char* action) {
    const std::string a = action;
    if (player == 0) {
        if (a == "trigger") return "mouse:left,pad:righttrigger";
        if (a == "reload") return "mouse:right,pad:lefttrigger";
        if (a == "pedal") return "key:Space,pad:a";
        if (a == "change") return "mouse:middle,key:E,pad:x";
        if (a == "start") return "key:1,pad:start,mouse:plus";
        if (a == "coin") return "key:5,pad:back,mouse:minus";
    } else {
        if (a == "trigger") return "mouse:left,pad:righttrigger";
        if (a == "reload") return "mouse:right,pad:lefttrigger";
        if (a == "pedal") return "pad:a";
        if (a == "change") return "mouse:middle,pad:x";
        if (a == "start") return "key:2,pad:start,mouse:plus";
        if (a == "coin") return "key:6,pad:back,mouse:minus";
    }
    return "";
}

namespace {

// ---------------------------------------------------------------- raw mice / lightguns
// Linux: each device read from its evdev node on a thread of its own, grabbed (so the
// system cursor doesn't also move with it). Relative mice drive a virtual crosshair;
// absolute devices (lightguns, tablets) point at a place in the window.
struct RawState {
    std::string name;
    std::atomic<float> x{320}, y{240};  // game pixels
    std::atomic<unsigned> buttons{0};   // bit 0 left, 1 right, 2 middle, 3 x1, 4 x2
    std::atomic<int> presses[16] = {};  // per button: presses since update() last took them
};
std::mutex g_raw_m;
std::map<std::string, RawState*> g_raw;  // by name ("Name" or "Name #2")

#ifdef __linux__
struct EvDev {
    std::string path, name;
    bool rel = false, abs = false;
};
bool test_bit(const unsigned long* bits, int b) { return (bits[b / (8 * sizeof(long))] >> (b % (8 * sizeof(long)))) & 1; }

std::vector<EvDev> scan_evdev(bool* access_ok) {
    std::vector<EvDev> out;
    bool any_denied = false, any_ok = false;
    DIR* d = opendir("/dev/input");
    if (!d) {
        if (access_ok) *access_ok = false;
        return out;
    }
    std::vector<std::string> nodes;
    while (dirent* e = readdir(d))
        if (!std::strncmp(e->d_name, "event", 5)) nodes.push_back(std::string("/dev/input/") + e->d_name);
    closedir(d);
    std::sort(nodes.begin(), nodes.end(), [](const std::string& a, const std::string& b) {
        return std::atoi(a.c_str() + 16) < std::atoi(b.c_str() + 16);
    });
    std::map<std::string, int> seen;
    for (const std::string& p : nodes) {
        const int fd = open(p.c_str(), O_RDONLY | O_NONBLOCK);
        if (fd < 0) {
            any_denied = true;
            continue;
        }
        any_ok = true;
        char name[256] = "?";
        ioctl(fd, EVIOCGNAME(sizeof name), name);
        unsigned long ev[(EV_MAX + 63) / 64] = {}, key[(KEY_MAX + 63) / 64] = {}, rel[(REL_MAX + 63) / 64] = {},
                      ab[(ABS_MAX + 63) / 64] = {};
        ioctl(fd, EVIOCGBIT(0, sizeof ev), ev);
        ioctl(fd, EVIOCGBIT(EV_KEY, sizeof key), key);
        ioctl(fd, EVIOCGBIT(EV_REL, sizeof rel), rel);
        ioctl(fd, EVIOCGBIT(EV_ABS, sizeof ab), ab);
        close(fd);
        const bool button = test_bit(ev, EV_KEY) && test_bit(key, BTN_LEFT);
        EvDev dv{p, name};
        dv.rel = test_bit(ev, EV_REL) && test_bit(rel, REL_X) && test_bit(rel, REL_Y);
        dv.abs = test_bit(ev, EV_ABS) && test_bit(ab, ABS_X) && test_bit(ab, ABS_Y);
        if (!button || !(dv.rel || dv.abs)) continue;
        if (dv.name.rfind("Nintendo Wii Remote", 0) == 0) continue;  // listed as Wiimotes
        const int n = ++seen[dv.name];
        if (n > 1) dv.name += " #" + std::to_string(n);
        out.push_back(dv);
    }
    if (access_ok) *access_ok = any_ok || !any_denied;
    return out;
}

void run_evdev(EvDev dev, RawState* st) {
    const int fd = open(dev.path.c_str(), O_RDONLY);
    if (fd < 0) {
        std::fprintf(stderr, "[input] can't open %s (%s): add yourself to the 'input' group\n", dev.path.c_str(), dev.name.c_str());
        return;
    }
    if (ioctl(fd, EVIOCGRAB, 1) != 0)
        std::fprintf(stderr, "[input] %s: not grabbed (the system cursor moves with it too)\n", dev.name.c_str());
    input_absinfo ax{}, ay{};
    if (dev.abs) {
        ioctl(fd, EVIOCGABS(ABS_X), &ax);
        ioctl(fd, EVIOCGABS(ABS_Y), &ay);
    }
    std::fprintf(stderr, "[input] reading %s (%s, %s)\n", dev.name.c_str(), dev.path.c_str(), dev.abs ? "absolute" : "relative");
    float ax_v = 0.5f, ay_v = 0.5f;
    for (;;) {
        input_event ev[64];
        const ssize_t n = read(fd, ev, sizeof ev);
        if (n <= 0) {
            std::fprintf(stderr, "[input] %s disconnected\n", dev.name.c_str());
            break;
        }
        for (ssize_t i = 0; i < n / ssize_t(sizeof(input_event)); ++i) {
            const input_event& e = ev[i];
            if (e.type == EV_REL && (e.code == REL_X || e.code == REL_Y)) {
                constexpr float kScale = 0.4f;  // game pixels per count (a mouse sweep ~1600 counts)
                auto& c = e.code == REL_X ? st->x : st->y;
                const float lim = e.code == REL_X ? 640.0f : 480.0f;
                c = std::clamp(c.load() + float(e.value) * kScale, -32.0f, lim + 32.0f);
            } else if (e.type == EV_ABS && (e.code == ABS_X || e.code == ABS_Y)) {
                const input_absinfo& a = e.code == ABS_X ? ax : ay;
                const float v = a.maximum > a.minimum ? float(e.value - a.minimum) / float(a.maximum - a.minimum) : 0.5f;
                (e.code == ABS_X ? ax_v : ay_v) = v;
                float gx, gy;
                gpu::window_to_game(ax_v, ay_v, gx, gy);
                st->x = gx;
                st->y = gy;
            } else if (e.type == EV_KEY) {
                int bit = -1;
                switch (e.code) {
                    case BTN_LEFT: case BTN_TRIGGER: bit = 0; break;
                    case BTN_RIGHT: bit = 1; break;
                    case BTN_MIDDLE: bit = 2; break;
                    case BTN_SIDE: bit = 3; break;
                    case BTN_EXTRA: bit = 4; break;
                    default: break;
                }
                if (bit >= 0) {
                    if (e.value) {
                        st->buttons |= 1u << bit;
                        ++st->presses[bit];
                    } else {
                        st->buttons &= ~(1u << bit);
                    }
                }
            }
        }
    }
    close(fd);
}

// Wiimotes (Linux hid-wiimote): a core device (buttons) and an IR device (the sensor bar's
// dots, ABS_HAT0X..HAT3Y, 1023 = not seen), matched by their Bluetooth address (uniq).
struct WiiDev { std::string core, ir, name; };
std::vector<WiiDev> scan_wii() {
    std::vector<WiiDev> out;
    DIR* d = opendir("/dev/input");
    if (!d) return out;
    std::map<std::string, WiiDev> by_uniq;
    std::vector<std::string> order;
    while (dirent* e = readdir(d)) {
        if (std::strncmp(e->d_name, "event", 5)) continue;
        const std::string p = std::string("/dev/input/") + e->d_name;
        const int fd = open(p.c_str(), O_RDONLY | O_NONBLOCK);
        if (fd < 0) continue;
        char name[256] = "", uniq[64] = "";
        ioctl(fd, EVIOCGNAME(sizeof name), name);
        ioctl(fd, EVIOCGUNIQ(sizeof uniq), uniq);
        close(fd);
        const std::string n = name;
        if (n.rfind("Nintendo Wii Remote", 0) != 0) continue;
        const std::string u = *uniq ? uniq : n;
        if (!by_uniq.count(u)) order.push_back(u);
        WiiDev& w = by_uniq[u];
        if (n == "Nintendo Wii Remote") w.core = p;
        else if (n == "Nintendo Wii Remote IR") w.ir = p;
    }
    closedir(d);
    std::sort(order.begin(), order.end());
    for (const std::string& u : order) {
        WiiDev w = by_uniq[u];
        if (w.core.empty() || w.ir.empty()) continue;
        w.name = "Wiimote " + std::to_string(out.size() + 1);
        out.push_back(w);
    }
    return out;
}

void run_wii(WiiDev w, RawState* st) {
    int fds[2] = {open(w.core.c_str(), O_RDONLY), open(w.ir.c_str(), O_RDONLY)};
    if (fds[0] < 0 || fds[1] < 0) {
        std::fprintf(stderr, "[input] can't open %s: add yourself to the 'input' group\n", w.name.c_str());
        return;
    }
    ioctl(fds[0], EVIOCGRAB, 1);
    std::fprintf(stderr, "[input] reading %s (%s + %s)\n", w.name.c_str(), w.core.c_str(), w.ir.c_str());
    int dot[4][2];
    for (auto& d : dot) d[0] = d[1] = 1023;
    float mx = 512, my = 384;
    for (;;) {
        pollfd pf[2] = {{fds[0], POLLIN, 0}, {fds[1], POLLIN, 0}};
        if (poll(pf, 2, -1) < 0) break;
        for (int k = 0; k < 2; ++k) {
            if (!(pf[k].revents & POLLIN)) continue;
            input_event ev[64];
            const ssize_t n = read(fds[k], ev, sizeof ev);
            if (n <= 0) {
                std::fprintf(stderr, "[input] %s disconnected\n", w.name.c_str());
                close(fds[0]);
                close(fds[1]);
                return;
            }
            for (ssize_t i = 0; i < n / ssize_t(sizeof(input_event)); ++i) {
                const input_event& e = ev[i];
                if (k == 1 && e.type == EV_ABS && e.code >= ABS_HAT0X && e.code <= ABS_HAT3Y) {
                    const int idx = (e.code - ABS_HAT0X) / 2;
                    dot[idx][(e.code - ABS_HAT0X) & 1] = e.value;
                } else if (k == 1 && e.type == EV_SYN) {
                    // The sensor bar: the midpoint of the (first two) visible dots.
                    float sx = 0, sy = 0;
                    int seen = 0;
                    for (auto& d : dot)
                        if (d[0] < 1023 && d[1] < 1023 && seen < 2) {
                            sx += float(d[0]);
                            sy += float(d[1]);
                            ++seen;
                        }
                    if (seen) {
                        mx = sx / float(seen);
                        my = sy / float(seen);
                        // Camera image -> pointing direction (mirrored), with some gain: the
                        // dots cover only part of the camera's view across a screen.
                        constexpr float kGain = 1.4f;
                        const float nx = 0.5f + (0.5f - mx / 1024.0f) * kGain, ny = 0.5f + (my / 768.0f - 0.5f) * kGain;
                        float gx, gy;
                        gpu::window_to_game(nx, ny, gx, gy);
                        st->x = gx;
                        st->y = gy;
                    }
                } else if (k == 0 && e.type == EV_KEY) {
                    int bit = -1;
                    switch (e.code) {
                        case BTN_B: bit = 0; break;
                        case BTN_A: bit = 1; break;
                        case BTN_MODE: bit = 2; break;
                        case BTN_1: bit = 3; break;
                        case BTN_2: bit = 4; break;
                        case KEY_NEXT: bit = 5; break;
                        case KEY_PREVIOUS: bit = 6; break;
                        case KEY_UP: bit = 7; break;
                        case KEY_DOWN: bit = 8; break;
                        case KEY_LEFT: bit = 9; break;
                        case KEY_RIGHT: bit = 10; break;
                        default: break;
                    }
                    if (bit >= 0) {
                        if (e.value) {
                            st->buttons |= 1u << bit;
                            ++st->presses[bit];
                        } else {
                            st->buttons &= ~(1u << bit);
                        }
                    }
                }
            }
        }
    }
}
#endif  // __linux__ (evdev mice / lightguns, Wiimotes)

#ifdef _WIN32
// ---- Windows: mice / lightguns through Raw Input, Wiimotes over HID (SDL's hidapi).
// Raw Input says which device every movement and click came from, so each player's mouse or
// lightgun drives their own crosshair. Windows can't grab a device the way Linux does: every
// mouse still moves the system cursor too (the cursor is kept inside the game window while a
// separate device is in use, so a gun pointed elsewhere can't click on other windows).
// Devices are named by their product name ("Name #2" for a second one), like on Linux.
std::string utf8(const wchar_t* w) {
    const int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
    std::string s(n > 0 ? size_t(n - 1) : 0, '\0');
    if (n > 1) WideCharToMultiByte(CP_UTF8, 0, w, -1, s.data(), n, nullptr, nullptr);
    return s;
}

struct WinMouse { HANDLE h; std::string name, path; };
std::vector<WinMouse> scan_rawinput() {
    std::vector<WinMouse> out;
    UINT n = 0;
    if (GetRawInputDeviceList(nullptr, &n, sizeof(RAWINPUTDEVICELIST)) != 0 || !n) return out;
    std::vector<RAWINPUTDEVICELIST> list(n);
    n = GetRawInputDeviceList(list.data(), &n, sizeof(RAWINPUTDEVICELIST));
    if (n == UINT(-1)) return out;
    for (UINT i = 0; i < n; ++i) {
        if (list[i].dwType != RIM_TYPEMOUSE) continue;
        wchar_t path[512];
        UINT len = UINT(std::size(path));
        if (GetRawInputDeviceInfoW(list[i].hDevice, RIDI_DEVICENAME, path, &len) == UINT(-1)) continue;
        const std::string p = utf8(path);
        std::string up = p;
        for (char& c : up) c = char(std::toupper((unsigned char)c));
        if (up.find("RDP_MOU") != std::string::npos || up.find("#ROOT#") != std::string::npos ||
            up.find("\\ROOT#") != std::string::npos)
            continue;  // remote-desktop / virtual mice
        std::string name;
        HANDLE f = CreateFileW(path, 0, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, 0, nullptr);
        if (f != INVALID_HANDLE_VALUE) {
            wchar_t prod[128] = {};
            if (HidD_GetProductString(f, prod, sizeof prod) && prod[0]) name = utf8(prod);
            CloseHandle(f);
        }
        if (name.empty()) {  // no product string: "Mouse (VID_xxxx&PID_yyyy)"
            const size_t v = up.find("VID_");
            name = v != std::string::npos && v + 17 <= up.size() ? "Mouse (" + up.substr(v, 17) + ")" : "Mouse";
        }
        while (!name.empty() && name.back() == ' ') name.pop_back();
        out.push_back({list[i].hDevice, name, p});
    }
    std::sort(out.begin(), out.end(), [](const WinMouse& a, const WinMouse& b) { return a.path < b.path; });
    std::map<std::string, int> seen;
    for (WinMouse& m : out)
        if (const int k = ++seen[m.name]; k > 1) m.name += " #" + std::to_string(k);
    return out;
}

std::mutex g_win_m;
std::map<std::string, RawState*> g_win_claimed;  // device name -> the player's state
std::map<HANDLE, RawState*> g_win_by_handle;      // resolved devices (nullptr: not used by a player)
std::atomic<HWND> g_game_hwnd{nullptr};
std::atomic<bool> g_win_uses_raw{false};

RawState* win_state_for(HANDLE h) {
    std::lock_guard lk(g_win_m);
    if (auto it = g_win_by_handle.find(h); it != g_win_by_handle.end()) return it->second;
    RawState* st = nullptr;
    for (const WinMouse& m : scan_rawinput())
        if (m.h == h) {
            if (auto c = g_win_claimed.find(m.name); c != g_win_claimed.end()) {
                st = c->second;
                std::fprintf(stderr, "[input] reading %s (Raw Input)\n", m.name.c_str());
            }
            break;
        }
    g_win_by_handle[h] = st;
    return st;
}

void win_mouse(HANDLE h, const RAWMOUSE& m) {
    RawState* st = win_state_for(h);
    if (!st) return;
    if (m.usFlags & MOUSE_MOVE_ABSOLUTE) {  // lightguns, tablets: a place on the screen
        const bool virt = m.usFlags & MOUSE_VIRTUAL_DESKTOP;
        const int ox = virt ? GetSystemMetrics(SM_XVIRTUALSCREEN) : 0, oy = virt ? GetSystemMetrics(SM_YVIRTUALSCREEN) : 0;
        const int w = GetSystemMetrics(virt ? SM_CXVIRTUALSCREEN : SM_CXSCREEN), hh = GetSystemMetrics(virt ? SM_CYVIRTUALSCREEN : SM_CYSCREEN);
        const float px = float(ox) + float(m.lLastX) / 65535.0f * float(w), py = float(oy) + float(m.lLastY) / 65535.0f * float(hh);
        float nx = w > 0 ? (px - float(ox)) / float(w) : 0.5f, ny = hh > 0 ? (py - float(oy)) / float(hh) : 0.5f;
        if (HWND gw = g_game_hwnd.load()) {  // the game window's client area, in screen pixels
            RECT r;
            POINT tl{0, 0};
            if (GetClientRect(gw, &r) && ClientToScreen(gw, &tl) && r.right > 0 && r.bottom > 0) {
                nx = (px - float(tl.x)) / float(r.right);
                ny = (py - float(tl.y)) / float(r.bottom);
            }
        }
        float gx, gy;
        gpu::window_to_game(nx, ny, gx, gy);
        st->x = gx;
        st->y = gy;
    } else if (m.lLastX || m.lLastY) {  // mice: a crosshair of their own
        constexpr float kScale = 0.4f;  // game pixels per count, as on Linux
        st->x = std::clamp(st->x.load() + float(m.lLastX) * kScale, -32.0f, 640.0f + 32.0f);
        st->y = std::clamp(st->y.load() + float(m.lLastY) * kScale, -32.0f, 480.0f + 32.0f);
    }
    static constexpr USHORT kDown[5] = {RI_MOUSE_BUTTON_1_DOWN, RI_MOUSE_BUTTON_2_DOWN, RI_MOUSE_BUTTON_3_DOWN,
                                        RI_MOUSE_BUTTON_4_DOWN, RI_MOUSE_BUTTON_5_DOWN};
    static constexpr USHORT kUp[5] = {RI_MOUSE_BUTTON_1_UP, RI_MOUSE_BUTTON_2_UP, RI_MOUSE_BUTTON_3_UP,
                                      RI_MOUSE_BUTTON_4_UP, RI_MOUSE_BUTTON_5_UP};
    for (int b = 0; b < 5; ++b) {  // left, right, middle, x1, x2
        if (m.usButtonFlags & kDown[b]) {
            st->buttons |= 1u << b;
            ++st->presses[b];
        }
        if (m.usButtonFlags & kUp[b]) st->buttons &= ~(1u << b);
    }
}

LRESULT CALLBACK raw_wndproc(HWND hw, UINT msg, WPARAM wp, LPARAM lp) {
    if (msg == WM_INPUT) {
        alignas(8) BYTE buf[256];
        UINT sz = sizeof buf;
        if (GetRawInputData(HRAWINPUT(lp), RID_INPUT, buf, &sz, sizeof(RAWINPUTHEADER)) != UINT(-1)) {
            const RAWINPUT* ri = reinterpret_cast<const RAWINPUT*>(buf);
            if (ri->header.dwType == RIM_TYPEMOUSE) win_mouse(ri->header.hDevice, ri->data.mouse);
        }
    } else if (msg == WM_INPUT_DEVICE_CHANGE) {  // plugged in or out: resolve devices again
        std::lock_guard lk(g_win_m);
        g_win_by_handle.clear();
    }
    return DefWindowProcW(hw, msg, wp, lp);
}

void raw_thread() {
    WNDCLASSW wc{};
    wc.lpfnWndProc = raw_wndproc;
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.lpszClassName = L"vc3_rawinput";
    RegisterClassW(&wc);
    HWND hw = CreateWindowExW(0, wc.lpszClassName, L"", 0, 0, 0, 0, 0, HWND_MESSAGE, nullptr, wc.hInstance, nullptr);
    // Mice (generic desktop page, usage 2), also while the game window isn't focused. SDL only
    // registers raw mouse input in relative mouse mode, which the game doesn't use.
    RAWINPUTDEVICE rid{0x01, 0x02, RIDEV_INPUTSINK | RIDEV_DEVNOTIFY, hw};
    if (!hw || !RegisterRawInputDevices(&rid, 1, sizeof rid)) {
        std::fprintf(stderr, "[input] Raw Input unavailable (error %lu): separate mice / lightguns won't work\n", GetLastError());
        return;
    }
    MSG m;
    while (GetMessageW(&m, nullptr, 0, 0) > 0) DispatchMessageW(&m);
}

// Wiimotes: Nintendo RVL-CNT-01 (and -TR), paired with Windows over Bluetooth. Set up the IR
// camera and read the buttons + IR dots (report 0x33); the pointer math is the Linux one.
struct WinWii { std::string path, name; };
std::vector<WinWii> scan_wii_win() {
    std::vector<WinWii> out;
    if (SDL_hid_init() != 0) return out;
    for (const unsigned short pid : {0x0306, 0x0330}) {
        SDL_hid_device_info* list = SDL_hid_enumerate(0x057E, pid);
        for (SDL_hid_device_info* d = list; d; d = d->next)
            if (d->path) out.push_back({d->path, ""});
        SDL_hid_free_enumeration(list);
    }
    std::sort(out.begin(), out.end(), [](const WinWii& a, const WinWii& b) { return a.path < b.path; });
    out.erase(std::unique(out.begin(), out.end(), [](const WinWii& a, const WinWii& b) { return a.path == b.path; }), out.end());
    for (size_t i = 0; i < out.size(); ++i) out[i].name = "Wiimote " + std::to_string(i + 1);
    return out;
}

void run_wii_win(WinWii w, RawState* st, int led) {
    SDL_hid_device* d = SDL_hid_open_path(w.path.c_str());
    if (!d) {
        std::fprintf(stderr, "[input] can't open %s (pair it with Windows over Bluetooth first)\n", w.name.c_str());
        return;
    }
    auto send = [&](std::initializer_list<unsigned char> bytes) {
        unsigned char b[22] = {};  // (hidapi pads to the report length the device declares)
        size_t n = 0;
        for (const unsigned char c : bytes) b[n++] = c;
        SDL_hid_write(d, b, n);
        SDL_Delay(30);
    };
    auto write_reg = [&](unsigned addr, std::initializer_list<unsigned char> data) {  // 0x16: write registers
        unsigned char b[22] = {0x16, 0x04, (unsigned char)(addr >> 16), (unsigned char)(addr >> 8), (unsigned char)addr,
                               (unsigned char)data.size()};
        size_t n = 6;
        for (const unsigned char c : data) b[n++] = c;
        SDL_hid_write(d, b, 22);
        SDL_Delay(50);
    };
    auto setup = [&] {
        send({0x11, (unsigned char)(0x10 << (led & 3))});  // player LED
        send({0x13, 0x04});                                 // IR camera on
        send({0x1A, 0x04});
        write_reg(0xB00030, {0x08});
        write_reg(0xB00000, {0x02, 0x00, 0x00, 0x71, 0x01, 0x00, 0xAA, 0x00, 0x64});  // sensitivity
        write_reg(0xB0001A, {0x63, 0x03});
        write_reg(0xB00033, {0x03});  // extended mode: x, y, size per dot
        write_reg(0xB00030, {0x08});
        send({0x12, 0x00, 0x33});  // report buttons + accelerometer + IR
    };
    setup();
    std::fprintf(stderr, "[input] reading %s (HID)\n", w.name.c_str());
    // Report bytes 1-2 -> the Linux button bits: 0 B, 1 A, 2 Home, 3 one, 4 two, 5 plus, 6 minus,
    // 7 up, 8 down, 9 left, 10 right.
    auto bits = [](unsigned char b1, unsigned char b2) {
        unsigned v = 0;
        if (b2 & 0x04) v |= 1u << 0;
        if (b2 & 0x08) v |= 1u << 1;
        if (b2 & 0x80) v |= 1u << 2;
        if (b2 & 0x02) v |= 1u << 3;
        if (b2 & 0x01) v |= 1u << 4;
        if (b1 & 0x10) v |= 1u << 5;
        if (b2 & 0x10) v |= 1u << 6;
        if (b1 & 0x08) v |= 1u << 7;
        if (b1 & 0x04) v |= 1u << 8;
        if (b1 & 0x01) v |= 1u << 9;
        if (b1 & 0x02) v |= 1u << 10;
        return v;
    };
    unsigned last = 0;
    for (;;) {
        unsigned char r[32];
        const int n = SDL_hid_read_timeout(d, r, sizeof r, 250);
        if (n < 0) {
            std::fprintf(stderr, "[input] %s disconnected\n", w.name.c_str());
            break;
        }
        if (n < 3) continue;
        if (r[0] == 0x20) send({0x12, 0x00, 0x33});  // status report (e.g. an extension plugged): resume IR reports
        if ((r[0] >= 0x30 && r[0] <= 0x37) || (r[0] >= 0x20 && r[0] <= 0x22)) {
            const unsigned now = bits(r[1], r[2]);
            for (int b = 0; b < 11; ++b)
                if ((now & ~last) >> b & 1) ++st->presses[b];
            st->buttons = now;
            last = now;
        }
        if (r[0] == 0x33 && n >= 18) {
            float sx = 0, sy = 0;
            int seen = 0;
            for (int i = 0; i < 4 && seen < 2; ++i) {  // the sensor bar: midpoint of the first two dots
                const unsigned char* o = r + 6 + 3 * i;
                const int x = o[0] | ((o[2] >> 4) & 3) << 8, y = o[1] | ((o[2] >> 6) & 3) << 8;
                if (x < 1023 && y < 1023) {
                    sx += float(x);
                    sy += float(y);
                    ++seen;
                }
            }
            if (seen) {
                const float mx = sx / float(seen), my = sy / float(seen);
                constexpr float kGain = 1.4f;  // as on Linux
                const float nx = 0.5f + (0.5f - mx / 1024.0f) * kGain, ny = 0.5f + (my / 768.0f - 0.5f) * kGain;
                float gx, gy;
                gpu::window_to_game(nx, ny, gx, gy);
                st->x = gx;
                st->y = gy;
            }
        }
    }
    SDL_hid_close(d);
}
#endif  // _WIN32

RawState* open_raw(const std::string& name) {
    std::lock_guard lk(g_raw_m);
    auto it = g_raw.find(name);
    if (it != g_raw.end()) return it->second;
    auto* st = new RawState;
    st->name = name;
    g_raw[name] = st;
#ifdef __linux__
    for (const WiiDev& w : scan_wii())
        if (w.name == name) {
            std::thread(run_wii, w, st).detach();
            return st;
        }
    for (const EvDev& d : scan_evdev(nullptr))
        if (d.name == name) {
            std::thread(run_evdev, d, st).detach();
            return st;
        }
#endif
#ifdef _WIN32
    const std::vector<WinWii> wiis = scan_wii_win();
    for (size_t i = 0; i < wiis.size(); ++i)
        if (wiis[i].name == name) {
            std::thread(run_wii_win, wiis[i], st, int(i)).detach();
            g_win_uses_raw = true;
            return st;
        }
    {
        // A mouse / lightgun: the Raw Input thread hands it its events (also if it is plugged in later).
        bool present = false;
        for (const WinMouse& m : scan_rawinput()) present |= m.name == name;
        {
            std::lock_guard lk2(g_win_m);
            g_win_claimed[name] = st;
            g_win_by_handle.clear();
        }
        static std::once_flag once;
        std::call_once(once, [] { std::thread(raw_thread).detach(); });
        g_win_uses_raw = true;
        if (present) return st;
    }
#endif
    std::fprintf(stderr, "[input] raw device \"%s\" not found\n", name.c_str());
    return st;
}

// ---------------------------------------------------------------- bindings
struct Input {
    enum Kind { kKey, kMouse, kPad } kind;
    int code;  // SDL keycode, mouse button bit, SDL gamepad button (or 100/101 = left/right trigger)
};
struct Player {
    std::string aim = "none";
    int pad = 0;  // 1-based gamepad slot
    RawState* raw = nullptr;
    std::vector<Input> bind[6];
    float x = 320, y = 240;  // aim (game pixels)
    unsigned mouse_buttons = 0;  // system mouse (players aiming with it)
    int presses[6] = {};         // per action: presses since the last update()
    Uint64 press_ns[6] = {};     // when the first of them happened (event time, for the log)
};
Player g_p[2];
std::vector<SDL_Gamepad*> g_pads;  // connection order
std::vector<SDL_Keycode> g_keys_down;
std::vector<Input> g_test_bind, g_service_bind;  // cabinet buttons (keys; gamepad 1)
Uint64 g_last_tick = 0;
Uint64 g_loop_gap_max_ns = 0;  // longest time between update() calls since the last report

bool pad_aim(const Player& p) { return p.aim == "pad" || p.aim == "pad_left"; }
// The system mouse drives the players aiming with it; when none does, it also moves (and its
// buttons fire for) the first player aiming with a gamepad, so mouse and stick both work.
bool takes_system_mouse(int i) {
    if (g_p[i].aim == "mouse") return true;
    if (g_p[0].aim == "mouse" || g_p[1].aim == "mouse" || !pad_aim(g_p[i])) return false;
    return i == 0 || !pad_aim(g_p[0]);
}

// Buttons of a player's aim device. Mice: left, right, middle, x1, x2. Wiimotes: B = left,
// A = right, Home = middle, 1 = x1, 2 = x2, plus the names plus, minus, up, down, left_d, right_d.
const char* const kDevButtons[] = {"left", "right", "middle", "x1", "x2", "plus", "minus", "up", "down", "dleft", "dright"};
int mouse_bit(const std::string& n) {
    for (int i = 0; i < int(std::size(kDevButtons)); ++i)
        if (n == kDevButtons[i]) return i;
    return -1;
}
int pad_code(const std::string& n) {
    if (n == "lefttrigger") return 100;
    if (n == "righttrigger") return 101;
    const SDL_GamepadButton b = SDL_GetGamepadButtonFromString(n.c_str());
    return b == SDL_GAMEPAD_BUTTON_INVALID ? -1 : int(b);
}
std::vector<Input> parse(const std::string& s) {
    std::vector<Input> out;
    size_t i = 0;
    while (i <= s.size()) {
        size_t j = s.find(',', i);
        if (j == std::string::npos) j = s.size();
        const std::string tok = s.substr(i, j - i);
        i = j + 1;
        const size_t c = tok.find(':');
        if (c == std::string::npos) continue;
        const std::string kind = tok.substr(0, c), name = tok.substr(c + 1);
        if (kind == "key") {
            const SDL_Keycode k = SDL_GetKeyFromName(name.c_str());
            if (k != SDLK_UNKNOWN) out.push_back({Input::kKey, int(k)});
        } else if (kind == "mouse") {
            if (const int b = mouse_bit(name); b >= 0) out.push_back({Input::kMouse, b});
        } else if (kind == "pad") {
            if (const int b = pad_code(name); b >= 0) out.push_back({Input::kPad, b});
        }
    }
    return out;
}

SDL_Gamepad* player_pad(const Player& p) {
    return p.pad >= 1 && p.pad <= int(g_pads.size()) ? g_pads[size_t(p.pad - 1)] : nullptr;
}

bool held(const Player& p, const Input& in) {
    switch (in.kind) {
        case Input::kKey: return std::find(g_keys_down.begin(), g_keys_down.end(), SDL_Keycode(in.code)) != g_keys_down.end();
        case Input::kMouse: {
            const unsigned b = p.raw ? p.raw->buttons.load() : p.mouse_buttons;
            return (b >> in.code) & 1;
        }
        case Input::kPad: {
            SDL_Gamepad* g = player_pad(p);
            if (!g) return false;
            if (in.code >= 100)
                return SDL_GetGamepadAxis(g, in.code == 100 ? SDL_GAMEPAD_AXIS_LEFT_TRIGGER : SDL_GAMEPAD_AXIS_RIGHT_TRIGGER) > 16000;
            return SDL_GetGamepadButton(g, SDL_GamepadButton(in.code));
        }
    }
    return false;
}

std::string setting(int p, const char* what) {
    const std::string key = "p" + std::to_string(p + 1) + "_" + what;
    return settings::get(key.c_str());
}

}  // namespace

void apply_api_hints() {
#ifdef _WIN32
    // A Wiimote aiming for a player is read here over HID: keep SDL's own Wii remote driver off it.
    for (int p = 0; p < 2; ++p)
        if (setting(p, "aim").rfind("raw:Wiimote", 0) == 0) SDL_SetHint(SDL_HINT_JOYSTICK_HIDAPI_WII, "0");
#endif
    const std::string api = settings::get("input_api");
    // SDL picks per controller among these on Windows; restricting the set selects the API.
    if (api == "xinput") {
        SDL_SetHint(SDL_HINT_JOYSTICK_RAWINPUT, "0");
        SDL_SetHint(SDL_HINT_JOYSTICK_DIRECTINPUT, "0");
        SDL_SetHint(SDL_HINT_XINPUT_ENABLED, "1");
    } else if (api == "dinput") {
        SDL_SetHint(SDL_HINT_JOYSTICK_RAWINPUT, "0");
        SDL_SetHint(SDL_HINT_XINPUT_ENABLED, "0");
        SDL_SetHint(SDL_HINT_JOYSTICK_DIRECTINPUT, "1");
    } else if (api == "rawinput") {
        SDL_SetHint(SDL_HINT_JOYSTICK_RAWINPUT, "1");
    }
}

void init() {
    SDL_InitSubSystem(SDL_INIT_GAMEPAD);
    int n = 0;
    if (SDL_JoystickID* ids = SDL_GetGamepads(&n)) {
        for (int i = 0; i < n; ++i)
            if (SDL_Gamepad* g = SDL_OpenGamepad(ids[i])) g_pads.push_back(g);
        SDL_free(ids);
    }
    for (int p = 0; p < 2; ++p) {
        Player& pl = g_p[p];
        pl.aim = setting(p, "aim");
        pl.pad = std::atoi(setting(p, "pad").c_str());
        for (int a = 0; a < 6; ++a) pl.bind[a] = parse(setting(p, kActions[a]));
        if (pl.aim.rfind("raw:", 0) == 0) pl.raw = open_raw(pl.aim.substr(4));
        std::fprintf(stderr, "[input] player %d: aim %s, gamepad %d\n", p + 1, pl.aim.c_str(), pl.pad);
    }
    {  // separate mice / lightguns / Wiimotes available (for bug reports)
        std::string list;
        for (const Device& d : aim_devices())
            if (d.id.rfind("raw:", 0) == 0) list += (list.empty() ? "" : ", ") + d.id.substr(4);
        std::fprintf(stderr, "[input] separate devices: %s\n", list.empty() ? "none" : list.c_str());
    }
    g_test_bind = parse(settings::get("test"));
    g_service_bind = parse(settings::get("service"));
    for (SDL_Gamepad* g : g_pads) std::fprintf(stderr, "[input] gamepad: %s\n", SDL_GetGamepadName(g));
    g_last_tick = SDL_GetTicksNS();
}

// A button went down: count a press for every action bound to it (update() hands each press
// to the game, so presses shorter than one update, or several within one, all count).
void count_press(Input::Kind kind, int code, Uint64 when_ns, SDL_JoystickID pad = 0) {
    for (int i = 0; i < 2; ++i) {
        Player& p = g_p[i];
        if (kind == Input::kMouse && (p.raw || !takes_system_mouse(i))) continue;
        if (kind == Input::kPad) {
            SDL_Gamepad* g = player_pad(p);
            if (!g || SDL_GetGamepadID(g) != pad) continue;
        }
        for (int a = 0; a < 6; ++a)
            for (const Input& in : p.bind[a])
                if (in.kind == kind && in.code == code) {
                    if (!p.presses[a]++) p.press_ns[a] = when_ns;
                    break;
                }
    }
}

void handle_event(const SDL_Event& e) {
    switch (e.type) {
        case SDL_EVENT_MOUSE_MOTION:  // already in game coordinates (gpu::to_screen_coords)
            for (int i = 0; i < 2; ++i)
                if (takes_system_mouse(i)) {
                    g_p[i].x = std::clamp(e.motion.x, 0.0f, 640.0f);
                    g_p[i].y = std::clamp(e.motion.y, 0.0f, 480.0f);
                }
            break;
        case SDL_EVENT_MOUSE_BUTTON_DOWN:
        case SDL_EVENT_MOUSE_BUTTON_UP: {
            int bit = -1;
            switch (e.button.button) {
                case SDL_BUTTON_LEFT: bit = 0; break;
                case SDL_BUTTON_RIGHT: bit = 1; break;
                case SDL_BUTTON_MIDDLE: bit = 2; break;
                case SDL_BUTTON_X1: bit = 3; break;
                case SDL_BUTTON_X2: bit = 4; break;
                default: break;
            }
            if (bit < 0) break;
            if (e.type == SDL_EVENT_MOUSE_BUTTON_DOWN) count_press(Input::kMouse, bit, e.button.timestamp);
            for (int i = 0; i < 2; ++i) {
                if (!takes_system_mouse(i)) continue;
                Player& p = g_p[i];
                if (e.type == SDL_EVENT_MOUSE_BUTTON_DOWN) p.mouse_buttons |= 1u << bit;
                else p.mouse_buttons &= ~(1u << bit);
            }
            break;
        }
        case SDL_EVENT_KEY_DOWN:
            if (e.key.repeat) break;
            if (std::find(g_keys_down.begin(), g_keys_down.end(), e.key.key) == g_keys_down.end()) g_keys_down.push_back(e.key.key);
            count_press(Input::kKey, int(e.key.key), e.key.timestamp);
            break;
        case SDL_EVENT_GAMEPAD_BUTTON_DOWN:
            count_press(Input::kPad, int(e.gbutton.button), e.gbutton.timestamp, e.gbutton.which);
            break;
        case SDL_EVENT_GAMEPAD_AXIS_MOTION:  // analog triggers bound as buttons: count each pull
            if (e.gaxis.axis == SDL_GAMEPAD_AXIS_LEFT_TRIGGER || e.gaxis.axis == SDL_GAMEPAD_AXIS_RIGHT_TRIGGER) {
                static std::map<std::pair<SDL_JoystickID, int>, bool> pulled;
                bool& was = pulled[{e.gaxis.which, int(e.gaxis.axis)}];
                const bool now = e.gaxis.value > 16000;
                if (now && !was)
                    count_press(Input::kPad, e.gaxis.axis == SDL_GAMEPAD_AXIS_LEFT_TRIGGER ? 100 : 101, e.gaxis.timestamp, e.gaxis.which);
                was = now;
            }
            break;
        case SDL_EVENT_KEY_UP:
            std::erase(g_keys_down, e.key.key);
            break;
        case SDL_EVENT_GAMEPAD_ADDED:
            if (std::any_of(g_pads.begin(), g_pads.end(), [&](SDL_Gamepad* g) { return SDL_GetGamepadID(g) == e.gdevice.which; }))
                break;  // already open (connected before startup)
            if (SDL_Gamepad* g = SDL_OpenGamepad(e.gdevice.which)) {
                g_pads.push_back(g);
                std::fprintf(stderr, "[input] gamepad connected: %s (slot %zu)\n", SDL_GetGamepadName(g), g_pads.size());
            }
            break;
        case SDL_EVENT_GAMEPAD_REMOVED:
            for (size_t i = 0; i < g_pads.size(); ++i)
                if (SDL_GetGamepadID(g_pads[i]) == e.gdevice.which) {
                    SDL_CloseGamepad(g_pads[i]);
                    g_pads.erase(g_pads.begin() + long(i));
                    break;
                }
            break;
        case SDL_EVENT_WINDOW_FOCUS_LOST:
            g_keys_down.clear();
            for (Player& p : g_p) p.mouse_buttons = 0;
            break;
        default: break;
    }
}

void update() {
    if (io::scripted()) return;  // scripted test input drives the cabinet
    const Uint64 now = SDL_GetTicksNS();
    g_loop_gap_max_ns = std::max(g_loop_gap_max_ns, now - g_last_tick);  // (for the log)
    const float dt = std::min(0.1f, float(now - g_last_tick) * 1e-9f);
    g_last_tick = now;
#ifdef _WIN32
    // Separate devices: the absolute ones map through the game window, and the system cursor (which
    // every mouse on Windows moves) stays inside it so nothing clicks outside the game.
    if (g_win_uses_raw)
        if (SDL_Window* w = SDL_GetKeyboardFocus()) {
            if (!g_game_hwnd)
                g_game_hwnd = static_cast<HWND>(SDL_GetPointerProperty(SDL_GetWindowProperties(w), SDL_PROP_WINDOW_WIN32_HWND_POINTER, nullptr));
            if (!SDL_GetWindowMouseGrab(w)) SDL_SetWindowMouseGrab(w, true);
        }
#endif
    // Presses on raw devices (counted by their reader threads), taken once per device.
    int raw_presses[2][16] = {};
    for (int i = 0; i < 2; ++i)
        if (RawState* r = g_p[i].raw; r && !(i == 1 && g_p[0].raw == r))
            for (int b = 0; b < 16; ++b) raw_presses[i][b] = r->presses[b].exchange(0);
    for (int i = 0; i < 2; ++i) {
        Player& p = g_p[i];
        // Aim
        if (p.raw) {
            p.x = p.raw->x;
            p.y = p.raw->y;
        } else if (p.aim == "pad" || p.aim == "pad_left") {
            if (SDL_Gamepad* g = player_pad(p)) {
                const bool left = p.aim == "pad_left";
                auto axis = [&](SDL_GamepadAxis a) {
                    const float v = float(SDL_GetGamepadAxis(g, a)) / 32767.0f;
                    const float dz = 0.15f;
                    return std::fabs(v) < dz ? 0.0f : std::copysign((std::fabs(v) - dz) / (1.0f - dz), v);
                };
                static const float sens = [] {
                    const float v = float(std::atof(settings::get("pad_sensitivity").c_str()));
                    return v > 0.05f ? v : 1.0f;
                }();
                const float kSpeed = 700.0f * sens;  // game pixels per second at full tilt
                p.x = std::clamp(p.x + axis(left ? SDL_GAMEPAD_AXIS_LEFTX : SDL_GAMEPAD_AXIS_RIGHTX) * kSpeed * dt, 0.0f, 640.0f);
                p.y = std::clamp(p.y + axis(left ? SDL_GAMEPAD_AXIS_LEFTY : SDL_GAMEPAD_AXIS_RIGHTY) * kSpeed * dt, 0.0f, 480.0f);
            }
        }
        bool st[6];
        int pr[6];
        for (int a = 0; a < 6; ++a) {
            st[a] = false;
            pr[a] = p.presses[a];
            p.presses[a] = 0;
            for (const Input& in : p.bind[a]) {
                st[a] = st[a] || held(p, in);
                if (in.kind == Input::kMouse && p.raw && in.code < 16) pr[a] += raw_presses[i][in.code];
            }
        }
        // Reload = the gun's screen sensor reading "off screen"; the game reloads from that
        // alone. No trigger with it: an automatic weapon (G36C) keeps a trigger pull made off
        // screen and fires that round once the gun points back at the screen.
        io::set_player(i, p.x, p.y, st[1]);
        const long long t = (long long)now;
        io::set_button(i, io::kTrigger, pr[0], st[0], (long long)p.press_ns[0], t);
        io::set_button(i, io::kPedal, pr[2], st[2], (long long)p.press_ns[2], t);
        io::set_button(i, io::kChange, pr[3], st[3], (long long)p.press_ns[3], t);
        io::set_button(i, io::kStart, pr[4], st[4], (long long)p.press_ns[4], t);
        io::set_button(i, io::kCoin, pr[5], st[5], (long long)p.press_ns[5], t);
    }
    Player cab;  // test / service: keys, gamepad 1, the system mouse's buttons
    cab.pad = 1;
    cab.mouse_buttons = takes_system_mouse(0) ? g_p[0].mouse_buttons : 0;
    bool test = false, service = false;
    for (const Input& in : g_test_bind) test = test || held(cab, in);
    for (const Input& in : g_service_bind) service = service || held(cab, in);
    io::set_test_service(test, service);
}

double take_loop_gap_ms() {
    const double ms = double(g_loop_gap_max_ns) / 1e6;
    g_loop_gap_max_ns = 0;
    return ms;
}

bool crosshair(int p, float& x, float& y) {
    const Player& pl = g_p[p & 1];
    x = pl.x;
    y = pl.y;
    if (pl.aim == "none") return false;
    if (pad_aim(pl)) return player_pad(pl) != nullptr || takes_system_mouse(p & 1);
    return true;  // mouse, raw mouse / lightgun / Wiimote
}


std::vector<Device> aim_devices() {
    std::vector<Device> d = {{"mouse", "System mouse"}, {"pad", "Gamepad (right stick)"},
                             {"pad_left", "Gamepad (left stick)"}, {"none", "None"}};
    for (const std::string& n : raw_device_names()) d.push_back({"raw:" + n, n + " (raw)"});
#ifdef __linux__
    for (const WiiDev& w : scan_wii()) d.push_back({"raw:" + w.name, w.name + " (IR pointer, sensor bar)"});
#endif
#ifdef _WIN32
    for (const WinWii& w : scan_wii_win()) d.push_back({"raw:" + w.name, w.name + " (IR pointer, sensor bar; experimental)"});
#endif
    return d;
}

std::vector<std::string> raw_device_names(bool* access_ok) {
    std::vector<std::string> out;
#ifdef __linux__
    for (const EvDev& d : scan_evdev(access_ok)) out.push_back(d.name);
#elif defined(_WIN32)
    for (const WinMouse& m : scan_rawinput()) out.push_back(m.name);
    if (access_ok) *access_ok = true;
#else
    if (access_ok) *access_ok = true;
#endif
    return out;
}

}  // namespace vc3::io::input
