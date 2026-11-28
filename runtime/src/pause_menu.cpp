#include "pause_menu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "audio/apu.h"
#include "gpu/gl_backend.h"
#include "gpu/nv2a.h"
#include "imgui.h"
#include "imgui_impl_sdlrenderer3.h"
#include "io/input.h"
#include "io/jvs.h"
#include "pause.h"
#include "settings.h"

namespace vc3::pause_menu {

namespace {

ImGuiContext* g_ctx = nullptr;
SDL_Surface* g_surf = nullptr;
SDL_Renderer* g_ren = nullptr;
int g_w = 0, g_h = 0, g_x = 0, g_y = 0;  // the panel: size and place in the window (pixels)
float g_scale = 0;
enum class Page { kMain, kOptions } g_page = Page::kMain;
bool g_focus_first = false, g_exit = false, g_saving = true;
std::vector<uint32_t> g_px;
uint64_t g_version = 0;
gpu::dev::OverlayImage g_img{};
bool g_has_img = false;
Uint64 g_last_ns = 0;
float g_content_h[2] = {0, 0};  // each page's height as last laid out (the panel fits it)

// The live settings (key, as in settings.cpp).
const char* const kOptionKeys[] = {"volume", "fullscreen", "vsync", "crosshair", "crosshair_size", "wide_aim", "wide_hud"};

void destroy() {
    if (g_ctx) {
        ImGuiContext* prev = ImGui::GetCurrentContext();
        ImGui::SetCurrentContext(g_ctx);
        ImGui_ImplSDLRenderer3_Shutdown();
        ImGui::DestroyContext(g_ctx);
        ImGui::SetCurrentContext(prev == g_ctx ? nullptr : prev);
        g_ctx = nullptr;
    }
    if (g_ren) SDL_DestroyRenderer(g_ren);
    if (g_surf) SDL_DestroySurface(g_surf);
    g_ren = nullptr;
    g_surf = nullptr;
}

void style(float scale) {  // the launcher's look
    ImGuiStyle& s = ImGui::GetStyle();
    ImGui::StyleColorsDark();
    s.WindowRounding = 0;  // (square: the picture is drawn without blending, see crosshair_image.h)
    s.FrameRounding = 6;
    s.GrabRounding = 6;
    s.FramePadding = ImVec2(12, 8);
    s.ItemSpacing = ImVec2(12, 10);
    s.WindowPadding = ImVec2(28, 24);
    s.WindowBorderSize = 2;
    ImVec4* c = s.Colors;
    c[ImGuiCol_WindowBg] = ImVec4(0.06f, 0.07f, 0.10f, 1.0f);
    c[ImGuiCol_PopupBg] = ImVec4(0.08f, 0.09f, 0.13f, 1.0f);
    c[ImGuiCol_Border] = ImVec4(0.75f, 0.10f, 0.12f, 1.0f);
    c[ImGuiCol_FrameBg] = ImVec4(0.13f, 0.15f, 0.21f, 1.0f);
    c[ImGuiCol_FrameBgHovered] = ImVec4(0.19f, 0.22f, 0.31f, 1.0f);
    c[ImGuiCol_Header] = ImVec4(0.75f, 0.10f, 0.12f, 0.55f);
    c[ImGuiCol_HeaderHovered] = ImVec4(0.85f, 0.15f, 0.17f, 0.75f);
    c[ImGuiCol_Button] = ImVec4(0.16f, 0.19f, 0.27f, 1.0f);
    c[ImGuiCol_ButtonHovered] = ImVec4(0.24f, 0.28f, 0.40f, 1.0f);
    c[ImGuiCol_NavCursor] = ImVec4(1.0f, 0.75f, 0.20f, 1.0f);
    s.ScaleAllSizes(scale);
}

void ensure(int w, int h, float scale) {
    if (g_ctx && w == g_w && h == g_h && scale == g_scale) return;
    if (std::getenv("VC3_TRACE_MENU")) std::fprintf(stderr, "[menu] new context %dx%d (was %dx%d)\n", w, h, g_w, g_h);
    destroy();
    g_surf = SDL_CreateSurface(w, h, SDL_PIXELFORMAT_ARGB8888);
    g_ren = g_surf ? SDL_CreateSoftwareRenderer(g_surf) : nullptr;
    if (!g_ren) return;
    g_ctx = ImGui::CreateContext();
    ImGui::SetCurrentContext(g_ctx);
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard | ImGuiConfigFlags_NavEnableGamepad;
    io.BackendFlags |= ImGuiBackendFlags_HasGamepad;
    io.FontGlobalScale = 1.25f * scale;
    style(scale);
    ImGui_ImplSDLRenderer3_Init(g_ren);
    g_w = w;
    g_h = h;
    g_scale = scale;
    g_focus_first = true;
}

void open() {
    pause::set(true);
    g_page = Page::kMain;
    g_focus_first = true;
    SDL_ShowCursor();
}
void close() {
    pause::set(false);
    io::input::release_held();
    g_has_img = false;
    SDL_HideCursor();
}
// Back: from Options to the menu, from the menu to the game.
void back() {
    if (g_page == Page::kOptions) {
        g_page = Page::kMain;
        g_focus_first = true;
    } else {
        close();
    }
}

void apply(const char* key, const std::string& v) {
    if (!std::strcmp(key, "volume")) audio::apu::set_volume(float(std::atof(v.c_str())));
    else if (!std::strcmp(key, "fullscreen")) {
        if ((v == "1") != gpu::fullscreen_on()) gpu::toggle_fullscreen();
    } else if (!std::strcmp(key, "vsync")) gpu::set_vsync(v != "0");
    else if (!std::strcmp(key, "crosshair") || !std::strcmp(key, "crosshair_size"))
        gpu::set_crosshair(settings::get("crosshair") != "off", float(std::atof(settings::get("crosshair_size").c_str())));
    else if (!std::strcmp(key, "wide_aim")) io::set_wide_aim(v != "0");
    else if (!std::strcmp(key, "wide_hud")) gpu::pgraph::glb::set_wide_hud(v != "0");
}

const settings::Option* find_option(const char* key) {
    for (const auto& o : settings::options())
        if (!std::strcmp(o.key, key)) return &o;
    return nullptr;
}

ImGuiKey nav_key(SDL_Keycode k) {
    switch (k) {
        case SDLK_UP: return ImGuiKey_UpArrow;
        case SDLK_DOWN: return ImGuiKey_DownArrow;
        case SDLK_LEFT: return ImGuiKey_LeftArrow;
        case SDLK_RIGHT: return ImGuiKey_RightArrow;
        case SDLK_RETURN: return ImGuiKey_Enter;
        case SDLK_KP_ENTER: return ImGuiKey_KeypadEnter;
        case SDLK_SPACE: return ImGuiKey_Space;
        case SDLK_TAB: return ImGuiKey_Tab;
        default: return ImGuiKey_None;
    }
}
ImGuiKey pad_key(Uint8 b) {
    switch (b) {
        case SDL_GAMEPAD_BUTTON_DPAD_UP: return ImGuiKey_GamepadDpadUp;
        case SDL_GAMEPAD_BUTTON_DPAD_DOWN: return ImGuiKey_GamepadDpadDown;
        case SDL_GAMEPAD_BUTTON_DPAD_LEFT: return ImGuiKey_GamepadDpadLeft;
        case SDL_GAMEPAD_BUTTON_DPAD_RIGHT: return ImGuiKey_GamepadDpadRight;
        case SDL_GAMEPAD_BUTTON_SOUTH: return ImGuiKey_GamepadFaceDown;   // A: select
        case SDL_GAMEPAD_BUTTON_EAST: return ImGuiKey_GamepadFaceRight;   // B: back
        default: return ImGuiKey_None;
    }
}

}  // namespace

void set_saving(bool on) { g_saving = on; }
void test_open(bool want, bool options) {
    if (want && !pause::active()) open();
    if (want && options) g_page = Page::kOptions;
    if (!want && pause::active()) close();
}
bool exit_requested() { return g_exit; }
const gpu::dev::OverlayImage* image() { return pause::active() && g_has_img ? &g_img : nullptr; }

bool handle_event(SDL_Window* win, const SDL_Event& e) {
    const bool toggle = io::input::is_pause_event(e);
    if (!pause::active()) {
        if (toggle) open();
        return toggle;
    }
    // While paused: quitting, window events and the fullscreen keys still work as usual.
    if (e.type == SDL_EVENT_QUIT || (e.type >= SDL_EVENT_WINDOW_FIRST && e.type <= SDL_EVENT_WINDOW_LAST)) return false;
    if (e.type == SDL_EVENT_KEY_DOWN && (e.key.key == SDLK_F11 || (e.key.key == SDLK_RETURN && (e.key.mod & SDL_KMOD_ALT))))
        return false;
    if (toggle) {
        back();
        return true;
    }
    if (!g_ctx) return true;
    ImGuiContext* prev = ImGui::GetCurrentContext();
    ImGui::SetCurrentContext(g_ctx);
    ImGuiIO& io = ImGui::GetIO();
    const float d = SDL_GetWindowPixelDensity(win);
    switch (e.type) {
        case SDL_EVENT_MOUSE_MOTION:
            io.AddMousePosEvent(e.motion.x * d - float(g_x), e.motion.y * d - float(g_y));
            break;
        case SDL_EVENT_MOUSE_BUTTON_DOWN:
        case SDL_EVENT_MOUSE_BUTTON_UP:
            io.AddMousePosEvent(e.button.x * d - float(g_x), e.button.y * d - float(g_y));
            if (e.button.button == SDL_BUTTON_LEFT) io.AddMouseButtonEvent(0, e.type == SDL_EVENT_MOUSE_BUTTON_DOWN);
            if (e.button.button == SDL_BUTTON_RIGHT && e.type == SDL_EVENT_MOUSE_BUTTON_DOWN &&
                !ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId))
                back();  // right click (reload): back, like the gamepad's B
            break;
        case SDL_EVENT_MOUSE_WHEEL: io.AddMouseWheelEvent(e.wheel.x, e.wheel.y); break;
        case SDL_EVENT_KEY_DOWN:
        case SDL_EVENT_KEY_UP:
            if (const ImGuiKey k = nav_key(e.key.key); k != ImGuiKey_None) io.AddKeyEvent(k, e.type == SDL_EVENT_KEY_DOWN);
            break;
        case SDL_EVENT_GAMEPAD_BUTTON_DOWN:
        case SDL_EVENT_GAMEPAD_BUTTON_UP: {
            const bool down = e.type == SDL_EVENT_GAMEPAD_BUTTON_DOWN;
            if (down && e.gbutton.button == SDL_GAMEPAD_BUTTON_EAST && !ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId)) {
                back();
                break;
            }
            if (const ImGuiKey k = pad_key(e.gbutton.button); k != ImGuiKey_None) io.AddKeyEvent(k, down);
            break;
        }
        case SDL_EVENT_GAMEPAD_AXIS_MOTION: {
            const float v = float(e.gaxis.value) / 32767.0f;
            if (e.gaxis.axis == SDL_GAMEPAD_AXIS_LEFTY) {
                io.AddKeyAnalogEvent(ImGuiKey_GamepadLStickUp, v < -0.5f, v < 0 ? -v : 0);
                io.AddKeyAnalogEvent(ImGuiKey_GamepadLStickDown, v > 0.5f, v > 0 ? v : 0);
            } else if (e.gaxis.axis == SDL_GAMEPAD_AXIS_LEFTX) {
                io.AddKeyAnalogEvent(ImGuiKey_GamepadLStickLeft, v < -0.5f, v < 0 ? -v : 0);
                io.AddKeyAnalogEvent(ImGuiKey_GamepadLStickRight, v > 0.5f, v > 0 ? v : 0);
            }
            break;
        }
        default: break;
    }
    ImGui::SetCurrentContext(prev);
    return true;
}

void frame(SDL_Window* win) {
    if (!pause::active()) return;
    int ww = 0, wh = 0;
    SDL_GetWindowSizeInPixels(win, &ww, &wh);
    if (ww <= 0 || wh <= 0) return;
    const float scale = std::clamp(float(wh) / 720.0f, 0.8f, 3.0f);
    const int page = g_page == Page::kMain ? 0 : 1;
    const float want_h = g_content_h[page] > 0 ? g_content_h[page] : (page ? 560.0f : 330.0f) * scale;
    const int pw = std::min(ww - 8, int(560 * scale)), ph = std::min(wh - 8, int(want_h + 0.5f));
    ensure(pw, ph, scale);
    if (!g_ctx) return;
    g_x = (ww - pw) / 2;
    g_y = (wh - ph) / 2;
    ImGuiContext* prev = ImGui::GetCurrentContext();
    ImGui::SetCurrentContext(g_ctx);
    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2(float(pw), float(ph));
    const Uint64 now = SDL_GetTicksNS();
    io.DeltaTime = g_last_ns ? std::clamp(float(now - g_last_ns) / 1e9f, 0.001f, 0.1f) : 1.0f / 60.0f;
    g_last_ns = now;
    ImGui_ImplSDLRenderer3_NewFrame();
    ImGui::NewFrame();
    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(io.DisplaySize);
    ImGui::Begin("pause", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings);
    ImGui::SetWindowFontScale(1.6f);
    ImGui::TextColored(ImVec4(0.95f, 0.25f, 0.25f, 1.0f), g_page == Page::kMain ? "PAUSED" : "OPTIONS");
    ImGui::SetWindowFontScale(1.0f);
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    const float bw = ImGui::GetContentRegionAvail().x;
    if (g_page == Page::kMain) {
        if (g_focus_first) ImGui::SetKeyboardFocusHere(), ImGui::SetNavCursorVisible(true), g_focus_first = false;
        if (ImGui::Button("Resume", ImVec2(bw, 0))) close();
        if (ImGui::Button("Options", ImVec2(bw, 0))) {
            g_page = Page::kOptions;
            g_focus_first = true;
        }
        ImGui::Spacing();
        if (ImGui::Button("Exit to desktop", ImVec2(bw, 0))) g_exit = true;
    } else {
        const float label_w = bw * 0.48f;
        bool first = g_focus_first;
        g_focus_first = false;
        for (const char* key : kOptionKeys) {
            const settings::Option* o = find_option(key);
            if (!o) continue;
            std::string cur = settings::get(key);
            if (!std::strcmp(key, "fullscreen")) cur = gpu::fullscreen_on() ? "1" : "0";  // (F11 may have switched)
            const char* cur_label = cur.c_str();
            for (const auto& c : o->choices)
                if (cur == c.value) cur_label = c.label;
            const bool locked = settings::overridden(key);
            ImGui::AlignTextToFramePadding();
            ImGui::TextUnformatted(o->label);
            ImGui::SameLine(label_w);
            ImGui::SetNextItemWidth(-1);
            ImGui::BeginDisabled(locked);
            if (first) ImGui::SetKeyboardFocusHere(), ImGui::SetNavCursorVisible(true), first = false;  // (keys work at once)
            if (ImGui::BeginCombo((std::string("##") + key).c_str(), cur_label)) {
                for (const auto& c : o->choices) {
                    const bool sel = cur == c.value;
                    if (ImGui::Selectable(c.label, sel) && !sel) {
                        settings::set(key, c.value);
                        if (g_saving) settings::save();
                        apply(key, c.value);
                    }
                    if (sel) ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }
            ImGui::EndDisabled();
        }
        ImGui::Spacing();
        if (ImGui::Button("Back", ImVec2(bw, 0))) back();
    }
    // The page's height, for the panel's size from the next frame on. A frame drawn at another size isn't
    // shown (the panel jumped when a page opened: first a default size, then the fitted one).
    g_content_h[page] = ImGui::GetCursorPosY() - ImGui::GetStyle().ItemSpacing.y + ImGui::GetStyle().WindowPadding.y;
    const bool fitted = std::abs(int(g_content_h[page] + 0.5f) - ph) <= 1 || ph >= wh - 8;
    ImGui::End();
    if (std::getenv("VC3_TRACE_MENU")) {
        static bool last_popup = false;
        const bool popup = ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId);
        if (popup != last_popup || ImGui::IsKeyPressed(ImGuiKey_Space) || ImGui::IsKeyPressed(ImGuiKey_DownArrow))
            std::fprintf(stderr, "[menu] frame %d: popup %d space %d down %d\n", ImGui::GetFrameCount(), popup,
                         ImGui::IsKeyPressed(ImGuiKey_Space), ImGui::IsKeyPressed(ImGuiKey_DownArrow));
        last_popup = popup;
    }
    ImGui::Render();
    SDL_SetRenderDrawColor(g_ren, 0, 0, 0, 0);
    SDL_RenderClear(g_ren);
    ImGui_ImplSDLRenderer3_RenderDrawData(ImGui::GetDrawData(), g_ren);
    SDL_FlushRenderer(g_ren);
    ImGui::SetCurrentContext(prev);
    // To the overlay picture: 0xAARRGGBB, alpha either drawn or not (no blending there).
    if (!pause::active()) return;  // (Resume was chosen this frame)
    if (!fitted) {
        g_has_img = g_has_img && g_img.h == ph;  // nothing new to show yet (the next frame is fitted)
        return;
    }
    g_px.resize(size_t(pw) * size_t(ph));
    if (SDL_LockSurface(g_surf)) {
        for (int y = 0; y < ph; ++y) {
            const auto* row = reinterpret_cast<const uint32_t*>(static_cast<const uint8_t*>(g_surf->pixels) + size_t(y) * size_t(g_surf->pitch));
            for (int x = 0; x < pw; ++x) {
                const uint32_t c = row[x];
                g_px[size_t(y) * size_t(pw) + size_t(x)] = (c >> 24) >= 128 ? (c | 0xFF000000u) : 0u;
            }
        }
        SDL_UnlockSurface(g_surf);
    }
    g_img = gpu::dev::OverlayImage{g_x, g_y, pw, ph, g_px.data(), 2, ++g_version};
    g_has_img = true;
}

}  // namespace vc3::pause_menu
