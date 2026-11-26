#include "platform.h"
#include "launcher.h"

#include <SDL3/SDL.h>
#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <imgui_impl_sdlrenderer3.h>

#include <algorithm>
#include <mutex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "install.h"
#include "io/input.h"
#include "settings.h"

namespace vc3::launcher {
namespace {

// Vulkan is offered only when the build includes it.
bool vulkan_built() {
#ifdef VC3_HAVE_VULKAN
    return true;
#else
    return false;
#endif
}

void style(float scale) {
    ImGuiStyle& s = ImGui::GetStyle();
    ImGui::StyleColorsDark();
    s.WindowRounding = 0;
    s.FrameRounding = 6;
    s.GrabRounding = 6;
    s.FramePadding = ImVec2(12, 8);
    s.ItemSpacing = ImVec2(12, 10);
    s.WindowPadding = ImVec2(28, 24);
    ImVec4* c = s.Colors;
    c[ImGuiCol_WindowBg] = ImVec4(0.06f, 0.07f, 0.10f, 1.0f);
    c[ImGuiCol_FrameBg] = ImVec4(0.13f, 0.15f, 0.21f, 1.0f);
    c[ImGuiCol_FrameBgHovered] = ImVec4(0.19f, 0.22f, 0.31f, 1.0f);
    c[ImGuiCol_Header] = ImVec4(0.75f, 0.10f, 0.12f, 0.55f);
    c[ImGuiCol_HeaderHovered] = ImVec4(0.85f, 0.15f, 0.17f, 0.75f);
    c[ImGuiCol_Button] = ImVec4(0.16f, 0.19f, 0.27f, 1.0f);
    c[ImGuiCol_ButtonHovered] = ImVec4(0.24f, 0.28f, 0.40f, 1.0f);
    c[ImGuiCol_NavCursor] = ImVec4(1.0f, 0.75f, 0.20f, 1.0f);
    s.ScaleAllSizes(scale);
}

// ---------------------------------------------------------------- controls page
// "Set" / "+": the next key, mouse button or gamepad button pressed becomes the binding.
struct Capture {
    int player = -1, action = -1;
    bool append = false;
    Uint64 since = 0;  // ignore the click that started it
} g_cap;
std::vector<SDL_Gamepad*> g_pads;

std::string pretty(const std::string& tok) {
    const size_t c = tok.find(':');
    if (c == std::string::npos) return tok;
    const std::string k = tok.substr(0, c), n = tok.substr(c + 1);
    if (k == "key") return n;
    if (k == "mouse") return "Mouse " + n;
    if (k == "pad") return "Pad " + n;
    return tok;
}
std::string pretty_list(const std::string& s) {
    std::string out;
    size_t i = 0;
    while (i < s.size()) {
        size_t j = s.find(',', i);
        if (j == std::string::npos) j = s.size();
        if (!out.empty()) out += ", ";
        out += pretty(s.substr(i, j - i));
        i = j + 1;
    }
    return out.empty() ? "(none)" : out;
}
std::string key_of(int p, const char* what) { return "p" + std::to_string(p + 1) + "_" + what; }

// An input event while capturing -> the binding text (empty: not a bindable input).
std::string captured(const SDL_Event& e) {
    switch (e.type) {
        case SDL_EVENT_KEY_DOWN: return std::string("key:") + SDL_GetKeyName(e.key.key);
        case SDL_EVENT_MOUSE_BUTTON_DOWN:
            switch (e.button.button) {
                case SDL_BUTTON_LEFT: return "mouse:left";
                case SDL_BUTTON_RIGHT: return "mouse:right";
                case SDL_BUTTON_MIDDLE: return "mouse:middle";
                case SDL_BUTTON_X1: return "mouse:x1";
                case SDL_BUTTON_X2: return "mouse:x2";
                default: return "";
            }
        case SDL_EVENT_GAMEPAD_BUTTON_DOWN:
            return std::string("pad:") + SDL_GetGamepadStringForButton(SDL_GamepadButton(e.gbutton.button));
        case SDL_EVENT_GAMEPAD_AXIS_MOTION:
            if (e.gaxis.value > 20000 && e.gaxis.axis == SDL_GAMEPAD_AXIS_LEFT_TRIGGER) return "pad:lefttrigger";
            if (e.gaxis.value > 20000 && e.gaxis.axis == SDL_GAMEPAD_AXIS_RIGHT_TRIGGER) return "pad:righttrigger";
            return "";
        default: return "";
    }
}

void controls_page(float label_w) {
    // Controller API (Windows), gamepad aim speed, ...
    for (const auto& o : settings::options()) {
        if (std::strcmp(o.section, "Controls") || o.choices.empty()) continue;
        ImGui::PushID(o.key);
        const std::string cur = settings::get(o.key);
        const char* lab = cur.c_str();
        for (const auto& c : o.choices)
            if (cur == c.value) lab = c.label;
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(o.label);
        ImGui::SameLine(label_w);
        ImGui::SetNextItemWidth(-1);
        if (ImGui::BeginCombo("##opt", lab)) {
            for (const auto& c : o.choices)
                if (ImGui::Selectable(c.label, cur == c.value)) settings::set(o.key, c.value);
            ImGui::EndCombo();
        }
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) ImGui::SetTooltip("%s", o.help);
        ImGui::PopID();
    }
    static std::vector<io::input::Device> devices;
    static bool raw_ok = true;
    static Uint64 scanned = 0;
    if (!scanned || SDL_GetTicks() - scanned > 3000) {  // devices come and go
        devices = io::input::aim_devices();
        io::input::raw_device_names(&raw_ok);
        scanned = SDL_GetTicks();
    }
    for (int p = 0; p < 2; ++p) {
        ImGui::Spacing();
        ImGui::SeparatorText(p == 0 ? "Player 1" : "Player 2");
        ImGui::PushID(p);
        // Aim device
        const std::string aim = settings::get(key_of(p, "aim").c_str());
        std::string aim_label = aim;
        for (const auto& d : devices)
            if (d.id == aim) aim_label = d.label;
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted("Aim with");
        ImGui::SameLine(label_w);
        ImGui::SetNextItemWidth(-1);
        if (ImGui::BeginCombo("##aim", aim_label.c_str())) {
            for (const auto& d : devices)
                if (ImGui::Selectable(d.label.c_str(), d.id == aim)) settings::set(key_of(p, "aim").c_str(), d.id);
            ImGui::EndCombo();
        }
        // Gamepad slot
        const std::string pad = settings::get(key_of(p, "pad").c_str());
        auto pad_label = [](int slot) {
            std::string l = "Gamepad " + std::to_string(slot);
            if (slot >= 1 && slot <= int(g_pads.size())) l += std::string(" - ") + SDL_GetGamepadName(g_pads[size_t(slot - 1)]);
            else l += " (not connected)";
            return l;
        };
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted("Gamepad");
        ImGui::SameLine(label_w);
        ImGui::SetNextItemWidth(-1);
        if (ImGui::BeginCombo("##pad", pad_label(std::atoi(pad.c_str())).c_str())) {
            for (int s = 1; s <= 4; ++s)
                if (ImGui::Selectable(pad_label(s).c_str(), std::atoi(pad.c_str()) == s))
                    settings::set(key_of(p, "pad").c_str(), std::to_string(s));
            ImGui::EndCombo();
        }
        // Buttons
        for (int a = 0; a < 6; ++a) {
            ImGui::PushID(a);
            const std::string k = key_of(p, io::input::kActions[a]);
            const bool cap = g_cap.player == p && g_cap.action == a;
            ImGui::AlignTextToFramePadding();
            ImGui::TextUnformatted(io::input::kActionLabels[a]);
            ImGui::SameLine(label_w);
            const float bw = ImGui::CalcTextSize("Clear").x + ImGui::GetStyle().FramePadding.x * 2;
            const float tw = ImGui::GetContentRegionAvail().x - 3 * (bw + ImGui::GetStyle().ItemSpacing.x);
            ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 1.0f);
            ImGui::Button(cap ? "Press a key, mouse or gamepad button... (Esc cancels)" : pretty_list(settings::get(k.c_str())).c_str(),
                          ImVec2(tw, 0));
            ImGui::PopStyleVar();
            if (!cap && ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) ImGui::SetTooltip("%s", pretty_list(settings::get(k.c_str())).c_str());
            ImGui::SameLine();
            if (ImGui::Button("Set", ImVec2(bw, 0))) g_cap = {p, a, false, SDL_GetTicks()};
            ImGui::SameLine();
            if (ImGui::Button("+", ImVec2(bw, 0))) g_cap = {p, a, true, SDL_GetTicks()};
            ImGui::SameLine();
            if (ImGui::Button("Clear", ImVec2(bw, 0))) settings::set(k.c_str(), "");
            ImGui::PopID();
        }
        if (ImGui::Button("Reset to defaults")) {
            for (int a = 0; a < 6; ++a) settings::set(key_of(p, io::input::kActions[a]).c_str(), io::input::default_binding(p, io::input::kActions[a]));
            settings::set(key_of(p, "aim").c_str(), p == 0 ? "mouse" : "pad");
            settings::set(key_of(p, "pad").c_str(), p == 0 ? "1" : "2");
        }
        ImGui::PopID();
    }
    // Cabinet buttons
    ImGui::Spacing();
    ImGui::SeparatorText("Cabinet");
    for (int c = 0; c < 2; ++c) {
        const char* k = c == 0 ? "test" : "service";
        ImGui::PushID(k);
        const bool cap = g_cap.player == 2 && g_cap.action == c;
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(c == 0 ? "Test (operator menu)" : "Service");
        ImGui::SameLine(label_w);
        const float bw = ImGui::CalcTextSize("Clear").x + ImGui::GetStyle().FramePadding.x * 2;
        const float tw = ImGui::GetContentRegionAvail().x - 3 * (bw + ImGui::GetStyle().ItemSpacing.x);
        ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 1.0f);
        ImGui::Button(cap ? "Press a key, mouse or gamepad button..." : pretty_list(settings::get(k)).c_str(), ImVec2(tw, 0));
        ImGui::PopStyleVar();
        ImGui::SameLine();
        if (ImGui::Button("Set", ImVec2(bw, 0))) g_cap = {2, c, false, SDL_GetTicks()};
        ImGui::SameLine();
        if (ImGui::Button("+", ImVec2(bw, 0))) g_cap = {2, c, true, SDL_GetTicks()};
        ImGui::SameLine();
        if (ImGui::Button("Clear", ImVec2(bw, 0))) settings::set(k, "");
        ImGui::PopID();
    }
    ImGui::Spacing();
    ImGui::TextWrapped("Mouse buttons are the buttons of the player's aim device (the system mouse, or the chosen "
                       "mouse / lightgun). Pad buttons are the player's gamepad. F2 / F3: test / service.");
    if (!raw_ok)
        ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.3f, 1.0f),
                           "Individual mice and lightguns can't be read: add yourself to the 'input' group\n"
                           "(sudo usermod -aG input $USER), then log out and back in.");
}

// ---------------------------------------------------------------- game files page
std::mutex g_pick_m;
std::string g_picked[3];  // set by the file dialog callbacks (any thread): zip, chd, install folder
install::Files g_files;
bool g_files_init = false;

void SDLCALL on_pick(void* which, const char* const* list, int) {
    if (!list || !list[0]) return;
    std::lock_guard lk(g_pick_m);
    g_picked[reinterpret_cast<intptr_t>(which)] = list[0];
}

void game_files_page(SDL_Window* win, float label_w) {
    if (!g_files_init) {
        g_files = install::find();
        g_files_init = true;
    }
    {
        std::lock_guard lk(g_pick_m);
        if (!g_picked[0].empty()) g_files.pic = std::move(g_picked[0]), g_picked[0].clear();
        if (!g_picked[1].empty()) g_files.chd = std::move(g_picked[1]), g_picked[1].clear();
        if (!g_picked[2].empty()) {
            settings::set("install_dir", g_picked[2]);
            install::use_location(install::location());
            g_picked[2].clear();
        }
    }
    install::Progress& pr = install::progress();
    std::string why;
    const bool ok = pr.state != 1 && install::installed(&why);
    ImGui::Spacing();
    if (ok) ImGui::TextColored(ImVec4(0.4f, 0.9f, 0.4f, 1.0f), "Game files installed and verified.");
    else ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.3f, 1.0f), "%s.", why.empty() ? "Installing" : why.c_str());
    ImGui::Spacing();
    ImGui::TextWrapped("Virtua Cop 3 needs your own dump of the game (MAME set): the vcop3 zip with the "
                       "security chip data (317-0354-com.data) and the GD-ROM image gdx-0003b.chd. They are "
                       "checked by SHA-1, then the game is decrypted and unpacked (about 500 MB).");
    ImGui::Spacing();
    ImGui::TextWrapped("Place them in the roms folder next to the program (also as roms/vcop3/gdx-0003b.chd), "
                       "or pick them below:");
    ImGui::TextDisabled("%s", install::roms_dir().c_str());
    ImGui::SameLine();
    if (ImGui::SmallButton("Open folder")) SDL_OpenURL(("file://" + install::roms_dir()).c_str());
    ImGui::SameLine();
    if (ImGui::SmallButton("Look again")) g_files = install::find();
    ImGui::Spacing();
    static const SDL_DialogFileFilter kZip[] = {{"vcop3 zip or PIC data", "zip;data"}, {"All files", "*"}};
    static const SDL_DialogFileFilter kChd[] = {{"GD-ROM image (CHD)", "chd"}, {"All files", "*"}};
    for (int i = 0; i < 2; ++i) {
        ImGui::PushID(i);
        std::string& path = i == 0 ? g_files.pic : g_files.chd;
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(i == 0 ? "vcop3.zip" : "gdx-0003b.chd");
        ImGui::SameLine(label_w);
        const float bw = ImGui::CalcTextSize("Browse...").x + ImGui::GetStyle().FramePadding.x * 2;
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - bw - ImGui::GetStyle().ItemSpacing.x);
        char buf[1024];
        std::snprintf(buf, sizeof buf, "%s", path.c_str());
        if (ImGui::InputText("##path", buf, sizeof buf)) path = buf;
        ImGui::SameLine();
        if (ImGui::Button("Browse...", ImVec2(bw, 0)))
            SDL_ShowOpenFileDialog(on_pick, reinterpret_cast<void*>(intptr_t(i)), win, i == 0 ? kZip : kChd, 2,
                                   install::roms_dir().c_str(), false);
        ImGui::PopID();
    }
    // Install location
    {
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted("Install to");
        ImGui::SameLine(label_w);
        const float bw = ImGui::CalcTextSize("Browse...").x + ImGui::GetStyle().FramePadding.x * 2;
        const float dw = ImGui::CalcTextSize("Default").x + ImGui::GetStyle().FramePadding.x * 2;
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - bw - dw - 2 * ImGui::GetStyle().ItemSpacing.x);
        char buf[1024];
        std::snprintf(buf, sizeof buf, "%s", install::location().c_str());
        if (ImGui::InputText("##loc", buf, sizeof buf, ImGuiInputTextFlags_EnterReturnsTrue)) {
            settings::set("install_dir", buf);
            install::use_location(install::location());
        }
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort))
            ImGui::SetTooltip("Where the game files (about 500 MB), save data and shader caches go.");
        ImGui::SameLine();
        if (ImGui::Button("Browse...##loc", ImVec2(bw, 0)) && pr.state != 1)
            SDL_ShowOpenFolderDialog(on_pick, reinterpret_cast<void*>(intptr_t(2)), win, install::location().c_str(), false);
        ImGui::SameLine();
        if (ImGui::Button("Default", ImVec2(dw, 0)) && pr.state != 1) {
            settings::set("install_dir", "");
            install::use_location(install::location());
        }
    }
    ImGui::Spacing();
    const bool busy = pr.state == 1;
    ImGui::BeginDisabled(busy || g_files.pic.empty() || g_files.chd.empty());
    if (ImGui::Button(ok ? "Re-install" : "Install", ImVec2(ImGui::CalcTextSize("Re-install").x * 2, 0))) {
        settings::save();  // keep the chosen install folder even if the launcher is closed afterwards
        install::start(g_files);
    }
    ImGui::EndDisabled();
    if (pr.state != 0) {
        std::string msg;
        {
            std::lock_guard lk(pr.m);
            msg = pr.message;
        }
        ImGui::SameLine();
        ImGui::ProgressBar(pr.fraction, ImVec2(-1, 0), pr.state == 1 ? nullptr : (pr.state == 2 ? "Done" : "Failed"));
        if (pr.state == 3) ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "%s", msg.c_str());
        else ImGui::TextDisabled("%s", msg.c_str());
    }
}

}  // namespace

bool run() {
    settings::load();
    const float scale = std::max(1.0f, SDL_GetDisplayContentScale(SDL_GetPrimaryDisplay()));
    SDL_Window* win = SDL_CreateWindow("Virtua Cop 3 (recomp) - Launcher", int(960 * scale), int(820 * scale),
                                       SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY);
    if (!win) {
        std::fprintf(stderr, "[launcher] window: %s\n", SDL_GetError());
        return true;  // no launcher: start with the saved settings
    }
    SDL_Renderer* ren = SDL_CreateRenderer(win, nullptr);
    if (!ren) {
        std::fprintf(stderr, "[launcher] renderer: %s\n", SDL_GetError());
        SDL_DestroyWindow(win);
        return true;
    }
    SDL_SetRenderVSync(ren, 1);
    SDL_InitSubSystem(SDL_INIT_GAMEPAD);
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;  // our own settings file, not imgui.ini
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard | ImGuiConfigFlags_NavEnableGamepad;
    io.FontGlobalScale = 1.25f * scale;
    style(scale);
    ImGui_ImplSDL3_InitForSDLRenderer(win, ren);
    ImGui_ImplSDLRenderer3_Init(ren);

    bool play = false, done = false;
    while (!done) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_EVENT_GAMEPAD_ADDED) {
                if (SDL_Gamepad* g = SDL_OpenGamepad(e.gdevice.which)) g_pads.push_back(g);
            } else if (e.type == SDL_EVENT_GAMEPAD_REMOVED) {
                std::erase_if(g_pads, [&](SDL_Gamepad* g) { return SDL_GetGamepadID(g) == e.gdevice.which; });
            }
            if (g_cap.player >= 0 && SDL_GetTicks() - g_cap.since > 250) {  // binding capture
                if (e.type == SDL_EVENT_KEY_DOWN && e.key.key == SDLK_ESCAPE) {
                    g_cap.player = -1;
                    continue;
                }
                const std::string b = captured(e);
                if (!b.empty()) {
                    const std::string k = g_cap.player == 2 ? std::string(g_cap.action == 0 ? "test" : "service")
                                                            : key_of(g_cap.player, io::input::kActions[g_cap.action]);
                    std::string cur = settings::get(k.c_str());
                    settings::set(k.c_str(), g_cap.append && !cur.empty() ? cur + "," + b : b);
                    g_cap.player = -1;
                    continue;
                }
                if (e.type == SDL_EVENT_KEY_DOWN || e.type == SDL_EVENT_MOUSE_BUTTON_DOWN ||
                    e.type == SDL_EVENT_GAMEPAD_BUTTON_DOWN)
                    continue;  // keep it from the UI while capturing
            }
            ImGui_ImplSDL3_ProcessEvent(&e);
            if (e.type == SDL_EVENT_QUIT || e.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED) done = true;
        }
        if (SDL_GetWindowFlags(win) & SDL_WINDOW_MINIMIZED) {
            SDL_Delay(20);
            continue;
        }
        ImGui_ImplSDLRenderer3_NewFrame();
        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(io.DisplaySize);
        ImGui::Begin("launcher", nullptr,
                     ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings);

        ImGui::SetWindowFontScale(1.6f);
        ImGui::TextColored(ImVec4(0.95f, 0.25f, 0.25f, 1.0f), "VIRTUA COP 3");
        ImGui::SetWindowFontScale(1.0f);
        ImGui::SameLine();
        ImGui::TextDisabled("  static recompilation");
        ImGui::Spacing();

        const float label_w = ImGui::GetContentRegionAvail().x * 0.34f;
        const float footer = ImGui::GetFrameHeightWithSpacing() * 2 + ImGui::GetTextLineHeightWithSpacing() * 2;
        ImGui::BeginTabBar("tabs");
        const bool on_settings = ImGui::BeginTabItem("Settings");
        if (on_settings) {
        ImGui::BeginChild("settings", ImVec2(0, -footer));
        // VC3_LAUNCHER_SCROLL=1 (testing): the end of the list, for screenshots.
        static const bool scroll_end = std::getenv("VC3_LAUNCHER_SCROLL") != nullptr;
        if (scroll_end && ImGui::GetFrameCount() > 3) ImGui::SetScrollY(ImGui::GetScrollMaxY());
        const char* section = "";
        for (const auto& o : settings::options()) {
            if (o.choices.empty() || !std::strcmp(o.section, "Controls")) continue;  // the Controls page edits these
            if (std::strcmp(section, o.section)) {
                section = o.section;
                ImGui::Spacing();
                ImGui::SeparatorText(section);
            }
            const bool locked = settings::overridden(o.key);
            const std::string cur = settings::get(o.key);
            const char* cur_label = cur.c_str();
            for (const auto& c : o.choices)
                if (cur == c.value) cur_label = c.label;
            ImGui::AlignTextToFramePadding();
            ImGui::TextUnformatted(o.label);
            ImGui::SameLine(label_w);
            ImGui::SetNextItemWidth(-1);
            ImGui::BeginDisabled(locked);
            if (ImGui::BeginCombo((std::string("##") + o.key).c_str(), cur_label)) {
                for (const auto& c : o.choices) {
                    if (!std::strcmp(o.key, "renderer") && !std::strcmp(c.value, "vulkan") && !vulkan_built()) continue;
                    if (!std::strcmp(o.key, "renderer") && !std::strcmp(c.value, "opengl") && platform::kVulkanOnly) continue;
                    const bool sel = cur == c.value;
                    if (ImGui::Selectable(c.label, sel)) settings::set(o.key, c.value);
                    if (sel) ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }
            ImGui::EndDisabled();
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled | ImGuiHoveredFlags_DelayShort)) {
                if (locked) ImGui::SetTooltip("Set by %s=%s in the environment", o.env, cur.c_str());
                else if (*o.help) ImGui::SetTooltip("%s", o.help);
            }
        }
        ImGui::EndChild();
        ImGui::EndTabItem();
        }
        static const bool open_controls = std::getenv("VC3_LAUNCHER_TAB") && !std::strcmp(std::getenv("VC3_LAUNCHER_TAB"), "controls");
        if (ImGui::BeginTabItem("Controls", nullptr, open_controls && ImGui::GetFrameCount() < 3 ? ImGuiTabItemFlags_SetSelected : 0)) {
            ImGui::BeginChild("controls", ImVec2(0, -footer));
            controls_page(label_w);
            ImGui::EndChild();
            ImGui::EndTabItem();
        }
        {
            static bool first = true;
            const bool missing = first && !install::installed();
            first = false;
            if (ImGui::BeginTabItem("Game files", nullptr, missing ? ImGuiTabItemFlags_SetSelected : 0)) {
                ImGui::BeginChild("files", ImVec2(0, -footer));
                game_files_page(win, label_w);
                ImGui::EndChild();
                ImGui::EndTabItem();
            }
        }
        ImGui::EndTabBar();

        ImGui::Spacing();
        const float bw = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) / 2;
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.70f, 0.10f, 0.12f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.85f, 0.15f, 0.17f, 1.0f));
        if (ImGui::IsWindowAppearing()) ImGui::SetKeyboardFocusHere();
        const bool ready = install::progress().state != 1 && install::installed();
        ImGui::BeginDisabled(!ready);
        if (ImGui::Button(ready ? "PLAY" : "PLAY (install the game files first)", ImVec2(bw, 0))) {
            play = true;
            done = true;
        }
        ImGui::EndDisabled();
        ImGui::PopStyleColor(2);
        ImGui::SameLine();
        if (ImGui::Button("Quit", ImVec2(bw, 0))) done = true;
        ImGui::Spacing();
        ImGui::TextDisabled("Settings: %s", settings::file_path().c_str());
        ImGui::End();

        ImGui::Render();
        SDL_SetRenderScale(ren, io.DisplayFramebufferScale.x, io.DisplayFramebufferScale.y);
        SDL_SetRenderDrawColor(ren, 15, 18, 26, 255);
        SDL_RenderClear(ren);
        ImGui_ImplSDLRenderer3_RenderDrawData(ImGui::GetDrawData(), ren);
        // VC3_LAUNCHER_SHOT=file.ppm: save the launcher as drawn and close it (testing).
        if (const char* shot = std::getenv("VC3_LAUNCHER_SHOT"); shot && ImGui::GetFrameCount() == 20) {
            if (SDL_Surface* s = SDL_RenderReadPixels(ren, nullptr)) {
                SDL_Surface* rgb = SDL_ConvertSurface(s, SDL_PIXELFORMAT_RGB24);
                if (std::FILE* f = rgb ? std::fopen(shot, "wb") : nullptr) {
                    std::fprintf(f, "P6\n%d %d\n255\n", rgb->w, rgb->h);
                    for (int y = 0; y < rgb->h; ++y) std::fwrite(static_cast<const char*>(rgb->pixels) + y * rgb->pitch, 1, size_t(rgb->w) * 3, f);
                    std::fclose(f);
                }
                if (rgb) SDL_DestroySurface(rgb);
                SDL_DestroySurface(s);
            }
            done = true;
        }
        SDL_RenderPresent(ren);
    }

    ImGui_ImplSDLRenderer3_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();
    SDL_DestroyRenderer(ren);
    SDL_DestroyWindow(win);
    if (play && !settings::save()) std::fprintf(stderr, "[launcher] could not save %s\n", settings::file_path().c_str());
    return play;
}

}  // namespace vc3::launcher
