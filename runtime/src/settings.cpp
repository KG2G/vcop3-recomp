#include "platform.h"
#include "settings.h"

#include <SDL3/SDL.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>

namespace vc3::settings {
namespace {
std::map<std::string, std::string> g_values;

const std::vector<Option> kOptions = {
#ifdef _WIN32  // Windows: Vulkan by default (most dependable there); OpenGL works too
    {"renderer", "VC3_RENDERER", "Graphics", "Renderer",
     "Vulkan (default on Windows) or OpenGL 4.5. Try the other one if a game scene looks wrong or stutters.",
     {{"opengl", "OpenGL"}, {"vulkan", "Vulkan"}}, "vulkan"},
#else
    {"renderer", "VC3_RENDERER", "Graphics", "Renderer",
     "OpenGL works everywhere. Vulkan can be smoother; if it isn't available the game uses OpenGL.",
     {{"opengl", "OpenGL"}, {"vulkan", "Vulkan"}}, "opengl"},
#endif
    {"res_scale", "VC3_RES_SCALE", "Graphics", "Resolution",
     "Internal 3D resolution as a multiple of the original 640x480. Match display picks the "
     "multiple closest to your screen's height (3x for 1440p, 2x for 1080p and the Steam Deck).",
     {{"auto", "Match display"}, {"1", "1x (640x480)"}, {"2", "2x (1280x960)"}, {"3", "3x (1920x1440)"}, {"4", "4x (2560x1920)"},
      {"6", "6x (3840x2880)"}, {"8", "8x (5120x3840)"}},
     "auto"},
    {"widescreen", "VC3_WIDESCREEN", "Graphics", "Aspect ratio",
     "Widescreen shows more of the 3D scene at the sides; aiming stays exact. Match display "
     "uses your screen's shape (16:9, 16:10 on the Steam Deck, ultrawide; 4:3 on 4:3 screens).",
     {{"auto", "Match display"}, {"0", "4:3 (original)"}, {"16:9", "16:9"}, {"16:10", "16:10"}, {"21:9", "21:9"}}, "auto"},
    {"wide_hud", "VC3_WIDE_HUD", "Graphics", "HUD in widescreen",
     "Move the score, ammo and lives panels out to the screen edges in widescreen.",
     {{"1", "At the edges"}, {"0", "In the 4:3 area"}}, "1"},
    {"anisotropic", "VC3_ANISOTROPIC", "Graphics", "Anisotropic filtering",
     "Sharper textures on surfaces seen at an angle (floors, roads, walls).",
     {{"1", "Off"}, {"2", "2x"}, {"4", "4x"}, {"8", "8x"}, {"16", "16x"}}, "16"},
    {"smaa", "VC3_SMAA", "Graphics", "Anti-aliasing (SMAA)",
     "Smooths jagged edges of the finished picture. Higher presets find more edges and cost a "
     "little more GPU time.",
     {{"off", "Off"}, {"low", "Low"}, {"medium", "Medium"}, {"high", "High"}, {"ultra", "Ultra"}}, "off"},
    // Shaders are always precompiled at startup (every one the game used before) and new ones
    // built in the background, for smooth play from the start. (VC3_SHADER_PRECOMPILE=0 and
    // VC3_ASYNC_SHADERS=0 turn those off for testing.)
    {"window_size", "VC3_WINDOW_SIZE", "Display", "Window size",
     "Size of the game window when not fullscreen (it can also be resized by dragging). Match "
     "display sizes it to the aspect ratio, filling most of the screen.",
     {{"auto", "Match display"}, {"640x480", "640x480"}, {"960x720", "960x720"}, {"1280x960", "1280x960"}, {"1280x720", "1280x720"},
      {"1600x900", "1600x900"}, {"1920x1080", "1920x1080"}, {"2560x1440", "2560x1440"}, {"3840x2160", "3840x2160"}},
     "auto"},
    {"fullscreen", "VC3_FULLSCREEN", "Display", "Window",
     "F11 or Alt+Enter switches while playing.", {{"0", "Windowed"}, {"1", "Fullscreen"}}, "0"},
    {"crosshair", "VC3_CROSSHAIR", "Display", "Crosshair",
     "A crosshair at each player's aim point (red player 1, blue player 2), shown while that player is "
     "in a game (from pressing start until game over; either side can play alone).",
     {{"on", "On"}, {"off", "Off"}}, "on"},
    {"crosshair_size", "VC3_CROSSHAIR_SIZE", "Display", "Crosshair size",
     "Relative to the window's height (the screen's when fullscreen); 100% is 5% of it across.",
     {{"0.5", "50%"}, {"0.75", "75%"}, {"1", "100%"}, {"1.25", "125%"}, {"1.5", "150%"}, {"2", "200%"}, {"2.5", "250%"},
      {"3", "300%"}}, "1"},
    {"sinden_border", "VC3_SINDEN_BORDER", "Display", "Sinden lightgun border",
     "A white border at the screen's edges for Sinden lightguns (their camera finds the screen by it), "
     "as thick as this share of the screen height: thicker tracks more reliably, thinner hides less "
     "of the picture. Play fullscreen and aim with the system mouse.",
     {{"0", "Off"}, {"1", "1%"}, {"1.5", "1.5%"}, {"2", "2%"}, {"3", "3%"}, {"4", "4%"}, {"5", "5%"}, {"7", "7%"}},
     "0"},
    {"vsync", "VC3_VSYNC", "Display", "Vertical sync",
     "Wait for the display refresh (no tearing). Leave on unless you know you need it off.",
     {{"1", "On"}, {"0", "Off"}}, "1"},
    {"audio", "VC3_AUDIO", "Audio", "Sound", "", {{"1", "On"}, {"0", "Off"}}, "1"},
    {"volume", "VC3_VOLUME", "Audio", "Volume", "Loudness of the game's sound and music.",
     {{"1", "100%"}, {"0.9", "90%"}, {"0.8", "80%"}, {"0.7", "70%"}, {"0.6", "60%"}, {"0.5", "50%"},
      {"0.4", "40%"}, {"0.3", "30%"}, {"0.2", "20%"}, {"0.1", "10%"}},
     "1"},
    {"check_updates", "VC3_CHECK_UPDATES", "Updates", "Check for updates",
     "When the program starts, look on GitHub for a newer version. If there is one, its changes are shown "
     "and you choose whether to update (only the program is replaced: the installed game, settings and "
     "saves stay as they are).",
     {{"1", "On"}, {"0", "Off"}}, "1"},
    {"freeplay", "VC3_FREEPLAY", "Game", "Free play",
     "Start and continue without coins (the game shows FREEPLAY instead of the credits).",
     {{"0", "Off (coins)"}, {"1", "On"}}, "0"},
    {"cheat_ammo", "VC3_CHEAT_AMMO", "Cheats", "Infinite ammo",
     "The magazine never runs out (no reloading needed); special weapons keep their ammo.",
     {{"0", "Off"}, {"1", "On"}}, "0"},
    {"cheat_lives", "VC3_CHEAT_LIVES", "Cheats", "Infinite lives",
     "Getting hit doesn't cost a life.", {{"0", "Off"}, {"1", "On"}}, "0"},
    {"cheat_slowmo", "VC3_CHEAT_SLOWMO", "Cheats", "Infinite E.S. slow motion",
     "The E.S. gauge (pedal slow motion) never runs down.", {{"0", "Off"}, {"1", "On"}}, "0"},
    {"cheat_final", "VC3_CHEAT_FINAL", "Cheats", "Start at the final stage",
     "Whichever mission is chosen, the final stage is played instead.", {{"0", "Off"}, {"1", "On"}}, "0"},
    {"cheat_weapon", "VC3_CHEAT_WEAPON", "Cheats", "Weapon",
     "Every player starts each mission with this weapon (the handgun stays as the second one; "
     "Change weapon switches). Given again whenever the player is back to the handgun alone. All "
     "weapons: Change weapon steps through every weapon (handgun, G36C, MP7, M4, M629, LMG249, SAM-92A).",
     // value = the game's weapon type + 1 (hooks.cpp)
     {{"0", "Off"}, {"3", "G36C assault rifle"}, {"4", "MP7 machine gun"}, {"5", "M4 shotgun"},
      {"6", "M629 magnum"}, {"11", "LMG249 machine gun"}, {"10", "SAM-92A missile launcher"},
      {"99", "All weapons"}}, "0"},
    // Controls (the launcher's Controls page; see io/input.h for the formats).
    {"input_api", nullptr, "Controls", "Controller API",
     "Windows: which system API reads controllers (Auto lets SDL choose per device). Linux always uses evdev.",
     {{"auto", "Auto"}, {"xinput", "XInput"}, {"dinput", "DirectInput"}, {"rawinput", "Raw Input"}}, "auto"},
    {"pad_sensitivity", nullptr, "Controls", "Gamepad aim speed",
     "How fast a gamepad stick moves the crosshair.",
     {{"0.5", "50%"}, {"0.75", "75%"}, {"1", "100%"}, {"1.25", "125%"}, {"1.5", "150%"}, {"2", "200%"}, {"3", "300%"}}, "1"},
    {"p1_aim", nullptr, "Controls", "", "", {}, "mouse"},
    {"p1_pad", nullptr, "Controls", "", "", {}, "1"},
    {"p1_trigger", nullptr, "Controls", "", "", {}, "mouse:left,pad:righttrigger"},
    {"p1_reload", nullptr, "Controls", "", "", {}, "mouse:right,pad:lefttrigger"},
    {"p1_pedal", nullptr, "Controls", "", "", {}, "key:Space,pad:a"},
    {"p1_change", nullptr, "Controls", "", "", {}, "mouse:middle,key:E,pad:x"},
    {"p1_start", nullptr, "Controls", "", "", {}, "key:1,pad:start,mouse:plus"},
    {"p1_coin", nullptr, "Controls", "", "", {}, "key:5,pad:back,mouse:minus"},
    {"p2_aim", nullptr, "Controls", "", "", {}, "pad"},
    {"p2_pad", nullptr, "Controls", "", "", {}, "2"},
    {"p2_trigger", nullptr, "Controls", "", "", {}, "mouse:left,pad:righttrigger"},
    {"p2_reload", nullptr, "Controls", "", "", {}, "mouse:right,pad:lefttrigger"},
    {"p2_pedal", nullptr, "Controls", "", "", {}, "pad:a"},
    {"p2_change", nullptr, "Controls", "", "", {}, "mouse:middle,pad:x"},
    {"p2_start", nullptr, "Controls", "", "", {}, "key:2,pad:start,mouse:plus"},
    {"p2_coin", nullptr, "Controls", "", "", {}, "key:6,pad:back,mouse:minus"},
    {"test", nullptr, "Controls", "", "", {}, "key:F2"},
    {"service", nullptr, "Controls", "", "", {}, "key:F3"},
    {"install_dir", nullptr, "Files", "", "", {}, ""},  // game data folder (empty: per-user data folder)
    {"skip_launcher", nullptr, "Launcher", "Show this launcher",
     "When skipped, start with --launcher to see it again.", {{"0", "At every start"}, {"1", "Skip it"}}, "0"},
};

std::string trim(const std::string& s) {
    const size_t a = s.find_first_not_of(" \t\r\n"), b = s.find_last_not_of(" \t\r\n");
    return a == std::string::npos ? std::string() : s.substr(a, b - a + 1);
}
const Option* find(const char* key) {
    for (const auto& o : kOptions)
        if (!std::strcmp(o.key, key)) return &o;
    return nullptr;
}
}  // namespace

const std::vector<Option>& options() { return kOptions; }

// Where everything lives: the data folder is wherever settings.ini is. The default is the per-user
// data folder. A folder of the player's own ("Install to") holds everything instead -- the game
// files (work/), settings.ini and vc3.log -- so it is portable. At startup it is found:
//   1. beside the program (installed into the program's own folder: the folder can be moved as is);
//   2. through location.txt in the per-user folder (one line, the folder's path), for a folder
//      anywhere else -- the only thing left in the per-user folder then;
//   3. otherwise the per-user folder.
namespace {
namespace fs = std::filesystem;
std::string g_file;  // the settings file in use (set by load())
fs::path pointer_file() { return platform::user_data_dir() / "location.txt"; }
fs::path default_file() { return platform::user_data_dir() / "settings.ini"; }
bool same_dir(const fs::path& a, const fs::path& b) {
    std::error_code ec;
    return fs::equivalent(a, b, ec) && !ec;
}
std::string locate() {
    std::error_code ec;
    const fs::path beside = platform::program_dir() / "settings.ini";
    if (fs::exists(beside, ec)) return beside.string();
    if (std::ifstream pf{pointer_file()}) {
        std::string dir;
        std::getline(pf, dir);
        dir = trim(dir);
        if (!dir.empty() && fs::exists(fs::path(dir) / "settings.ini", ec)) return (fs::path(dir) / "settings.ini").string();
    }
    return default_file().string();
}
}  // namespace

std::string file_path() {
    if (g_file.empty()) g_file = locate();
    return g_file;
}

void load() {
    for (const auto& o : kOptions) g_values[o.key] = o.def;
    g_file = locate();
    std::ifstream f(g_file);
    std::string line;
    while (std::getline(f, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#' || line[0] == '[') continue;
        const size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        const std::string k = trim(line.substr(0, eq)), v = trim(line.substr(eq + 1));
        const Option* o = find(k.c_str());
        if (!o) continue;
        if (o->choices.empty()) g_values[k] = v;  // free-form (control bindings)
        for (const auto& c : o->choices)  // only values the option offers
            if (v == c.value) g_values[k] = v;
    }
    f.close();
    if (!same_dir(fs::path(g_file).parent_path(), platform::user_data_dir())) {
        // A folder of the player's own: it is where settings.ini is (also after the folder was moved).
        g_values["install_dir"] = fs::path(g_file).parent_path().string();
    } else if (!g_values["install_dir"].empty()) {
        // Settings from before (v0.1.6 and older) with an install folder chosen: everything moves there.
        relocate(g_values["install_dir"]);
    }
}

void tidy_default_folder() {
    std::error_code ec;
    fs::remove(platform::user_data_dir(), ec);  // (only if empty)
    fs::remove(platform::user_data_dir().parent_path(), ec);
}

bool relocate(const std::string& dir_in) {
    std::error_code ec;
    const std::string old_file = file_path();
    const std::string old_dir_setting = g_values["install_dir"];
    fs::path dir = dir_in.empty() ? platform::user_data_dir() : fs::absolute(fs::path(dir_in), ec);
    const bool custom = !dir_in.empty() && !same_dir(dir, platform::user_data_dir());
    fs::create_directories(dir, ec);
    g_file = (dir / "settings.ini").string();
    g_values["install_dir"] = custom ? dir.string() : "";
    if (!save()) {  // (a folder that can't be written: stay where we were)
        g_file = old_file;
        g_values["install_dir"] = old_dir_setting;
        return false;
    }
    // The pointer in the per-user folder: only for a folder of the player's own that isn't the
    // program's folder (that one is found by itself).
    if (custom && !same_dir(dir, platform::program_dir())) {
        fs::create_directories(platform::user_data_dir(), ec);
        std::ofstream(pointer_file()) << dir.string() << "\n";
    } else {
        fs::remove(pointer_file(), ec);
    }
    if (old_file != g_file) fs::remove(old_file, ec);  // the old settings file (game files stay where they are)
    return true;
}

bool save() {
    const std::string p = file_path();
    std::ofstream f(p);
    if (!f) return false;
    f << "# Virtua Cop 3 (recomp) settings - edited by the launcher\n";
    for (const auto& o : kOptions) f << o.key << " = " << g_values[o.key] << "\n";
    return bool(f);
}

std::string get(const char* key) {
    const Option* o = find(key);
    if (o && o->env && overridden(key)) return std::getenv(o->env);
    auto it = g_values.find(key);
    return it != g_values.end() ? it->second : (o ? o->def : "");
}

void set(const char* key, const std::string& value) { g_values[key] = value; }

bool overridden(const char* key) {
    const Option* o = find(key);
    return o && o->env && std::getenv(o->env) != nullptr;
}

void apply() {
    for (const auto& o : kOptions)
        if (o.env && !overridden(o.key)) platform::set_env(o.env, get(o.key).c_str());
}

}  // namespace vc3::settings
