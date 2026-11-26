// vc3 — runs the recompiled Virtua Cop 3 executable.
//
//   vc3 [path/to/vc3.xbe] [--headless] [--launcher | --no-launcher] [--no-settings]
//
// The launcher (settings window) shows first unless the settings say to skip it or
// --no-launcher is given; --launcher shows it regardless. --no-settings ignores the
// settings file (environment variables and built-in defaults only: tests use it).
#include <SDL3/SDL.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <ctime>
#include <chrono>
#include <cmath>
#include <thread>
#include <cstring>
#include <filesystem>
#include <string>

#include "profiler.h"
#include "audio/apu.h"
#include "gpu/gl_backend.h"
#include "gpu/nv2a.h"
#include "io/baseboard.h"
#include "io/input.h"
#include "io/jvs.h"
#include "io/mediaboard.h"
#include "io/ohci.h"
#include "kernel/ke.h"
#include "kernel/kernel.h"
#include "kernel/ps.h"
#include "install.h"
#include "launcher.h"
#include "loader.h"
#include "settings.h"
#include "mmio.h"
#include "vc3/memory.h"
#include "vc3/recomp.h"
#include "xbe.h"

using namespace vc3;

// Everything the program prints (stdout and stderr) also goes to vc3.log in the data
// folder (the previous run's log is kept as vc3.prev.log), so a crash can be reported
// without a terminal.
struct LogSink { int term, file; };
static ssize_t log_write(void* c, const char* buf, size_t n) {
    const auto* s = static_cast<const LogSink*>(c);
    if (write(s->term, buf, n) < 0) {}
    if (write(s->file, buf, n) < 0) {}
    return ssize_t(n);
}
static void start_log() {
    std::error_code ec;
    std::filesystem::rename("vc3.log", "vc3.prev.log", ec);
    const int file = open("vc3.log", O_WRONLY | O_CREAT | O_TRUNC | O_APPEND, 0644);
    if (file < 0) return;
    // Written straight through (no pipe, no buffering), so the lines before a crash land too.
    static LogSink err{dup(STDERR_FILENO), file}, out{dup(STDOUT_FILENO), file};
    cookie_io_functions_t io{nullptr, log_write, nullptr, nullptr};
    if (FILE* e = fopencookie(&err, "w", io)) {
        std::setvbuf(e, nullptr, _IONBF, 0);
        stderr = e;
    }
    if (FILE* o = fopencookie(&out, "w", io)) {
        std::setvbuf(o, nullptr, _IOLBF, 0);
        stdout = o;
    }
}

namespace vc3::hwwatch { void init(); }

namespace {
// F9: save what's needed to look into a problem seen on screen, in work/captures/<time>/:
// every draw of the next frame (images + draws.txt, with the widescreen HUD decisions), the
// window as shown, guest RAM 0-16 MiB, and the settings in effect.
void capture() {
    char stamp[32];
    const std::time_t t = std::time(nullptr);
    std::strftime(stamp, sizeof stamp, "%Y%m%d-%H%M%S", std::localtime(&t));
    const std::string dir = std::string("work/captures/") + stamp;
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) return;
    vc3::gpu::pgraph::glb::request_draw_trace(dir);
    vc3::gpu::request_window_shot(dir + "/window.ppm");
    if (std::FILE* f = std::fopen((dir + "/ram.bin").c_str(), "wb")) {
        std::fwrite(vc3::host_ptr(0), 1, 16u << 20, f);
        std::fclose(f);
    }
    if (std::FILE* f = std::fopen((dir + "/settings.txt").c_str(), "w")) {
        for (const auto& o : vc3::settings::options())
            std::fprintf(f, "%s = %s\n", o.key, vc3::settings::get(o.key).c_str());
        std::fprintf(f, "debug: depth test %s (F6), alpha test %s (F7), face culling %s (F8)\n",
                     vc3::gpu::pgraph::glb::debug_no_depth() ? "OFF" : "on", vc3::gpu::pgraph::glb::debug_no_alpha() ? "OFF" : "on",
                     vc3::gpu::pgraph::glb::no_cull() ? "OFF" : "on");
        std::fclose(f);
    }
    std::fprintf(stderr, "[capture] saved to %s\n", dir.c_str());
}
}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);  // keep logs when the guest aborts
    std::string xbe_path = "work/fs/vc3.xbe";
    bool headless = false, want_launcher = false, no_launcher = false, no_settings = false;
    std::string install_pic, install_chd;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--headless")) headless = true;
        else if (!std::strcmp(argv[i], "--launcher")) want_launcher = true;
        else if (!std::strcmp(argv[i], "--no-launcher")) no_launcher = true;
        else if (!std::strcmp(argv[i], "--no-settings")) no_settings = true;
        else if (!std::strcmp(argv[i], "--install") && i + 2 < argc) {
            // vc3 --install <vcop3.zip | 317-0354-com.data> <gdx-0003b.chd>: import the game files
            install_pic = std::filesystem::absolute(argv[++i]).string();
            install_chd = std::filesystem::absolute(argv[++i]).string();
        }
        else xbe_path = std::filesystem::absolute(argv[i]).string();  // before any chdir below
    }
    // Game data and state live under work/ in a data folder: the current directory or the
    // project root above the executable if either has the game or is a source checkout, otherwise
    // the per-user data folder (installed builds: ~/.local/share/vcop3-recomp/vcop3-recomp/).
    if (!no_settings) settings::load();
    if (!no_settings && !settings::get("install_dir").empty()) {
        install::use_location(settings::get("install_dir"));  // the player's chosen folder
    } else if (!std::filesystem::exists("work/fs")) {
        std::error_code ec;
        bool found = false;
        auto dir = std::filesystem::read_symlink("/proc/self/exe", ec).parent_path();
        for (; !ec && !dir.empty() && dir != dir.root_path(); dir = dir.parent_path())
            // (a source checkout counts before the game is installed: --install fills its work/)
            if (std::filesystem::exists(dir / "work" / "fs") || std::filesystem::exists(dir / "config" / "vc3_entries.txt")) {
                std::filesystem::current_path(dir, ec);
                std::printf("[main] using project directory %s\n", dir.c_str());
                found = true;
                break;
            }
        if (!found) install::use_location(install::default_location());
    }
    if (no_settings || settings::get("install_dir").empty()) install::set_default_location(std::filesystem::current_path().string());
    if (!getenv("VC3_NO_LOG")) start_log();
    if (!install_pic.empty()) return install::run({install_pic, install_chd}) ? 0 : 1;

    if (vc3_func_table_size == 0) {
        std::fprintf(stderr,
                     "No recompiled code is linked in. Run the recompiler first:\n"
                     "  build/recompiler/vc3recomp work/fs/vc3.xbe generated\n"
                     "then re-run cmake and rebuild.\n");
        return 1;
    }

    std::string err;
    if (!mem::init(err)) { std::fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
    hwwatch::init();  // VC3_HW_WATCH (dev): before any guest thread exists
    kernel::init();
    mmio::install();
    io::ohci::init();
    io::baseboard::init();
    io::mediaboard::init();
    // SDL would turn SIGINT/SIGTERM into a quit event, which nobody polls in headless
    // mode; let them end the process directly (shutdown is an exit anyway).
    SDL_SetHint(SDL_HINT_NO_SIGNAL_HANDLERS, "1");
    profiler::init();
    // The GL renderer shares an EGL context with the window: before SDL's video starts.
    SDL_SetHint(SDL_HINT_VIDEO_FORCE_EGL, "1");
    // Settings, and the launcher to change them (before anything reads the options).
    if (!no_settings) {
        const bool show = !headless && !no_launcher &&
                          (want_launcher || settings::get("skip_launcher") != "1" || !install::installed());
        if (show) {
            if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS)) {
                std::fprintf(stderr, "SDL_Init: %s\n", SDL_GetError());
                return 1;
            }
            if (!launcher::run()) {
                SDL_Quit();
                return 0;
            }
        }
        settings::apply();
    }
    if (!install::installed()) {
        std::string why;
        install::installed(&why);
        std::fprintf(stderr, "%s. Start without --no-launcher to import them, or:\n"
                             "  vc3 --install <vcop3.zip> <gdx-0003b.chd>\n", why.c_str());
        return 1;
    }
    auto xbe = Xbe::load(xbe_path, err);
    if (!xbe) { std::fprintf(stderr, "error: %s: %s\n", xbe_path.c_str(), err.c_str()); return 1; }
    if (xbe->entry != vc3_entry_point) {
        std::fprintf(stderr, "error: %s entry %08x does not match the recompiled code (%08x)\n",
                     xbe_path.c_str(), xbe->entry, vc3_entry_point);
        return 1;
    }
    gpu::pgraph::glb::configure();
    // The window (and its GL context, which the renderer shares) comes first, so the
    // GPU thread can share with it when the guest starts drawing.
    SDL_Window* win = nullptr;
    if (!headless) {
        if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS)) {
            std::fprintf(stderr, "SDL_Init: %s\n", SDL_GetError());
            return 1;
        }
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 5);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
        SDL_GL_SetAttribute(SDL_GL_ALPHA_SIZE, 0);  // an alpha channel would make the window translucent
        SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 0);
        const bool vulkan = gpu::wants_vulkan_window();
        int ww = 640, wh = 480;  // VC3_WINDOW_SIZE=WxH
        if (const char* ws = std::getenv("VC3_WINDOW_SIZE"); ws && std::strchr(ws, 'x')) {
            ww = std::max(320, std::atoi(ws));
            wh = std::max(240, std::atoi(std::strchr(ws, 'x') + 1));
        }
        win = SDL_CreateWindow("Virtua Cop 3 (recomp)", ww, wh,
                               SDL_WINDOW_RESIZABLE | (vulkan ? SDL_WINDOW_VULKAN : SDL_WINDOW_OPENGL));
        if (vulkan && (!win || !gpu::init(win))) {
            // No usable Vulkan: a Vulkan window can't host GL, so start over with OpenGL
            // (the renderer reads VC3_RENDERER too).
            std::fprintf(stderr, "[main] Vulkan unavailable (%s), using OpenGL\n", SDL_GetError());
            if (win) SDL_DestroyWindow(win);
            setenv("VC3_RENDERER", "opengl", 1);
            win = SDL_CreateWindow("Virtua Cop 3 (recomp)", ww, wh, SDL_WINDOW_RESIZABLE | SDL_WINDOW_OPENGL);
            if (win && !gpu::init(win)) win = nullptr;
        } else if (win && !vulkan && !gpu::init(win)) {
            win = nullptr;
        }
        if (!win) {
            std::fprintf(stderr, "window/graphics: %s\n", SDL_GetError());
            return 1;
        }
    }
    if (!headless) {
        io::input::apply_api_hints();
        io::input::init();
    }
    audio::apu::init(!headless);
    gpu::init_hardware();
    if (!load_xbe(*xbe, err)) { std::fprintf(stderr, "error: %s\n", err.c_str()); return 1; }

    std::printf("[main] starting guest at %08x\n", xbe->entry);
    kernel::ke::start_worker();
    kernel::ps::start_boot_thread(xbe->entry, xbe->stack_size);
    if (headless) {
        kernel::ps::wait_all_threads();
        std::printf("[main] all guest threads exited\n");
        return 0;
    }


    if (const char* fs = std::getenv("VC3_FULLSCREEN"); fs && std::strcmp(fs, "0")) gpu::toggle_fullscreen();
    SDL_HideCursor();  // no system cursor over the game (the crosshair shows the aim)
    bool running = true;
    // VC3_QUIT_AFTER=S (testing): quit the normal way (as closing the window does) after S seconds.
    const double quit_after = std::getenv("VC3_QUIT_AFTER") ? std::atof(std::getenv("VC3_QUIT_AFTER")) : 0;
    const Uint64 started = SDL_GetTicks();
    while (running) {
        if (quit_after > 0 && double(SDL_GetTicks() - started) >= quit_after * 1000) running = false;
        // VC3_CAPTURE_AFTER=S (testing): an F9 capture after S seconds.
        static const double capture_after = std::getenv("VC3_CAPTURE_AFTER") ? std::atof(std::getenv("VC3_CAPTURE_AFTER")) : 0;
        static bool captured = false;
        if (capture_after > 0 && !captured && double(SDL_GetTicks() - started) >= capture_after * 1000) {
            captured = true;
            capture();
        }
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_EVENT_QUIT) running = false;
            // F11 or Alt+Enter: toggle fullscreen (desktop resolution; picture scaled to fit).
            if (e.type == SDL_EVENT_KEY_DOWN && !e.key.repeat &&
                (e.key.key == SDLK_F11 || (e.key.key == SDLK_RETURN && (e.key.mod & SDL_KMOD_ALT))))
                gpu::toggle_fullscreen();
            if (e.type == SDL_EVENT_KEY_DOWN && !e.key.repeat && e.key.key == SDLK_F9) capture();
            // F6 (developer): depth test off / on, for telling a surface hidden by a wrong depth.
            if (e.type == SDL_EVENT_KEY_DOWN && !e.key.repeat && e.key.key == SDLK_F6) {
                const bool off = gpu::pgraph::glb::toggle_no_depth_test();
                std::fprintf(stderr, "[debug] depth test %s (F6)\n", off ? "OFF" : "on");
                SDL_SetWindowTitle(win, off ? "Virtua Cop 3 (recomp) - depth test OFF (F6)" : "Virtua Cop 3 (recomp)");
            }
            // F7 (developer): alpha test off / on, for telling a surface discarded by its alpha.
            if (e.type == SDL_EVENT_KEY_DOWN && !e.key.repeat && e.key.key == SDLK_F7) {
                const bool off = gpu::pgraph::glb::toggle_no_alpha_test();
                std::fprintf(stderr, "[debug] alpha test %s (F7)\n", off ? "OFF" : "on");
                SDL_SetWindowTitle(win, off ? "Virtua Cop 3 (recomp) - alpha test OFF (F7)" : "Virtua Cop 3 (recomp)");
            }
            // F8 (developer): face culling off / on, for telling a wrongly culled surface.
            if (e.type == SDL_EVENT_KEY_DOWN && !e.key.repeat && e.key.key == SDLK_F8) {
                const bool off = gpu::pgraph::glb::toggle_no_cull();
                std::fprintf(stderr, "[debug] face culling %s (F8)\n", off ? "OFF" : "on");
                SDL_SetWindowTitle(win, off ? "Virtua Cop 3 (recomp) - face culling OFF (F8)" : "Virtua Cop 3 (recomp)");
            }
            gpu::to_screen_coords(e);
            io::input::handle_event(e);
        }
        io::input::update();
        // Startup shader compilation (new machine or driver): show progress in the title.
        {
            static bool shown = false;
            int done = 0, total = 0;
            gpu::pgraph::glb::shader_warmup_progress(done, total);
            static int last = -1;
            if (total > 0 && done < total && done != last) {
                last = done;
                shown = true;
                const std::string t = "Virtua Cop 3 (recomp) - compiling shaders " + std::to_string(done) + "/" + std::to_string(total);
                SDL_SetWindowTitle(win, t.c_str());
            } else if (shown && (total == 0 || done >= total)) {
                shown = false;
                SDL_SetWindowTitle(win, "Virtua Cop 3 (recomp)");
            }
        }
        // A present blocks for the display refresh (vsync); with no new frame, poll again shortly.
        if (!gpu::present()) SDL_Delay(1);
    }
    // Exit without tearing the renderer down: guest threads (the GPU thread included) are
    // detached and keep running until the process ends, so destroying the Vulkan device or
    // the GL context under them can crash. Only the audio thread is stopped (it polls the
    // SDL audio stream), and fullscreen is left so a changed display mode is restored.
    audio::apu::shutdown();
    SDL_SetWindowFullscreen(win, false);
    SDL_SyncWindow(win);
    std::_Exit(0);
}
