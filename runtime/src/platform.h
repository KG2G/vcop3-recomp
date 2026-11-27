// Small platform differences (Linux / Windows) shared by several files.
#pragma once
#include <chrono>
#include <filesystem>
#include <string>
#include <vector>

namespace vc3::platform {

// The running program's own file (Linux: /proc/self/exe; Windows: GetModuleFileName).
std::filesystem::path exe_path();
// The folder the player sees the program in (Linux AppImage: the .AppImage file's folder, not
// its temporary mount).
std::filesystem::path program_dir();
// The default data folder, computed without creating it (SDL_GetPrefPath creates it, which would
// leave an empty folder behind for installs that live elsewhere): Windows
// %APPDATA%\vcop3-recomp\vcop3-recomp, Linux $XDG_DATA_HOME (~/.local/share)/vcop3-recomp/vcop3-recomp.
std::filesystem::path user_data_dir();
#ifdef _WIN32
// Set an open file's size without writing the new part: marked sparse first (unwritten
// parts take no space and read as zeros, like Linux), then the end moved.
bool set_file_size(int fd, long long size);
#endif
// Process setup: on Windows, 1 ms timer precision (the default rounds every short sleep
// and timed wait up to ~15.6 ms: laggy input and stutter). Nothing elsewhere.
void init_process();
// Precise sleeps. Windows rounds ordinary sleeps up to whole milliseconds and overshoots
// (Sleep(1) is often 2 ms), which adds up across the many short waits in a game frame;
// here they use a high-resolution waitable timer instead. Linux: std::this_thread.
// VC3_COARSE_SLEEP=1 (testing, Linux): sleep the way Windows' default timer does.
void sleep_until(std::chrono::steady_clock::time_point t);
void sleep_for(std::chrono::nanoseconds d);
bool coarse_sleep();  // the testing switch above
#ifdef _WIN32
// Auto-reset Win32 events for fast thread wake-ups (scheduler, kernel waits), without
// pulling windows.h into files whose names clash with it. wait_event: ms ~0u = forever.
void* make_event();
void set_event(void* e);
void wait_event(void* e, unsigned ms);
#endif
// The program's command line (main), and a restart of the program with it -- the launcher skipped,
// VC3_TEST_MODE set or cleared -- the way the cabinet reboots into or out of its test menu.
// Returns only if the restart failed.
void set_args(int argc, char** argv);
std::vector<std::string> args();
void relaunch(bool test_mode);
// setenv(name, value, overwrite=1) / unsetenv (Windows: _putenv_s).
void set_env(const char* name, const char* value);
void unset_env(const char* name);

}  // namespace vc3::platform
