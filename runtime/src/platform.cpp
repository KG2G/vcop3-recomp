#include "platform.h"

#include <chrono>
#include <cstdlib>

#include <cstdio>
#include <thread>
#include <vector>
#include <string>
#ifndef _WIN32
#include <unistd.h>
#endif

#ifdef _WIN32
#include <io.h>
#include <windows.h>
#include <dwmapi.h>
#include <winioctl.h>
#include <timeapi.h>
#endif

namespace vc3::platform {

std::filesystem::path exe_path() {
#ifdef _WIN32
    wchar_t buf[32768];
    const DWORD n = GetModuleFileNameW(nullptr, buf, DWORD(std::size(buf)));
    return n ? std::filesystem::path(std::wstring(buf, n)) : std::filesystem::path();
#else
    std::error_code ec;
    return std::filesystem::read_symlink("/proc/self/exe", ec);
#endif
}

std::filesystem::path program_dir() {
    if (const char* ai = std::getenv("APPIMAGE"); ai && *ai) return std::filesystem::path(ai).parent_path();
    const std::filesystem::path p = exe_path();
    return p.empty() ? std::filesystem::current_path() : p.parent_path();
}

std::filesystem::path user_data_dir() {
#ifdef _WIN32
    const char* base = std::getenv("APPDATA");
    const std::filesystem::path root = base && *base ? std::filesystem::path(base) : std::filesystem::current_path();
#else
    std::filesystem::path root;
    if (const char* x = std::getenv("XDG_DATA_HOME"); x && *x) root = x;
    else if (const char* h = std::getenv("HOME"); h && *h) root = std::filesystem::path(h) / ".local" / "share";
    else root = std::filesystem::current_path();
#endif
    return root / "vcop3-recomp" / "vcop3-recomp";
}

#ifdef _WIN32
bool set_file_size(int fd, long long size) {
    const HANDLE h = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
    if (h == INVALID_HANDLE_VALUE) return false;
    DWORD ret = 0;
    DeviceIoControl(h, FSCTL_SET_SPARSE, nullptr, 0, nullptr, 0, &ret, nullptr);  // (fails on FAT32: fine)
    LARGE_INTEGER end;
    end.QuadPart = size;
    return SetFilePointerEx(h, end, nullptr, FILE_BEGIN) && SetEndOfFile(h);
}
#endif

bool coarse_sleep() {
    static const bool on = std::getenv("VC3_COARSE_SLEEP") != nullptr;
    return on;
}

void sleep_until(std::chrono::steady_clock::time_point t) {
    const auto now = std::chrono::steady_clock::now();
    if (t <= now) return;
    sleep_for(t - now);
}

void sleep_for(std::chrono::nanoseconds d) {
    if (d.count() <= 0) return;
#ifdef _WIN32
    // A high-resolution waitable timer (Windows 10 1803+) per thread; the last few
    // microseconds are yielded away rather than overslept.
    thread_local HANDLE timer = [] {
        HANDLE h = CreateWaitableTimerExW(nullptr, nullptr, 0x00000002 /*CREATE_WAITABLE_TIMER_HIGH_RESOLUTION*/,
                                          TIMER_ALL_ACCESS);
        return h ? h : CreateWaitableTimerExW(nullptr, nullptr, 0, TIMER_ALL_ACCESS);
    }();
    const auto end = std::chrono::steady_clock::now() + d;
    const auto slack = std::chrono::microseconds(300);
    if (timer && d > slack) {
        LARGE_INTEGER due;
        due.QuadPart = -(long long)((d - slack).count() / 100);  // relative, 100 ns units
        if (SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE)) WaitForSingleObject(timer, INFINITE);
    }
    while (std::chrono::steady_clock::now() < end) std::this_thread::yield();
#else
    if (coarse_sleep()) {  // like Windows' Sleep(): whole milliseconds, rounded up, then late
        const long long ms = (d.count() + 999999) / 1000000;
        d = std::chrono::milliseconds(ms + 1);
    }
    std::this_thread::sleep_for(d);
#endif
}

#ifdef _WIN32
void* make_event() { return CreateEventW(nullptr, FALSE, FALSE, nullptr); }
void set_event(void* e) { SetEvent(static_cast<HANDLE>(e)); }
void wait_event(void* e, unsigned ms) { WaitForSingleObject(static_cast<HANDLE>(e), ms == ~0u ? INFINITE : DWORD(ms)); }
#endif

void set_thread_priority(Priority p) {
#ifdef _WIN32
    SetThreadPriority(GetCurrentThread(), p == Priority::kTimeCritical ? THREAD_PRIORITY_HIGHEST : THREAD_PRIORITY_BELOW_NORMAL);
#else
    (void)p;
#endif
}

#ifdef _WIN32
bool dwm_timing(long long& vblank_ns, long long& period_ns, unsigned long long& count) {
    using Fn = HRESULT(WINAPI*)(HWND, DWM_TIMING_INFO*);
    static const Fn fn = [] {
        HMODULE m = LoadLibraryW(L"dwmapi.dll");
        return m ? reinterpret_cast<Fn>(reinterpret_cast<void*>(GetProcAddress(m, "DwmGetCompositionTimingInfo"))) : Fn(nullptr);
    }();
    if (!fn) return false;
    DWM_TIMING_INFO info{};
    info.cbSize = sizeof info;
    if (FAILED(fn(nullptr, &info)) || !info.qpcRefreshPeriod || !info.qpcVBlank) return false;
    LARGE_INTEGER freq, now;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&now);
    const long long steady =
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    const double ns_per_tick = 1e9 / double(freq.QuadPart);
    vblank_ns = steady - (long long)(double(now.QuadPart - (long long)info.qpcVBlank) * ns_per_tick);
    period_ns = (long long)(double(info.qpcRefreshPeriod) * ns_per_tick);
    count = info.cRefresh;
    return true;
}
#endif

void init_process() {
#ifdef _WIN32
    timeBeginPeriod(1);  // for the life of the process (Windows restores it on exit)
#endif
}

std::vector<std::string> g_args;
void set_args(int argc, char** argv) { g_args.assign(argv, argv + argc); }
std::vector<std::string> args() { return g_args; }

void relaunch(bool test_mode) {
    if (test_mode) set_env("VC3_TEST_MODE", "1");
    else unset_env("VC3_TEST_MODE");
    std::vector<std::string> args;
    for (size_t i = 0; i < g_args.size(); ++i)
        if (!i || (g_args[i] != "--launcher" && g_args[i] != "--no-launcher")) args.push_back(g_args[i]);
    if (args.empty()) args.push_back("vc3");
    args.push_back("--no-launcher");  // straight back into the game
    std::fflush(stderr);
    std::fflush(stdout);
#ifdef _WIN32
    std::wstring cmd;
    for (const std::string& a : args) {
        const int n = MultiByteToWideChar(CP_UTF8, 0, a.c_str(), -1, nullptr, 0);
        std::wstring w(n > 0 ? size_t(n - 1) : 0, L'\0');
        if (n > 1) MultiByteToWideChar(CP_UTF8, 0, a.c_str(), -1, w.data(), n);
        if (!cmd.empty()) cmd += L' ';
        cmd += L'"' + w + L'"';
    }
    STARTUPINFOW si{};
    si.cb = sizeof si;
    PROCESS_INFORMATION pi{};
    const std::wstring exe = exe_path().wstring();
    if (CreateProcessW(exe.c_str(), cmd.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr, &si, &pi)) {
        CloseHandle(pi.hThread);
        CloseHandle(pi.hProcess);
        std::_Exit(0);
    }
#else
    std::vector<char*> argv;
    for (std::string& a : args) argv.push_back(a.data());
    argv.push_back(nullptr);
    execv(exe_path().c_str(), argv.data());  // (inside an AppImage: its mounted binary, still mounted)
#endif
}

void set_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

void unset_env(const char* name) {
#ifdef _WIN32
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

}  // namespace vc3::platform
