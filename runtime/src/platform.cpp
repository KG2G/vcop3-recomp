#include "platform.h"

#include <cstdlib>
#include <thread>

#ifdef _WIN32
#include <io.h>
#include <windows.h>
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

void init_process() {
#ifdef _WIN32
    timeBeginPeriod(1);  // for the life of the process (Windows restores it on exit)
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
