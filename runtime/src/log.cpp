#include "log.h"

#include <SDL3/SDL.h>
#include <fcntl.h>

#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>

#include "platform.h"
#include "privacy.h"
#include "settings.h"

#ifdef _WIN32
#include <windows.h>
#include <intrin.h>
#else
#include <cpuid.h>
#include <execinfo.h>
#include <sys/utsname.h>
#include <unistd.h>
#endif

#ifndef VC3_VERSION
#define VC3_VERSION "dev"
#endif

namespace vc3::log {
namespace {
const auto g_start = std::chrono::steady_clock::now();
std::string g_folder;

#ifndef _WIN32
// Linux: every write goes to the terminal as is and to the file scrubbed, straight
// through (no buffering), so the lines before a crash land too.
struct Sink { int term, file; };
ssize_t sink_write(void* c, const char* buf, size_t n) {
    const auto* s = static_cast<const Sink*>(c);
    if (write(s->term, buf, n) < 0) {}
    const std::string clean = privacy::scrub(std::string(buf, n));
    if (write(s->file, clean.data(), clean.size()) < 0) {}
    return ssize_t(n);
}
#endif

std::string cpu_name() {
    unsigned r[12] = {};
#ifdef _WIN32
    int regs[4];
    __cpuid(regs, 0x80000000);
    if (unsigned(regs[0]) < 0x80000004u) return "unknown CPU";
    for (unsigned i = 0; i < 3; ++i) {
        __cpuid(regs, int(0x80000002u + i));
        std::memcpy(&r[i * 4], regs, 16);
    }
#else
    unsigned a, b, c, d;
    if (!__get_cpuid(0x80000000u, &a, &b, &c, &d) || a < 0x80000004u) return "unknown CPU";
    for (unsigned i = 0; i < 3; ++i) __get_cpuid(0x80000002u + i, &r[i * 4], &r[i * 4 + 1], &r[i * 4 + 2], &r[i * 4 + 3]);
#endif
    std::string s(reinterpret_cast<const char*>(r), 48);
    s = s.c_str();
    const size_t b0 = s.find_first_not_of(' ');
    return b0 == std::string::npos ? "unknown CPU" : s.substr(b0);
}

std::string os_name() {
#ifdef _WIN32
    using RtlGetVersionFn = LONG(WINAPI*)(OSVERSIONINFOW*);
    OSVERSIONINFOW v{sizeof v};
    if (HMODULE nt = GetModuleHandleA("ntdll.dll")) {
        if (auto f = reinterpret_cast<RtlGetVersionFn>(reinterpret_cast<void*>(GetProcAddress(nt, "RtlGetVersion")))) f(&v);
        std::string s = "Windows " + std::to_string(v.dwMajorVersion) + "." + std::to_string(v.dwMinorVersion) +
                        " build " + std::to_string(v.dwBuildNumber);
        using WineFn = const char*(CDECL*)();
        if (auto w = reinterpret_cast<WineFn>(reinterpret_cast<void*>(GetProcAddress(nt, "wine_get_version"))))
            s += std::string(" (Wine ") + w() + ")";
        return s;
    }
    return "Windows";
#else
    std::string name = "Linux";
    std::ifstream f("/etc/os-release");
    for (std::string line; std::getline(f, line);)
        if (line.rfind("PRETTY_NAME=", 0) == 0) {
            name = line.substr(12);
            if (name.size() >= 2 && name.front() == '"') name = name.substr(1, name.size() - 2);
        }
    utsname u{};
    if (uname(&u) == 0) name += std::string(", kernel ") + u.release;
    return name;
#endif
}

void crash(const char* what) {
    static volatile std::sig_atomic_t once = 0;
    if (once) return;
    once = 1;
    std::fprintf(stderr, "[crash] %s after %.1f s\n", what, uptime());
#ifndef _WIN32
    void* bt[32];
    const int n = backtrace(bt, 32);
    if (char** sy = backtrace_symbols(bt, n)) {
        for (int i = 1; i < n; ++i) {
            // Only the function part: the binary's own path can be personal.
            std::string s = sy[i];
            const size_t slash = s.rfind('/', s.find('('));
            if (slash != std::string::npos) s = s.substr(slash + 1);
            std::fprintf(stderr, "    %s\n", s.c_str());
        }
    }
#endif
    std::fflush(nullptr);
}

void on_signal(int sig) {
    crash(sig == SIGABRT ? "aborted" : sig == SIGFPE ? "arithmetic error" : sig == SIGILL ? "illegal instruction"
                                                                                           : "signal");
    std::signal(sig, SIG_DFL);
    std::raise(sig);
}

#ifdef _WIN32
LONG WINAPI on_exception(EXCEPTION_POINTERS* e) {
    char what[160];
    const auto addr = reinterpret_cast<uintptr_t>(e->ExceptionRecord->ExceptionAddress);
    HMODULE mod = nullptr;
    GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       reinterpret_cast<LPCSTR>(addr), &mod);
    char name[MAX_PATH] = "?";
    if (mod) GetModuleFileNameA(mod, name, MAX_PATH);
    const char* base = std::strrchr(name, '\\');
    std::snprintf(what, sizeof what, "exception %08lx at %s+0x%llx", e->ExceptionRecord->ExceptionCode,
                  base ? base + 1 : name, (unsigned long long)(addr - reinterpret_cast<uintptr_t>(mod)));
    crash(what);
    return EXCEPTION_CONTINUE_SEARCH;
}
#endif
}  // namespace

std::string folder() { return g_folder; }

double uptime() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - g_start).count();
}

void start(const std::string& dir) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path log = fs::path(dir) / "vc3.log";
    g_folder = fs::absolute(dir, ec).string();
    fs::rename(log, fs::path(dir) / "vc3.prev.log", ec);
#ifdef _WIN32
    // Windows: a GUI program without a console; both streams append to the file.
    const std::string p = log.string();
    if (std::FILE* t = std::fopen(p.c_str(), "w")) std::fclose(t);
    if (std::freopen(p.c_str(), "a", stderr)) std::setvbuf(stderr, nullptr, _IONBF, 0);
    if (FILE* o = std::freopen(p.c_str(), "a", stdout)) std::setvbuf(o, nullptr, _IONBF, 0);
    SetUnhandledExceptionFilter(on_exception);
#else
    const int file = open(log.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_APPEND, 0644);
    if (file >= 0) {
        static Sink err{dup(STDERR_FILENO), file}, out{dup(STDOUT_FILENO), file};
        cookie_io_functions_t io{nullptr, sink_write, nullptr, nullptr};
        if (FILE* e = fopencookie(&err, "w", io)) {
            std::setvbuf(e, nullptr, _IONBF, 0);
            stderr = e;
        }
        if (FILE* o = fopencookie(&out, "w", io)) {
            std::setvbuf(o, nullptr, _IOLBF, 0);
            stdout = o;
        }
    }
#endif
    for (int sig : {SIGABRT, SIGFPE, SIGILL}) std::signal(sig, on_signal);
}

void header() {
    std::fprintf(stderr, "[log] vcop3-recomp %s, %s\n", VC3_VERSION, platform::kVulkanOnly ? "Windows build" : "Linux build");
    std::fprintf(stderr, "[log] system: %s\n", os_name().c_str());
    std::fprintf(stderr, "[log] CPU: %s, %u threads; RAM: %.0f GB\n", cpu_name().c_str(), std::thread::hardware_concurrency(),
                 double(SDL_GetSystemRAM()) / 1024.0);
    if (const char* v = SDL_GetCurrentVideoDriver()) std::fprintf(stderr, "[log] video: %s\n", v);
    std::string s;
    for (const auto& o : settings::options()) {
        std::string v = settings::get(o.key);
        if (!std::strcmp(o.key, "install_dir")) v = v.empty() ? "default" : "custom";  // (no paths)
        if (!std::strcmp(o.key, "renderer") && platform::kVulkanOnly) v = "vulkan";  // (the only one there)
        s += std::string(" ") + o.key + "=" + v;
    }
    std::fprintf(stderr, "[log] settings:%s\n", s.c_str());
    std::fprintf(stderr, "[log] times below are seconds since start; no names, paths or clock times are logged\n");
}

void crashed(const char* what) { crash(what); }

void end(const char* how) {
    std::fprintf(stderr, "[log] %s after %.0f s\n", how, uptime());
    std::fflush(nullptr);
}

}  // namespace vc3::log

