// Sampling profiler (VC3_PROFILE=from:to, in vblanks; "1" = whole run).
//
// ITIMER_PROF fires SIGPROF every 1 ms of process CPU time, on whichever thread is
// running; the handler records the interrupted instruction pointer (and the
// thread). Every 10 s, and when the window closes, work/profile.txt gets a
// histogram of program-relative addresses; tools/profile.py (or addr2line -f)
// turns them into function names.
#ifndef _WIN32  // (Linux-only developer tool; Windows gets the empty version below)
#include "profiler.h"

#include <signal.h>
#include <sys/time.h>
#include <ucontext.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace vc3::profiler {
namespace {

constexpr size_t kRing = 1 << 20;
uintptr_t g_pc[kRing];
std::atomic<size_t> g_head{0};
std::atomic<bool> g_on{false};
uintptr_t g_base = 0;  // load address of the executable
long g_from = -1, g_to = -1;

void on_sigprof(int, siginfo_t*, void* uc) {
    if (!g_on.load(std::memory_order_relaxed)) return;
    const auto* ctx = static_cast<const ucontext_t*>(uc);
    const size_t i = g_head.fetch_add(1, std::memory_order_relaxed);
    if (i < kRing) g_pc[i] = uintptr_t(ctx->uc_mcontext.gregs[REG_RIP]);
}

uintptr_t exe_base() {
    // First mapping of our own executable in /proc/self/maps.
    char exe[4096];
    const ssize_t n = readlink("/proc/self/exe", exe, sizeof exe - 1);
    if (n <= 0) return 0;
    exe[n] = 0;
    std::FILE* f = std::fopen("/proc/self/maps", "r");
    if (!f) return 0;
    char line[8192];
    uintptr_t base = 0;
    while (std::fgets(line, sizeof line, f))
        if (std::strstr(line, exe)) {
            base = std::strtoull(line, nullptr, 16);
            break;
        }
    std::fclose(f);
    return base;
}

}  // namespace

void init() {
    const char* spec = std::getenv("VC3_PROFILE");
    if (!spec) return;
    if (std::strchr(spec, ':')) {
        g_from = std::atol(spec);
        g_to = std::atol(std::strchr(spec, ':') + 1);
    } else {
        g_from = 0;
        g_to = -1;
    }
    g_base = exe_base();
    struct sigaction sa{};
    sa.sa_sigaction = on_sigprof;
    sa.sa_flags = SA_SIGINFO | SA_RESTART;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGPROF, &sa, nullptr);
    itimerval t{};
    t.it_interval.tv_usec = 1000;
    t.it_value.tv_usec = 1000;
    setitimer(ITIMER_PROF, &t, nullptr);
    std::fprintf(stderr, "[profile] sampling vblanks %ld..%ld, exe base %#lx\n", g_from, g_to, (unsigned long)g_base);
}

void dump() {
    if (g_from < 0) return;
    const size_t n = std::min(g_head.load(), kRing);
    std::unordered_map<uintptr_t, size_t> hist;
    for (size_t i = 0; i < n; ++i) ++hist[g_pc[i] - g_base];
    std::vector<std::pair<uintptr_t, size_t>> v(hist.begin(), hist.end());
    std::sort(v.begin(), v.end(), [](auto& a, auto& b) { return a.second > b.second; });
    std::filesystem::create_directories("work");
    std::FILE* f = std::fopen("work/profile.txt", "w");
    if (!f) return;
    std::fprintf(f, "# %zu samples (1 ms CPU each); offset count\n", n);
    for (auto& [pc, c] : v) std::fprintf(f, "%#lx %zu\n", (unsigned long)pc, c);
    std::fclose(f);
}

void vblank(long frame) {
    if (g_from < 0) return;
    const bool on = frame >= g_from && (g_to < 0 || frame < g_to);
    if (on != g_on.load(std::memory_order_relaxed)) {
        g_on.store(on, std::memory_order_relaxed);
        if (!on) {
            dump();
            std::fprintf(stderr, "[profile] wrote work/profile.txt (%zu samples)\n", std::min(g_head.load(), kRing));
        }
    } else if (on && frame % 600 == 0) {
        dump();
    }
}

}  // namespace vc3::profiler

#else
#include "profiler.h"
namespace vc3::profiler {
void init() {}
void vblank(long) {}
void dump() {}
}  // namespace vc3::profiler
#endif
