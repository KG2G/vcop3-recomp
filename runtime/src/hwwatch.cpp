// VC3_HW_WATCH=ADDR[,ADDR...] (hex guest addresses, dev): hardware write breakpoints on guest
// bytes. Every new host instruction address that writes one is logged once (the instruction
// after the write); `addr2line -f -e build/runtime/vc3 0xOFFSET` names the recompiled
// function (OFFSET: the logged address minus the executable's load base, also logged).
// Set up before any guest thread starts: threads created later inherit the breakpoints.
#ifndef _WIN32  // (Linux-only developer tool; Windows gets the empty version below)
#include <linux/hw_breakpoint.h>
#include <linux/perf_event.h>
#include <signal.h>
#include <sys/syscall.h>
#include <ucontext.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "vc3/recomp.h"

namespace vc3::hwwatch {
namespace {
std::atomic<uintptr_t> g_seen[64];
uintptr_t g_base = 0;

void on_trap(int, siginfo_t* si, void* ctx) {
    const auto* uc = static_cast<const ucontext_t*>(ctx);
    const uintptr_t ip = uintptr_t(uc->uc_mcontext.gregs[REG_RIP]);
    for (auto& s : g_seen) {
        uintptr_t cur = s.load();
        if (cur == ip) return;
        if (cur == 0 && s.compare_exchange_strong(cur, ip)) {
            char b[160];
            const int n = std::snprintf(b, sizeof b, "[hwwatch] write to guest %#lx by host ip %#lx (offset %#lx)\n",
                                        (unsigned long)(uintptr_t(si->si_addr) - uintptr_t(g_mem)), (unsigned long)ip,
                                        (unsigned long)(ip - g_base));
            if (n > 0) (void)!write(2, b, size_t(n));
            return;
        }
    }
}
}  // namespace

void init() {
    const char* e = std::getenv("VC3_HW_WATCH");
    if (!e) return;
    // The executable's load base (for addr2line offsets): the first mapping of /proc/self/exe.
    if (std::FILE* f = std::fopen("/proc/self/maps", "r")) {
        char line[512];
        if (std::fgets(line, sizeof line, f)) g_base = uintptr_t(std::strtoull(line, nullptr, 16));
        std::fclose(f);
    }
    struct sigaction sa{};
    sa.sa_sigaction = on_trap;
    sa.sa_flags = SA_SIGINFO | SA_RESTART;
    sigaction(SIGTRAP, &sa, nullptr);
    for (const char* p = e; *p;) {
        char* end;
        const uint32_t a = uint32_t(std::strtoul(p, &end, 16));
        p = *end ? end + 1 : end;
        perf_event_attr attr{};
        attr.type = PERF_TYPE_BREAKPOINT;
        attr.size = sizeof attr;
        attr.bp_type = HW_BREAKPOINT_W;
        attr.bp_addr = uint64_t(uintptr_t(g_mem + a));
        attr.bp_len = HW_BREAKPOINT_LEN_1;
        attr.sample_period = 1;
        attr.exclude_kernel = 1;
        attr.exclude_hv = 1;
        attr.inherit = 1;
        attr.remove_on_exec = 1;
        attr.sigtrap = 1;
        const long fd = syscall(SYS_perf_event_open, &attr, 0, -1, -1, 0);
        std::fprintf(stderr, "[hwwatch] guest %#x: %s (exe base %#lx)\n", a, fd >= 0 ? "watching" : std::strerror(errno),
                     (unsigned long)g_base);
    }
}

}  // namespace vc3::hwwatch

#else
namespace vc3::hwwatch {
void init() {}
}  // namespace vc3::hwwatch
#endif
