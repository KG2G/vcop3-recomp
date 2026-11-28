#include "pause.h"

#include <atomic>
#include <cstdio>

#include "audio/apu.h"
#include "kernel/ke.h"
#include "kernel/ps.h"
#include "log.h"

namespace vc3::pause {

namespace {
std::atomic<bool> g_active{false};
}

bool active() { return g_active.load(std::memory_order_relaxed); }

void set(bool paused) {
    if (paused == active()) return;
    if (paused) {
        kernel::ke::pause_clock(true);  // clocks first: nothing that still runs sees time pass
        g_active = true;
        audio::apu::set_paused(true);
        kernel::ps::pause_threads(true);
    } else {
        kernel::ps::pause_threads(false);
        audio::apu::set_paused(false);
        g_active = false;
        kernel::ke::pause_clock(false);
    }
    std::fprintf(stderr, "[pause] %.1f s: %s\n", log::uptime(), paused ? "paused" : "resumed");
}

}  // namespace vc3::pause
