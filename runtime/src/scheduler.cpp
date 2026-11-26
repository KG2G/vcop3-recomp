#include "scheduler.h"

#include <atomic>
#include <chrono>
#include <climits>
#include <condition_variable>
#include <cstdlib>
#include <cstdio>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "vc3/recomp.h"

namespace vc3 {

std::atomic<bool> g_yield_request{false};

namespace sched {
namespace {

// Host threads that aren't guest threads (the kernel worker delivering ISRs and
// DPCs) run above every guest thread, like DISPATCH_LEVEL work on the real kernel.
constexpr int kHostPriority = 64;

// A queued thread's priority is read live from its source, so raising the
// priority of a thread that is already waiting for the lock (CRI boosts its idle
// thread that way) takes effect immediately.
struct Waiter {
    const std::atomic<int>* source;
    int fixed;  // priority for threads without a source
    int64_t seq;  // order within a priority: lower first (front placements are negative)
    bool granted = false;
    int priority() const { return source ? source->load(std::memory_order_relaxed) : fixed; }
};

std::mutex g_m;
std::condition_variable g_cv;
bool g_owned = false;
int g_owner_priority = 0;
int64_t g_seq = 0, g_front_seq = 0;
std::vector<Waiter*> g_waiters;
std::atomic<int> g_best_waiting{INT_MIN};  // highest priority among waiters (INT_MIN: none)
void (*g_hook)() = nullptr;
bool g_fifo = false;  // VC3_SCHED_FIFO=1: ignore priorities (the old behaviour; debugging aid)

thread_local const std::atomic<int>* t_priority = nullptr;

}  // namespace
std::atomic<uint64_t> g_yields{0}, g_refused_irql{0}, g_refused_prio{0};
std::atomic<uint64_t> g_io_ns{0};
namespace {
// VC3_TRACE_SCHED=1: once a second, how long each thread held the guest lock.
bool g_trace = false;
struct HoldStat { int priority; uint64_t ns; };
std::map<const void*, HoldStat> g_hold;  // key: priority source (nullptr: host threads)
thread_local std::chrono::steady_clock::time_point t_acquired;
std::atomic<uint64_t> g_max_hold_ns{0};
std::atomic<int> g_max_hold_prio{0};
void trace_report() {
    static auto last = std::chrono::steady_clock::now();
    const auto now = std::chrono::steady_clock::now();
    if (now - last < std::chrono::seconds(1)) return;
    last = now;
    char hdr[160];
    std::snprintf(hdr, sizeof hdr, "[sched] yields %llu, refused: irql %llu prio %llu; best waiting %d; lock held (ms):",
                  (unsigned long long)g_yields.exchange(0), (unsigned long long)g_refused_irql.exchange(0),
                  (unsigned long long)g_refused_prio.exchange(0), g_best_waiting.load());
    std::string r = hdr;
    for (auto& [k, v] : g_hold) {
        char b[64];
        std::snprintf(b, sizeof b, " %p/p%d=%.0f", k, v.priority, double(v.ns) / 1e6);
        r += b;
        v.ns = 0;
    }
    std::fprintf(stderr, "%s\n", r.c_str());
}

int my_priority() { return g_fifo ? 0 : t_priority ? t_priority->load(std::memory_order_relaxed) : kHostPriority; }

void update_best() {
    int best = INT_MIN;
    for (const Waiter* w : g_waiters) best = std::max(best, g_fifo ? 0 : w->priority());
    g_best_waiting.store(best, std::memory_order_relaxed);
}

}  // namespace

void set_priority_source(const std::atomic<int>* p) { t_priority = p; }

void priority_changed() {
    std::lock_guard lk(g_m);
    update_best();
    if (g_owned && g_best_waiting.load(std::memory_order_relaxed) > g_owner_priority)
        g_yield_request.store(true, std::memory_order_relaxed);
}

static void acquire_raw(bool front = false) {
    const int prio = my_priority();
    std::unique_lock lk(g_m);
    if (!g_owned && g_waiters.empty()) {
        g_owned = true;
        g_owner_priority = prio;
        t_acquired = std::chrono::steady_clock::now();
        return;
    }
    // NT ready-queue placement: a thread preempted by a higher-priority one goes
    // back to the front of its priority's queue; one whose time slice ran out (or
    // that just became ready) goes to the back.
    Waiter w{g_fifo ? nullptr : t_priority, prio, front ? --g_front_seq : g_seq++};
    g_waiters.push_back(&w);
    update_best();
    // A higher-priority thread became ready: ask the running one to yield at its next
    // back-edge instead of waiting for the time slice to end.
    if (g_owned && prio > g_owner_priority) g_yield_request.store(true, std::memory_order_relaxed);
    g_cv.wait(lk, [&] { return w.granted; });
    t_acquired = std::chrono::steady_clock::now();
}

void gil_acquire() {
    acquire_raw();
    if (g_hook) g_hook();
}

// Hand the lock to a higher-priority waiter and get it back ahead of our equals.
static void yield_preempted() {
    gil_release();
    acquire_raw(true);
    if (g_hook) g_hook();
}

void set_acquire_hook(void (*hook)()) { g_hook = hook; }

// Hand the lock to the highest-priority waiter (FIFO among equals), or free it.
void take_max_hold(uint64_t& ns, int& priority) {
    ns = g_max_hold_ns.exchange(0);
    priority = g_max_hold_prio.load();
}

void gil_release() {
    {
        const uint64_t held = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - t_acquired).count());
        if (held > g_max_hold_ns.load(std::memory_order_relaxed)) {
            g_max_hold_ns.store(held, std::memory_order_relaxed);
            g_max_hold_prio.store(my_priority(), std::memory_order_relaxed);
        }
        std::lock_guard lk(g_m);
        if (g_trace) {
            HoldStat& h = g_hold[t_priority];
            h.priority = my_priority();
            h.ns += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t_acquired).count());
            trace_report();
        }
        if (g_waiters.empty()) {
            g_owned = false;
        } else {
            size_t best = 0;
            for (size_t i = 1; i < g_waiters.size(); ++i) {
                const Waiter* a = g_waiters[i];
                const Waiter* b = g_waiters[best];
                const int pa = g_fifo ? 0 : a->priority(), pb = g_fifo ? 0 : b->priority();
                if (pa > pb || (pa == pb && a->seq < b->seq)) best = i;
            }
            Waiter* w = g_waiters[best];
            g_waiters.erase(g_waiters.begin() + std::ptrdiff_t(best));
            update_best();
            w->granted = true;
            g_owned = true;
            g_owner_priority = g_fifo ? 0 : w->priority();
        }
    }
    g_cv.notify_all();
}

void start_preemption() {
    static std::once_flag once;
    std::call_once(once, [] {
        g_fifo = std::getenv("VC3_SCHED_FIFO") != nullptr;
        g_trace = std::getenv("VC3_TRACE_SCHED") != nullptr;
        // Time slice for round-robin among equal-priority threads: about the Xbox
        // kernel's quantum (~20 ms). Higher-priority threads don't wait for it (they
        // preempt at the next back-edge). Much shorter slices let freshly created
        // threads run before their creator can suspend them, which the real kernel
        // wouldn't, and which deadlocked CRI's thread setup. VC3_SLICE_US overrides.
        static const int slice_us = std::getenv("VC3_SLICE_US") ? std::atoi(std::getenv("VC3_SLICE_US")) : 20000;
        std::thread([] {
            for (;;) {
                std::this_thread::sleep_for(std::chrono::microseconds(slice_us));
                if (g_best_waiting.load(std::memory_order_relaxed) != INT_MIN)
                    g_yield_request.store(true, std::memory_order_relaxed);
            }
        }).detach();
    });
}

// A thread of priority `prio` was just made ready (resumed / woken). On the real
// single-core kernel it preempts the caller immediately if it outranks it; here it
// is a host thread that must first queue for the lock, so wait briefly for that
// and then hand over.
void preempt_for(int prio) {
    if (g_fifo || prio <= my_priority()) return;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(5);
    while (g_best_waiting.load(std::memory_order_relaxed) < prio) {
        if (std::chrono::steady_clock::now() > deadline) return;
        std::this_thread::yield();
    }
    yield_preempted();
}

// A strictly higher-priority thread is waiting (preemption, not time-slice end).
bool preempted() { return g_best_waiting.load(std::memory_order_relaxed) > my_priority(); }
void yield_preempted_public() { yield_preempted(); }

bool should_yield() {
    // Time slices only rotate among threads of equal priority; a lower-priority
    // thread never runs while a higher one is ready.
    return g_best_waiting.load(std::memory_order_relaxed) >= my_priority();
}

}  // namespace sched

void yield_point(Cpu& c) {
    // Code at IRQL >= DISPATCH_LEVEL (DPCs, ISRs, KeRaiseIrql sections) is never
    // preempted on the real kernel; that is how drivers protect shared state.
    // The IRQL lives in the KPCR (fs:[0x24]). Keep the request pending for later.
    if (c.fs_base && rd<uint8_t>(c.fs_base + 0x24) >= 2) {
        sched::g_refused_irql.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    g_yield_request.store(false, std::memory_order_relaxed);
    if (!sched::should_yield()) {
        sched::g_refused_prio.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    sched::g_yields.fetch_add(1, std::memory_order_relaxed);
    if (sched::preempted()) {
        sched::yield_preempted_public();
    } else {
        sched::gil_release();
        sched::gil_acquire();
    }
}

}  // namespace vc3
