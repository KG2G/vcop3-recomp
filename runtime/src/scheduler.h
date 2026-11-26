// Guest thread scheduling.
//
// Guest threads are host threads, but only the holder of the "guest lock"
// runs guest code, like the Chihiro's single-core Pentium III. That keeps
// lifted `lock`-prefixed instructions and guest spinlocks correct without
// atomics.
//
// The lock follows the Xbox kernel's strict priority scheduling: it is handed
// to the highest-priority waiter (FIFO among equals), and a lower-priority
// thread never runs while a higher-priority one is ready. Middleware relies on
// that (CRI's ADX server thread runs time-critical and expects not to be
// interleaved with lower-priority threads mid-update).
//
// Preemption: a ticker sets g_yield_request every few ms while threads are
// queued (immediately when a higher-priority thread becomes ready); lifted code
// checks it on loop back-edges (VC3_BACKEDGE) and calls yield_point(), which
// only yields to an equal- or higher-priority waiter. Kernel waits/sleeps
// release the lock while blocked. VC3_SCHED_FIFO=1 restores plain FIFO.
#pragma once
#include <atomic>
#include <cstdint>

namespace vc3::sched {

void gil_acquire();
// Called after every acquisition (lets the kernel park suspended threads). Set once at startup.
void set_acquire_hook(void (*hook)());
void gil_release();
void start_preemption();  // idempotent
// The calling host thread's scheduling priority (a guest thread's current priority;
// read on every acquisition). Threads without one outrank all guest threads.
void set_priority_source(const std::atomic<int>* priority);
// A thread of equal or higher priority is waiting (time-slice rotation may hand over).
bool should_yield();
// A strictly higher-priority thread is waiting; yield_preempted() hands it the lock and
// gets it back ahead of the caller's equals (NT puts a preempted thread at the front).
bool preempted();
void yield_preempted();
// Give the lock to a waiter of equal or higher priority, if any (Sleep(0) / NtYieldExecution
// / an unsatisfied zero-timeout wait). Never lets a lower-priority thread in.
void yield_to_equals();
// A thread of priority `prio` was just made ready: if it outranks the caller, let it
// run first (as the single-core kernel's immediate preemption would).
void preempt_for(int prio);
// A guest thread's priority changed: re-rank waiters (and preempt if needed).
void priority_changed();
// Handoffs since the last call: how long from being granted the lock to running (for the log).
void take_handoff_stats(double& avg_us, double& max_us, unsigned long long& count);

// Hitch diagnostics: the longest single guest-lock hold since the last call (ns) and
// the holder's priority; and cumulative host file-read time.
void take_max_hold(uint64_t& ns, int& priority);
extern std::atomic<uint64_t> g_io_ns;

// RAII: release the guest lock around a blocking host operation.
struct Unlocked {
    Unlocked() { gil_release(); }
    ~Unlocked() { gil_acquire(); }
    Unlocked(const Unlocked&) = delete;
    Unlocked& operator=(const Unlocked&) = delete;
};

}  // namespace vc3::sched
