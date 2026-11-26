// Ke*: time, timers, DPCs, interrupts, misc.
#include "platform.h"
#include "ke.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <map>
#include <mutex>
#include <thread>

#include "kernel.h"
#include "objects.h"
#include "ps.h"
#include "scheduler.h"

namespace vc3::kernel::ke {

namespace {

using Clock = std::chrono::steady_clock;

// KDPC { s16 Type; u8 Inserted@2; u8 Importance; LIST_ENTRY@4; DeferredRoutine@0xC; DeferredContext@0x10;
//        SystemArgument1@0x14; SystemArgument2@0x18 }
constexpr u32 kDpcInserted = 0x2, kDpcRoutine = 0xC, kDpcContext = 0x10, kDpcArg1 = 0x14, kDpcArg2 = 0x18;
// KTIMER { DISPATCHER_HEADER; DueTime@0x10; TimerListEntry@0x18; Dpc@0x20; Period@0x24 }
constexpr u32 kTimerInserted = 0x3, kTimerDue = 0x10, kTimerDpc = 0x20, kTimerPeriod = 0x24;
// KINTERRUPT { ServiceRoutine@0; ServiceContext@4; BusInterruptLevel@8; Irql@0xC; Connected@0x10; ... }

std::mutex g_m;
std::condition_variable g_cv;
std::deque<u32> g_dpcs;
std::deque<u32> g_irqs;
// Interrupt latency (queued -> ISR start), reported with VC3_TRACE_SCHED.
std::map<u32, Clock::time_point> g_irq_raised;
u64 g_lat_n = 0, g_lat_over1 = 0, g_lat_over4 = 0;
double g_lat_max_ms = 0, g_lat_sum_ms = 0;
const bool g_lat_trace = std::getenv("VC3_TRACE_SCHED") != nullptr;
std::map<u32, Clock::time_point> g_timers;  // KTIMER -> due
std::map<u32, u32> g_interrupts;            // vector -> KINTERRUPT
const Clock::time_point g_boot = Clock::now();

Clock::time_point due_from(s64 due100ns) {
    const s64 rel = due100ns < 0 ? -due100ns : std::max<s64>(0, due100ns - system_time_100ns());
    return Clock::now() + std::chrono::nanoseconds(rel * 100);
}

void queue_dpc(u32 dpc, u32 arg1, u32 arg2) {
    wr<u8>(dpc + kDpcInserted, 1);
    wr<u32>(dpc + kDpcArg1, arg1);
    wr<u32>(dpc + kDpcArg2, arg2);
    g_dpcs.push_back(dpc);
    g_cv.notify_all();
}

void worker() {
    const u32 tick = data_export_addr("KeTickCount");
    std::unique_lock lk(g_m);
    for (;;) {
        const auto now = Clock::now();
        wr<u32>(tick, u32(std::chrono::duration_cast<std::chrono::milliseconds>(now - g_boot).count()));

        // Expire timers.
        for (auto it = g_timers.begin(); it != g_timers.end();) {
            if (it->second > now) { ++it; continue; }
            const u32 t = it->first;
            signal_object(t, 1);
            const s32 period = rd<s32>(t + kTimerPeriod);
            if (const u32 dpc = rd<u32>(t + kTimerDpc); dpc && !rd<u8>(dpc + kDpcInserted)) {
                const s64 st = system_time_100ns();
                queue_dpc(dpc, u32(st), u32(st >> 32));
            }
            if (period > 0) {
                it->second = now + std::chrono::milliseconds(period);
                ++it;
            } else {
                wr<u8>(t + kTimerInserted, 0);
                it = g_timers.erase(it);
            }
        }

        // Deliver interrupts, then DPCs, under the guest lock.
        while (!g_irqs.empty() || !g_dpcs.empty()) {
            u32 irq = 0, dpc = 0;
            if (!g_irqs.empty()) { irq = g_irqs.front(); g_irqs.pop_front(); }
            else { dpc = g_dpcs.front(); g_dpcs.pop_front(); }
            lk.unlock();
            sched::gil_acquire();
            if (g_lat_trace && irq) {
                std::lock_guard l2(g_m);
                auto r = g_irq_raised.find(irq);
                if (r != g_irq_raised.end()) {
                    const double ms = std::chrono::duration<double, std::milli>(Clock::now() - r->second).count();
                    ++g_lat_n; g_lat_sum_ms += ms; g_lat_max_ms = std::max(g_lat_max_ms, ms);
                    if (ms > 1) ++g_lat_over1;
                    if (ms > 4) ++g_lat_over4;
                    g_irq_raised.erase(r);
                }
                static auto last = Clock::now();
                if (Clock::now() - last > std::chrono::seconds(5)) {
                    last = Clock::now();
                    std::fprintf(stderr, "[ke] interrupt latency: %llu, avg %.2f ms, max %.1f ms, >1 ms %llu, >4 ms %llu\n",
                                 (unsigned long long)g_lat_n, g_lat_n ? g_lat_sum_ms / double(g_lat_n) : 0.0, g_lat_max_ms,
                                 (unsigned long long)g_lat_over1, (unsigned long long)g_lat_over4);
                    g_lat_n = g_lat_over1 = g_lat_over4 = 0;
                    g_lat_sum_ms = g_lat_max_ms = 0;
                }
            }
            if (trace_enabled()) std::fprintf(stderr, "[ke] deliver %s %08x\n", irq ? "ISR" : "DPC", irq ? rd<u32>(irq) : rd<u32>(dpc + kDpcRoutine));
            if (irq) {
                if (rd<u8>(irq + 0x10))
                    ps::call_guest_stdcall(rd<u32>(irq + 0x0), {irq, rd<u32>(irq + 0x4)}, u8(rd<u32>(irq + 0xC)));
            } else {
                wr<u8>(dpc + kDpcInserted, 0);
                ps::call_guest_stdcall(rd<u32>(dpc + kDpcRoutine),
                                       {dpc, rd<u32>(dpc + kDpcContext), rd<u32>(dpc + kDpcArg1), rd<u32>(dpc + kDpcArg2)}, 2);
            }
            sched::gil_release();
            lk.lock();
        }

        auto next = Clock::now() + std::chrono::milliseconds(1);
        for (const auto& [t, due] : g_timers) next = std::min(next, due);
        if (platform::coarse_sleep()) next += std::chrono::microseconds(1500);  // (testing: Windows-like)
        g_cv.wait_until(lk, next);
    }
}

}  // namespace

void start_worker() {
    static std::once_flag once;
    std::call_once(once, [] { std::thread(worker).detach(); });
}

void raise_interrupt(u32 vector) {
    std::lock_guard lk(g_m);
    auto it = g_interrupts.find(vector);
    if (it == g_interrupts.end()) return;
    // Level-triggered lines may be raised repeatedly while asserted: one pending
    // delivery per interrupt object is enough (the ISR reads the cause registers).
    if (std::find(g_irqs.begin(), g_irqs.end(), it->second) != g_irqs.end()) return;
    g_irqs.push_back(it->second);
    if (g_lat_trace) g_irq_raised[it->second] = Clock::now();
    g_cv.notify_all();
}

// ================================================================ exports
// --- time
KEXPORT(KeQuerySystemTime) { wr<s64>(a[0], system_time_100ns()); return 0; }
KEXPORT(KeQueryInterruptTime) {
    const u64 t = u64(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - g_boot).count() / 100);
    c.edx = u32(t >> 32);
    return u32(t);
}
constexpr u64 kAcpiFrequency = 3375000;  // Xbox ACPI timer
KEXPORT(KeQueryPerformanceCounter) {
    const u64 ns = u64(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - g_boot).count());
    const u64 t = u64((unsigned __int128)ns * kAcpiFrequency / 1000000000u);
    c.edx = u32(t >> 32);
    return u32(t);
}
KEXPORT(KeQueryPerformanceFrequency) { c.edx = 0; return u32(kAcpiFrequency); }

KEXPORT(KeStallExecutionProcessor) {
    const auto until = Clock::now() + std::chrono::microseconds(a[0]);
    while (Clock::now() < until) {}
    return 0;
}

// --- DPCs
// KeInitializeDpc(PKDPC Dpc, PKDEFERRED_ROUTINE DeferredRoutine, PVOID DeferredContext)
KEXPORT(KeInitializeDpc) {
    wr<s16>(a[0], 19);  // DpcObject
    wr<u8>(a[0] + kDpcInserted, 0);
    wr<u32>(a[0] + kDpcRoutine, a[1]);
    wr<u32>(a[0] + kDpcContext, a[2]);
    return 0;
}
// KeInsertQueueDpc(PRKDPC Dpc, PVOID SystemArgument1, PVOID SystemArgument2)
KEXPORT(KeInsertQueueDpc) {
    std::lock_guard lk(g_m);
    if (rd<u8>(a[0] + kDpcInserted)) return 0;
    queue_dpc(a[0], a[1], a[2]);
    return 1;
}
KEXPORT(KeRemoveQueueDpc) {
    std::lock_guard lk(g_m);
    if (!rd<u8>(a[0] + kDpcInserted)) return 0;
    std::erase(g_dpcs, a[0]);
    wr<u8>(a[0] + kDpcInserted, 0);
    return 1;
}

// --- timers
// KeInitializeTimerEx(PKTIMER Timer, TIMER_TYPE Type)
KEXPORT(KeInitializeTimerEx) {
    init_dispatcher_header(a[0], u8(kTimerNotification + (a[1] ? 1 : 0)), 10, 0);
    wr<u64>(a[0] + kTimerDue, 0);
    wr<u32>(a[0] + kTimerDpc, 0);
    wr<s32>(a[0] + kTimerPeriod, 0);
    return 0;
}
static u32 set_timer(u32 timer, s64 due, s32 period, u32 dpc) {
    std::lock_guard lk(g_m);
    const bool was = g_timers.erase(timer) != 0;
    {
        std::lock_guard klk(kmutex());
        wr<s32>(timer + kHdrSignal, 0);
    }
    wr<u32>(timer + kTimerDpc, dpc);
    wr<s32>(timer + kTimerPeriod, period);
    wr<u8>(timer + kTimerInserted, 1);
    wr<s64>(timer + kTimerDue, due);
    g_timers[timer] = due_from(due);
    g_cv.notify_all();
    return was;
}
// KeSetTimer(PKTIMER, LARGE_INTEGER DueTime, PKDPC)          [DueTime occupies a[1..2]]
KEXPORT(KeSetTimer) { return set_timer(a[0], s64(u64(a[1]) | u64(a[2]) << 32), 0, a[3]); }
// KeSetTimerEx(PKTIMER, LARGE_INTEGER DueTime, LONG Period, PKDPC)
KEXPORT(KeSetTimerEx) { return set_timer(a[0], s64(u64(a[1]) | u64(a[2]) << 32), s32(a[3]), a[4]); }
KEXPORT(KeCancelTimer) {
    std::lock_guard lk(g_m);
    wr<u8>(a[0] + kTimerInserted, 0);
    return g_timers.erase(a[0]) != 0;
}

// NtCreateTimer(PHANDLE, POBJECT_ATTRIBUTES, TIMER_TYPE)
KEXPORT(NtCreateTimer) {
    const u32 t = alloc_object(0x28, data_export_addr("ExTimerObjectType"));
    init_dispatcher_header(t, u8(kTimerNotification + (a[2] ? 1 : 0)), 10, 0);
    wr<u32>(a[0], create_handle(t));
    return STATUS_SUCCESS;
}
// NtSetTimerEx(HANDLE, PLARGE_INTEGER DueTime, PTIMER_APC_ROUTINE, KPROCESSOR_MODE, PVOID TimerContext,
//              BOOLEAN ResumeTimer, LONG Period, PBOOLEAN PreviousState)
KEXPORT(NtSetTimerEx) {
    const u32 t = handle_to_object(a[0]);
    if (!t) return STATUS_INVALID_HANDLE;
    if (a[2]) std::fprintf(stderr, "[ke] NtSetTimerEx with an APC routine (%08x) is not supported\n", a[2]);
    const u32 prev = rd<s32>(t + kHdrSignal) != 0;
    set_timer(t, rd<s64>(a[1]), s32(a[6]), 0);
    if (a[7]) wr<u8>(a[7], u8(prev));
    return STATUS_SUCCESS;
}
// NtCancelTimer(HANDLE, PBOOLEAN CurrentState)
KEXPORT(NtCancelTimer) {
    const u32 t = handle_to_object(a[0]);
    if (!t) return STATUS_INVALID_HANDLE;
    {
        std::lock_guard lk(g_m);
        g_timers.erase(t);
    }
    if (a[1]) wr<u8>(a[1], u8(rd<s32>(t + kHdrSignal) != 0));
    return STATUS_SUCCESS;
}

// --- interrupts
// HalGetInterruptVector(ULONG BusInterruptLevel, PKIRQL Irql)
KEXPORT(HalGetInterruptVector) {
    if (a[1]) wr<u8>(a[1], u8(0x1B - std::min<u32>(a[0], 0x1B)));
    return 0x30 + a[0];
}
// KeInitializeInterrupt(PKINTERRUPT, PKSERVICE_ROUTINE, PVOID ServiceContext, ULONG Vector, KIRQL Irql,
//                       KINTERRUPT_MODE, BOOLEAN ShareVector)
KEXPORT(KeInitializeInterrupt) {
    const u32 i = a[0];
    wr<u32>(i + 0x00, a[1]);
    wr<u32>(i + 0x04, a[2]);
    wr<u32>(i + 0x08, a[3]);        // vector (we store it where BusInterruptLevel lives)
    wr<u32>(i + 0x0C, a[4] & 0xFF); // Irql
    wr<u8>(i + 0x10, 0);            // Connected
    wr<u8>(i + 0x11, u8(a[6]));
    wr<u8>(i + 0x12, u8(a[5]));
    return 0;
}
KEXPORT(KeConnectInterrupt) {
    std::lock_guard lk(g_m);
    if (rd<u8>(a[0] + 0x10)) return 0;
    wr<u8>(a[0] + 0x10, 1);
    g_interrupts[rd<u32>(a[0] + 0x08)] = a[0];
    if (trace_enabled())
        std::fprintf(stderr, "[ke] ISR %08x connected to vector %#x\n", rd<u32>(a[0]), rd<u32>(a[0] + 0x08));
    return 1;
}
KEXPORT(KeDisconnectInterrupt) {
    std::lock_guard lk(g_m);
    if (!rd<u8>(a[0] + 0x10)) return 0;
    wr<u8>(a[0] + 0x10, 0);
    g_interrupts.erase(rd<u32>(a[0] + 0x08));
    return 1;
}
// KeSynchronizeExecution(PKINTERRUPT, PKSYNCHRONIZE_ROUTINE, PVOID Context): run the routine
// directly on the caller's stack (the guest lock already excludes the ISR).
KEXPORT(KeSynchronizeExecution) {
    const u32 routine = a[1], ctx = a[2];
    const u32 ret_addr = rd<u32>(c.esp);
    // Temporarily step below our own frame so the callee can use the stack.
    const u32 saved_esp = c.esp;
    push32(c, ctx);
    call_indirect(c, routine, ret_addr);
    c.esp = saved_esp;
    return c.eax & 0xFF;
}

// --- misc
KEXPORT(KeSaveFloatingPointState) { return STATUS_SUCCESS; }
KEXPORT(KeRestoreFloatingPointState) { return STATUS_SUCCESS; }
KEXPORT(KeBugCheck) {
    std::fprintf(stderr, "[ke] KeBugCheck(%08x) from %08x\n", a[0], rd<u32>(c.esp));
    std::abort();
}

}  // namespace vc3::kernel::ke
