#include "ps.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <list>
#include <mutex>
#include <thread>

#include "kernel.h"
#include "mm.h"
#include "objects.h"
#include "scheduler.h"

namespace vc3::kernel::ps {

namespace {
const bool g_trace_threads = std::getenv("VC3_TRACE_THREADS") != nullptr;

constexpr u32 kThreadExitSentinel = 0xFFFEFFF0;
constexpr u32 kKthreadSize = 0x110;

struct ThreadExit {
    u32 status;
};

struct GuestThread {
    u32 kthread = 0, kpcr = 0, id = 0;
    u32 stack_lo = 0, stack_hi = 0;
    u32 start = 0, context = 0, system_routine = 0;
    int suspend_count = 0;
    std::condition_variable resume_cv;
    // Scheduling priority: process base (8) + base-priority increment, 1..15.
    std::atomic<int> priority{8};
    int increment = 0;
    int cs_held = 0;  // critical sections currently owned (a suspension waits for 0)
};

thread_local GuestThread* t_self = nullptr;
std::mutex g_threads_m;
std::condition_variable g_threads_cv;
std::list<GuestThread> g_threads;
int g_live = 0;
std::atomic<u32> g_next_id{1};

u32 alloc_kpcr(u32 kthread, u32 stack_lo, u32 stack_hi, u8 irql) {
    const u32 pcr = mm::pool_alloc(0x300);
    wr<u32>(pcr + 0x00, 0xFFFFFFFF);  // NtTib.ExceptionList: end of SEH chain
    wr<u32>(pcr + 0x04, stack_hi);    // NtTib.StackBase
    wr<u32>(pcr + 0x08, stack_lo);    // NtTib.StackLimit
    wr<u32>(pcr + 0x18, pcr);         // NtTib.Self
    wr<u32>(pcr + 0x1C, pcr);         // SelfPcr
    wr<u32>(pcr + 0x20, pcr + 0x28);  // Prcb
    wr<u8>(pcr + kKpcrIrql, irql);
    wr<u32>(pcr + 0x28, kthread);     // PrcbData.CurrentThread
    return pcr;
}

GuestThread& create_thread(u32 stack_size, u32 tls_size, u32 start, u32 context, u32 system_routine, bool suspended) {
    const u32 tls = (tls_size + 15) & ~15u;
    const u32 total = ((stack_size ? stack_size : 0x10000) + tls + 0xFFF) & ~0xFFFu;
    const u32 lo = mm::alloc_pages(total, 0x10000, 0xFFFFFFFF, mm::kPage, true);
    const u32 hi = lo + total;

    const u32 kt = alloc_object(kKthreadSize, data_export_addr("PsThreadObjectType"));
    init_dispatcher_header(kt, kThreadObject, kKthreadSize / 4, 0);
    wr<u32>(kt + 0x1C, hi);
    wr<u32>(kt + 0x20, lo);
    wr<u32>(kt + 0x28, tls ? hi - tls : 0);  // TlsData at the top of the stack
    wr<u8>(kt + 0x32, 8);                    // Priority

    std::lock_guard lk(g_threads_m);
    GuestThread& t = g_threads.emplace_back();
    t.kthread = kt;
    t.kpcr = alloc_kpcr(kt, lo, hi, 0);
    t.id = g_next_id++;
    t.stack_lo = lo;
    t.stack_hi = hi - tls;
    t.start = start;
    t.context = context;
    t.system_routine = system_routine;
    t.suspend_count = suspended ? 1 : 0;
    ++g_live;
    return t;
}

GuestThread* find_by_kthread(u32 kt) {
    std::lock_guard lk(g_threads_m);
    for (auto& t : g_threads)
        if (t.kthread == kt) return &t;
    return nullptr;
}

void thread_main(GuestThread* t) {
    t_self = t;
    sched::set_priority_source(&t->priority);
    sched::gil_acquire();  // parks here while created suspended (acquire hook)
    _mm_setcsr(0x1F80);  // host MXCSR = guest reset value

    Cpu c{};
    c.fs_base = t->kpcr;
    c.esp = (t->stack_hi - 16) & ~15u;
    u32 status = 0;
    try {
        if (t->system_routine) {
            push32(c, t->context);
            push32(c, t->start);
            call_indirect(c, t->system_routine, kThreadExitSentinel);
        } else {
            push32(c, t->context);
            call_indirect(c, t->start, kThreadExitSentinel);
        }
    } catch (const ThreadExit& e) {
        status = e.status;
    }
    if (trace_enabled()) std::fprintf(stderr, "[ps] thread %u exited (%08x)\n", t->id, status);

    signal_object(t->kthread, 1);
    sched::gil_release();
    std::lock_guard lk(g_threads_m);
    --g_live;
    g_threads_cv.notify_all();
}

void launch(GuestThread& t) { std::thread(thread_main, &t).detach(); }

// Guest-lock acquire hook: a suspended thread gives the lock back and waits
// until resumed. The suspender always holds the lock, so a thread can only be
// stopped between guest instructions, exactly as on the single-core original.
void park_if_suspended() {
    GuestThread* t = t_self;
    thread_local bool in_park = false;  // re-acquiring the lock below re-runs this hook
    if (!t || in_park) return;
    // A thread suspended while it owns a critical section stops once it has left
    // them all (RtlLeaveCriticalSection parks it). Parking it inside one can block
    // every other thread that needs the lock; CRI suspends its threads at arbitrary
    // points, which on a multi-core host can land inside XAPI's thread-startup lock.
    if (t->cs_held > 0) return;
    in_park = true;
    if (g_trace_threads && t->suspend_count > 0) std::fprintf(stderr, "[thr] %u parks (count %d)\n", t->id, t->suspend_count);
    struct Reset { ~Reset() { in_park = false; } } reset;
    std::unique_lock lk(g_threads_m);
    while (t->suspend_count > 0) {
        lk.unlock();
        sched::gil_release();
        lk.lock();
        t->resume_cv.wait(lk, [&] { return t->suspend_count <= 0; });
        lk.unlock();
        sched::gil_acquire();
        lk.lock();  // loop: the thread may have been suspended again meanwhile
    }
}

}  // namespace

u32 current_kthread() { return t_self ? t_self->kthread : 0; }

void critical_section_held(int delta) {
    GuestThread* t = t_self;
    if (!t) return;
    t->cs_held = std::max(0, t->cs_held + delta);
    if (t->cs_held == 0 && delta < 0 && t->suspend_count > 0) park_if_suspended();  // deferred suspension
}
u32 current_kpcr() { return t_self ? t_self->kpcr : 0; }

void start_boot_thread(u32 entry, u32 stack_size) {
    sched::set_acquire_hook(park_if_suspended);
    sched::start_preemption();
    launch(create_thread(stack_size, 0, entry, 0, 0, false));
}

void wait_all_threads() {
    std::unique_lock lk(g_threads_m);
    g_threads_cv.wait(lk, [] { return g_live == 0; });
}

u32 call_guest_stdcall(u32 fn, std::initializer_list<u32> args, u8 irql) {
    // One helper context per host thread (DPC/ISR dispatcher threads).
    thread_local u32 stack_lo = 0, pcr = 0;
    if (!stack_lo) {
        stack_lo = mm::alloc_pages(0x10000, 0x10000, 0xFFFFFFFF, mm::kPage, true);
        pcr = alloc_kpcr(0, stack_lo, stack_lo + 0x10000, irql);
    }
    wr<u8>(pcr + kKpcrIrql, irql);
    Cpu c{};
    c.fs_base = pcr;
    c.esp = stack_lo + 0x10000 - 16;
    std::vector<u32> v(args);
    for (auto it = v.rbegin(); it != v.rend(); ++it) push32(c, *it);
    call_indirect(c, fn, kThreadExitSentinel);
    return c.eax;
}

// ================================================================ exports

// PsCreateSystemThreadEx(PHANDLE ThreadHandle, ULONG ThreadExtensionSize, ULONG KernelStackSize, ULONG TlsDataSize,
//                        PHANDLE ThreadId, PKSTART_ROUTINE StartRoutine, PVOID StartContext, BOOLEAN CreateSuspended,
//                        BOOLEAN DebuggerThread, PKSYSTEM_ROUTINE SystemRoutine)
KEXPORT(PsCreateSystemThreadEx) {
    GuestThread& t = create_thread(a[2], a[3], a[5], a[6], a[9], (a[7] & 0xFF) != 0);
    if (g_trace_threads)
        std::fprintf(stderr, "[thr] %u created by %u suspended=%d start=%08x\n", t.id, t_self ? t_self->id : 0, (a[7] & 0xFF) != 0, a[5]);
    if (a[0]) wr<u32>(a[0], create_handle(t.kthread));
    if (a[4]) wr<u32>(a[4], t.id);
    if (trace_enabled())
        std::fprintf(stderr, "[ps] thread %u: start=%08x ctx=%08x sysroutine=%08x stack=%08x-%08x tls=%u\n", t.id,
                     t.start, t.context, t.system_routine, t.stack_lo, t.stack_hi, a[3]);
    launch(t);
    return STATUS_SUCCESS;
}

KEXPORT(PsTerminateSystemThread) { throw ThreadExit{a[0]}; }

// NtResumeThread(HANDLE, PULONG PreviousSuspendCount) / NtSuspendThread
KEXPORT(NtResumeThread) {
    GuestThread* t = find_by_kthread(handle_to_object(a[0]));
    if (!t) return STATUS_INVALID_HANDLE;
    bool resumed = false;
    if (g_trace_threads) std::fprintf(stderr, "[thr] %u resumes %u (count %d, prio %d)\n", t_self ? t_self->id : 0, t->id, t->suspend_count, t->priority.load());
    {
        std::lock_guard lk(g_threads_m);
        if (a[1]) wr<u32>(a[1], u32(t->suspend_count));
        if (t->suspend_count > 0 && --t->suspend_count == 0) {
            t->resume_cv.notify_all();
            resumed = true;
        }
    }
    // A resumed thread that outranks us runs right away on the real kernel. CRI
    // relies on it: it boosts a new thread, resumes it and suspends it again, which
    // must not catch the thread halfway through XAPI's startup (holding its lock).
    if (resumed && t != t_self) sched::preempt_for(t->priority.load());
    return STATUS_SUCCESS;
}
KEXPORT(NtSuspendThread) {
    GuestThread* t = find_by_kthread(handle_to_object(a[0]));
    if (!t) return STATUS_INVALID_HANDLE;
    static const bool no_suspend = std::getenv("VC3_NO_SUSPEND") != nullptr;  // debugging aid
    if (g_trace_threads) std::fprintf(stderr, "[thr] %u suspends %u (count %d)\n", t_self ? t_self->id : 0, t->id, t->suspend_count);
    if (no_suspend) { if (a[1]) wr<u32>(a[1], 0); return STATUS_SUCCESS; }
    {
        std::lock_guard lk(g_threads_m);
        if (a[1]) wr<u32>(a[1], u32(t->suspend_count));
        ++t->suspend_count;  // takes effect when the thread next acquires the guest lock
    }
    if (t == t_self) park_if_suspended();  // self-suspension stops right here
    return STATUS_SUCCESS;
}

// KeDelayExecutionThread(KPROCESSOR_MODE, BOOLEAN Alertable, PLARGE_INTEGER Interval)
KEXPORT(KeDelayExecutionThread) {
    const s64 iv = rd<s64>(a[2]);
    const s64 rel = iv < 0 ? -iv : std::max<s64>(0, iv - system_time_100ns());
    if (rel == 0) {  // Sleep(0): only threads of equal or higher priority get to run
        sched::yield_to_equals();
        return STATUS_SUCCESS;
    }
    sched::Unlocked unlocked;
    std::this_thread::sleep_for(std::chrono::nanoseconds(rel * 100));
    return STATUS_SUCCESS;
}

KEXPORT(NtYieldExecution) {
    sched::yield_to_equals();
    return STATUS_SUCCESS;
}

// Base priority = process base (8) + increment, kept within the variable range
// 1..15; an increment of +-16 saturates (THREAD_PRIORITY_TIME_CRITICAL / IDLE).
KEXPORT(KeQueryBasePriorityThread) {
    GuestThread* t = find_by_kthread(a[0]);
    return t ? u32(t->increment) : 0;
}
KEXPORT(KeSetBasePriorityThread) {
    GuestThread* t = find_by_kthread(a[0]);
    if (!t) return 0;
    const int inc = int(s32(a[1]));
    const int old = t->increment;
    t->increment = inc;
    const int prio = inc >= 16 ? 15 : inc <= -16 ? 1 : std::clamp(8 + inc, 1, 15);
    t->priority.store(prio, std::memory_order_relaxed);
    wr<u8>(t->kthread + 0x32, u8(prio));
    sched::priority_changed();
    if (trace_enabled()) std::fprintf(stderr, "[ps] thread %u priority %d (increment %d)\n", t->id, prio, inc);
    // Lowering our own priority below a waiting thread's lets it run now; an equal one
    // waits for the time slice, as on NT. CRI's lock drops back from 15 to the server
    // threads' shared priority on every unlock, thousands of times a second: handing
    // over there let one server thread run inside another's loop (and free a stream it
    // was walking -- a crash right as a mission starts).
    if (t == t_self && sched::preempted()) sched::yield_preempted();
    return u32(old);
}
KEXPORT(KeSetDisableBoostThread) { return 0; }

// IRQL lives in the current KPCR.
static u8 get_irql() { return t_self ? rd<u8>(t_self->kpcr + kKpcrIrql) : 2; }
static void set_irql(u8 v) { if (t_self) wr<u8>(t_self->kpcr + kKpcrIrql, v); }
KEXPORT(KfRaiseIrql) { const u8 o = get_irql(); set_irql(u8(a[0])); return o; }
KEXPORT(KfLowerIrql) { set_irql(u8(a[0])); return 0; }
KEXPORT(KeRaiseIrqlToDpcLevel) { const u8 o = get_irql(); set_irql(2); return o; }

}  // namespace vc3::kernel::ps
