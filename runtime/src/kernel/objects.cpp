#include "platform.h"
#include "objects.h"

#include <chrono>
#include <string>
#include <map>
#include <unordered_map>

#include "mm.h"
#include "ps.h"
#include "scheduler.h"

namespace vc3::kernel {

namespace {
std::mutex g_kmutex;
// Threads blocked in wait_for_objects, each woken through its own wake-up object when any
// object is signalled (they re-check their own objects). Faster than one shared condition
// variable on Windows, where a thread wake-up costs much more (see scheduler.cpp).
#ifdef _WIN32
thread_local void* t_kwake = platform::make_event();
std::vector<void*> g_kwaiting;  // guarded by g_kmutex
#else
thread_local std::condition_variable t_kwake;
std::vector<std::condition_variable*> g_kwaiting;  // guarded by g_kmutex
#endif
std::unordered_map<u32, std::pair<u32, std::shared_ptr<HostObject>>> g_handles;
u32 g_next_handle = 0x100;
}  // namespace

std::mutex& kmutex() { return g_kmutex; }

// Release generations for notification events (guarded by g_kmutex). KeSetEvent and
// KePulseEvent release the threads waiting at that moment, whatever happens to the
// signal state afterwards: waiters note each object's generation when they block,
// and a changed generation satisfies them.
std::unordered_map<u32, u64> g_pulse_gen;
std::unordered_map<u32, int> g_blocked;  // threads currently blocked on each object
void notify_waiters() {
#ifdef _WIN32
    void* wake[64];
    size_t n = 0;
    {
        std::lock_guard lk(g_kmutex);
        for (void* h : g_kwaiting) if (n < 64) wake[n++] = h;
    }
    for (size_t i = 0; i < n; ++i) platform::set_event(wake[i]);  // (the events live as long as their threads)
#else
    std::lock_guard lk(g_kmutex);
    for (std::condition_variable* cv : g_kwaiting) cv->notify_one();  // under the lock: waiters can't go meanwhile
#endif
}

void init_dispatcher_header(u32 obj, u8 type, u8 size_dwords, s32 signal) {
    wr<u8>(obj + kHdrType, type);
    wr<u8>(obj + 1, 0);
    wr<u8>(obj + kHdrSize, size_dwords);
    wr<u8>(obj + 3, 0);
    wr<s32>(obj + kHdrSignal, signal);
    wr<u32>(obj + 8, obj + 8);  // WaitListHead: empty list points at itself
    wr<u32>(obj + 12, obj + 8);
}

u32 alloc_object(u32 body_size, u32 type_object) {
    const u32 hdr = mm::pool_alloc(0x10 + body_size);
    if (!hdr) return 0;
    wr<u32>(hdr + 0x0, 1);  // PointerCount
    wr<u32>(hdr + 0x4, 0);  // HandleCount
    wr<u32>(hdr + 0x8, type_object);
    wr<u32>(hdr + 0xC, 0);
    return hdr + 0x10;
}

u32 object_type_of(u32 body) { return rd<u32>(body - 0x10 + 0x8); }

u32 create_handle(u32 body, std::shared_ptr<HostObject> host) {
    std::lock_guard lk(g_kmutex);
    const u32 h = g_next_handle;
    g_next_handle += 4;
    g_handles[h] = {body, std::move(host)};
    return h;
}

u32 handle_to_object(u32 handle) {
    if (handle == kCurrentThreadHandle) return ps::current_kthread();
    std::lock_guard lk(g_kmutex);
    auto it = g_handles.find(handle);
    return it == g_handles.end() ? 0 : it->second.first;
}

std::shared_ptr<HostObject> handle_host(u32 handle) {
    std::lock_guard lk(g_kmutex);
    auto it = g_handles.find(handle);
    return it == g_handles.end() ? nullptr : it->second.second;
}

bool close_handle(u32 handle) {
    if (handle == kCurrentThreadHandle) return true;
    std::lock_guard lk(g_kmutex);
    return g_handles.erase(handle) != 0;
}

std::optional<s64> read_timeout(u32 p) {
    if (!p) return std::nullopt;
    return rd<s64>(p);
}

// --- dispatcher logic (kmutex held) ---
static bool is_signaled(u32 obj, u32 thread) {
    const u8 type = rd<u8>(obj + kHdrType);
    const s32 sig = rd<s32>(obj + kHdrSignal);
    if (type == kMutantObject) return sig > 0 || rd<u32>(obj + kMutOwner) == thread;
    return sig > 0;
}

static void satisfy(u32 obj, u32 thread) {
    switch (rd<u8>(obj + kHdrType)) {
        case kEventSynchronization: case kTimerSynchronization: wr<s32>(obj + kHdrSignal, 0); break;
        case kSemaphoreObject: wr<s32>(obj + kHdrSignal, rd<s32>(obj + kHdrSignal) - 1); break;
        case kMutantObject:
            wr<s32>(obj + kHdrSignal, rd<s32>(obj + kHdrSignal) - 1);
            wr<u32>(obj + kMutOwner, thread);
            break;
        default: break;
    }
}

u32 wait_for_objects(const std::vector<u32>& objs, bool wait_all, std::optional<s64> timeout) {
    using namespace std::chrono;
    const u32 me = ps::current_kthread();
    std::optional<steady_clock::time_point> deadline;
    if (timeout) {
        s64 rel = *timeout < 0 ? -*timeout : *timeout - system_time_100ns();
        if (rel < 0) rel = 0;
        deadline = steady_clock::now() + nanoseconds(rel * 100);
    }

    std::unique_lock lk(g_kmutex);
    auto try_satisfy = [&]() -> std::optional<u32> {
        if (wait_all) {
            for (u32 o : objs)
                if (!is_signaled(o, me)) return std::nullopt;
            for (u32 o : objs) satisfy(o, me);
            return STATUS_SUCCESS;
        }
        for (size_t i = 0; i < objs.size(); ++i)
            if (is_signaled(objs[i], me)) {
                satisfy(objs[i], me);
                return u32(i);  // STATUS_WAIT_0 + i
            }
        return std::nullopt;
    };

    if (auto r = try_satisfy()) return *r;
    if (deadline && *deadline <= steady_clock::now()) {
        lk.unlock();
        sched::yield_to_equals();  // a zero-timeout poll lets equals run, never lower priorities
        return STATUS_TIMEOUT;
    }

    // Blocking: record pulse generations so a pulse during the wait counts.
    std::vector<u64> gen0(objs.size());
    for (size_t i = 0; i < objs.size(); ++i) {
        gen0[i] = g_pulse_gen[objs[i]];
        ++g_blocked[objs[i]];
    }
    auto pulsed = [&]() -> std::optional<u32> {
        if (wait_all) return std::nullopt;
        for (size_t i = 0; i < objs.size(); ++i)
            if (g_pulse_gen[objs[i]] != gen0[i]) return u32(i);
        return std::nullopt;
    };
    struct Unblock {
        const std::vector<u32>& objs;
        ~Unblock() { for (u32 o : objs) if (--g_blocked[o] <= 0) g_blocked.erase(o); }
    };

    sched::gil_release();
    std::optional<u32> result;
    {
    Unblock unblock{objs};
#ifdef _WIN32
    g_kwaiting.push_back(t_kwake);
    struct Unregister { ~Unregister() { std::erase(g_kwaiting, t_kwake); } } unregister;  // (lk held at exit)
#else
    g_kwaiting.push_back(&t_kwake);
    struct Unregister { ~Unregister() { std::erase(g_kwaiting, &t_kwake); } } unregister;
#endif
    for (;;) {
        if ((result = try_satisfy()) || (result = pulsed())) break;
        if (deadline && steady_clock::now() >= *deadline) {
            result = STATUS_TIMEOUT;
            break;
        }
#ifdef _WIN32
        unsigned ms = ~0u;
        if (deadline) ms = unsigned(std::max<s64>(0, (duration_cast<microseconds>(*deadline - steady_clock::now()).count() + 999) / 1000));
        lk.unlock();
        platform::wait_event(t_kwake, ms);
        lk.lock();
#else
        if (deadline) t_kwake.wait_until(lk, platform::coarse_sleep() ? *deadline + std::chrono::microseconds(1500) : *deadline);
        else t_kwake.wait(lk);
#endif
    }
    }
    lk.unlock();
    sched::gil_acquire();
    return *result;
}

s32 set_event(u32 ev) {
    s32 prev;
    {
        std::lock_guard lk(g_kmutex);
        prev = rd<s32>(ev + kHdrSignal);
        wr<s32>(ev + kHdrSignal, 1);
        // Setting a notification event releases every thread waiting on it right
        // now, even if one of them (or anyone) resets it before the others run.
        // D3D's BlockUntilVerticalBlank resets the vblank event inline and waits;
        // without this the first waiter to wake re-armed it and the rest (the
        // game's main loop among them) slept through the vblank.
        if (rd<u8>(ev) == kEventNotification && g_blocked.count(ev)) ++g_pulse_gen[ev];
    }
    notify_waiters();
    return prev;
}

s32 reset_event(u32 ev) {
    std::lock_guard lk(g_kmutex);
    const s32 prev = rd<s32>(ev + kHdrSignal);
    wr<s32>(ev + kHdrSignal, 0);
    return prev;
}

void signal_object(u32 obj, s32 state) {
    {
        std::lock_guard lk(g_kmutex);
        wr<s32>(obj + kHdrSignal, state);
    }
    notify_waiters();
}

// ================================================================ exports
// Events
// KeInitializeEvent(PRKEVENT Event, EVENT_TYPE Type, BOOLEAN State)
KEXPORT(KeInitializeEvent) {
    init_dispatcher_header(a[0], u8(a[1] ? kEventSynchronization : kEventNotification), 4, s32(a[2] & 0xFF));
    return 0;
}
KEXPORT(KeSetEvent) { return u32(set_event(a[0])); }
KEXPORT(KeResetEvent) { return u32(reset_event(a[0])); }
// KePulseEvent: release the current waiters, then leave the event unsignalled.
KEXPORT(KePulseEvent) {
    s32 prev;
    {
        std::lock_guard lk(g_kmutex);
        const u32 ev = a[0];
        prev = rd<s32>(ev + kHdrSignal);
        const bool waiters = g_blocked.count(ev) != 0;
        if (rd<u8>(ev) == kEventSynchronization) {
            // One waiter is released: it consumes the signal when it wakes.
            wr<s32>(ev + kHdrSignal, waiters ? 1 : 0);
        } else {
            if (waiters) ++g_pulse_gen[ev];
            wr<s32>(ev + kHdrSignal, 0);
        }
    }
    notify_waiters();
    return u32(prev);
}

// NtCreateEvent(PHANDLE EventHandle, POBJECT_ATTRIBUTES, EVENT_TYPE EventType, BOOLEAN InitialState)
KEXPORT(NtCreateEvent) {
    const u32 ev = alloc_object(0x10, data_export_addr("ExEventObjectType"));
    init_dispatcher_header(ev, u8(a[2] ? kEventSynchronization : kEventNotification), 4, s32(a[3] & 0xFF));
    wr<u32>(a[0], create_handle(ev));
    return STATUS_SUCCESS;
}
// NtSetEvent(HANDLE, PLONG PreviousState)
KEXPORT(NtSetEvent) {
    const u32 ev = handle_to_object(a[0]);
    if (!ev) return STATUS_INVALID_HANDLE;
    const s32 p = set_event(ev);
    if (a[1]) wr<s32>(a[1], p);
    return STATUS_SUCCESS;
}
KEXPORT(NtClearEvent) {
    const u32 ev = handle_to_object(a[0]);
    if (!ev) return STATUS_INVALID_HANDLE;
    reset_event(ev);
    return STATUS_SUCCESS;
}
KEXPORT(NtPulseEvent) {
    const u32 ev = handle_to_object(a[0]);
    if (!ev) return STATUS_INVALID_HANDLE;
    const s32 p = set_event(ev);
    reset_event(ev);
    if (a[1]) wr<s32>(a[1], p);
    return STATUS_SUCCESS;
}

// Semaphores: KSEMAPHORE = header + Limit
// NtCreateSemaphore(PHANDLE, POBJECT_ATTRIBUTES, LONG InitialCount, LONG MaximumCount)
KEXPORT(NtCreateSemaphore) {
    const u32 s = alloc_object(0x14, data_export_addr("ExSemaphoreObjectType"));
    init_dispatcher_header(s, kSemaphoreObject, 5, s32(a[2]));
    wr<s32>(s + kSemLimit, s32(a[3]));
    wr<u32>(a[0], create_handle(s));
    return STATUS_SUCCESS;
}
// NtReleaseSemaphore(HANDLE, LONG ReleaseCount, PLONG PreviousCount)
KEXPORT(NtReleaseSemaphore) {
    const u32 s = handle_to_object(a[0]);
    if (!s) return STATUS_INVALID_HANDLE;
    s32 prev;
    {
        std::lock_guard lk(kmutex());
        prev = rd<s32>(s + kHdrSignal);
        if (prev + s32(a[1]) > rd<s32>(s + kSemLimit)) return STATUS_SEMAPHORE_LIMIT_EXCEEDED;
        wr<s32>(s + kHdrSignal, prev + s32(a[1]));
    }
    notify_waiters();
    if (a[2]) wr<s32>(a[2], prev);
    return STATUS_SUCCESS;
}

// Mutants: KMUTANT = header + MutantListEntry + OwnerThread + Abandoned
// NtCreateMutant(PHANDLE, POBJECT_ATTRIBUTES, BOOLEAN InitialOwner)
KEXPORT(NtCreateMutant) {
    const u32 m = alloc_object(0x20, data_export_addr("ExMutantObjectType"));
    const bool own = a[2] & 0xFF;
    init_dispatcher_header(m, kMutantObject, 8, own ? 0 : 1);
    wr<u32>(m + kMutOwner, own ? ps::current_kthread() : 0);
    wr<u32>(a[0], create_handle(m));
    return STATUS_SUCCESS;
}
// NtReleaseMutant(HANDLE, PLONG PreviousCount)
KEXPORT(NtReleaseMutant) {
    const u32 m = handle_to_object(a[0]);
    if (!m) return STATUS_INVALID_HANDLE;
    s32 prev;
    {
        std::lock_guard lk(kmutex());
        if (rd<u32>(m + kMutOwner) != ps::current_kthread()) return STATUS_MUTANT_NOT_OWNED;
        prev = rd<s32>(m + kHdrSignal);
        wr<s32>(m + kHdrSignal, prev + 1);
        if (prev + 1 > 0) wr<u32>(m + kMutOwner, 0);
    }
    notify_waiters();
    if (a[1]) wr<s32>(a[1], prev);
    return STATUS_SUCCESS;
}

// Waits
// KeWaitForSingleObject(PVOID Object, KWAIT_REASON, KPROCESSOR_MODE, BOOLEAN Alertable, PLARGE_INTEGER Timeout)
KEXPORT(KeWaitForSingleObject) { return wait_for_objects({a[0]}, false, read_timeout(a[4])); }
// KeWaitForMultipleObjects(ULONG Count, PVOID Object[], WAIT_TYPE, KWAIT_REASON, KPROCESSOR_MODE, BOOLEAN Alertable,
//                          PLARGE_INTEGER Timeout, PKWAIT_BLOCK)
KEXPORT(KeWaitForMultipleObjects) {
    std::vector<u32> objs;
    for (u32 i = 0; i < a[0]; ++i) objs.push_back(rd<u32>(a[1] + 4 * i));
    return wait_for_objects(objs, a[2] == 0, read_timeout(a[6]));
}
// NtWaitForSingleObject(HANDLE, BOOLEAN Alertable, PLARGE_INTEGER Timeout)
KEXPORT(NtWaitForSingleObject) {
    const u32 o = handle_to_object(a[0]);
    if (!o) return STATUS_INVALID_HANDLE;
    return wait_for_objects({o}, false, read_timeout(a[2]));
}
// NtWaitForSingleObjectEx(HANDLE, KPROCESSOR_MODE, BOOLEAN Alertable, PLARGE_INTEGER Timeout)
KEXPORT(NtWaitForSingleObjectEx) {
    const u32 o = handle_to_object(a[0]);
    if (!o) return STATUS_INVALID_HANDLE;
    return wait_for_objects({o}, false, read_timeout(a[3]));
}
// NtWaitForMultipleObjectsEx(ULONG Count, HANDLE Handles[], WAIT_TYPE, KPROCESSOR_MODE, BOOLEAN Alertable, PLARGE_INTEGER)
KEXPORT(NtWaitForMultipleObjectsEx) {
    std::vector<u32> objs;
    for (u32 i = 0; i < a[0]; ++i) {
        const u32 o = handle_to_object(rd<u32>(a[1] + 4 * i));
        if (!o) return STATUS_INVALID_HANDLE;
        objs.push_back(o);
    }
    return wait_for_objects(objs, a[2] == 0, read_timeout(a[5]));
}
// NtSignalAndWaitForSingleObjectEx(HANDLE SignalHandle, HANDLE WaitHandle, KPROCESSOR_MODE, BOOLEAN, PLARGE_INTEGER)
KEXPORT(NtSignalAndWaitForSingleObjectEx) {
    const u32 s = handle_to_object(a[0]), w = handle_to_object(a[1]);
    if (!s || !w) return STATUS_INVALID_HANDLE;
    const u8 t = rd<u8>(s + kHdrType);
    if (t == kEventNotification || t == kEventSynchronization) set_event(s);
    else if (t == kSemaphoreObject) signal_object(s, rd<s32>(s + kHdrSignal) + 1);
    else if (t == kMutantObject) signal_object(s, rd<s32>(s + kHdrSignal) + 1);
    return wait_for_objects({w}, false, read_timeout(a[4]));
}

// Handles / object manager
KEXPORT(NtClose) { return close_handle(a[0]) ? STATUS_SUCCESS : STATUS_INVALID_HANDLE; }

// NtDuplicateObject(HANDLE Source, PHANDLE Target, ULONG Options)
KEXPORT(NtDuplicateObject) {
    const u32 o = handle_to_object(a[0]);
    if (!o) return STATUS_INVALID_HANDLE;
    wr<u32>(a[1], create_handle(o, handle_host(a[0])));
    if (a[2] & 1) close_handle(a[0]);  // DUPLICATE_CLOSE_SOURCE
    return STATUS_SUCCESS;
}

// ObReferenceObjectByHandle(HANDLE, POBJECT_TYPE ObjectType, PVOID* ReturnedObject)
KEXPORT(ObReferenceObjectByHandle) {
    const u32 o = handle_to_object(a[0]);
    if (!o) return STATUS_INVALID_HANDLE;
    if (a[1] && a[0] != kCurrentThreadHandle && object_type_of(o) != a[1]) return STATUS_OBJECT_TYPE_MISMATCH;
    wr<u32>(o - 0x10, rd<u32>(o - 0x10) + 1);
    wr<u32>(a[2], o);
    return STATUS_SUCCESS;
}
// ObfDereferenceObject(PVOID Object)  [fastcall]
KEXPORT(ObfDereferenceObject) {
    if (a[0]) wr<u32>(a[0] - 0x10, rd<u32>(a[0] - 0x10) - 1);
    return 0;
}

}  // namespace vc3::kernel
