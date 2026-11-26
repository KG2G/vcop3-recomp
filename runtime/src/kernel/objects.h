// Kernel objects, handles and waits.
//
// Waitable objects live in guest memory with a real DISPATCHER_HEADER
// ({u8 Type, u8 Absolute, u8 Size, u8 Inserted, s32 SignalState, LIST_ENTRY}),
// so Ke* calls on guest-allocated KEVENTs and Nt* calls on handles share one
// implementation. Handle-created objects get an OBJECT_HEADER
// ({PointerCount, HandleCount, Type*, Flags}, 0x10 bytes) in front, as on Xbox.
#pragma once
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include "kernel.h"

namespace vc3::kernel {

enum DispatcherType : u8 {
    kEventNotification = 0, kEventSynchronization = 1, kMutantObject = 2, kQueueObject = 4,
    kSemaphoreObject = 5, kThreadObject = 6, kTimerNotification = 8, kTimerSynchronization = 9,
};

// Dispatcher-object field offsets.
constexpr u32 kHdrType = 0x00, kHdrSize = 0x02, kHdrSignal = 0x04;
constexpr u32 kSemLimit = 0x10;      // KSEMAPHORE.Limit
constexpr u32 kMutOwner = 0x18;      // KMUTANT.OwnerThread
constexpr u32 kMutAbandoned = 0x1C;  // KMUTANT.Abandoned

constexpr u32 kCurrentThreadHandle = 0xFFFFFFFE;

// Host-side payload for objects that are not plain dispatcher objects (files, ...).
struct HostObject {
    virtual ~HostObject() = default;
};

std::mutex& kmutex();             // guards dispatcher state and the handle table
void notify_waiters();            // call after signalling (kmutex need not be held)
void init_dispatcher_header(u32 obj, u8 type, u8 size_dwords, s32 signal);

// Allocate an object body (with OBJECT_HEADER) in the kernel pool.
u32 alloc_object(u32 body_size, u32 type_object);
u32 object_type_of(u32 body);

u32 create_handle(u32 body, std::shared_ptr<HostObject> host = {});
u32 handle_to_object(u32 handle);  // 0 if invalid; resolves kCurrentThreadHandle
std::shared_ptr<HostObject> handle_host(u32 handle);
bool close_handle(u32 handle);

// Waits. Timeout is NT-style: nullopt = infinite, negative = relative 100 ns,
// positive = absolute system time. Returns STATUS_WAIT_0 + index, STATUS_TIMEOUT...
std::optional<s64> read_timeout(u32 plarge_integer);
u32 wait_for_objects(const std::vector<u32>& objects, bool wait_all, std::optional<s64> timeout);

// Signalling helpers (take kmutex internally).
s32 set_event(u32 event);       // returns previous state
s32 reset_event(u32 event);
void signal_object(u32 obj, s32 state);

}  // namespace vc3::kernel
