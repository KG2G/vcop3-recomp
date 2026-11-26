// Guest threads.
//
// Each guest thread is a host std::thread with its own Cpu, a guest stack,
// a KPCR (fs: base) and a KTHREAD. Layouts follow the Xbox kernel:
//   KPCR:    NtTib{ExceptionList, StackBase, StackLimit, ..., Self@0x18}, SelfPcr@0x1C,
//            Prcb@0x20, Irql@0x24, PrcbData@0x28 {CurrentThread@0x28, ...}
//   KTHREAD: Header@0, StackBase@0x1C, StackLimit@0x20, TlsData@0x28, Priority@0x32
#pragma once
#include <cstdint>
#include <initializer_list>

#include "vc3/recomp.h"

namespace vc3::kernel::ps {

constexpr uint32_t kKpcrIrql = 0x24;

uint32_t current_kthread();
// RtlEnter/LeaveCriticalSection bookkeeping (+1 / -1) for deferred suspension.
void critical_section_held(int delta);
uint32_t current_kpcr();

// Start the XBE entry point on a new guest thread (does not wait).
void start_boot_thread(uint32_t entry, uint32_t stack_size);
// Block the calling host thread until every guest thread has exited.
void wait_all_threads();

// Run guest code `fn(args...)` (stdcall) on a host-owned helper context, e.g.
// DPCs and interrupt service routines. Caller must hold the guest lock.
uint32_t call_guest_stdcall(uint32_t fn, std::initializer_list<uint32_t> args, uint8_t irql);

}  // namespace vc3::kernel::ps
