#pragma once
#include <cstdint>

namespace vc3::kernel::ke {

// Start the kernel worker (tick count, timers, DPC delivery). Idempotent.
void start_worker();
// Queue an interrupt for delivery to the ISR connected to `vector` (from any host thread).
void raise_interrupt(uint32_t vector);
// The pause menu: the guest's clocks (tick count, interrupt time, performance counter, the CPU's
// time stamp counter) stop while paused, and timers don't expire; resuming moves the timers on by
// the time paused, so the game sees no gap.
void pause_clock(bool paused);
// Total host time spent paused so far (ns), the current pause included: guest clocks subtract it.
int64_t paused_ns();

}  // namespace vc3::kernel::ke
