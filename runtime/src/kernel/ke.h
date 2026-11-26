#pragma once
#include <cstdint>

namespace vc3::kernel::ke {

// Start the kernel worker (tick count, timers, DPC delivery). Idempotent.
void start_worker();
// Queue an interrupt for delivery to the ISR connected to `vector` (from any host thread).
void raise_interrupt(uint32_t vector);

}  // namespace vc3::kernel::ke
