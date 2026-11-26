// Built-in sampling profiler; see profiler.cpp. All functions are no-ops unless
// VC3_PROFILE is set.
#pragma once

namespace vc3::profiler {

void init();             // at startup, before threads are created
void vblank(long frame); // from the vblank timer: starts/stops sampling by frame range
void dump();             // write work/profile.txt now

}  // namespace vc3::profiler
