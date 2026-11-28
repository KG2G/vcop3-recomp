// The pause menu's freeze: while paused the game stands still -- its threads park (never inside
// a critical section), its clocks and timers stop, no vblank ticks, no sound is made -- and the
// window keeps showing the last picture with the menu over it (pause_menu.h).
#pragma once

namespace vc3::pause {

bool active();
void set(bool paused);

}  // namespace vc3::pause
