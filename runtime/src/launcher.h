// Launcher: a settings window shown before the game starts (Dear ImGui on SDL's 2D
// renderer, so it works whichever graphics API the game will use). Keyboard, mouse and
// gamepad all work.
#pragma once

namespace vc3::launcher {

// Shows the launcher (SDL video must be initialised). Returns true to start the game,
// false to quit. Settings are saved when the player starts the game.
bool run();

}  // namespace vc3::launcher
