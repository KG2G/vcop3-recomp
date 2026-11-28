# vcop3-recomp

A static recompilation of **Virtua Cop 3** (Sega Chihiro arcade, 2003) to native x86-64 code,
with a small runtime that stands in for the Chihiro hardware: the Xbox kernel, the NV2A GPU
(rendered with OpenGL 4.5 or Vulkan 1.3), the APU, the JVS I/O board and the media board.

**No game data is included.** You need your own dump of the game, as a MAME set:

| File | What it is | SHA-1 checked |
|---|---|---|
| `vcop3.zip` (containing `317-0354-com.data`) | the security PIC data | `9f0f4bf6b15f3b6eeea81eaa27b3d25bd94110da` (`317-0354-com.data`) |
| `gdx-0003b.chd` | the GD-ROM | `4268aadb83c880d4f3dab1d43ddd0a3a2f8befa2` (CHD data hash) |

The installer checks both, builds the game and unpacks it, then checks the game executable
(`vc3.xbe`, SHA-1 `e22c2b0ecf091a2388d9e3ca7eb9e07ce54d4727`) against the one this code was
recompiled from.

## Playing a release build

1. Download the AppImage (Linux) or the Windows build from the Releases page.
2. Start it. The launcher opens on **Game files**: pick your `vcop3.zip` and `gdx-0003b.chd`
   with Browse (or put them in a `roms` folder next to the program) and choose where to install.
3. Press **INSTALL**. When it's done the launcher shows **Settings** so you can set things up
   (renderer, window, controls, ...); then press **PLAY**. From then on the button is just **PLAY**.

By default the game, its settings and the log live in the per-user data folder (Linux:
`~/.local/share/vcop3-recomp/vcop3-recomp/`, Windows: `%APPDATA%\vcop3-recomp\vcop3-recomp\`).
Choose another **Install to** folder and everything goes there instead (the game files, saves and
shader caches in `work`, plus `settings.ini` and `vc3.log`): it's portable. Install into the
program's own folder and that folder can be moved or copied as a whole; for a folder elsewhere, a
one-line `location.txt` in the per-user folder points the program to it.
Command-line options and environment variables: [docs/OPTIONS.md](docs/OPTIONS.md).

### Launcher

- **Settings**: renderer (OpenGL / Vulkan), 3D resolution (Match display or 1x-8x), aspect
  ratio (4:3, 16:9, 16:10, 21:9) with the HUD at the screen edges, anisotropic filtering, SMAA,
  window size, windowed / fullscreen, crosshair, its size and a custom crosshair image per player, Sinden lightgun border, vsync,
  sound, free play, cheats (infinite ammo / lives / E.S. slow motion, start at the final stage, a
  starting weapon or all of them).
- **Controls**: per player, the aim device (system mouse, a gamepad stick, one specific mouse or
  lightgun read on its own, a Wiimote's IR pointer) and the inputs for trigger, reload, pedal,
  change weapon, start and coin; test / service; gamepad aim speed; controller API. On Linux,
  separate mice, lightguns and Wiimotes are read from `/dev/input` (be in the `input` group); on
  Windows mice and lightguns through Raw Input and Wiimotes over HID (experimental on Windows; pair
  them with Windows over Bluetooth first). Windows moves its cursor with every mouse, so there give each player their own
  device rather than pairing one with "System mouse".
- **Game files**: install as above.
- **Updates**: when the launcher starts it checks GitHub for a newer version (Settings > Updates >
  Check for updates, on by default). If there is one, its changes are shown and you choose whether
  to update: only the program file is downloaded (its SHA-256 checked against the release) and
  replaced, then it restarts -- the installed game, settings and saves stay.

Default controls (player 1): mouse aims, left button fires, right button reloads, middle button
or `E` changes weapon, Space is the pedal (E.S. slow motion), `1` starts, `5` inserts a coin
(player 2: `2` and `6`); F2 / F3 test / service; F11 or Alt+Enter toggles fullscreen.
TEST (F2) during play restarts into the game's test menu, as on the cabinet (SERVICE moves, TEST
selects; EXIT restarts the game). Its settings (difficulty, lives, gun adjustment, bookkeeping) are
kept like the cabinet's battery-backed memory, in `work/hdd` of the install folder.
Gamepads: right trigger fires, left trigger reloads, A pedal, X change weapon, Start, Back coin.

## Building from source

Requirements: CMake 3.24+, Ninja, a C++20 compiler, SDL3, libepoxy (EGL + OpenGL 4.5). For the
Vulkan renderer (optional): Vulkan headers and loader, and shaderc. Other libraries (Zydis, Dear
ImGui, miniz, libchdr, SMAA) are fetched by CMake.

The recompiled game code is generated from your own `vc3.xbe`, so the build has three steps:

```sh
# 1. Build the recompiler and a first runtime (without game code), and install your game files
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
build/runtime/vc3 --install /path/to/vcop3.zip /path/to/gdx-0003b.chd

# 2. Recompile the game executable to C++
build/recompiler/vc3recomp work/fs/vc3.xbe generated \
    --entries-file config/vc3_entries.txt --hooks-file config/vc3_hooks.txt

# 3. Build the runtime with the game code, and play
cmake -S . -B build
cmake --build build
build/runtime/vc3
```

`tools/appimage/build.sh` builds the Linux AppImage inside an Ubuntu 22.04 container, and
`tools/windows/build.sh` cross-compiles the Windows `.exe` with MinGW-w64 (both need Docker and
step 2 done).

The Windows build renders with Vulkan by default (any current NVIDIA, AMD or Intel driver) or
OpenGL 4.5, and needs Windows 10 version 1803 or newer.

## License

[PolyForm Noncommercial 1.0.0](LICENSE): fork it, mod it, share it, build on it -- freely, for any
non-commercial purpose. Selling it, or using it to make money, is not allowed: Virtua Cop 3 belongs
to Sega, and this project isn't something to profit from. When you pass it on, include the licence
(or its link) and the `Required Notice` line at the top of it.

That covers this project's source only. Virtua Cop 3 itself belongs to Sega (you supply your own
dump; nothing of the game is in this repository), and the third-party libraries below keep their
own licences (all permissive: MIT, BSD, zlib, Apache 2.0, public domain).

## Third-party code

Fetched at build time: Zydis, Dear ImGui, miniz, libchdr, stb_image (Sean Barrett, public domain:
custom crosshair images), and SMAA (Jorge Jimenez et al., MIT licence: the reference shader and its
lookup tables).

Virtua Cop 3 is a trademark of Sega. This project contains no Sega code or data; it only works
with a dump you supply.
