# vcop3-recomp

A static recompilation of **Virtua Cop 3** (Sega Chihiro arcade, 2003) to native x86-64 code,
with a small runtime that stands in for the Chihiro hardware: the Xbox kernel, the NV2A GPU
(rendered with OpenGL 4.5 or Vulkan 1.3), the APU, the JVS I/O board and the media board.

**No game data is included.** You need your own dump of the game, as a MAME set:

| File | What it is | SHA-1 checked |
|---|---|---|
| `vcop3.zip` (containing `317-0354-com.data`) | the security PIC data | `9f0f4bf6b15f3b6eeea81eaa27b3d25bd94110da` (`317-0354-com.data`) |
| `gdx-0003b.chd` | the GD-ROM | `4268aadb83c880d4f3dab1d43ddd0a3a2f8befa2` (CHD data hash) |

The installer checks both, decrypts the game and unpacks it, then checks the game executable
(`vc3.xbe`, SHA-1 `e22c2b0ecf091a2388d9e3ca7eb9e07ce54d4727`) against the one this code was
recompiled from.

## Playing a release build

1. Download the AppImage (Linux) or the Windows build from the Releases page.
2. Start it. The launcher opens on **Game files**: pick your `vcop3.zip` and `gdx-0003b.chd`
   with Browse (or put them in a `roms` folder next to the program), choose where to install,
   and press Install.
3. Press **PLAY**.

Settings and the installed game live in the per-user data folder (Linux:
`~/.local/share/vcop3-recomp/vcop3-recomp/`) unless you choose another install folder.
Command-line options and environment variables: [docs/OPTIONS.md](docs/OPTIONS.md).

### Launcher

- **Settings**: renderer (OpenGL / Vulkan), 3D resolution (Match display or 1x-8x), aspect
  ratio (4:3, 16:9, 16:10, 21:9) with the HUD at the screen edges, anisotropic filtering, SMAA,
  window size, windowed / fullscreen, crosshair and its size, Sinden lightgun border, vsync,
  sound, free play, cheats (infinite ammo / lives / slow motion, start at the final stage, a
  starting weapon or all of them).
- **Controls**: per player, the aim device (system mouse, a gamepad stick, one specific mouse or
  lightgun read on its own, a Wiimote's IR pointer) and the inputs for trigger, reload, pedal,
  change weapon, start and coin; test / service; gamepad aim speed; controller API. On Linux,
  separate mice, lightguns and Wiimotes are read from `/dev/input` (be in the `input` group).
- **Game files**: install as above.

Default controls (player 1): mouse aims, left button fires, right button reloads, middle button
or `E` changes weapon, Space is the pedal (E.S. slow motion), `1` starts, `5` inserts a coin
(player 2: `2` and `6`); F2 / F3 test / service; F11 or Alt+Enter toggles fullscreen.
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

`tools/appimage/build.sh` builds the Linux AppImage inside an Ubuntu 22.04 container (needs
Docker and step 2 done).

## Third-party code

Fetched at build time: Zydis, Dear ImGui, miniz, libchdr, and SMAA (Jorge Jimenez et al., MIT
licence: the reference shader and its lookup tables).

Virtua Cop 3 is a trademark of Sega. This project contains no Sega code or data; it only works
with a dump you supply.
