# Command-line options and environment variables

## Command line

```
vc3 [options]
```

| Option | Effect |
|---|---|
| `--install <vcop3.zip> <gdx-0003b.chd>` | Install the game from your own files and exit (no window). The first file can also be the loose `317-0354-com.data`. Both are checked by SHA-1. |
| `--launcher` | Always open the launcher, even if it was set to be skipped. |
| `--no-launcher` | Start the game straight away (the launcher still opens if the game isn't installed). |
| `--no-settings` | Ignore `settings.ini` and use the defaults (environment variables still apply). |
| `--headless` | Run without a window, audio or input (testing). |

In the game: **F11** or **Alt+Enter** toggles fullscreen.

## Settings as environment variables

Every launcher setting can also be given as an environment variable. One that is set wins over
`settings.ini`, and the launcher shows it locked. Example:

```sh
VC3_RENDERER=vulkan VC3_RES_SCALE=3 VC3_WIDESCREEN=16:9 ./vc3
```

| Variable | Setting | Values |
|---|---|---|
| `VC3_RENDERER` | Renderer | `opengl`, `vulkan` |
| `VC3_RES_SCALE` | 3D resolution | `auto` (match display), `1`, `2`, `3`, `4`, `6`, `8` (times 640x480) |
| `VC3_WIDESCREEN` | Aspect ratio | `0` (4:3), `16:9`, `16:10`, `21:9` |
| `VC3_WIDE_HUD` | HUD in widescreen | `1` at the screen edges, `0` in place |
| `VC3_ANISOTROPIC` | Anisotropic filtering | `1` (off), `2`, `4`, `8`, `16` |
| `VC3_SMAA` | Anti-aliasing (SMAA) | `off`, `low`, `medium`, `high`, `ultra` |
| `VC3_WINDOW_SIZE` | Window size | e.g. `1280x960`, `1920x1080` |
| `VC3_FULLSCREEN` | Window | `0` windowed, `1` fullscreen |
| `VC3_CROSSHAIR` | Crosshair | `on`, `off` |
| `VC3_CROSSHAIR_SIZE` | Crosshair size | `0.5` - `3` (1 = 5% of the screen height) |
| `VC3_SINDEN_BORDER` | Sinden lightgun border | `0` off, or its thickness in percent of the screen height (`1` - `7`) |
| `VC3_VSYNC` | Vertical sync | `1`, `0` |
| `VC3_AUDIO` | Sound | `1`, `0` |
| `VC3_FREEPLAY` | Free play | `1`, `0` |
| `VC3_CHEAT_AMMO` | Infinite ammo | `1`, `0` |
| `VC3_CHEAT_LIVES` | Infinite lives | `1`, `0` |
| `VC3_CHEAT_SLOWMO` | Infinite E.S. slow motion | `1`, `0` |
| `VC3_CHEAT_FINAL` | Start at the final stage | `1`, `0` |
| `VC3_CHEAT_WEAPON` | Weapon | `0` none, `3` G36C, `4` MP7, `5` M4, `6` M629, `10` SAM-92A, `11` LMG249, `99` all weapons |

Controls are set in the launcher's Controls tab (stored in `settings.ini`).

## Other variables

| Variable | Effect |
|---|---|
| `VC3_VOLUME` | Sound volume factor (`1` normal, e.g. `0.5` half). |
| `VC3_VK_DEVICE` | Vulkan: index of the graphics card to use (`0`, `1`, ...), when there are several. |
| `VC3_VK_VALIDATION` | Vulkan: `1` enables the validation layer (needs it installed; for reporting renderer bugs). |
| `VC3_TRACE_HITCH` | `1` writes every slow frame, with its cause, to `vc3.log` (for reporting stutter). |

## Files

- `settings.ini`: in the per-user data folder (Linux: `~/.local/share/vcop3-recomp/vcop3-recomp/`).
- `vc3.log` (and `vc3.prev.log` from the run before): the log, in the data folder.
- The installed game: `work/` in the data folder, or in the folder chosen in the launcher.
