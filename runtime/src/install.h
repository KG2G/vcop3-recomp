// Game file import: the player's own dump -> the files the recompiled game runs from.
//
//   vcop3.zip       MAME set holding the security PIC data (317-0354-com.data; the loose
//                   .data file works too)
//   gdx-0003b.chd   the GD-ROM (MAME CHD)
//
// Both are checked by SHA-1 (the PIC file, and the CHD's own data hash from its header),
// the game file is read from the disc's high-density area (ISO 9660), decrypted (DES, key
// from the PIC), and its FATX file system is unpacked to work/fs; the executable's SHA-1
// must then match the one the code was recompiled from.
#pragma once
#include <atomic>
#include <mutex>
#include <string>

namespace vc3::install {

struct Files {
    std::string pic;  // zip (or .data file) with the PIC data
    std::string chd;
};
// Looks in the roms folder (and the MAME layout roms/vcop3/) for the files.
Files find();
// The roms folder: next to the program (the .AppImage file, or the executable).
std::string roms_dir();
// Where the game is installed (work/ and its caches live there): the chosen folder or, when
// none is chosen, the per-user data folder. use_location() makes it the current directory.
std::string default_location();
void set_default_location(const std::string& dir);  // main: where it starts (a project tree in development)
std::string location();  // from the settings (install_dir) or the default
bool use_location(const std::string& dir);

// The game is installed and is the version this build was recompiled from.
bool installed(std::string* problem = nullptr);

// Import in the background (start) or on this thread (run: returns false on failure).
struct Progress {
    std::atomic<int> state{0};  // 0 idle, 1 working, 2 done, 3 failed
    std::atomic<float> fraction{0};
    std::mutex m;
    std::string message;  // current step / error (guarded by m)
};
Progress& progress();
void start(const Files& f);  // into location()
bool run(const Files& f);

}  // namespace vc3::install
