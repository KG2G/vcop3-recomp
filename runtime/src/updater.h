// Updates: looks for a newer release on GitHub (the public releases API, no personal data: a
// generic "vcop3-recomp/<version>" user agent), shows its changelog, and on request downloads the
// new program file (the AppImage / the .exe), checks its SHA-256 against the one GitHub publishes,
// puts it in place of the running program and restarts it. The installed game files, settings,
// saves and shader caches are left as they are: nothing is reinstalled or rebuilt.
// Setting "check_updates" (VC3_CHECK_UPDATES): check when the program starts (default on).
#pragma once
#include <string>

namespace vc3::updater {

enum class State { kIdle, kChecking, kUpToDate, kAvailable, kFailed, kDownloading, kInstalled, kInstallFailed };

void start_check();  // in the background (does nothing if already started)
State state();
// The newest release (when kAvailable / downloading / installed).
std::string latest_version();   // e.g. "0.1.7"
std::string current_version();  // this program's
std::string changelog();        // release notes of every newer release, newest first
std::string error();            // why a check or an install failed (for the player)
float progress();               // download progress 0..1
// The program can replace itself: running from an AppImage (Linux) or an .exe (Windows) in a
// folder it can write to, and a file for this platform is in the release.
bool can_install();
void start_install();           // download, verify, swap in (background)
// Start the new program and quit this one (after kInstalled). Returns only if starting it failed.
void restart();
std::string releases_page();
// At startup: removes what an earlier update left behind (Windows: the replaced .exe).
void cleanup();

}  // namespace vc3::updater
