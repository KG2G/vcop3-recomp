#pragma once
// vc3.log: everything the program prints, for players to attach to bug reports. It holds
// what's needed to look into a problem (version, OS, CPU, GPU and driver, display mode,
// settings, what the game was doing, slow frames, crashes) and nothing that identifies
// the player: no names, folder paths, computer name or clock times (lines carry the time
// since start). See privacy.h.
#include <string>

namespace vc3::log {
// Send stdout/stderr to <dir>/vc3.log (the run before is kept as vc3.prev.log) and set up
// crash reports.
void start(const std::string& dir);
// Version, system and settings: once the settings are final.
void header();
// A crash report (what happened, the call stack on Linux), once; then the program dies.
void crashed(const char* what);
// The last line: how the session ended.
void end(const char* how);
// Continue the log in another folder (the data folder moved): the log so far (and vc3.prev.log)
// is moved there.
void move_to(const std::string& dir);
// The folder vc3.log is in (empty: no log).
std::string folder();
// Seconds since start, for log lines.
double uptime();
}  // namespace vc3::log
