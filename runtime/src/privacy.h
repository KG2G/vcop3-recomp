#pragma once
// The log (vc3.log) is for players to attach to bug reports: hardware and settings yes,
// nothing that identifies the person. scrub() removes the home folder and user name from
// a line (paths become ~/..., the name <user>); the log sink runs every line through it.
#include <string>

namespace vc3::privacy {
std::string scrub(std::string s);
// A path for the log: only its last component (folders a player picked can be personal).
std::string short_path(const std::string& path);
}  // namespace vc3::privacy
