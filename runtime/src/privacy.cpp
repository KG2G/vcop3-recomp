#include "privacy.h"

#include <cctype>
#include <cstdlib>
#include <filesystem>

namespace vc3::privacy {
namespace {
std::string env(const char* n) {
    const char* v = std::getenv(n);
    return v ? v : "";
}
void replace_all(std::string& s, const std::string& from, const std::string& to, bool whole_word) {
    if (from.size() < 3) return;  // too short to replace safely
    for (size_t p = 0; (p = s.find(from, p)) != std::string::npos;) {
        auto word = [&](size_t i) { return i < s.size() && (std::isalnum(static_cast<unsigned char>(s[i])) || s[i] == '_'); };
        if (whole_word && ((p > 0 && word(p - 1)) || word(p + from.size()))) {
            p += from.size();
            continue;
        }
        s.replace(p, from.size(), to);
        p += to.size();
    }
}
struct Ids {
    std::string home, home_alt, user;
    Ids() {
#ifdef _WIN32
        home = env("USERPROFILE");
        user = env("USERNAME");
        home_alt = home;
        for (char& c : home_alt) if (c == '\\') c = '/';
#else
        home = env("HOME");
        user = env("USER");
        if (user.empty()) user = env("LOGNAME");
#endif
        while (home.size() > 1 && (home.back() == '/' || home.back() == '\\')) home.pop_back();
        while (home_alt.size() > 1 && home_alt.back() == '/') home_alt.pop_back();
    }
};
const Ids& ids() {
    static const Ids i;
    return i;
}
}  // namespace

std::string scrub(std::string s) {
    const Ids& i = ids();
    if (!i.home.empty() && i.home != "/") replace_all(s, i.home, "~", false);
    if (!i.home_alt.empty() && i.home_alt != i.home) replace_all(s, i.home_alt, "~", false);
    if (!i.user.empty()) replace_all(s, i.user, "<user>", true);
    return s;
}

std::string short_path(const std::string& path) {
    const std::string name = std::filesystem::path(path).filename().string();
    return name.empty() ? std::string("(folder)") : ".../" + name;
}
}  // namespace vc3::privacy
