#include "platform.h"

#include <cstdlib>

#ifdef _WIN32
#include <windows.h>
#endif

namespace vc3::platform {

std::filesystem::path exe_path() {
#ifdef _WIN32
    wchar_t buf[32768];
    const DWORD n = GetModuleFileNameW(nullptr, buf, DWORD(std::size(buf)));
    return n ? std::filesystem::path(std::wstring(buf, n)) : std::filesystem::path();
#else
    std::error_code ec;
    return std::filesystem::read_symlink("/proc/self/exe", ec);
#endif
}

void set_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

void unset_env(const char* name) {
#ifdef _WIN32
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

}  // namespace vc3::platform
