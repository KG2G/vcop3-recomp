// Small platform differences (Linux / Windows) shared by several files.
#pragma once
#include <filesystem>

namespace vc3::platform {

// The running program's own file (Linux: /proc/self/exe; Windows: GetModuleFileName).
std::filesystem::path exe_path();
// setenv(name, value, overwrite=1) / unsetenv (Windows: _putenv_s).
void set_env(const char* name, const char* value);
void unset_env(const char* name);
// Only one renderer can run here (Windows: Vulkan; the OpenGL renderer needs EGL).
constexpr bool kVulkanOnly =
#ifdef _WIN32
    true;
#else
    false;
#endif

}  // namespace vc3::platform
