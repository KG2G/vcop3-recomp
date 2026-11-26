// OpenGL device: context set-up, plus the shader source store shared by both backends.
#pragma once
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <string>

namespace vc3::gpu::dev {

namespace gl {
// Windowed mode: the window's EGL display/config/context, set before create_context().
// The GPU thread's context then shares objects with the window's.
void set_share(void* egl_display, void* egl_config, void* egl_context);
bool shared();
// Creates an EGL context current on the calling thread (shared with the window's if set).
bool create_context();
}  // namespace gl

// Every generated program's GLSL is kept (work/shadercache/v4/sources/, any driver and
// backend): on a new machine or after a driver update, startup compiles them all up
// front instead of hitching through each scene once.
namespace shader_store {
extern std::atomic<int> warm_done, warm_total;  // startup compilation progress
extern std::atomic<uint64_t> compile_ns, compile_n;  // compile time and count (VC3_TRACE_HITCH)
struct Timer {  // adds its lifetime to compile_ns
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    ~Timer();
};
void save(const std::string& vs, const std::string& fs);
// Calls fn for every stored program (and enables saving new ones).
void for_each(const std::function<void(const std::string& vs, const std::string& fs)>& fn);
void forget(const std::string& vs, const std::string& fs);  // no longer compiles
bool write_file(const std::string& path, const void* a, size_t na, const void* b = nullptr, size_t nb = 0);
}  // namespace shader_store

}  // namespace vc3::gpu::dev
