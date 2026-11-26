// OpenGL 4.5 implementation of the device layer (device.h), on an EGL context of the GPU
// thread (shared with the window's when it can be, so frames are displayed straight from
// GPU memory).
#include "gpu/device.h"
#include "gpu/gl_device.h"
#include "gpu/pipe_recipe.h"

#include <epoxy/egl.h>
#include <epoxy/gl.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace vc3::gpu::dev {

namespace {

// ================================================================ context
bool make_context(EGLDisplay dpy) {
    if (dpy == EGL_NO_DISPLAY) return false;
    EGLint major, minor;
    if (!eglInitialize(dpy, &major, &minor)) return false;
    if (!eglBindAPI(EGL_OPENGL_API)) return false;
    const EGLint cfg_attr[] = {EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT, EGL_NONE};
    EGLConfig cfg;
    EGLint n = 0;
    if (!eglChooseConfig(dpy, cfg_attr, &cfg, 1, &n) || n == 0) {
        const EGLint any[] = {EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT, EGL_NONE};
        if (!eglChooseConfig(dpy, any, &cfg, 1, &n) || n == 0) return false;
    }
    const EGLint ctx_attr[] = {EGL_CONTEXT_MAJOR_VERSION, 4, EGL_CONTEXT_MINOR_VERSION, 5,
                               EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT, EGL_NONE};
    EGLContext ctx = eglCreateContext(dpy, cfg, EGL_NO_CONTEXT, ctx_attr);
    if (ctx == EGL_NO_CONTEXT) return false;
    return eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx);
}

EGLDisplay g_share_dpy = EGL_NO_DISPLAY;
EGLConfig g_share_cfg = nullptr;
EGLContext g_share_ctx = EGL_NO_CONTEXT;
bool g_shared = false;

// A context sharing objects with the window's, current without a surface.
bool make_shared_context() {
    if (g_share_ctx == EGL_NO_CONTEXT) return false;
    if (!epoxy_has_egl_extension(g_share_dpy, "EGL_KHR_surfaceless_context")) {
        std::fprintf(stderr, "[gl] no EGL_KHR_surfaceless_context: not sharing with the window\n");
        return false;
    }
    if (!eglBindAPI(EGL_OPENGL_API)) return false;
    const EGLint ctx_attr[] = {EGL_CONTEXT_MAJOR_VERSION, 4, EGL_CONTEXT_MINOR_VERSION, 5,
                               EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT, EGL_NONE};
    EGLContext ctx = eglCreateContext(g_share_dpy, g_share_cfg, g_share_ctx, ctx_attr);
    if (ctx == EGL_NO_CONTEXT) {
        std::fprintf(stderr, "[gl] shared context: EGL error %#x\n", unsigned(eglGetError()));
        return false;
    }
    if (!eglMakeCurrent(g_share_dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx)) return false;
    g_shared = true;
    return true;
}

// ================================================================ program cache
// Programs are found by a hash of their GLSL source. Linked programs are saved as
// driver binaries in work/shadercache/v4/<driver>/ and all of them are loaded at
// startup, so a state combination seen in any earlier run never compiles during
// play (compiles at scene changes were the main source of frame hitches).
u64 hash_bytes(const u8* p, size_t n) {
    u64 h0 = 0x9E3779B97F4A7C15ull ^ n, h1 = 0xC2B2AE3D27D4EB4Full, h2 = 0x165667B19E3779F9ull, h3 = 0x27D4EB2F165667C5ull;
    size_t i = 0;
    auto mix = [](u64 h, u64 v) { h = (h ^ v) * 0xFF51AFD7ED558CCDull; return h ^ (h >> 29); };
    for (; i + 32 <= n; i += 32) {
        u64 v[4];
        std::memcpy(v, p + i, 32);
        h0 = mix(h0, v[0]);
        h1 = mix(h1, v[1]);
        h2 = mix(h2, v[2]);
        h3 = mix(h3, v[3]);
    }
    u64 h = mix(mix(mix(h0, h1), h2), h3);
    for (; i < n; ++i) h = (h ^ p[i]) * 0x100000001B3ull;
    return h;
}

struct SrcKey {
    u64 a, b;
    bool operator==(const SrcKey& o) const { return a == o.a && b == o.b; }
};
struct SrcKeyHash { size_t operator()(const SrcKey& k) const { return size_t(k.a); } };
std::unordered_map<SrcKey, GLuint, SrcKeyHash> g_src_programs;
std::string g_cache_dir;  // empty: binaries unsupported or disabled (VC3_SHADER_CACHE=0)
std::unordered_set<SrcKey, SrcKeyHash> g_have_source;

SrcKey src_key(const std::string& vs, const std::string& fs) {
    std::string all = vs;
    all += '\0';
    all += fs;
    const u8* p = reinterpret_cast<const u8*>(all.data());
    u64 b = 0xCBF29CE484222325ull;
    for (size_t i = 0; i < all.size(); i += 61) b = (b ^ p[i]) * 0x100000001B3ull;  // second, independent hash
    return {hash_bytes(p, all.size()), b ^ all.size()};
}

std::string key_name(const SrcKey& k) {
    char name[40];
    std::snprintf(name, sizeof name, "%016llx%016llx", (unsigned long long)k.a, (unsigned long long)k.b);
    return name;
}
std::string cache_path(const SrcKey& k) { return g_cache_dir + "/" + key_name(k) + ".bin"; }

GLuint compile(GLenum type, const std::string& src) {
    GLuint sh = glCreateShader(type);
    const char* s = src.c_str();
    glShaderSource(sh, 1, &s, nullptr);
    glCompileShader(sh);
    GLint ok = 0;
    glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetShaderInfoLog(sh, sizeof log, nullptr, log);
        std::fprintf(stderr, "[gl] shader compile failed:\n%s\n%s\n", log, src.c_str());
        std::abort();
    }
    return sh;
}

GLuint load_binary(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return 0;
    GLenum format = 0;
    std::vector<u8> data;
    if (std::fread(&format, sizeof format, 1, f) == 1) {
        std::fseek(f, 0, SEEK_END);
        const long size = std::ftell(f) - long(sizeof format);
        std::fseek(f, long(sizeof format), SEEK_SET);
        if (size > 0) {
            data.resize(size_t(size));
            if (std::fread(data.data(), 1, data.size(), f) != data.size()) data.clear();
        }
    }
    std::fclose(f);
    if (data.empty()) return 0;
    const GLuint prog = glCreateProgram();
    glProgramBinary(prog, format, data.data(), GLsizei(data.size()));
    GLint ok = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {  // stale or rejected by the driver: rebuild from source
        glDeleteProgram(prog);
        std::remove(path.c_str());
        return 0;
    }
    return prog;
}

void save_binary(GLuint prog, const SrcKey& k) {
    GLint len = 0;
    glGetProgramiv(prog, GL_PROGRAM_BINARY_LENGTH, &len);
    if (len <= 0) return;
    std::vector<u8> data(static_cast<size_t>(len));
    GLenum format = 0;
    glGetProgramBinary(prog, len, nullptr, &format, data.data());
    shader_store::write_file(cache_path(k), &format, sizeof format, data.data(), data.size());
}

// Background compilation (VC3_ASYNC_SHADERS, default on where the driver compiles in
// parallel: GL_KHR/ARB_parallel_shader_compile): a program seen for the first time is
// compiled by the driver's threads while the game keeps running; a draw that needs it
// waits for it (ShaderWait, device.h).
bool g_async = false;
struct PendingProg { GLuint vs = 0, fs = 0; SrcKey k{}; };
std::unordered_map<GLuint, PendingProg> g_pending;
ShaderWait g_shader_wait;

GLuint start_shader(GLenum type, const std::string& src) {
    const GLuint sh = glCreateShader(type);
    const char* c = src.c_str();
    glShaderSource(sh, 1, &c, nullptr);
    glCompileShader(sh);
    return sh;
}

// The program can be drawn with (finishes a background compile once the driver is done).
bool program_ready(GLuint prog) {
    auto it = g_pending.find(prog);
    if (it == g_pending.end()) return true;
    GLint done = 0;
    glGetProgramiv(prog, GL_COMPLETION_STATUS_KHR, &done);
    if (!done) return false;
    const PendingProg p = it->second;
    g_pending.erase(it);
    GLint ok = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[4096];
        for (GLuint sh : {p.vs, p.fs}) {
            glGetShaderInfoLog(sh, sizeof log, nullptr, log);
            if (*log) std::fprintf(stderr, "[gl] shader: %s\n", log);
        }
        glGetProgramInfoLog(prog, sizeof log, nullptr, log);
        std::fprintf(stderr, "[gl] link failed: %s\n", log);
        std::abort();
    }
    glDetachShader(prog, p.vs);
    glDetachShader(prog, p.fs);
    glDeleteShader(p.vs);
    glDeleteShader(p.fs);
    if (!g_cache_dir.empty()) save_binary(prog, p.k);
    return true;
}

GLuint build_program(const std::string& vs_src, const std::string& fs_src) {
    const SrcKey k = src_key(vs_src, fs_src);
    GLuint& prog = g_src_programs[k];
    if (prog) return prog;
    shader_store::Timer timer;
    if (!g_cache_dir.empty() && (prog = load_binary(cache_path(k)))) return prog;
    if (g_async) {
        PendingProg p;
        p.vs = start_shader(GL_VERTEX_SHADER, vs_src);
        p.fs = start_shader(GL_FRAGMENT_SHADER, fs_src);
        p.k = k;
        prog = glCreateProgram();
        if (!g_cache_dir.empty()) glProgramParameteri(prog, GL_PROGRAM_BINARY_RETRIEVABLE_HINT, GL_TRUE);
        glAttachShader(prog, p.vs);
        glAttachShader(prog, p.fs);
        glLinkProgram(prog);
        g_pending[prog] = p;
        return prog;
    }
    const GLuint vs = compile(GL_VERTEX_SHADER, vs_src), fs = compile(GL_FRAGMENT_SHADER, fs_src);
    prog = glCreateProgram();
    if (!g_cache_dir.empty()) glProgramParameteri(prog, GL_PROGRAM_BINARY_RETRIEVABLE_HINT, GL_TRUE);
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    GLint ok = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    glDetachShader(prog, vs);
    glDetachShader(prog, fs);
    glDeleteShader(vs);
    glDeleteShader(fs);
    if (!ok) {
        char log[2048];
        glGetProgramInfoLog(prog, sizeof log, nullptr, log);
        std::fprintf(stderr, "[gl] link failed: %s\n", log);
        std::abort();
    }
    if (!g_cache_dir.empty()) save_binary(prog, k);
    return prog;
}

// The generated GLSL names the vertex index VERTEX_ID (gl_VertexIndex in Vulkan); stored
// sources are backend-neutral, so OpenGL defines it before compiling.
std::string gl_vertex_source(const std::string& vs_in) {
    std::string vs = vs_in;
    const size_t nl = vs.find('\n');
    vs.insert(nl == std::string::npos ? 0 : nl + 1, "#define VERTEX_ID gl_VertexID\n");
    return vs;
}

// Compile every known program source that has no binary for this driver yet.
void warm_from_sources() {
    struct Pending { SrcKey k; std::string vs, fs, stored_vs; GLuint v = 0, f = 0, prog = 0; };
    std::vector<Pending> todo;
    shader_store::for_each([&](const std::string& stored_vs, const std::string& fs) {
        std::string vs = gl_vertex_source(stored_vs);
        Pending p{src_key(vs, fs), std::move(vs), fs, stored_vs};
        if (!g_src_programs.count(p.k)) todo.push_back(std::move(p));
    });
    if (todo.empty()) return;
    std::fprintf(stderr, "[gl] compiling %zu shader programs for this driver...\n", todo.size());
    const auto t0 = std::chrono::steady_clock::now();
    shader_store::warm_total = int(todo.size());
    // Let the driver compile on several threads if it can: issue a batch, then collect it.
    if (epoxy_has_gl_extension("GL_KHR_parallel_shader_compile")) glMaxShaderCompilerThreadsKHR(0xFFFFFFFFu);
    else if (epoxy_has_gl_extension("GL_ARB_parallel_shader_compile")) glMaxShaderCompilerThreadsARB(0xFFFFFFFFu);
    constexpr size_t kBatch = 64;
    for (size_t b = 0; b < todo.size(); b += kBatch) {
        const size_t e = std::min(todo.size(), b + kBatch);
        for (size_t i = b; i < e; ++i) {
            Pending& p = todo[i];
            auto start = [](GLenum type, const std::string& src) {
                const GLuint sh = glCreateShader(type);
                const char* c = src.c_str();
                glShaderSource(sh, 1, &c, nullptr);
                glCompileShader(sh);
                return sh;
            };
            p.v = start(GL_VERTEX_SHADER, p.vs);
            p.f = start(GL_FRAGMENT_SHADER, p.fs);
            p.prog = glCreateProgram();
            glProgramParameteri(p.prog, GL_PROGRAM_BINARY_RETRIEVABLE_HINT, GL_TRUE);
            glAttachShader(p.prog, p.v);
            glAttachShader(p.prog, p.f);
            glLinkProgram(p.prog);
        }
        for (size_t i = b; i < e; ++i) {
            Pending& p = todo[i];
            GLint ok = 0;
            glGetProgramiv(p.prog, GL_LINK_STATUS, &ok);  // waits for this one
            glDetachShader(p.prog, p.v);
            glDetachShader(p.prog, p.f);
            glDeleteShader(p.v);
            glDeleteShader(p.f);
            if (ok) {
                if (!g_cache_dir.empty()) save_binary(p.prog, p.k);
                g_src_programs[p.k] = p.prog;
            } else {  // generated by an older build and no longer valid: forget it
                glDeleteProgram(p.prog);
                shader_store::forget(p.stored_vs, p.fs);
            }
            ++shader_store::warm_done;
        }
    }
    std::fprintf(stderr, "[gl] compiled %zu shader programs in %.1f s\n", todo.size(),
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
}

// At startup: pick the cache directory for this driver and load every program in it.
void init_program_cache() {
    GLint formats = 0;
    glGetIntegerv(GL_NUM_PROGRAM_BINARY_FORMATS, &formats);
    const char* env = std::getenv("VC3_SHADER_CACHE");
    if (formats > 0 && !(env && !std::strcmp(env, "0"))) {
        std::string id = reinterpret_cast<const char*>(glGetString(GL_VENDOR));
        id += '|';
        id += reinterpret_cast<const char*>(glGetString(GL_RENDERER));
        id += '|';
        id += reinterpret_cast<const char*>(glGetString(GL_VERSION));
        char dir[64];
        std::snprintf(dir, sizeof dir, "work/shadercache/v4/gl-%016llx",
                      (unsigned long long)hash_bytes(reinterpret_cast<const u8*>(id.data()), id.size()));
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        if (!ec) {
            g_cache_dir = dir;
            const auto t0 = std::chrono::steady_clock::now();
            size_t n = 0;
            for (const auto& e : std::filesystem::directory_iterator(g_cache_dir, ec)) {
                const std::string name = e.path().filename().string();
                if (name.size() != 36 || e.path().extension() != ".bin") continue;
                const SrcKey k{std::strtoull(name.substr(0, 16).c_str(), nullptr, 16),
                               std::strtoull(name.substr(16, 16).c_str(), nullptr, 16)};
                if (const GLuint prog = load_binary(e.path().string())) {
                    g_src_programs[k] = prog;
                    ++n;
                }
            }
            std::fprintf(stderr, "[gl] shader cache %s: %zu programs loaded in %.0f ms\n", g_cache_dir.c_str(), n,
                         std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
    }
    // VC3_SHADER_PRECOMPILE=0: don't compile stored programs that lack a binary at startup
    // (they compile when first drawn instead).
    const char* pc = std::getenv("VC3_SHADER_PRECOMPILE");
    if (pc && !std::strcmp(pc, "0")) shader_store::for_each([](const std::string&, const std::string&) {});
    else warm_from_sources();
}

// ================================================================ device
GLenum gl_compare(u8 f) { return GLenum(0x200 + (f & 7)); }
GLenum gl_stencil_op(u8 op) {
    static constexpr GLenum k[] = {GL_KEEP, GL_ZERO, GL_REPLACE, GL_INCR, GL_DECR, GL_INVERT, GL_INCR_WRAP, GL_DECR_WRAP};
    return k[op & 7];
}
GLenum gl_blend_factor(u8 f) {
    static constexpr GLenum k[] = {GL_ZERO, GL_ONE, GL_SRC_COLOR, GL_ONE_MINUS_SRC_COLOR, GL_SRC_ALPHA,
                                   GL_ONE_MINUS_SRC_ALPHA, GL_DST_ALPHA, GL_ONE_MINUS_DST_ALPHA, GL_DST_COLOR,
                                   GL_ONE_MINUS_DST_COLOR, GL_SRC_ALPHA_SATURATE, GL_CONSTANT_COLOR,
                                   GL_ONE_MINUS_CONSTANT_COLOR, GL_CONSTANT_ALPHA, GL_ONE_MINUS_CONSTANT_ALPHA};
    return f < 15 ? k[f] : GL_ONE;
}
GLenum gl_blend_eq(u8 e) {
    static constexpr GLenum k[] = {GL_FUNC_ADD, GL_FUNC_SUBTRACT, GL_FUNC_REVERSE_SUBTRACT, GL_MIN, GL_MAX};
    return e < 5 ? k[e] : GL_FUNC_ADD;
}
GLenum gl_topology(u8 t) {
    static constexpr GLenum k[] = {GL_POINTS, GL_LINES, GL_LINE_LOOP, GL_LINE_STRIP, GL_TRIANGLES, GL_TRIANGLE_STRIP,
                                   GL_TRIANGLE_FAN};
    return t < 7 ? k[t] : GL_TRIANGLES;
}
GLint gl_wrap(u8 w) {
    switch (w) {
        case kRepeat: return GL_REPEAT;
        case kMirror: return GL_MIRRORED_REPEAT;
        case kClampBorder: return GL_CLAMP_TO_BORDER;
        default: return GL_CLAMP_TO_EDGE;
    }
}

class GlDevice final : public Device {
    float max_aniso_ = 1.0f;  // VC3_ANISOTROPIC (1 when the driver can't)

public:
    GlDevice() {
        if (epoxy_gl_version() >= 46 || epoxy_has_gl_extension("GL_ARB_texture_filter_anisotropic") ||
            epoxy_has_gl_extension("GL_EXT_texture_filter_anisotropic")) {
            GLfloat lim = 1.0f;
            glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY, &lim);
            max_aniso_ = std::clamp(anisotropy_setting(), 1.0f, std::max(1.0f, lim));
        }
        std::fprintf(stderr, "[gl] anisotropic filtering %.0fx (GL %d)\n", double(max_aniso_), epoxy_gl_version());
        g_s3tc_ = epoxy_has_gl_extension("GL_EXT_texture_compression_s3tc") && !std::getenv("VC3_NO_S3TC");
        std::fprintf(stderr, "[gl] renderer: %s (%s)\n", reinterpret_cast<const char*>(glGetString(GL_RENDERER)),
                     reinterpret_cast<const char*>(glGetString(GL_VERSION)));
        // Lower-left origin maps NDC y = -1 to framebuffer row 0 = guest memory row 0 (the
        // top line on the NV2A).
        glClipControl(GL_LOWER_LEFT, GL_ZERO_TO_ONE);
        glEnable(GL_PROGRAM_POINT_SIZE);
        glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
        glGetIntegerv(GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT, &ubo_align_);
        ubo_align_ = std::max(ubo_align_, 16);
        init_program_cache();
        {
            const char* a = std::getenv("VC3_ASYNC_SHADERS");
            const bool par = epoxy_has_gl_extension("GL_KHR_parallel_shader_compile") ||
                             epoxy_has_gl_extension("GL_ARB_parallel_shader_compile");
            g_async = par && !(a && !std::strcmp(a, "0"));
            if (par) {
                if (epoxy_has_gl_extension("GL_KHR_parallel_shader_compile")) glMaxShaderCompilerThreadsKHR(0xFFFFFFFFu);
                else glMaxShaderCompilerThreadsARB(0xFFFFFFFFu);
            }
            std::fprintf(stderr, "[gl] new shaders compile %s\n", g_async ? "in the background" : "immediately");
        }
        glCreateVertexArrays(1, &vao_);
        glCreateBuffers(1, &ring_);
        const GLbitfield flags = GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT;
        glNamedBufferStorage(ring_, GLsizeiptr(kRingBytes), nullptr, flags);
        ring_ptr_ = static_cast<u8*>(glMapNamedBufferRange(ring_, 0, GLsizeiptr(kRingBytes), flags));
    }

    const char* name() const override { return "OpenGL"; }
    bool supports_bc() const override { return g_s3tc_; }

    // ---- textures
    Tex create_texture(int w, int h, int levels, Fmt fmt, bool cube) override {
        GLuint t = 0;
        glCreateTextures(cube ? GL_TEXTURE_CUBE_MAP : GL_TEXTURE_2D, 1, &t);
        GLenum f = GL_RGBA8;
        switch (fmt) {
            case Fmt::D24S8: f = GL_DEPTH24_STENCIL8; break;
            case Fmt::BC1: f = GL_COMPRESSED_RGBA_S3TC_DXT1_EXT; break;
            case Fmt::BC2: f = GL_COMPRESSED_RGBA_S3TC_DXT3_EXT; break;
            case Fmt::BC3: f = GL_COMPRESSED_RGBA_S3TC_DXT5_EXT; break;
            default: break;
        }
        glTextureStorage2D(t, std::max(1, levels), f, w, h);
        info_[t] = {fmt, cube};
        return Tex{t};
    }
    void destroy_texture(Tex t) override {
        if (!t) return;
        for (auto it = fbos_.begin(); it != fbos_.end();)
            if (it->first.first == t.id || it->first.second == t.id) {
                glDeleteFramebuffers(1, &it->second);
                it = fbos_.erase(it);
            } else {
                ++it;
            }
        info_.erase(t.id);
        glDeleteTextures(1, &t.id);
    }
    void upload(Tex t, int level, int face, int x, int y, int w, int h, const void* data, size_t bytes) override {
        const auto& in = info_[t.id];
        GLenum cf = 0;
        switch (in.fmt) {
            case Fmt::BC1: cf = GL_COMPRESSED_RGBA_S3TC_DXT1_EXT; break;
            case Fmt::BC2: cf = GL_COMPRESSED_RGBA_S3TC_DXT3_EXT; break;
            case Fmt::BC3: cf = GL_COMPRESSED_RGBA_S3TC_DXT5_EXT; break;
            default: break;
        }
        if (cf) {
            if (in.cube) glCompressedTextureSubImage3D(t.id, level, x, y, face, w, h, 1, cf, GLsizei(bytes), data);
            else glCompressedTextureSubImage2D(t.id, level, x, y, w, h, cf, GLsizei(bytes), data);
        } else if (in.cube) {
            glTextureSubImage3D(t.id, level, x, y, face, w, h, 1, GL_BGRA, GL_UNSIGNED_BYTE, data);
        } else {
            glTextureSubImage2D(t.id, level, x, y, w, h, GL_BGRA, GL_UNSIGNED_BYTE, data);
        }
    }
    void copy(Tex src, Tex dst, int w, int h) override {
        glCopyImageSubData(src.id, GL_TEXTURE_2D, 0, 0, 0, 0, dst.id, GL_TEXTURE_2D, 0, 0, 0, 0, w, h, 1);
    }
    void copy_region(Tex src, Tex dst, int x, int y, int w, int h) override {
        glCopyImageSubData(src.id, GL_TEXTURE_2D, 0, x, y, 0, dst.id, GL_TEXTURE_2D, 0, x, y, 0, w, h, 1);
    }
    void blit(Tex src, int sx0, int sy0, int sx1, int sy1, Tex dst, int dx0, int dy0, int dx1, int dy1, bool linear) override {
        glDisable(GL_SCISSOR_TEST);  // blits obey the scissor a draw may have left on
        glBlitNamedFramebuffer(fbo(src.id, 0), fbo(dst.id, 0), sx0, sy0, sx1, sy1, dx0, dy0, dx1, dy1, GL_COLOR_BUFFER_BIT,
                               linear ? GL_LINEAR : GL_NEAREST);
    }

    // ---- read-back
    void read(Tex t, int w, int h, void* dst) override {
        glGetTextureImage(t.id, 0, GL_BGRA, GL_UNSIGNED_BYTE, GLsizei(size_t(w) * size_t(h) * 4), dst);
    }
    u32 read_start(Tex t, int w, int h) override {
        const size_t bytes = size_t(w) * size_t(h) * 4;
        u32 id = 0;
        for (u32 i = 0; i < reads_.size(); ++i)
            if (!reads_[i].busy && reads_[i].size == bytes) { id = i + 1; break; }
        if (!id) {
            for (u32 i = 0; i < reads_.size(); ++i)
                if (!reads_[i].busy) {
                    if (reads_[i].pbo) glDeleteBuffers(1, &reads_[i].pbo);
                    reads_[i] = {};
                    id = i + 1;
                    break;
                }
            if (!id) { reads_.push_back({}); id = u32(reads_.size()); }
            Read& r = reads_[id - 1];
            glCreateBuffers(1, &r.pbo);
            glNamedBufferStorage(r.pbo, GLsizeiptr(bytes), nullptr, GL_MAP_READ_BIT | GL_CLIENT_STORAGE_BIT);
            r.size = bytes;
        }
        Read& r = reads_[id - 1];
        glBindBuffer(GL_PIXEL_PACK_BUFFER, r.pbo);
        glGetTextureImage(t.id, 0, GL_BGRA, GL_UNSIGNED_BYTE, GLsizei(bytes), nullptr);
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
        r.fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
        glFlush();
        r.busy = true;
        return id;
    }
    const void* read_poll(u32 id, bool wait) override {
        Read& r = reads_[id - 1];
        if (!r.map) {
            const GLenum s = glClientWaitSync(r.fence, wait ? GL_SYNC_FLUSH_COMMANDS_BIT : 0, wait ? 1000000000ull : 0);
            if (s == GL_TIMEOUT_EXPIRED || s == GL_WAIT_FAILED) return nullptr;
            glDeleteSync(r.fence);
            r.fence = nullptr;
            r.map = glMapNamedBufferRange(r.pbo, 0, GLsizeiptr(r.size), GL_MAP_READ_BIT);
        }
        return r.map;
    }
    void read_release(u32 id) override {
        Read& r = reads_[id - 1];
        if (r.map) glUnmapNamedBuffer(r.pbo);
        if (r.fence) glDeleteSync(r.fence);
        r.map = nullptr;
        r.fence = nullptr;
        r.busy = false;
    }

    // ---- rendering
    void set_target(Tex color, Tex depth, int w, int h) override {
        depth_bound_ = depth.id != 0;
        glBindFramebuffer(GL_FRAMEBUFFER, fbo(color.id, depth.id));
        glViewport(0, 0, w, h);
        target_w_ = w;
        target_h_ = h;
    }
    void clear(int x, int y, int w, int h, bool color, u8 color_mask, const float rgba[4], bool depth, float z, bool stencil,
               u8 s) override {
        glEnable(GL_SCISSOR_TEST);
        glScissor(x, y, w, h);
        GLbitfield mask = 0;
        if (color) {
            glColorMask(color_mask & 1, (color_mask >> 1) & 1, (color_mask >> 2) & 1, (color_mask >> 3) & 1);
            glClearColor(rgba[0], rgba[1], rgba[2], rgba[3]);
            mask |= GL_COLOR_BUFFER_BIT;
        }
        if (depth) {
            glDepthMask(GL_TRUE);
            glClearDepth(z);
            mask |= GL_DEPTH_BUFFER_BIT;
        }
        if (stencil) {
            glStencilMask(0xFF);
            glClearStencil(GLint(s));
            mask |= GL_STENCIL_BUFFER_BIT;
        }
        if (mask) glClear(mask);
        glDisable(GL_SCISSOR_TEST);
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    }
    Program program(const std::string& vs_in, const std::string& fs) override {
        shader_store::save(vs_in, fs);
        const GLuint prog = build_program(gl_vertex_source(vs_in), fs);
        vk_prog_key_[prog] = program_key(vs_in, fs);
        return Program{prog};
    }
    size_t ubo_align() const override { return size_t(ubo_align_); }

    u8* stream(size_t bytes, size_t align, Slice& out) override {
        if (bytes > kChunk) {  // larger than a ring chunk: a buffer of its own, freed once used
            GLuint b = 0;
            glCreateBuffers(1, &b);
            glNamedBufferStorage(b, GLsizeiptr(bytes), nullptr, GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT);
            u8* p = static_cast<u8*>(glMapNamedBufferRange(b, 0, GLsizeiptr(bytes),
                                                            GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT));
            big_.push_back(b);
            out = {b, 0, bytes};
            return p;
        }
        size_t pos = (pos_ + align - 1) / align * align;
        size_t chunk = chunk_;  // (pos_ can sit exactly on the next chunk's start)
        if (pos + bytes > (chunk + 1) * kChunk) {
            // Crossing into the next chunk: fence the one we leave, wait for the one we enter.
            if (fence_[chunk]) glDeleteSync(fence_[chunk]);
            fence_[chunk] = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
            chunk = (chunk + 1) % kChunks;
            pos = chunk * kChunk;
            if (GLsync f = fence_[chunk]) {
                glClientWaitSync(f, GL_SYNC_FLUSH_COMMANDS_BIT, 1000000000ull);
                glDeleteSync(f);
                fence_[chunk] = nullptr;
            }
            chunk_ = chunk;
        }
        pos_ = pos + bytes;
        out = {ring_, pos, bytes};
        return ring_ptr_ + pos;
    }

    bool draw(const DrawCall& dc) override {
        if (!g_shader_wait.wait([&] { return program_ready(dc.prog.id); })) return false;  // still compiling
        // The Vulkan pipeline this draw would need, for the cache (and the shader seed).
        if (auto it = vk_prog_key_.find(dc.prog.id); it != vk_prog_key_.end()) {
            Recipe r{it->second, canonical_pipe_key(dc, depth_bound_)};
            r.key.prog = 0;
            shader_store::record_recipe(&r, sizeof r);
        }
        glUseProgram(dc.prog.id);
        // Textures (2D units 4-7, cube 8-11, as the shaders' bindings)
        for (int i = 0; i < 8; ++i) {
            const Tex t = dc.tex[i];
            if (!t) continue;
            const SamplerDesc& s = dc.samp[i];
            glBindTextureUnit(GLuint(4 + i), t.id);
            GLint minf = s.min == kNearest ? GL_NEAREST : GL_LINEAR;
            if (s.mip == kMipNearest) minf = s.min == kNearest ? GL_NEAREST_MIPMAP_NEAREST : GL_LINEAR_MIPMAP_NEAREST;
            else if (s.mip == kMipLinear) minf = s.min == kNearest ? GL_NEAREST_MIPMAP_LINEAR : GL_LINEAR_MIPMAP_LINEAR;
            glTextureParameteri(t.id, GL_TEXTURE_MIN_FILTER, minf);
            glTextureParameteri(t.id, GL_TEXTURE_MAG_FILTER, s.mag == kNearest ? GL_NEAREST : GL_LINEAR);
            glTextureParameterf(t.id, GL_TEXTURE_LOD_BIAS, s.lod_bias);
            glTextureParameterf(t.id, GL_TEXTURE_MIN_LOD, s.min_lod);
            glTextureParameterf(t.id, GL_TEXTURE_MAX_LOD, s.max_lod);
            glTextureParameteri(t.id, GL_TEXTURE_WRAP_S, gl_wrap(s.wrap_u));
            glTextureParameteri(t.id, GL_TEXTURE_WRAP_T, gl_wrap(s.wrap_v));
            glTextureParameterfv(t.id, GL_TEXTURE_BORDER_COLOR, s.border);
            glTextureParameteri(t.id, GL_TEXTURE_SWIZZLE_A, s.alpha_one ? GL_ONE : GL_ALPHA);
            // Anisotropic filtering (VC3_ANISOTROPIC) for linearly filtered textures. The game
            // rarely uses mipmaps; on level 0 alone it still samples along the direction a
            // surface recedes (less shimmer on floors and walls seen at an angle).
            glTextureParameterf(t.id, GL_TEXTURE_MAX_ANISOTROPY, s.min == kLinear ? max_aniso_ : 1.0f);
        }
        // Uniform blocks / HUD storage
        if (dc.ubo0) glBindBufferRange(GL_UNIFORM_BUFFER, 0, dc.ubo0.buf, GLintptr(dc.ubo0.off), GLsizeiptr(dc.ubo0.size));
        if (dc.ubo1) glBindBufferRange(GL_UNIFORM_BUFFER, 1, dc.ubo1.buf, GLintptr(dc.ubo1.off), GLsizeiptr(dc.ubo1.size));
        if (dc.hud) glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 3, dc.hud.buf, GLintptr(dc.hud.off), GLsizeiptr(dc.hud.size));

        // Fixed-function state
        const RasterState& r = dc.rs;
        if (r.scissor) {
            glEnable(GL_SCISSOR_TEST);
            glScissor(r.sx, r.sy, r.sw, r.sh);
        } else {
            glDisable(GL_SCISSOR_TEST);
        }
        if (r.depth_test || r.depth_write) {  // writes without test: GL needs the test enabled
            glEnable(GL_DEPTH_TEST);
            glDepthFunc(r.depth_test ? gl_compare(r.depth_func) : GL_ALWAYS);
        } else {
            glDisable(GL_DEPTH_TEST);
        }
        glDepthMask(r.depth_write ? GL_TRUE : GL_FALSE);
        if (r.poly_offset) {
            glEnable(GL_POLYGON_OFFSET_FILL);
            glEnable(GL_POLYGON_OFFSET_LINE);
            glEnable(GL_POLYGON_OFFSET_POINT);
            glPolygonOffset(r.poly_offset_scale, r.poly_offset_units);
        } else {
            glDisable(GL_POLYGON_OFFSET_FILL);
            glDisable(GL_POLYGON_OFFSET_LINE);
            glDisable(GL_POLYGON_OFFSET_POINT);
        }
        if (r.stencil_test) {
            glEnable(GL_STENCIL_TEST);
            glStencilFunc(gl_compare(r.stencil_func), GLint(r.stencil_ref), r.stencil_read_mask);
            glStencilOp(gl_stencil_op(r.stencil_fail), gl_stencil_op(r.stencil_zfail), gl_stencil_op(r.stencil_zpass));
            glStencilMask(r.stencil_write_mask);
        } else {
            glDisable(GL_STENCIL_TEST);
        }
        if (r.blend) {
            glEnable(GL_BLEND);
            glBlendFunc(gl_blend_factor(r.blend_src), gl_blend_factor(r.blend_dst));
            glBlendEquation(gl_blend_eq(r.blend_eq));
            glBlendColor(r.blend_color[0], r.blend_color[1], r.blend_color[2], r.blend_color[3]);
        } else {
            glDisable(GL_BLEND);
        }
        glColorMask(r.color_mask & 1, (r.color_mask >> 1) & 1, (r.color_mask >> 2) & 1, (r.color_mask >> 3) & 1);
        // Window y runs the same way as the guest's screen y here (lower-left origin with
        // row 0 at the bottom), so GL's winding is the mirror of the NV2A's: swap CW/CCW.
        glFrontFace(r.front_ccw ? GL_CW : GL_CCW);
        if (r.cull != kCullNone) {
            glEnable(GL_CULL_FACE);
            glCullFace(r.cull == kCullFront ? GL_FRONT : r.cull == kCullBack ? GL_BACK : GL_FRONT_AND_BACK);
        } else {
            glDisable(GL_CULL_FACE);
        }

        // Vertex input
        for (u32 a = 0; a < 16; ++a) {
            bool used = false;
            for (int i = 0; i < dc.nattr; ++i) used = used || dc.attr[i].location == a;
            if (!used && (attr_on_ >> a & 1)) {
                glDisableVertexArrayAttrib(vao_, a);
                attr_on_ &= ~(1u << a);
            }
        }
        for (int i = 0; i < dc.nattr; ++i) {
            const VertexAttr& at = dc.attr[i];
            const GLuint loc = at.location;
            if (at.constant) {
                if (attr_on_ >> loc & 1) {
                    glDisableVertexArrayAttrib(vao_, loc);
                    attr_on_ &= ~(1u << loc);
                }
                glVertexAttrib4fv(loc, at.value);
                continue;
            }
            if (!(attr_on_ >> loc & 1)) {
                glEnableVertexArrayAttrib(vao_, loc);
                attr_on_ |= 1u << loc;
            }
            const GLuint rel = at.offset;
            switch (at.fmt) {
                case kF32x1: case kF32x2: case kF32x3: case kF32x4:
                    glVertexArrayAttribFormat(vao_, loc, 1 + (at.fmt - kF32x1), GL_FLOAT, GL_FALSE, rel); break;
                case kUN8x1: case kUN8x2: case kUN8x3: case kUN8x4:
                    glVertexArrayAttribFormat(vao_, loc, 1 + (at.fmt - kUN8x1), GL_UNSIGNED_BYTE, GL_TRUE, rel); break;
                case kBGRA8: glVertexArrayAttribFormat(vao_, loc, GL_BGRA, GL_UNSIGNED_BYTE, GL_TRUE, rel); break;
                case kSN16x1: case kSN16x2: case kSN16x3: case kSN16x4:
                    glVertexArrayAttribFormat(vao_, loc, 1 + (at.fmt - kSN16x1), GL_SHORT, GL_TRUE, rel); break;
                case kS16x1: case kS16x2: case kS16x3: case kS16x4:
                    glVertexArrayAttribFormat(vao_, loc, 1 + (at.fmt - kS16x1), GL_SHORT, GL_FALSE, rel); break;
                case kU32x1: glVertexArrayAttribIFormat(vao_, loc, 1, GL_UNSIGNED_INT, rel); break;
                default: break;
            }
            glVertexArrayAttribBinding(vao_, loc, at.binding);
        }
        for (int b = 0; b < dc.nbind; ++b)
            glVertexArrayVertexBuffer(vao_, GLuint(b), dc.vbuf[b].buf, GLintptr(dc.vbuf[b].off), GLsizei(dc.stride[b]));
        glBindVertexArray(vao_);
        const GLenum prim = gl_topology(dc.rs.topology);
        if (dc.index) {
            glVertexArrayElementBuffer(vao_, dc.index.buf);
            glDrawElements(prim, GLsizei(dc.count), GL_UNSIGNED_INT, reinterpret_cast<const void*>(dc.index.off));
        } else {
            glDrawArrays(prim, GLint(dc.first), GLsizei(dc.count));
        }
        return true;
    }

    Sync fence() override {
        GLsync f = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
        glFlush();  // another context may wait on it
        return Sync(reinterpret_cast<uintptr_t>(f));
    }
    bool wait(Sync s, u64 timeout_ns) override {
        if (!s) return true;
        GLsync f = reinterpret_cast<GLsync>(uintptr_t(s));
        const GLenum r = glClientWaitSync(f, GL_SYNC_FLUSH_COMMANDS_BIT, timeout_ns);
        if (r == GL_TIMEOUT_EXPIRED || r == GL_WAIT_FAILED) return false;
        glDeleteSync(f);
        return true;
    }
    void gpu_wait(Sync s) override {
        if (!s) return;
        GLsync f = reinterpret_cast<GLsync>(uintptr_t(s));
        glWaitSync(f, 0, GL_TIMEOUT_IGNORED);
        glDeleteSync(f);
    }
    void release(Sync s) override {
        if (s) glDeleteSync(reinterpret_cast<GLsync>(uintptr_t(s)));
    }
    void flush() override {
        g_shader_wait.new_frame();
        // Buffers of oversized draws: free those the GPU has finished with.
        if (!big_.empty()) {
            big_fences_.push_back({glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0), std::move(big_)});
            big_.clear();
        }
        while (!big_fences_.empty()) {
            auto& [f, bufs] = big_fences_.front();
            if (glClientWaitSync(f, 0, 0) == GL_TIMEOUT_EXPIRED) break;
            glDeleteSync(f);
            glDeleteBuffers(GLsizei(bufs.size()), bufs.data());
            big_fences_.erase(big_fences_.begin());
        }
    }

private:
    GLuint fbo(GLuint color, GLuint depth) {
        auto it = fbos_.find({color, depth});
        if (it != fbos_.end()) return it->second;
        GLuint f;
        glCreateFramebuffers(1, &f);
        glNamedFramebufferTexture(f, GL_COLOR_ATTACHMENT0, color, 0);
        if (depth) glNamedFramebufferTexture(f, GL_DEPTH_STENCIL_ATTACHMENT, depth, 0);
        fbos_[{color, depth}] = f;
        return f;
    }

    struct Info { Fmt fmt = Fmt::BGRA8; bool cube = false; };
    struct Read { GLuint pbo = 0; GLsync fence = nullptr; size_t size = 0; void* map = nullptr; bool busy = false; };
    static constexpr size_t kRingBytes = 128u << 20, kChunks = 4, kChunk = kRingBytes / kChunks;

    bool g_s3tc_ = false;
    GLint ubo_align_ = 256;
    GLuint vao_ = 0, ring_ = 0;
    u8* ring_ptr_ = nullptr;
    size_t pos_ = 0, chunk_ = 0;
    GLsync fence_[kChunks] = {};
    u32 attr_on_ = 0;
    int target_w_ = 0, target_h_ = 0;
    std::unordered_map<GLuint, Info> info_;
    std::map<std::pair<GLuint, GLuint>, GLuint> fbos_;
    std::vector<Read> reads_;
    std::vector<GLuint> big_;
    std::unordered_map<GLuint, u64> vk_prog_key_;  // program -> program_key() of its sources
    bool depth_bound_ = false;
    std::vector<std::pair<GLsync, std::vector<GLuint>>> big_fences_;
};

}  // namespace

// ================================================================ shader source store
}  // namespace vc3::gpu::dev
// Shader seed: the shader cache of a build machine, embedded by release builds
// (tools/make_shader_seed.py writes generated/shader_seed.cpp, which registers it here).
namespace vc3::shader_seed {
namespace {
const unsigned char* g_data = nullptr;
std::size_t g_size = 0;
const char* g_id = "";
}  // namespace
void set(const unsigned char* data, std::size_t size, const char* id) {
    g_data = data;
    g_size = size;
    g_id = id;
}
}  // namespace vc3::shader_seed
namespace vc3::gpu::dev {
namespace shader_store {
std::atomic<int> warm_done{0}, warm_total{0};
std::atomic<u64> compile_ns{0}, compile_n{0};
namespace {
const char* const kSourceDir = "work/shadercache/v4/sources";
std::unordered_set<SrcKey, SrcKeyHash> g_saved;
bool g_enabled = false;
}  // namespace

Timer::~Timer() {
    compile_ns += u64(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
    ++compile_n;
}

bool write_file(const std::string& path, const void* a, size_t na, const void* b, size_t nb) {
    const std::string tmp = path + ".tmp";
    std::FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) return false;
    const bool ok = std::fwrite(a, 1, na, f) == na && (!nb || std::fwrite(b, 1, nb, f) == nb);
    std::fclose(f);
    if (ok) std::rename(tmp.c_str(), path.c_str());
    else std::remove(tmp.c_str());
    return ok;
}

// Source file: vertex shader, a NUL byte, fragment shader.
void save(const std::string& vs, const std::string& fs) {
    if (!g_enabled) return;
    const SrcKey k = src_key(vs, fs);
    if (g_saved.count(k)) return;
    const std::string all = vs + '\0' + fs;
    if (write_file(std::string(kSourceDir) + "/" + key_name(k) + ".glsl", all.data(), all.size())) g_saved.insert(k);
}

// Copy the embedded seed into this install's cache once (per seed): its programs and
// pipeline recipes then build at startup with the rest, so a fresh install doesn't meet
// them for the first time mid-game.
void apply_seed() {
    namespace sd = vc3::shader_seed;
    if (!sd::g_data || sd::g_size < 12 || std::memcmp(sd::g_data, "VC3SEED1", 8) != 0) return;
    const std::string marker = std::string("work/shadercache/v4/seeded-") + sd::g_id;
    std::error_code ec;
    if (std::filesystem::exists(marker, ec)) return;
    const unsigned char* p = sd::g_data + 8;
    const unsigned char* const end = sd::g_data + sd::g_size;
    auto u16 = [](const unsigned char* q) { return u32(q[0]) | u32(q[1]) << 8; };
    auto u32le = [](const unsigned char* q) { return u32(q[0]) | u32(q[1]) << 8 | u32(q[2]) << 16 | u32(q[3]) << 24; };
    const u32 count = u32le(p);
    p += 4;
    int sources = 0;
    for (u32 i = 0; i < count && end - p >= 3; ++i) {
        const u8 kind = p[0];
        const u32 nlen = u16(p + 1);
        p += 3;
        if (u32(end - p) < nlen + 4) return;
        const std::string name(reinterpret_cast<const char*>(p), nlen);
        p += nlen;
        const u32 size = u32le(p);
        p += 4;
        if (u32(end - p) < size) return;
        if (name.find_first_of("/\\") == std::string::npos && !name.empty()) {
            if (kind == 0) {
                const std::string path = std::string(kSourceDir) + "/" + name;
                if (!std::filesystem::exists(path, ec) && write_file(path, p, size)) ++sources;
            } else if (kind == 1) {  // appended: duplicates are dropped when the file is read
                if (std::FILE* f = std::fopen("work/shadercache/v4/vk-recipes.bin", "ab")) {
                    std::fwrite(p, 1, size, f);
                    std::fclose(f);
                }
            }
        }
        p += size;
    }
    write_file(marker, "", 0);
    std::fprintf(stderr, "[shaders] added %d known shader programs to the cache\n", sources);
}

void for_each(const std::function<void(const std::string&, const std::string&)>& fn) {
    std::error_code ec;
    std::filesystem::create_directories(kSourceDir, ec);
    if (ec) return;
    apply_seed();
    g_enabled = true;
    for (const auto& e : std::filesystem::directory_iterator(kSourceDir, ec)) {
        if (e.path().extension() != ".glsl") continue;
        std::FILE* f = std::fopen(e.path().string().c_str(), "rb");
        if (!f) continue;
        std::string all;
        char buf[65536];
        for (size_t r; (r = std::fread(buf, 1, sizeof buf, f)) > 0;) all.append(buf, r);
        std::fclose(f);
        const size_t z = all.find('\0');
        if (z == std::string::npos) continue;
        const std::string vs = all.substr(0, z), fs = all.substr(z + 1);
        g_saved.insert(src_key(vs, fs));
        fn(vs, fs);
    }
}

void record_recipe(const void* recipe, size_t size) {
    if (!g_enabled) return;
    static std::unordered_set<u64> known;
    static bool loaded = false;
    const char* const path = "work/shadercache/v4/vk-recipes.bin";
    if (!loaded) {  // what earlier runs (and the seed) recorded
        loaded = true;
        if (std::FILE* f = std::fopen(path, "rb")) {
            std::vector<u8> r(size);
            while (std::fread(r.data(), 1, size, f) == size) known.insert(recipe_fnv(r.data(), size));
            std::fclose(f);
        }
    }
    if (!known.insert(recipe_fnv(recipe, size)).second) return;
    if (std::FILE* f = std::fopen(path, "ab")) {
        std::fwrite(recipe, 1, size, f);
        std::fclose(f);
    }
}

void forget(const std::string& vs, const std::string& fs) {
    std::remove((std::string(kSourceDir) + "/" + key_name(src_key(vs, fs)) + ".glsl").c_str());
}
}  // namespace shader_store

// ================================================================ entry points
namespace gl {
void set_share(void* dpy, void* cfg, void* ctx) {
    g_share_dpy = static_cast<EGLDisplay>(dpy);
    g_share_cfg = static_cast<EGLConfig>(cfg);
    g_share_ctx = static_cast<EGLContext>(ctx);
}
bool shared() { return g_shared; }

bool create_context() {
    if (make_shared_context()) return true;
    // Prefer a hardware device (EGL device platform), fall back to Mesa surfaceless.
    if (epoxy_has_egl_extension(EGL_NO_DISPLAY, "EGL_EXT_device_enumeration") &&
        epoxy_has_egl_extension(EGL_NO_DISPLAY, "EGL_EXT_platform_device")) {
        EGLDeviceEXT devs[16];
        EGLint n = 0;
        eglQueryDevicesEXT(16, devs, &n);
        for (EGLint i = 0; i < n; ++i) {
            EGLDisplay d = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, devs[i], nullptr);
            if (!make_context(d)) {
                std::fprintf(stderr, "[gl] EGL device %d: no context (EGL error %#x)\n", int(i), unsigned(eglGetError()));
                continue;
            }
            const char* r = reinterpret_cast<const char*>(glGetString(GL_RENDERER));
            if (r && !std::strstr(r, "llvmpipe")) return true;  // hardware: done
            eglMakeCurrent(d, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        }
    }
    if (make_context(eglGetPlatformDisplayEXT(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, nullptr))) return true;
    std::fprintf(stderr, "[gl] surfaceless EGL: no context (EGL error %#x)\n", unsigned(eglGetError()));
    return false;
}
}  // namespace gl

float anisotropy_setting() {
    const char* e = std::getenv("VC3_ANISOTROPIC");
    const float v = e ? float(std::atof(e)) : 1.0f;
    return std::clamp(v, 1.0f, 16.0f);
}

Device* create_gl_device() { return new GlDevice(); }

}  // namespace vc3::gpu::dev
