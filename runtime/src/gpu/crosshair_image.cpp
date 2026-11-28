#include "gpu/crosshair_image.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#define STBI_ONLY_TGA
#define STBI_ONLY_JPEG
#define STBI_ONLY_GIF
#define STBI_WINDOWS_UTF8  // file names are UTF-8 (the launcher's file dialog)
#include "stb_image.h"

namespace vc3::gpu::crosshair {

namespace {

struct Source {
    bool tried = false;
    int w = 0, h = 0;
    std::vector<uint8_t> rgba;  // as loaded (straight alpha)
    Image fitted;
    int fitted_size = -1;
};
Source g_src[2];
uint64_t g_version = 0;

void load(int p, Source& s) {
    s.tried = true;
    const char* path = std::getenv(p == 0 ? "VC3_CROSSHAIR_IMAGE_P1" : "VC3_CROSSHAIR_IMAGE_P2");
    if (!path || !*path) return;
    int w = 0, h = 0, n = 0;
    stbi_uc* data = stbi_load(path, &w, &h, &n, 4);
    if (!data || w <= 0 || h <= 0) {
        // (no path in the log: it can name the player)
        std::fprintf(stderr, "[crosshair] player %d: the image couldn't be read (%s); using the built-in crosshair\n", p + 1,
                     stbi_failure_reason() ? stbi_failure_reason() : "unknown");
        if (data) stbi_image_free(data);
        return;
    }
    s.w = w;
    s.h = h;
    s.rgba.assign(data, data + size_t(w) * size_t(h) * 4);
    stbi_image_free(data);
    std::fprintf(stderr, "[crosshair] player %d: custom image %dx%d\n", p + 1, w, h);
}

// Shrink to fit `size` (box filter, alpha-weighted colour), then make it drawable without blending.
void fit(Source& s, int size) {
    Image& out = s.fitted;
    int nw = s.w, nh = s.h;
    if (s.w > size || s.h > size) {
        const double k = double(size) / double(std::max(s.w, s.h));
        nw = std::max(1, int(s.w * k + 0.5));
        nh = std::max(1, int(s.h * k + 0.5));
    }
    out.w = nw;
    out.h = nh;
    out.px.assign(size_t(nw) * size_t(nh), 0);
    for (int y = 0; y < nh; ++y) {
        const int sy0 = int(int64_t(y) * s.h / nh), sy1 = std::max(sy0 + 1, int(int64_t(y + 1) * s.h / nh));
        for (int x = 0; x < nw; ++x) {
            const int sx0 = int(int64_t(x) * s.w / nw), sx1 = std::max(sx0 + 1, int(int64_t(x + 1) * s.w / nw));
            double r = 0, g = 0, b = 0, a = 0;
            int n = 0;
            for (int yy = sy0; yy < sy1; ++yy)
                for (int xx = sx0; xx < sx1; ++xx) {
                    const uint8_t* q = &s.rgba[(size_t(yy) * size_t(s.w) + size_t(xx)) * 4];
                    const double qa = q[3] / 255.0;
                    r += q[0] * qa;
                    g += q[1] * qa;
                    b += q[2] * qa;
                    a += qa;
                    ++n;
                }
            const double alpha = a / n;  // coverage
            if (alpha < 0.25) continue;  // transparent: not drawn
            // Colour over black by coverage (r,g,b are alpha-weighted sums: average = sum / n).
            auto c = [&](double v) { return uint32_t(std::clamp(v / n + 0.5, 0.0, 255.0)); };
            out.px[size_t(y) * size_t(nw) + size_t(x)] = 0xFF000000u | c(r) << 16 | c(g) << 8 | c(b);
        }
    }
    out.version = ++g_version;
    s.fitted_size = size;
}

}  // namespace

bool probe(const char* path, int& w, int& h) {
    int n = 0;
    return path && *path && stbi_info(path, &w, &h, &n) && w > 0 && h > 0;
}

const Image* get(int player, int size) {
    if (player < 0 || player > 1 || size < 1) return nullptr;
    Source& s = g_src[player];
    if (!s.tried) load(player, s);
    if (s.rgba.empty()) return nullptr;
    if (s.fitted_size != size) fit(s, size);
    return &s.fitted;
}

}  // namespace vc3::gpu::crosshair
