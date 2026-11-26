#include "gpu/smaa.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "AreaTex.h"    // SMAA lookup tables (the SMAA checkout's Textures/)
#include "SearchTex.h"
#include "smaa_hlsl.inc"  // kSmaaHlsl: SMAA.hlsl (generated, runtime/CMakeLists.txt)

namespace vc3::gpu::smaa {
namespace {

const char* preset() {
    static const char* p = [] () -> const char* {
        const char* e = std::getenv("VC3_SMAA");
        if (!e) return nullptr;
        const std::string v = e;
        if (v == "low") return "SMAA_PRESET_LOW";
        if (v == "medium") return "SMAA_PRESET_MEDIUM";
        if (v == "high" || v == "1") return "SMAA_PRESET_HIGH";
        if (v == "ultra") return "SMAA_PRESET_ULTRA";
        return nullptr;
    }();
    return p;
}

// Device conventions match SMAA's Direct3D ones: texture row 0 is the picture's top line and
// texture coordinate v grows downwards, so the lookup tables go up as they are.
std::string source(bool vertex, const char* body) {
    std::string s = "#version 450\n";
    s += "layout(std140, binding = 0) uniform SmaaBlock { vec4 uRt; };\n";
    s += "#define SMAA_RT_METRICS uRt\n#define SMAA_GLSL_4\n";
    s += std::string("#define ") + preset() + "\n";
    s += vertex ? "#define SMAA_INCLUDE_PS 0\n" : "#define SMAA_INCLUDE_VS 0\n";
    s += kSmaaHlsl;
    s += "\n";
    s += body;
    return s;
}
// A triangle covering the target; texture coordinates 0..1 across it (clip y -1 = row 0).
constexpr const char* kFullscreen =
    "    vec2 p = vec2((VERTEX_ID << 1) & 2, VERTEX_ID & 2);\n"
    "    vTex = p;\n"
    "    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);\n";

struct State {
    dev::Program edge, weight, blend;
    dev::Tex edges, weights, area, search;
    int w = 0, h = 0;
};
State g;

void make_programs(dev::Device& d) {
    if (g.edge) return;
    g.edge = d.program(source(true, (std::string("layout(location = 0) out vec2 vTex;\nlayout(location = 1) out vec4 vOff[3];\n"
                                                 "void main() {\n") + kFullscreen + "    SMAAEdgeDetectionVS(vTex, vOff);\n}\n").c_str()),
                       source(false, "layout(binding = 4) uniform sampler2D colorTex;\n"
                                     "layout(location = 0) in vec2 vTex;\nlayout(location = 1) in vec4 vOff[3];\n"
                                     "layout(location = 0) out vec4 oColor;\n"
                                     "void main() { oColor = vec4(SMAALumaEdgeDetectionPS(vTex, vOff, colorTex), 0.0, 0.0); }\n"));
    g.weight = d.program(source(true, (std::string("layout(location = 0) out vec2 vTex;\nlayout(location = 1) out vec2 vPix;\n"
                                                   "layout(location = 2) out vec4 vOff[3];\nvoid main() {\n") + kFullscreen +
                                       "    SMAABlendingWeightCalculationVS(vTex, vPix, vOff);\n}\n").c_str()),
                         source(false, "layout(binding = 4) uniform sampler2D edgesTex;\n"
                                       "layout(binding = 5) uniform sampler2D areaTex;\n"
                                       "layout(binding = 6) uniform sampler2D searchTex;\n"
                                       "layout(location = 0) in vec2 vTex;\nlayout(location = 1) in vec2 vPix;\n"
                                       "layout(location = 2) in vec4 vOff[3];\nlayout(location = 0) out vec4 oColor;\n"
                                       "void main() { oColor = SMAABlendingWeightCalculationPS(vTex, vPix, vOff, edgesTex, "
                                       "areaTex, searchTex, vec4(0.0)); }\n"));
    g.blend = d.program(source(true, (std::string("layout(location = 0) out vec2 vTex;\nlayout(location = 1) out vec4 vOff;\n"
                                                  "void main() {\n") + kFullscreen + "    SMAANeighborhoodBlendingVS(vTex, vOff);\n}\n").c_str()),
                        source(false, "layout(binding = 4) uniform sampler2D colorTex;\n"
                                      "layout(binding = 5) uniform sampler2D blendTex;\n"
                                      "layout(location = 0) in vec2 vTex;\nlayout(location = 1) in vec4 vOff;\n"
                                      "layout(location = 0) out vec4 oColor;\n"
                                      "void main() { oColor = SMAANeighborhoodBlendingPS(vTex, vOff, colorTex, blendTex); }\n"));
}

void make_tables(dev::Device& d) {
    if (g.area) return;
    // Textures hold BGRA8 (A8R8G8B8 words): the tables' first channel goes to red, the
    // second to green, as the shader reads them (.rg / .r).
    std::vector<uint32_t> px(size_t(AREATEX_WIDTH) * AREATEX_HEIGHT);
    for (size_t i = 0; i < px.size(); ++i)
        px[i] = 0xFF000000u | uint32_t(areaTexBytes[2 * i]) << 16 | uint32_t(areaTexBytes[2 * i + 1]) << 8;
    g.area = d.create_texture(AREATEX_WIDTH, AREATEX_HEIGHT, 1, dev::Fmt::BGRA8);
    d.upload(g.area, 0, 0, 0, 0, AREATEX_WIDTH, AREATEX_HEIGHT, px.data(), px.size() * 4);
    px.assign(size_t(SEARCHTEX_WIDTH) * SEARCHTEX_HEIGHT, 0);
    for (size_t i = 0; i < px.size(); ++i) px[i] = 0xFF000000u | uint32_t(searchTexBytes[i]) << 16;
    g.search = d.create_texture(SEARCHTEX_WIDTH, SEARCHTEX_HEIGHT, 1, dev::Fmt::BGRA8);
    d.upload(g.search, 0, 0, 0, 0, SEARCHTEX_WIDTH, SEARCHTEX_HEIGHT, px.data(), px.size() * 4);
}

dev::SamplerDesc linear() {
    dev::SamplerDesc s;
    s.min = s.mag = dev::kLinear;
    s.mip = dev::kMipNone;
    s.wrap_u = s.wrap_v = dev::kClampEdge;
    return s;
}

bool pass(dev::Device& d, dev::Program p, dev::Tex target, std::initializer_list<dev::Tex> tex, const dev::Slice& ubo) {
    d.set_target(target, dev::Tex{}, g.w, g.h);
    const float zero[4] = {0, 0, 0, 0};
    d.clear(0, 0, g.w, g.h, true, 0xF, zero, false, 1.0f, false, 0);
    dev::DrawCall dc;
    dc.prog = p;
    dc.ubo0 = ubo;
    int i = 0;
    for (dev::Tex t : tex) {
        dc.tex[i] = t;
        dc.samp[i] = linear();
        ++i;
    }
    dc.first = 0;
    dc.count = 3;
    return d.draw(dc);
}

}  // namespace

bool enabled() { return preset() != nullptr; }

bool apply(dev::Device& d, dev::Tex src, dev::Tex dst, int w, int h) {
    if (!enabled() || w <= 0 || h <= 0) return false;
    make_programs(d);
    make_tables(d);
    if (!g.edges || g.w != w || g.h != h) {
        if (g.edges) d.destroy_texture(g.edges);
        if (g.weights) d.destroy_texture(g.weights);
        g.edges = d.create_texture(w, h, 1, dev::Fmt::BGRA8);
        g.weights = d.create_texture(w, h, 1, dev::Fmt::BGRA8);
        g.w = w;
        g.h = h;
    }
    dev::Slice ubo;
    float* rt = reinterpret_cast<float*>(d.stream(16, d.ubo_align(), ubo));
    if (!rt) return false;
    rt[0] = 1.0f / float(w);
    rt[1] = 1.0f / float(h);
    rt[2] = float(w);
    rt[3] = float(h);
    // Each pass needs the one before; the last writes `dst` (a pass not drawn yet: copy).
    return pass(d, g.edge, g.edges, {src}, ubo) && pass(d, g.weight, g.weights, {g.edges, g.area, g.search}, ubo) &&
           pass(d, g.blend, dst, {src, g.weights}, ubo);
}

}  // namespace vc3::gpu::smaa
