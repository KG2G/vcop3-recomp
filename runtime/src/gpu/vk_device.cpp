// Vulkan 1.3 implementation of the device layer (device.h).
//
// One command buffer is recorded at a time on the GPU thread and submitted at the end of
// each guest frame (flush) or when the CPU needs results (read-backs, fences, a full
// streaming chunk). Every image lives in VK_IMAGE_LAYOUT_GENERAL, so synchronisation is
// a global memory barrier, issued only where an image is about to be used in a way that
// conflicts with how it was used since the last barrier (sampling a render target,
// writing a texture that was read, ...). Rendering uses dynamic rendering; everything
// the NV2A changes per draw is dynamic state except blending, colour masks, vertex
// layouts and topology, which select a cached pipeline (saved across runs).
#include "gpu/vk_device.h"
#include "gpu/gl_device.h"  // shader_store

#include <SDL3/SDL_vulkan.h>
#include <shaderc/shaderc.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_set>
#include <string>
#include <unordered_map>
#include <vector>

namespace vc3::gpu::dev {

namespace vk {
namespace {
Context g_ctx;
bool g_init_done = false, g_init_ok = false;
VkSemaphore g_gpu_tl = VK_NULL_HANDLE, g_present_tl = VK_NULL_HANDLE;

#define VKCHECK(x)                                                                                  \
    do {                                                                                            \
        const VkResult r_ = (x);                                                                    \
        if (r_ != VK_SUCCESS) {                                                                     \
            std::fprintf(stderr, "[vk] %s failed: %d (%s:%d)\n", #x, int(r_), __FILE__, __LINE__);  \
            std::abort();                                                                           \
        }                                                                                           \
    } while (0)

bool has_ext(const std::vector<VkExtensionProperties>& exts, const char* name) {
    for (const auto& e : exts)
        if (!std::strcmp(e.extensionName, name)) return true;
    return false;
}

VkSemaphore make_timeline() {
    VkSemaphoreTypeCreateInfo t{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    t.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    t.initialValue = 0;
    VkSemaphoreCreateInfo ci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    ci.pNext = &t;
    VkSemaphore s;
    VKCHECK(vkCreateSemaphore(g_ctx.device, &ci, nullptr, &s));
    return s;
}
}  // namespace

Context& ctx() { return g_ctx; }
VkSemaphore gpu_timeline() { return g_gpu_tl; }
VkSemaphore presenter_timeline() { return g_present_tl; }

namespace {
std::mutex g_psub_m;
std::condition_variable g_psub_cv;
uint64_t g_psub = 0;  // highest presenter value submitted
}  // namespace
void presenter_submitted(uint64_t v) {
    {
        std::lock_guard lk(g_psub_m);
        g_psub = std::max(g_psub, v);
    }
    g_psub_cv.notify_all();
}
uint64_t presenter_submitted_value() {
    std::lock_guard lk(g_psub_m);
    return g_psub;
}
// Blocks until the presenter has submitted value v (see presenter_submitted).
void wait_presenter_submitted(uint64_t v) {
    std::unique_lock lk(g_psub_m);
    for (int tries = 0; g_psub < v; ++tries) {
        if (g_psub_cv.wait_for(lk, std::chrono::seconds(2), [&] { return g_psub >= v; })) break;
        std::fprintf(stderr, "[vk] waiting for the window to take frame %llu (%d s)\n", (unsigned long long)v, 2 * (tries + 1));
    }
}

uint32_t memory_type(uint32_t bits, VkMemoryPropertyFlags flags) {
    for (uint32_t i = 0; i < g_ctx.mem.memoryTypeCount; ++i)
        if ((bits >> i & 1) && (g_ctx.mem.memoryTypes[i].propertyFlags & flags) == flags) return i;
    return UINT32_MAX;
}

bool init(SDL_Window* window) {
    if (g_init_done) return g_init_ok;
    g_init_done = true;
    Context& c = g_ctx;
    // ---- instance
    uint32_t n = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> iexts(n);
    vkEnumerateInstanceExtensionProperties(nullptr, &n, iexts.data());
    std::vector<const char*> want;
    if (window) {
        Uint32 sn = 0;
        const char* const* se = SDL_Vulkan_GetInstanceExtensions(&sn);
        for (Uint32 i = 0; i < sn; ++i) want.push_back(se[i]);
    }
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    if (has_ext(iexts, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {  // MoltenVK
        want.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        ici.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    }
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "vc3-recomp";
    app.apiVersion = VK_API_VERSION_1_3;
    ici.pApplicationInfo = &app;
    ici.enabledExtensionCount = uint32_t(want.size());
    ici.ppEnabledExtensionNames = want.data();
    const char* validation = "VK_LAYER_KHRONOS_validation";
    const char* v = std::getenv("VC3_VK_VALIDATION");
    if (v && !std::strcmp(v, "1")) {
        ici.enabledLayerCount = 1;
        ici.ppEnabledLayerNames = &validation;
    }
    if (vkCreateInstance(&ici, nullptr, &c.instance) != VK_SUCCESS) {
        std::fprintf(stderr, "[vk] no Vulkan instance\n");
        return false;
    }
    if (window && !SDL_Vulkan_CreateSurface(window, c.instance, nullptr, &c.surface)) {
        std::fprintf(stderr, "[vk] window surface: %s\n", SDL_GetError());
        c.surface = VK_NULL_HANDLE;
    }
    // ---- physical device: Vulkan 1.3, a graphics queue (that can present), discrete first
    vkEnumeratePhysicalDevices(c.instance, &n, nullptr);
    std::vector<VkPhysicalDevice> devs(n);
    vkEnumeratePhysicalDevices(c.instance, &n, devs.data());
    const char* pick = std::getenv("VC3_VK_DEVICE");
    int best_score = -1;
    for (uint32_t i = 0; i < devs.size(); ++i) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(devs[i], &p);
        if (p.apiVersion < VK_API_VERSION_1_3) continue;
        uint32_t qn = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(devs[i], &qn, nullptr);
        std::vector<VkQueueFamilyProperties> qf(qn);
        vkGetPhysicalDeviceQueueFamilyProperties(devs[i], &qn, qf.data());
        int fam = -1;
        for (uint32_t q = 0; q < qn && fam < 0; ++q) {
            if (!(qf[q].queueFlags & VK_QUEUE_GRAPHICS_BIT)) continue;
            VkBool32 present = VK_TRUE;
            if (c.surface) vkGetPhysicalDeviceSurfaceSupportKHR(devs[i], q, c.surface, &present);
            if (present) fam = int(q);
        }
        if (fam < 0) continue;
        int score = p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 3
                    : p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 2 : 1;
        if (pick && std::atoi(pick) == int(i)) score = 100;
        if (score > best_score) {
            best_score = score;
            c.phys = devs[i];
            c.queue_family = uint32_t(fam);
            c.props = p;
        }
    }
    if (!c.phys) {
        std::fprintf(stderr, "[vk] no Vulkan 1.3 device with a graphics queue\n");
        return false;
    }
    vkGetPhysicalDeviceMemoryProperties(c.phys, &c.mem);
    // ---- device
    vkEnumerateDeviceExtensionProperties(c.phys, nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> dexts(n);
    vkEnumerateDeviceExtensionProperties(c.phys, nullptr, &n, dexts.data());
    std::vector<const char*> dext;
    if (c.surface && has_ext(dexts, VK_KHR_SWAPCHAIN_EXTENSION_NAME)) {
        dext.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
        c.swapchain = true;
    }
    if (!has_ext(dexts, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME)) {
        std::fprintf(stderr, "[vk] %s lacks VK_KHR_push_descriptor\n", c.props.deviceName);
        return false;
    }
    dext.push_back(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
    if (has_ext(dexts, "VK_KHR_portability_subset")) dext.push_back("VK_KHR_portability_subset");
    VkPhysicalDeviceCustomBorderColorFeaturesEXT cb{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_FEATURES_EXT};
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.pNext = &f12;
    f12.pNext = &f13;
    if (has_ext(dexts, VK_EXT_CUSTOM_BORDER_COLOR_EXTENSION_NAME)) f13.pNext = &cb;
    vkGetPhysicalDeviceFeatures2(c.phys, &f2);
    if (!f13.dynamicRendering || !f13.synchronization2 || !f12.timelineSemaphore) {
        std::fprintf(stderr, "[vk] %s lacks dynamic rendering / synchronization2 / timeline semaphores\n", c.props.deviceName);
        return false;
    }
    VkPhysicalDeviceCustomBorderColorFeaturesEXT cb_on{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_FEATURES_EXT};
    VkPhysicalDeviceVulkan13Features on13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    on13.dynamicRendering = VK_TRUE;
    on13.synchronization2 = VK_TRUE;
    VkPhysicalDeviceVulkan12Features on12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    on12.timelineSemaphore = VK_TRUE;
    on12.pNext = &on13;
    if (cb.customBorderColors && cb.customBorderColorWithoutFormat) {
        cb_on.customBorderColors = VK_TRUE;
        cb_on.customBorderColorWithoutFormat = VK_TRUE;
        on13.pNext = &cb_on;
        dext.push_back(VK_EXT_CUSTOM_BORDER_COLOR_EXTENSION_NAME);
        c.custom_border = true;
    }
    VkPhysicalDeviceFeatures2 on{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    on.pNext = &on12;
    on.features.textureCompressionBC = f2.features.textureCompressionBC;
    on.features.largePoints = f2.features.largePoints;
    on.features.samplerAnisotropy = f2.features.samplerAnisotropy;  // VC3_ANISOTROPIC
    c.aniso = f2.features.samplerAnisotropy;
    c.bc = f2.features.textureCompressionBC && !std::getenv("VC3_NO_S3TC");
    const float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = c.queue_family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &on;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = uint32_t(dext.size());
    dci.ppEnabledExtensionNames = dext.data();
    if (vkCreateDevice(c.phys, &dci, nullptr, &c.device) != VK_SUCCESS) {
        std::fprintf(stderr, "[vk] vkCreateDevice failed\n");
        return false;
    }
    vkGetDeviceQueue(c.device, c.queue_family, 0, &c.queue);
    c.push_descriptor = reinterpret_cast<PFN_vkCmdPushDescriptorSetKHR>(vkGetDeviceProcAddr(c.device, "vkCmdPushDescriptorSetKHR"));
    g_gpu_tl = make_timeline();
    g_present_tl = make_timeline();
    std::fprintf(stderr, "[vk] %s (Vulkan %u.%u.%u, driver %#x)%s%s\n", c.props.deviceName, VK_API_VERSION_MAJOR(c.props.apiVersion),
                 VK_API_VERSION_MINOR(c.props.apiVersion), VK_API_VERSION_PATCH(c.props.apiVersion), c.props.driverVersion,
                 c.bc ? ", BC textures" : "", c.custom_border ? ", custom border colours" : "");
    g_init_ok = true;
    return true;
}
}  // namespace vk

namespace {
using vk::g_ctx;

// ================================================================ helpers
VkCompareOp vk_compare(u8 f) { return VkCompareOp(f & 7); }  // GL order == Vulkan order
VkStencilOp vk_stencil_op(u8 op) {
    static constexpr VkStencilOp k[] = {VK_STENCIL_OP_KEEP, VK_STENCIL_OP_ZERO, VK_STENCIL_OP_REPLACE,
                                        VK_STENCIL_OP_INCREMENT_AND_CLAMP, VK_STENCIL_OP_DECREMENT_AND_CLAMP,
                                        VK_STENCIL_OP_INVERT, VK_STENCIL_OP_INCREMENT_AND_WRAP, VK_STENCIL_OP_DECREMENT_AND_WRAP};
    return k[op & 7];
}
VkBlendFactor vk_blend_factor(u8 f) {
    static constexpr VkBlendFactor k[] = {
        VK_BLEND_FACTOR_ZERO, VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_SRC_COLOR, VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR,
        VK_BLEND_FACTOR_SRC_ALPHA, VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, VK_BLEND_FACTOR_DST_ALPHA,
        VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA, VK_BLEND_FACTOR_DST_COLOR, VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR,
        VK_BLEND_FACTOR_SRC_ALPHA_SATURATE, VK_BLEND_FACTOR_CONSTANT_COLOR, VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR,
        VK_BLEND_FACTOR_CONSTANT_ALPHA, VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA};
    return f < 15 ? k[f] : VK_BLEND_FACTOR_ONE;
}
VkBlendOp vk_blend_op(u8 e) {
    static constexpr VkBlendOp k[] = {VK_BLEND_OP_ADD, VK_BLEND_OP_SUBTRACT, VK_BLEND_OP_REVERSE_SUBTRACT, VK_BLEND_OP_MIN,
                                      VK_BLEND_OP_MAX};
    return e < 5 ? k[e] : VK_BLEND_OP_ADD;
}
VkPrimitiveTopology vk_topology(u8 t) {
    switch (t) {
        case kPoints: return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
        case kLines: return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
        case kLineLoop:  // drawn as a strip with the first vertex repeated
        case kLineStrip: return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
        case kTriStrip: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
        case kTriFan: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;
        default: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    }
}
VkFormat vk_attr_format(u8 f) {
    switch (f) {
        case kF32x1: return VK_FORMAT_R32_SFLOAT;
        case kF32x2: return VK_FORMAT_R32G32_SFLOAT;
        case kF32x3: return VK_FORMAT_R32G32B32_SFLOAT;
        case kF32x4: return VK_FORMAT_R32G32B32A32_SFLOAT;
        case kUN8x1: return VK_FORMAT_R8_UNORM;
        case kUN8x2: return VK_FORMAT_R8G8_UNORM;
        case kUN8x3: return VK_FORMAT_R8G8B8_UNORM;
        case kUN8x4: return VK_FORMAT_R8G8B8A8_UNORM;
        case kBGRA8: return VK_FORMAT_B8G8R8A8_UNORM;
        case kSN16x1: return VK_FORMAT_R16_SNORM;
        case kSN16x2: return VK_FORMAT_R16G16_SNORM;
        case kSN16x3: return VK_FORMAT_R16G16B16_SNORM;
        case kSN16x4: return VK_FORMAT_R16G16B16A16_SNORM;
        case kS16x1: return VK_FORMAT_R16_SSCALED;
        case kS16x2: return VK_FORMAT_R16G16_SSCALED;
        case kS16x3: return VK_FORMAT_R16G16B16_SSCALED;
        case kS16x4: return VK_FORMAT_R16G16B16A16_SSCALED;
        case kU32x1: return VK_FORMAT_R32_UINT;
        default: return VK_FORMAT_R32G32B32A32_SFLOAT;
    }
}
VkSamplerAddressMode vk_wrap(u8 w) {
    switch (w) {
        case kRepeat: return VK_SAMPLER_ADDRESS_MODE_REPEAT;
        case kMirror: return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
        case kClampBorder: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        default: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    }
}

u64 fnv(const void* p, size_t n, u64 h = 0xCBF29CE484222325ull) {
    const u8* b = static_cast<const u8*>(p);
    for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 0x100000001B3ull;
    return h;
}

// ================================================================ the device
class VulkanDevice final : public Device {
public:
    VulkanDevice() : c_(g_ctx), dev_(g_ctx.device) {
        // Command buffers (a small ring, each reused once its submission completed)
        VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pci.queueFamilyIndex = c_.queue_family;
        VKCHECK(vkCreateCommandPool(dev_, &pci, nullptr, &pool_));
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = pool_;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = kCmdBufs;
        VkCommandBuffer cbs[kCmdBufs];
        VKCHECK(vkAllocateCommandBuffers(dev_, &ai, cbs));
        for (u32 i = 0; i < kCmdBufs; ++i) cmd_[i].cb = cbs[i];
        // Streaming ring
        ring_ = make_buffer(kRingBytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        // Descriptor set layout (push): 0, 1 uniform, 3 storage, 4-11 combined samplers
        VkDescriptorSetLayoutBinding b[11]{};
        int nb = 0;
        auto bind = [&](u32 binding, VkDescriptorType t, VkShaderStageFlags st) {
            b[nb].binding = binding;
            b[nb].descriptorType = t;
            b[nb].descriptorCount = 1;
            b[nb].stageFlags = st;
            ++nb;
        };
        bind(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
        bind(1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT);
        bind(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT);
        for (u32 i = 4; i < 12; ++i) bind(i, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT);
        VkDescriptorSetLayoutCreateInfo dl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dl.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
        dl.bindingCount = u32(nb);
        dl.pBindings = b;
        VKCHECK(vkCreateDescriptorSetLayout(dev_, &dl, nullptr, &set_layout_));
        VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pl.setLayoutCount = 1;
        pl.pSetLayouts = &set_layout_;
        VKCHECK(vkCreatePipelineLayout(dev_, &pl, nullptr, &layout_));
        // Masked clears draw a quad: colour from a push constant
        VkPushConstantRange pc{VK_SHADER_STAGE_FRAGMENT_BIT, 0, 16};
        VkPipelineLayoutCreateInfo cl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        cl.pushConstantRangeCount = 1;
        cl.pPushConstantRanges = &pc;
        VKCHECK(vkCreatePipelineLayout(dev_, &cl, nullptr, &clear_layout_));
        // Depth format: D24S8 where supported (AMD has none: D32F S8)
        VkFormatProperties fp;
        vkGetPhysicalDeviceFormatProperties(c_.phys, VK_FORMAT_D24_UNORM_S8_UINT, &fp);
        depth_fmt_ = (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) && !std::getenv("VC3_VK_DEPTH32")
                         ? VK_FORMAT_D24_UNORM_S8_UINT
                         : VK_FORMAT_D32_SFLOAT_S8_UINT;
        // Vertex formats the driver can't fetch (AMD lacks 3-component 8/16-bit ones) are
        // converted to floats on the CPU per draw. VC3_VK_CONVERT_ATTRS=1 converts every
        // non-float format (to test that path on any GPU).
        const bool force = std::getenv("VC3_VK_CONVERT_ATTRS") != nullptr;
        std::string conv;
        for (u8 f = kF32x1; f <= kU32x1; ++f) {
            vkGetPhysicalDeviceFormatProperties(c_.phys, vk_attr_format(f), &fp);
            attr_ok_[f] = (fp.bufferFeatures & VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT) &&
                          !(force && f != kU32x1 && (f < kF32x1 || f > kF32x4));
            if (!attr_ok_[f]) conv += " " + std::to_string(int(f));
        }
        if (!conv.empty()) std::fprintf(stderr, "[vk] vertex formats converted on the CPU:%s\n", conv.c_str());
        compiler_ = shaderc_compiler_initialize();
        init_pipeline_cache();
        // Placeholder textures for unused sampler bindings (shaders declare all eight)
        dummy2d_ = create_texture(1, 1, 1, Fmt::BGRA8, false);
        dummycube_ = create_texture(1, 1, 1, Fmt::BGRA8, true);
        const u32 zero[6] = {};
        upload(dummy2d_, 0, 0, 0, 0, 1, 1, zero, 4);
        for (int f = 0; f < 6; ++f) upload(dummycube_, 0, f, 0, 0, 1, 1, zero, 4);
        std::fprintf(stderr, "[vk] device ready (depth %s)\n", depth_fmt_ == VK_FORMAT_D24_UNORM_S8_UINT ? "D24S8" : "D32F S8");
        max_aniso_ = anisotropy_setting();
        {
            const char* a = std::getenv("VC3_ASYNC_SHADERS");
            async_ = !(a && !std::strcmp(a, "0"));
            const int n = std::clamp(int(std::thread::hardware_concurrency()) / 2, 1, 6);
            workers_.start(n);
            std::fprintf(stderr, "[vk] new shaders compile %s (%d workers)\n", async_ ? "in the background" : "immediately", n);
        }
        warm_up();
    }

    const char* name() const override { return "Vulkan"; }
    bool supports_bc() const override { return c_.bc; }
    size_t ubo_align() const override {
        return std::max<size_t>(16, size_t(c_.props.limits.minUniformBufferOffsetAlignment));
    }

    // ================================================================ textures
    Tex create_texture(int w, int h, int levels, Fmt fmt, bool cube) override {
        Image im;
        im.w = w;
        im.h = h;
        im.levels = std::max(1, levels);
        im.cube = cube;
        im.depth = fmt == Fmt::D24S8;
        switch (fmt) {
            case Fmt::D24S8: im.fmt = depth_fmt_; break;
            case Fmt::BC1: im.fmt = VK_FORMAT_BC1_RGBA_UNORM_BLOCK; break;
            case Fmt::BC2: im.fmt = VK_FORMAT_BC2_UNORM_BLOCK; break;
            case Fmt::BC3: im.fmt = VK_FORMAT_BC3_UNORM_BLOCK; break;
            default: im.fmt = VK_FORMAT_B8G8R8A8_UNORM; break;
        }
        const bool bc = fmt == Fmt::BC1 || fmt == Fmt::BC2 || fmt == Fmt::BC3;
        VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ci.flags = cube ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0;
        ci.imageType = VK_IMAGE_TYPE_2D;
        ci.format = im.fmt;
        ci.extent = {u32(w), u32(h), 1};
        ci.mipLevels = u32(im.levels);
        ci.arrayLayers = cube ? 6 : 1;
        ci.samples = VK_SAMPLE_COUNT_1_BIT;
        ci.tiling = VK_IMAGE_TILING_OPTIMAL;
        ci.usage = im.depth ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
                            : VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                  (bc || cube ? 0 : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
        ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VKCHECK(vkCreateImage(dev_, &ci, nullptr, &im.img));
        VkMemoryRequirements mr;
        vkGetImageMemoryRequirements(dev_, im.img, &mr);
        im.mem = alloc(mr, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VKCHECK(vkBindImageMemory(dev_, im.img, im.mem, 0));
        im.view = make_view(im, false);
        if (!im.depth) im.view_a1 = make_view(im, true);
        // Into GENERAL (depth: attachment-optimal) once, before anything uses it.
        end_rendering();
        VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        b.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
        b.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        b.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
        b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout = im.depth ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_GENERAL;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = im.img;
        b.subresourceRange = {aspect(im), 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
        VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        di.imageMemoryBarrierCount = 1;
        di.pImageMemoryBarriers = &b;
        vkCmdPipelineBarrier2(cb(), &di);
        u32 id;
        std::lock_guard lk(images_m_);  // the window thread looks images up
        if (!free_images_.empty()) {
            id = free_images_.back();
            free_images_.pop_back();
            images_[id - 1] = im;
        } else {
            images_.push_back(im);
            id = u32(images_.size());
        }
        return Tex{id};
    }
    void destroy_texture(Tex t) override {
        if (!t) return;
        Image im = images_[t.id - 1];
        {
            std::lock_guard lk(images_m_);
            images_[t.id - 1] = Image{};
        }
        if (cur_color_ == t.id || cur_depth_ == t.id) {
            end_rendering();
            if (cur_color_ == t.id) cur_color_ = 0;
            if (cur_depth_ == t.id) cur_depth_ = 0;
        }
        // Freed once the GPU is past everything recorded so far.
        later([this, im] {
            if (im.view) vkDestroyImageView(dev_, im.view, nullptr);
            if (im.view_a1) vkDestroyImageView(dev_, im.view_a1, nullptr);
            vkDestroyImage(dev_, im.img, nullptr);
            vkFreeMemory(dev_, im.mem, nullptr);
        });
        free_images_.push_back(t.id);
    }
    void upload(Tex t, int level, int face, int x, int y, int w, int h, const void* data, size_t bytes) override {
        Image& im = images_[t.id - 1];
        Slice s;
        u8* p = stream(bytes, 16, s);
        std::memcpy(p, data, bytes);
        transfer_write(im);
        VkBufferImageCopy r{};
        r.bufferOffset = s.off;
        r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, u32(level), u32(face), 1};
        r.imageOffset = {x, y, 0};
        r.imageExtent = {u32(w), u32(h), 1};
        vkCmdCopyBufferToImage(cb(), buffer_of(s), im.img, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
    }
    void copy(Tex src, Tex dst, int w, int h) override { copy_region(src, dst, 0, 0, w, h); }
    void copy_region(Tex src, Tex dst, int x, int y, int w, int h) override {
        Image &s = images_[src.id - 1], &d = images_[dst.id - 1];
        transfer_read(s);
        transfer_write(d);
        VkImageCopy r{};
        r.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        r.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        r.srcOffset = {x, y, 0};
        r.dstOffset = {x, y, 0};
        r.extent = {u32(w), u32(h), 1};
        vkCmdCopyImage(cb(), s.img, VK_IMAGE_LAYOUT_GENERAL, d.img, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
    }
    void blit(Tex src, int sx0, int sy0, int sx1, int sy1, Tex dst, int dx0, int dy0, int dx1, int dy1, bool linear) override {
        Image &s = images_[src.id - 1], &d = images_[dst.id - 1];
        transfer_read(s);
        transfer_write(d);
        VkImageBlit r{};
        r.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        r.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        r.srcOffsets[0] = {sx0, sy0, 0};
        r.srcOffsets[1] = {sx1, sy1, 1};
        r.dstOffsets[0] = {dx0, dy0, 0};
        r.dstOffsets[1] = {dx1, dy1, 1};
        vkCmdBlitImage(cb(), s.img, VK_IMAGE_LAYOUT_GENERAL, d.img, VK_IMAGE_LAYOUT_GENERAL, 1, &r,
                       linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST);
    }

    // ================================================================ read-back
    void read(Tex t, int w, int h, void* dst) override {
        const u32 id = read_start(t, w, h);
        const void* p = read_poll(id, true);
        if (p) std::memcpy(dst, p, size_t(w) * size_t(h) * 4);
        read_release(id);
    }
    u32 read_start(Tex t, int w, int h) override {
        const size_t bytes = size_t(w) * size_t(h) * 4;
        u32 id = 0;
        for (u32 i = 0; i < reads_.size() && !id; ++i)
            if (!reads_[i].busy && reads_[i].buf.size == bytes) id = i + 1;
        if (!id) {
            for (u32 i = 0; i < reads_.size() && !id; ++i)
                if (!reads_[i].busy) {
                    free_buffer(reads_[i].buf);
                    id = i + 1;
                }
            if (!id) {
                reads_.push_back({});
                id = u32(reads_.size());
            }
            reads_[id - 1].buf = make_buffer(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
        }
        Read& r = reads_[id - 1];
        Image& im = images_[t.id - 1];
        transfer_read(im);
        VkBufferImageCopy c{};
        c.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        c.imageExtent = {u32(w), u32(h), 1};
        vkCmdCopyImageToBuffer(cb(), im.img, VK_IMAGE_LAYOUT_GENERAL, r.buf.buf, 1, &c);
        // Make the copy visible to the host once it completes.
        VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        mb.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        mb.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        mb.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
        mb.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
        VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        di.memoryBarrierCount = 1;
        di.pMemoryBarriers = &mb;
        vkCmdPipelineBarrier2(cb(), &di);
        r.value = submit();
        r.busy = true;
        return id;
    }
    const void* read_poll(u32 id, bool wait_done) override {
        Read& r = reads_[id - 1];
        if (!reached(r.value)) {
            if (!wait_done) return nullptr;
            wait_value(g_gpu_tl(), r.value);
        }
        if (!r.buf.coherent) {
            VkMappedMemoryRange mr{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
            mr.memory = r.buf.mem;
            mr.size = VK_WHOLE_SIZE;
            vkInvalidateMappedMemoryRanges(dev_, 1, &mr);
        }
        return r.buf.map;
    }
    void read_release(u32 id) override { reads_[id - 1].busy = false; }

    // ================================================================ rendering
    void set_target(Tex color, Tex depth, int w, int h) override {
        if (color.id == cur_color_ && depth.id == cur_depth_) return;
        end_rendering();
        cur_color_ = color.id;
        cur_depth_ = depth.id;
        target_w_ = w;
        target_h_ = h;
    }
    void clear(int x, int y, int w, int h, bool color, u8 color_mask, const float rgba[4], bool depth, float z, bool stencil,
               u8 s) override {
        x = std::max(0, x);
        y = std::max(0, y);
        w = std::min(w, target_w_ - x);
        h = std::min(h, target_h_ - y);
        if (w <= 0 || h <= 0) return;
        if (!cur_depth_) depth = stencil = false;
        if (!color && !depth && !stencil) return;
        begin_rendering();
        VkClearRect rect{{{x, y}, {u32(w), u32(h)}}, 0, 1};
        VkClearAttachment at[2];
        u32 n = 0;
        const bool quad = color && (color_mask & 0xF) != 0xF;
        if (color && !quad) {
            at[n].aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            at[n].colorAttachment = 0;
            std::memcpy(at[n].clearValue.color.float32, rgba, 16);
            ++n;
        }
        if (depth || stencil) {
            at[n].aspectMask = (depth ? VK_IMAGE_ASPECT_DEPTH_BIT : 0) | (stencil ? VK_IMAGE_ASPECT_STENCIL_BIT : 0);
            at[n].clearValue.depthStencil = {z, u32(s)};
            ++n;
        }
        if (n) vkCmdClearAttachments(cb(), n, at, 1, &rect);
        if (quad && (color_mask & 0xF)) clear_quad(x, y, w, h, color_mask, rgba);
    }

    Program program(const std::string& vs, const std::string& fs) override {
        shader_store::save(vs, fs);
        return build(vs, fs, async_);
    }

    u8* stream(size_t bytes, size_t align, Slice& out) override {
        if (bytes > kChunk) {  // a buffer of its own, freed once used
            Buffer b = make_buffer(bytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                              VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                              VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            const u32 id = next_big_++;
            big_[id] = b;
            out = {id, 0, bytes};
            later([this, b, id] {
                free_buffer(b);
                big_.erase(id);
            });
            return static_cast<u8*>(b.map);
        }
        size_t pos = (pos_ + align - 1) / align * align;
        size_t chunk = chunk_;  // (pos_ can sit exactly on the next chunk's start)
        if (pos + bytes > (chunk + 1) * kChunk) {
            // Leaving this chunk: it is in use until the submission now being recorded
            // completes; the next chunk must be free (the GPU past its last user).
            chunk_value_[chunk] = submitted_ + 1;
            chunk = (chunk + 1) % kChunks;
            pos = chunk * kChunk;
            if (chunk_value_[chunk]) wait_value(g_gpu_tl(), chunk_value_[chunk]);
            chunk_ = chunk;
        }
        pos_ = pos + bytes;
        out = {kRingId, pos, bytes};
        return static_cast<u8*>(ring_.map) + pos;
    }

    bool draw(const DrawCall& dc) override {
        if (!cur_color_) return false;
        // Textures first: sampling something written since the last barrier needs one
        // (and that ends the rendering pass).
        VkDescriptorImageInfo ii[8];
        for (int i = 0; i < 8; ++i) {
            const bool cube = i >= 4;
            Tex t = dc.tex[i];
            if (!t || images_[t.id - 1].cube != cube) t = cube ? dummycube_ : dummy2d_;
            Image& im = images_[t.id - 1];
            sample_read(im);
            ii[i].sampler = sampler(dc.samp[i]);
            ii[i].imageView = dc.samp[i].alpha_one && im.view_a1 ? im.view_a1 : im.view;
            ii[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        }
        const RasterState& r = dc.rs;
        // Constant attributes: one stride-0 binding each, after the draw's own bindings.
        VkBuffer bufs[32];
        VkDeviceSize offs[32], strides[32];
        u32 nb = dc.nbind;
        for (u32 b = 0; b < dc.nbind; ++b) {
            bufs[b] = buffer_of(dc.vbuf[b]);
            offs[b] = dc.vbuf[b].off;
            strides[b] = dc.stride[b];
        }
        DrawCall key_dc = dc;  // constant attributes get their binding numbers here
        for (int i = 0; i < dc.nattr; ++i)
            if (dc.attr[i].constant) {
                Slice s;
                std::memcpy(stream(16, 16, s), dc.attr[i].value, 16);
                key_dc.attr[i].binding = u8(nb);
                bufs[nb] = buffer_of(s);
                offs[nb] = s.off;
                strides[nb] = 0;
                ++nb;
            }
        // Attributes in formats the driver can't fetch: converted to float4 in a binding of their own.
        for (int i = 0; i < dc.nattr; ++i) {
            const VertexAttr& a = dc.attr[i];
            if (a.constant || attr_ok_[a.fmt]) continue;
            const Slice& src = dc.vbuf[a.binding];
            const u32 stride = dc.stride[a.binding];
            const size_t need = attr_bytes(a.fmt);
            if (a.offset + need > src.size) continue;
            const size_t count = stride ? (src.size - a.offset - need) / stride + 1 : 1;
            const u8* in = static_cast<const u8*>(map_of(src)) + src.off + a.offset;
            Slice s;
            float* out = reinterpret_cast<float*>(stream(count * 16, 16, s));
            for (size_t v = 0; v < count; ++v) decode_attr(a.fmt, in + v * stride, out + v * 4);
            key_dc.attr[i].fmt = kF32x4;
            key_dc.attr[i].binding = u8(nb);
            key_dc.attr[i].offset = 0;
            bufs[nb] = buffer_of(s);
            offs[nb] = s.off;
            strides[nb] = 16;
            ++nb;
        }
        Slice hud = dc.hud;
        if (!hud) std::memset(stream(16, 256, hud), 0, 16);
        // Line loops: drawn as a strip closed by the first vertex again.
        Slice idx = dc.index;
        u32 count = dc.count;
        if (r.topology == kLineLoop && count > 1) {
            const u32* src = dc.index ? reinterpret_cast<const u32*>(static_cast<const u8*>(map_of(dc.index)) + dc.index.off)
                                      : nullptr;
            u32* p = reinterpret_cast<u32*>(stream(size_t(count + 1) * 4, 4, idx));
            for (u32 i = 0; i < count; ++i) p[i] = src ? src[i] : dc.first + i;
            p[count] = src ? src[0] : dc.first;
            ++count;
        }
        // Everything streamed: now record.
        begin_rendering();
        VkCommandBuffer c = cb();
        const VkPipeline pipe = progs_[dc.prog.id - 1]->state == 1 ? pipeline(key_dc, nb) : VK_NULL_HANDLE;
        if (!pipe) return false;  // still being built in the background
        if (pipe != bound_pipe_) {
            vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
            bound_pipe_ = pipe;
        }
        if (nb) vkCmdBindVertexBuffers2(c, 0, nb, bufs, offs, nullptr, strides);
        // Descriptors
        VkDescriptorBufferInfo ub0{buffer_of(dc.ubo0), dc.ubo0.off, dc.ubo0.size};
        VkDescriptorBufferInfo ub1{buffer_of(dc.ubo1), dc.ubo1.off, std::max<size_t>(dc.ubo1.size, 16)};
        VkDescriptorBufferInfo hb{buffer_of(hud), hud.off, hud.size};
        VkWriteDescriptorSet w[11]{};
        u32 nw = 0;
        auto wbuf = [&](u32 binding, VkDescriptorType t, const VkDescriptorBufferInfo* bi) {
            w[nw].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[nw].dstBinding = binding;
            w[nw].descriptorCount = 1;
            w[nw].descriptorType = t;
            w[nw].pBufferInfo = bi;
            ++nw;
        };
        wbuf(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, &ub0);
        if (dc.ubo1) wbuf(1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, &ub1);
        wbuf(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &hb);
        for (u32 i = 0; i < 8; ++i) {
            w[nw].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[nw].dstBinding = 4 + i;
            w[nw].descriptorCount = 1;
            w[nw].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[nw].pImageInfo = &ii[i];
            ++nw;
        }
        c_.push_descriptor(c, VK_PIPELINE_BIND_POINT_GRAPHICS, layout_, 0, nw, w);
        // Dynamic state
        VkViewport vp{0, 0, float(target_w_), float(target_h_), 0.0f, 1.0f};
        vkCmdSetViewport(c, 0, 1, &vp);
        VkRect2D sc{{0, 0}, {u32(target_w_), u32(target_h_)}};
        if (r.scissor) {
            const int x0 = std::clamp(r.sx, 0, target_w_), y0 = std::clamp(r.sy, 0, target_h_);
            const int x1 = std::clamp(r.sx + r.sw, 0, target_w_), y1 = std::clamp(r.sy + r.sh, 0, target_h_);
            sc = {{x0, y0}, {u32(std::max(0, x1 - x0)), u32(std::max(0, y1 - y0))}};
        }
        vkCmdSetScissor(c, 0, 1, &sc);
        vkCmdSetCullMode(c, r.cull == kCullFront ? VK_CULL_MODE_FRONT_BIT
                            : r.cull == kCullBack ? VK_CULL_MODE_BACK_BIT
                            : r.cull == kCullBoth ? VK_CULL_MODE_FRONT_AND_BACK : VK_CULL_MODE_NONE);
        // Framebuffer y runs down like the NV2A's screen: same winding sense.
        vkCmdSetFrontFace(c, r.front_ccw ? VK_FRONT_FACE_COUNTER_CLOCKWISE : VK_FRONT_FACE_CLOCKWISE);
        const bool has_depth = cur_depth_ != 0;
        vkCmdSetDepthTestEnable(c, has_depth && (r.depth_test || r.depth_write));
        vkCmdSetDepthWriteEnable(c, has_depth && r.depth_write);
        vkCmdSetDepthCompareOp(c, r.depth_test ? vk_compare(r.depth_func) : VK_COMPARE_OP_ALWAYS);
        vkCmdSetDepthBoundsTestEnable(c, VK_FALSE);
        vkCmdSetDepthBiasEnable(c, has_depth && r.poly_offset);
        if (r.poly_offset) vkCmdSetDepthBias(c, r.poly_offset_units, 0.0f, r.poly_offset_scale);
        vkCmdSetStencilTestEnable(c, has_depth && r.stencil_test);
        if (r.stencil_test) {
            vkCmdSetStencilOp(c, VK_STENCIL_FACE_FRONT_AND_BACK, vk_stencil_op(r.stencil_fail), vk_stencil_op(r.stencil_zpass),
                              vk_stencil_op(r.stencil_zfail), vk_compare(r.stencil_func));
            vkCmdSetStencilCompareMask(c, VK_STENCIL_FACE_FRONT_AND_BACK, r.stencil_read_mask);
            vkCmdSetStencilWriteMask(c, VK_STENCIL_FACE_FRONT_AND_BACK, r.stencil_write_mask);
            vkCmdSetStencilReference(c, VK_STENCIL_FACE_FRONT_AND_BACK, r.stencil_ref);
        }
        vkCmdSetBlendConstants(c, r.blend_color);
        vkCmdSetPrimitiveRestartEnable(c, VK_FALSE);
        vkCmdSetRasterizerDiscardEnable(c, VK_FALSE);
        vkCmdSetLineWidth(c, 1.0f);
        if (idx) {
            vkCmdBindIndexBuffer(c, buffer_of(idx), idx.off, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(c, count, 1, 0, 0, 0);
        } else {
            vkCmdDraw(c, count, 1, dc.first, 0);
        }
        Image& tgt = images_[cur_color_ - 1];
        tgt.write_epoch = epoch_;
        if (cur_depth_) images_[cur_depth_ - 1].write_epoch = epoch_;
        return true;
    }

    // ================================================================ sync
    Sync fence() override { return submit(); }
    bool wait(Sync s, u64 timeout_ns) override {
        if (!s) return true;
        const VkSemaphore sem = (s & vk::kPresenterBit) ? g_present_tl_() : g_gpu_tl();
        const u64 v = s & ~vk::kPresenterBit;
        if (!(s & vk::kPresenterBit) && v > submitted_) submit();
        VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
        wi.semaphoreCount = 1;
        wi.pSemaphores = &sem;
        wi.pValues = &v;
        return vkWaitSemaphores(dev_, &wi, timeout_ns) == VK_SUCCESS;
    }
    void gpu_wait(Sync s) override {
        if (!s) return;
        if (s & vk::kPresenterBit) waits_.push_back(s & ~vk::kPresenterBit);
        // Our own timeline: the queue already runs our submissions in order.
    }
    void release(Sync) override {}
    void flush() override {
        submit();
        save_pipeline_cache(false);
    }

private:
    // ---------------------------------------------------------------- resources
    struct Image {
        VkImage img = VK_NULL_HANDLE;
        VkDeviceMemory mem = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE, view_a1 = VK_NULL_HANDLE;
        VkFormat fmt = VK_FORMAT_UNDEFINED;
        int w = 0, h = 0, levels = 1;
        bool cube = false, depth = false;
        u64 write_epoch = ~0ull, read_epoch = ~0ull;  // barrier epoch of the last write / read
    };
    struct Buffer {
        VkBuffer buf = VK_NULL_HANDLE;
        VkDeviceMemory mem = VK_NULL_HANDLE;
        void* map = nullptr;
        size_t size = 0;
        bool coherent = true;
    };
    struct Read { Buffer buf; u64 value = 0; bool busy = false; };
    struct Prog {
        VkShaderModule vs = VK_NULL_HANDLE, fs = VK_NULL_HANDLE;
        u64 key = 0;                // hash of the GLSL
        std::atomic<int> state{0};  // 0 building, 1 ready, 2 failed
    };
    // Worker threads for shader modules and pipelines.
    struct Workers {
        std::vector<std::thread> threads;
        std::mutex m;
        std::condition_variable cv;
        std::deque<std::function<void()>> q;
        void start(int n) {
            for (int i = 0; i < n; ++i)
                threads.emplace_back([this] {
                    for (;;) {
                        std::function<void()> job;
                        {
                            std::unique_lock lk(m);
                            cv.wait(lk, [this] { return !q.empty(); });
                            job = std::move(q.front());
                            q.pop_front();
                        }
                        job();
                    }
                });
        }
        void push(std::function<void()> job) {
            {
                std::lock_guard lk(m);
                q.push_back(std::move(job));
            }
            cv.notify_one();
        }
    };
    struct Cmd { VkCommandBuffer cb = VK_NULL_HANDLE; u64 value = 0; };

    static constexpr u32 kCmdBufs = 6;
    static constexpr size_t kRingBytes = 128u << 20, kChunks = 4, kChunk = kRingBytes / kChunks;
    static constexpr u32 kRingId = 1, kBigBase = 1000;

    static VkSemaphore g_gpu_tl() { return vk::gpu_timeline(); }
    static VkSemaphore g_present_tl_() { return vk::presenter_timeline(); }

    VkDeviceMemory alloc(const VkMemoryRequirements& mr, VkMemoryPropertyFlags flags, bool* coherent = nullptr) {
        u32 type = vk::memory_type(mr.memoryTypeBits, flags);
        if (type == UINT32_MAX && (flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT))
            type = vk::memory_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (type == UINT32_MAX) {
            std::fprintf(stderr, "[vk] no memory type for flags %#x\n", unsigned(flags));
            std::abort();
        }
        if (coherent) *coherent = c_.mem.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = mr.size;
        ai.memoryTypeIndex = type;
        VkDeviceMemory m;
        VKCHECK(vkAllocateMemory(dev_, &ai, nullptr, &m));
        return m;
    }
    Buffer make_buffer(size_t size, VkBufferUsageFlags usage, VkMemoryPropertyFlags flags) {
        Buffer b;
        b.size = size;
        VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        ci.size = size;
        ci.usage = usage;
        VKCHECK(vkCreateBuffer(dev_, &ci, nullptr, &b.buf));
        VkMemoryRequirements mr;
        vkGetBufferMemoryRequirements(dev_, b.buf, &mr);
        b.mem = alloc(mr, flags, &b.coherent);
        VKCHECK(vkBindBufferMemory(dev_, b.buf, b.mem, 0));
        VKCHECK(vkMapMemory(dev_, b.mem, 0, VK_WHOLE_SIZE, 0, &b.map));
        return b;
    }
    void free_buffer(const Buffer& b) {
        if (!b.buf) return;
        vkDestroyBuffer(dev_, b.buf, nullptr);
        vkFreeMemory(dev_, b.mem, nullptr);
    }
    VkBuffer buffer_of(const Slice& s) {
        if (s.buf == kRingId) return ring_.buf;
        auto it = big_.find(s.buf);
        return it != big_.end() ? it->second.buf : ring_.buf;
    }
    const void* map_of(const Slice& s) {
        if (s.buf == kRingId) return ring_.map;
        auto it = big_.find(s.buf);
        return it != big_.end() ? it->second.map : ring_.map;
    }
    static VkImageAspectFlags aspect(const Image& im) {
        return im.depth ? VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    }
    VkImageView make_view(const Image& im, bool alpha_one) {
        VkImageViewCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        ci.image = im.img;
        ci.viewType = im.cube ? VK_IMAGE_VIEW_TYPE_CUBE : VK_IMAGE_VIEW_TYPE_2D;
        ci.format = im.fmt;
        ci.components = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                         alpha_one ? VK_COMPONENT_SWIZZLE_ONE : VK_COMPONENT_SWIZZLE_IDENTITY};
        ci.subresourceRange = {aspect(im), 0, u32(im.levels), 0, im.cube ? 6u : 1u};
        VkImageView v;
        VKCHECK(vkCreateImageView(dev_, &ci, nullptr, &v));
        return v;
    }
    VkSampler sampler(const SamplerDesc& s) {
        const u64 key = fnv(&s, sizeof s);
        auto it = samplers_.find(key);
        if (it != samplers_.end()) return it->second;
        VkSamplerCreateInfo ci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        ci.magFilter = s.mag == kNearest ? VK_FILTER_NEAREST : VK_FILTER_LINEAR;
        ci.minFilter = s.min == kNearest ? VK_FILTER_NEAREST : VK_FILTER_LINEAR;
        ci.mipmapMode = s.mip == kMipLinear ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
        ci.addressModeU = vk_wrap(s.wrap_u);
        ci.addressModeV = vk_wrap(s.wrap_v);
        ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        const float max_bias = c_.props.limits.maxSamplerLodBias;
        ci.mipLodBias = std::clamp(s.lod_bias, -max_bias, max_bias);
        ci.minLod = s.min_lod;
        ci.maxLod = s.mip == kMipNone ? 0.25f : s.max_lod;  // no mipmapping: level 0 only (GL's rule)
        // Anisotropic filtering (VC3_ANISOTROPIC) for linearly filtered textures (as the GL device).
        if (c_.aniso && s.min == kLinear && max_aniso_ > 1.0f) {
            ci.anisotropyEnable = VK_TRUE;
            ci.maxAnisotropy = std::min(max_aniso_, c_.props.limits.maxSamplerAnisotropy);
        }
        VkSamplerCustomBorderColorCreateInfoEXT cbc{VK_STRUCTURE_TYPE_SAMPLER_CUSTOM_BORDER_COLOR_CREATE_INFO_EXT};
        if (c_.custom_border) {
            std::memcpy(cbc.customBorderColor.float32, s.border, 16);
            cbc.format = VK_FORMAT_UNDEFINED;
            ci.borderColor = VK_BORDER_COLOR_FLOAT_CUSTOM_EXT;
            ci.pNext = &cbc;
        } else {
            const bool white = s.border[0] > 0.5f, opaque = s.border[3] > 0.5f;
            ci.borderColor = white ? VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE
                             : opaque ? VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK : VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
        }
        VkSampler smp;
        VKCHECK(vkCreateSampler(dev_, &ci, nullptr, &smp));
        samplers_[key] = smp;
        return smp;
    }

    // ---------------------------------------------------------------- command recording
    VkCommandBuffer cb() {
        if (!recording_) {
            Cmd& cm = cmd_[cur_cmd_];
            if (cm.value) wait_value(g_gpu_tl(), cm.value);
            VKCHECK(vkResetCommandBuffer(cm.cb, 0));
            VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            VKCHECK(vkBeginCommandBuffer(cm.cb, &bi));
            recording_ = true;
            bound_pipe_ = VK_NULL_HANDLE;
            // Order everything here after all earlier submissions on the queue (a barrier
            // covers prior submissions), then hazards start afresh.
            record_barrier(cm.cb);
        }
        return cmd_[cur_cmd_].cb;
    }
    // Submits what has been recorded; returns the timeline value that marks its end.
    u64 submit() {
        if (!recording_) return submitted_;
        end_rendering();
        Cmd& cm = cmd_[cur_cmd_];
        VKCHECK(vkEndCommandBuffer(cm.cb));
        cm.value = ++submitted_;
        VkCommandBufferSubmitInfo ci{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
        ci.commandBuffer = cm.cb;
        VkSemaphoreSubmitInfo sig{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
        sig.semaphore = g_gpu_tl();
        sig.value = cm.value;
        sig.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        std::vector<VkSemaphoreSubmitInfo> wt;
        for (u64 v : waits_) {
            vk::wait_presenter_submitted(v);  // never queue a wait ahead of its signal
            sub_waits_.push_back({cm.value, v});
            if (sub_waits_.size() > 8) sub_waits_.erase(sub_waits_.begin());
            VkSemaphoreSubmitInfo w{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
            w.semaphore = g_present_tl_();
            w.value = v;
            w.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            wt.push_back(w);
        }
        waits_.clear();
        VkSubmitInfo2 si{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
        si.waitSemaphoreInfoCount = u32(wt.size());
        si.pWaitSemaphoreInfos = wt.data();
        si.commandBufferInfoCount = 1;
        si.pCommandBufferInfos = &ci;
        si.signalSemaphoreInfoCount = 1;
        si.pSignalSemaphoreInfos = &sig;
        {
            std::lock_guard lk(c_.queue_m);
            VKCHECK(vkQueueSubmit2(c_.queue, 1, &si, VK_NULL_HANDLE));
        }
        recording_ = false;
        cur_cmd_ = (cur_cmd_ + 1) % kCmdBufs;
        // Each submission starts with no barrier-tracked hazards pending: the queue's
        // next command buffer begins with a full barrier (see begin of cb()).
        pending_.push_back({cm.value, std::move(later_)});
        later_.clear();
        collect();
        return cm.value;
    }
    bool reached(u64 v) {
        u64 cur = 0;
        vkGetSemaphoreCounterValue(dev_, g_gpu_tl(), &cur);
        return cur >= v;
    }
    void wait_value(VkSemaphore s, u64 v) {
        if (s == g_gpu_tl() && v > submitted_) submit();
        VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
        wi.semaphoreCount = 1;
        wi.pSemaphores = &s;
        wi.pValues = &v;
        for (int tries = 0;; ++tries) {  // a long wait is logged, not fatal (a lost device is)
            const VkResult r = vkWaitSemaphores(dev_, &wi, 2000000000ull);
            if (r == VK_SUCCESS) break;
            if (r != VK_TIMEOUT) VKCHECK(r);
            u64 gpu = 0, pres = 0;
            vkGetSemaphoreCounterValue(dev_, g_gpu_tl(), &gpu);
            vkGetSemaphoreCounterValue(dev_, g_present_tl_(), &pres);
            std::string pw;
            for (const auto& [val, w] : sub_waits_) pw += " " + std::to_string(val) + "<-p" + std::to_string(w);
            std::fprintf(stderr, "[vk] still waiting for the GPU (%s value %llu, %d s): gpu at %llu (submitted %llu), "
                         "window at %llu (submitted %llu); recent waits on the window:%s\n",
                         s == g_gpu_tl() ? "frame" : "window", (unsigned long long)v, 2 * (tries + 1), (unsigned long long)gpu,
                         (unsigned long long)submitted_, (unsigned long long)pres,
                         (unsigned long long)vk::presenter_submitted_value(), pw.c_str());
        }
    }
    void later(std::function<void()> fn) { later_.push_back(std::move(fn)); }
    void collect() {
        u64 cur = 0;
        vkGetSemaphoreCounterValue(dev_, g_gpu_tl(), &cur);
        while (!pending_.empty() && pending_.front().first <= cur) {
            for (auto& fn : pending_.front().second) fn();
            pending_.pop_front();
        }
    }

    // ---------------------------------------------------------------- hazards
    void full_barrier() {
        end_rendering();
        record_barrier(cb());
    }
    void record_barrier(VkCommandBuffer c) {
        VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        mb.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        mb.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
        mb.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        mb.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
        VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        di.memoryBarrierCount = 1;
        di.pMemoryBarriers = &mb;
        vkCmdPipelineBarrier2(c, &di);
        ++epoch_;
    }
    void sample_read(Image& im) {
        if (im.write_epoch == epoch_) full_barrier();
        im.read_epoch = epoch_;
    }
    void transfer_read(Image& im) {
        end_rendering();
        if (im.write_epoch == epoch_) full_barrier();
        im.read_epoch = epoch_;
    }
    void transfer_write(Image& im) {
        end_rendering();
        if (im.write_epoch == epoch_ || im.read_epoch == epoch_) full_barrier();
        im.write_epoch = epoch_;
    }
    void begin_rendering() {
        if (rendering_) return;
        Image& col = images_[cur_color_ - 1];
        Image* dep = cur_depth_ ? &images_[cur_depth_ - 1] : nullptr;
        // Attachments used otherwise since the last barrier (sampled, copied, or rendered
        // in an earlier pass): order against that first.
        if (col.write_epoch == epoch_ || col.read_epoch == epoch_ || (dep && dep->write_epoch == epoch_)) full_barrier();
        VkRenderingAttachmentInfo ca{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        ca.imageView = col.view;
        ca.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        ca.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        ca.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        VkRenderingAttachmentInfo da{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        if (dep) {
            da.imageView = dep->view;
            da.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            da.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            da.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        }
        VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
        ri.renderArea = {{0, 0}, {u32(target_w_), u32(target_h_)}};
        ri.layerCount = 1;
        ri.colorAttachmentCount = 1;
        ri.pColorAttachments = &ca;
        ri.pDepthAttachment = dep ? &da : nullptr;
        ri.pStencilAttachment = dep ? &da : nullptr;
        vkCmdBeginRendering(cb(), &ri);
        rendering_ = true;
        col.write_epoch = epoch_;
        if (dep) dep->write_epoch = epoch_;
    }
    void end_rendering() {
        if (!rendering_) return;
        vkCmdEndRendering(cmd_[cur_cmd_].cb);
        rendering_ = false;
    }

    // ---------------------------------------------------------------- shaders and pipelines
    VkShaderModule module(const std::string& src_in, bool vertex) {
        std::string src = src_in;
        if (vertex) {  // the generated GLSL names the vertex index VERTEX_ID
            const size_t nl = src.find('\n');
            src.insert(nl == std::string::npos ? 0 : nl + 1, "#define VERTEX_ID gl_VertexIndex\n");
        }
        const u64 h = fnv(src.data(), src.size());
        char path[160];
        std::snprintf(path, sizeof path, "work/shadercache/v4/spirv/%016llx.spv", (unsigned long long)h);
        std::vector<u32> code;
        if (std::FILE* f = std::fopen(path, "rb")) {
            std::fseek(f, 0, SEEK_END);
            const long n = std::ftell(f);
            std::fseek(f, 0, SEEK_SET);
            if (n > 0 && n % 4 == 0) {
                code.resize(size_t(n) / 4);
                if (std::fread(code.data(), 1, size_t(n), f) != size_t(n)) code.clear();
            }
            std::fclose(f);
        }
        if (code.empty()) {
            shader_store::Timer timer;
            shaderc_compile_options_t o = shaderc_compile_options_initialize();
            shaderc_compile_options_set_target_env(o, shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_3);
            shaderc_compile_options_set_optimization_level(o, shaderc_optimization_level_zero);
            shaderc_compilation_result_t r =
                shaderc_compile_into_spv(compiler_, src.data(), src.size(),
                                         vertex ? shaderc_glsl_vertex_shader : shaderc_glsl_fragment_shader, "gen", "main", o);
            shaderc_compile_options_release(o);
            if (shaderc_result_get_compilation_status(r) != shaderc_compilation_status_success) {
                if (!quiet_)
                    std::fprintf(stderr, "[vk] GLSL -> SPIR-V failed:\n%s\n%s\n", shaderc_result_get_error_message(r), src.c_str());
                shaderc_result_release(r);
                return VK_NULL_HANDLE;
            }
            const size_t n = shaderc_result_get_length(r);
            code.resize(n / 4);
            std::memcpy(code.data(), shaderc_result_get_bytes(r), n);
            shaderc_result_release(r);
            std::error_code ec;
            std::filesystem::create_directories("work/shadercache/v4/spirv", ec);
            shader_store::write_file(path, code.data(), code.size() * 4);
        }
        VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        ci.codeSize = code.size() * 4;
        ci.pCode = code.data();
        VkShaderModule m;
        VKCHECK(vkCreateShaderModule(dev_, &ci, nullptr, &m));
        return m;
    }
    // Programs: shader modules built on a worker (async) or right away; draws using one
    // that isn't ready are skipped.
    Program build(const std::string& vs, const std::string& fs, bool async) {
        const u64 key = fnv(fs.data(), fs.size(), fnv(vs.data(), vs.size()));
        auto it = programs_by_key_.find(key);
        if (it != programs_by_key_.end()) return Program{it->second};
        auto p = std::make_unique<Prog>();
        p->key = key;
        Prog* raw = p.get();
        progs_.push_back(std::move(p));
        const u32 id = u32(progs_.size());
        programs_by_key_[key] = id;
        auto job = [this, raw, vs, fs] {
            raw->vs = module(vs, true);
            raw->fs = module(fs, false);
            if (!raw->vs || !raw->fs) {
                std::fprintf(stderr, "[vk] shader compile failed (draws using it are skipped)\n");
                raw->state = 2;
            } else {
                raw->state = 1;
            }
        };
        if (async) workers_.push(job);
        else job();
        return Program{id};
    }
    // Startup: shader modules for every stored program, then every pipeline recorded in
    // earlier runs (vk-recipes.bin), built on all workers while the game waits (progress
    // in the window title). VC3_SHADER_PRECOMPILE=0 skips this.
    void warm_up() {
        std::vector<std::pair<std::string, std::string>> all;
        shader_store::for_each([&](const std::string& vs, const std::string& fs) { all.emplace_back(vs, fs); });
        load_recipes();
        const char* pc = std::getenv("VC3_SHADER_PRECOMPILE");
        if (pc && !std::strcmp(pc, "0")) return;
        const auto t0 = std::chrono::steady_clock::now();
        shader_store::warm_total = int(all.size() + recipes_.size());
        quiet_ = true;
        std::vector<Prog*> built;
        for (const auto& [vs, fs] : all) {
            const Program p = build(vs, fs, true);
            built.push_back(progs_[p.id - 1].get());
        }
        for (Prog* p : built) {  // wait for the modules; drop sources an older build made
            while (p->state == 0) std::this_thread::sleep_for(std::chrono::milliseconds(1));
            ++shader_store::warm_done;
        }
        int bad = 0;
        for (size_t i = 0; i < all.size(); ++i)
            if (built[i]->state == 2) {
                shader_store::forget(all[i].first, all[i].second);
                ++bad;
            }
        quiet_ = false;
        std::vector<Pipe*> pipes;
        for (const auto& r : recipes_) {
            auto it = programs_by_key_.find(r.prog_key);
            if (it == programs_by_key_.end() || progs_[it->second - 1]->state != 1) continue;
            PipeKey k = r.key;
            k.prog = it->second;
            pipes.push_back(get_pipe(k, true));
        }
        for (Pipe* p : pipes) {
            while (p->state == 0) std::this_thread::sleep_for(std::chrono::milliseconds(1));
            ++shader_store::warm_done;
        }
        shader_store::warm_done = shader_store::warm_total.load();
        if (!all.empty())
            std::fprintf(stderr, "[vk] precompiled %zu programs (%d stale dropped) and %zu pipelines in %.1f s\n",
                         all.size() - bad, bad, pipes.size(),
                         std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }

    struct PipeKey {
        u32 prog;
        u8 topology, blend, src, dst, eq, mask, depth, nattr, nbind, pad[3];
        u8 attr[16][3];   // location, binding, format
        u32 offset[16];
    };
    struct Pipe {
        PipeKey key{};
        Prog* prog = nullptr;
        std::atomic<int> state{0};  // 0 building, 1 ready, 2 failed
        VkPipeline pipe = VK_NULL_HANDLE;
    };
    // The pipeline for a draw (VK_NULL_HANDLE: not built yet, skip the draw).
    VkPipeline pipeline(const DrawCall& dc, u32 nbind) {
        PipeKey k{};
        k.prog = dc.prog.id;
        k.topology = u8(vk_topology(dc.rs.topology));
        k.blend = dc.rs.blend;
        if (dc.rs.blend) {
            k.src = dc.rs.blend_src;
            k.dst = dc.rs.blend_dst;
            k.eq = dc.rs.blend_eq;
        }
        k.mask = dc.rs.color_mask & 0xF;
        k.depth = cur_depth_ != 0;
        k.nattr = dc.nattr;
        k.nbind = u8(nbind);
        for (int i = 0; i < dc.nattr; ++i) {
            k.attr[i][0] = dc.attr[i].location;
            k.attr[i][1] = dc.attr[i].binding;
            k.attr[i][2] = dc.attr[i].constant ? u8(kF32x4) : dc.attr[i].fmt;
            k.offset[i] = dc.attr[i].constant ? 0 : dc.attr[i].offset;
        }
        Pipe* p = get_pipe(k, async_);
        return p->state == 1 ? p->pipe : VK_NULL_HANDLE;
    }
    Pipe* get_pipe(const PipeKey& k, bool async) {
        const u64 h = fnv(&k, sizeof k);
        auto it = pipes_.find(h);
        if (it != pipes_.end()) return it->second.get();
        auto e = std::make_unique<Pipe>();
        e->key = k;
        e->prog = progs_[k.prog - 1].get();
        Pipe* raw = e.get();
        pipes_[h] = std::move(e);
        record_recipe(k, raw->prog->key);
        auto job = [this, raw] {
            while (raw->prog->state == 0) std::this_thread::sleep_for(std::chrono::microseconds(200));
            if (raw->prog->state != 1) {
                raw->state = 2;
                return;
            }
            raw->pipe = create_pipeline(raw->key, *raw->prog);
            raw->state = 1;
        };
        if (async) workers_.push(job);
        else job();
        return raw;
    }
    VkPipeline create_pipeline(const PipeKey& k, const Prog& p) {
        const u32 nbind = k.nbind;
        VkPipelineShaderStageCreateInfo st[2]{};
        st[0].sType = st[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        st[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        st[0].module = p.vs;
        st[0].pName = "main";
        st[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        st[1].module = p.fs;
        st[1].pName = "main";
        VkVertexInputBindingDescription bd[32];
        for (u32 b = 0; b < nbind; ++b) bd[b] = {b, 0, VK_VERTEX_INPUT_RATE_VERTEX};  // stride: dynamic
        VkVertexInputAttributeDescription ad[16];
        for (int i = 0; i < k.nattr; ++i)
            ad[i] = {k.attr[i][0], k.attr[i][1], vk_attr_format(k.attr[i][2]), k.offset[i]};
        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        vi.vertexBindingDescriptionCount = nbind;
        vi.pVertexBindingDescriptions = bd;
        vi.vertexAttributeDescriptionCount = k.nattr;
        vi.pVertexAttributeDescriptions = ad;
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        ia.topology = VkPrimitiveTopology(k.topology);
        VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vp.viewportCount = 1;
        vp.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.lineWidth = 1.0f;
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        VkPipelineColorBlendAttachmentState ba{};
        ba.blendEnable = k.blend;
        ba.srcColorBlendFactor = ba.srcAlphaBlendFactor = vk_blend_factor(k.src);
        ba.dstColorBlendFactor = ba.dstAlphaBlendFactor = vk_blend_factor(k.dst);
        ba.colorBlendOp = ba.alphaBlendOp = vk_blend_op(k.eq);
        ba.colorWriteMask = k.mask;  // bit order R G B A matches VkColorComponentFlagBits
        VkPipelineColorBlendStateCreateInfo cbs{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        cbs.attachmentCount = 1;
        cbs.pAttachments = &ba;
        const VkDynamicState dyn[] = {
            VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_CULL_MODE, VK_DYNAMIC_STATE_FRONT_FACE,
            VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE, VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE, VK_DYNAMIC_STATE_DEPTH_COMPARE_OP,
            VK_DYNAMIC_STATE_DEPTH_BOUNDS_TEST_ENABLE, VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE, VK_DYNAMIC_STATE_DEPTH_BIAS,
            VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE, VK_DYNAMIC_STATE_STENCIL_OP, VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK,
            VK_DYNAMIC_STATE_STENCIL_WRITE_MASK, VK_DYNAMIC_STATE_STENCIL_REFERENCE, VK_DYNAMIC_STATE_BLEND_CONSTANTS,
            VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE, VK_DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE,
            VK_DYNAMIC_STATE_RASTERIZER_DISCARD_ENABLE, VK_DYNAMIC_STATE_LINE_WIDTH};
        VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dy.dynamicStateCount = u32(std::size(dyn));
        dy.pDynamicStates = dyn;
        const VkFormat cf = VK_FORMAT_B8G8R8A8_UNORM;
        VkPipelineRenderingCreateInfo pr{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        pr.colorAttachmentCount = 1;
        pr.pColorAttachmentFormats = &cf;
        pr.depthAttachmentFormat = k.depth ? depth_fmt_ : VK_FORMAT_UNDEFINED;
        pr.stencilAttachmentFormat = k.depth ? depth_fmt_ : VK_FORMAT_UNDEFINED;
        VkGraphicsPipelineCreateInfo gp{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        gp.pNext = &pr;
        gp.stageCount = 2;
        gp.pStages = st;
        gp.pVertexInputState = &vi;
        gp.pInputAssemblyState = &ia;
        gp.pViewportState = &vp;
        gp.pRasterizationState = &rs;
        gp.pMultisampleState = &ms;
        gp.pDepthStencilState = &ds;
        gp.pColorBlendState = &cbs;
        gp.pDynamicState = &dy;
        gp.layout = layout_;
        VkPipeline pipe;
        {
            shader_store::Timer timer;
            VKCHECK(vkCreateGraphicsPipelines(dev_, pcache_, 1, &gp, nullptr, &pipe));
        }
        ++new_pipes_;
        return pipe;
    }

    // Pipeline recipes (state + program), so later runs can build them all up front.
    struct Recipe { u64 prog_key; PipeKey key; };
    void load_recipes() {
        recipes_path_ = "work/shadercache/v4/vk-recipes.bin";
        std::FILE* f = std::fopen(recipes_path_.c_str(), "rb");
        if (!f) return;
        Recipe r;
        while (std::fread(&r, sizeof r, 1, f) == 1)
            if (known_recipes_.insert(fnv(&r, sizeof r)).second) recipes_.push_back(r);
        std::fclose(f);
    }
    void record_recipe(const PipeKey& k, u64 prog_key) {
        Recipe r{prog_key, k};
        r.key.prog = 0;  // program ids differ between runs: stored by source hash
        if (!known_recipes_.insert(fnv(&r, sizeof r)).second) return;
        if (recipes_path_.empty()) return;
        if (std::FILE* f = std::fopen(recipes_path_.c_str(), "ab")) {
            std::fwrite(&r, sizeof r, 1, f);
            std::fclose(f);
        }
    }

    // Masked colour clear: a quad writing the colour through the colour write mask.
    void clear_quad(int x, int y, int w, int h, u8 mask, const float rgba[4]) {
        VkPipeline& pipe = clear_pipes_[mask & 0xF][cur_depth_ ? 1 : 0];
        if (!pipe) {
            static const char* kVs = "#version 450\nvoid main() { vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);"
                                     " gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0); }\n";
            static const char* kFs = "#version 450\nlayout(push_constant) uniform C { vec4 color; };\n"
                                     "layout(location = 0) out vec4 o;\nvoid main() { o = color; }\n";
            if (!clear_vs_) clear_vs_ = module(kVs, true);
            if (!clear_fs_) clear_fs_ = module(kFs, false);
            VkPipelineShaderStageCreateInfo st[2]{};
            st[0].sType = st[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            st[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
            st[0].module = clear_vs_;
            st[0].pName = "main";
            st[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
            st[1].module = clear_fs_;
            st[1].pName = "main";
            VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
            VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
            ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
            vp.viewportCount = vp.scissorCount = 1;
            VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
            rs.polygonMode = VK_POLYGON_MODE_FILL;
            rs.cullMode = VK_CULL_MODE_NONE;
            rs.lineWidth = 1.0f;
            VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
            ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
            VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
            VkPipelineColorBlendAttachmentState ba{};
            ba.colorWriteMask = mask & 0xF;
            VkPipelineColorBlendStateCreateInfo cbs{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
            cbs.attachmentCount = 1;
            cbs.pAttachments = &ba;
            const VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
            VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
            dy.dynamicStateCount = 2;
            dy.pDynamicStates = dyn;
            const VkFormat cf = VK_FORMAT_B8G8R8A8_UNORM;
            VkPipelineRenderingCreateInfo pr{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
            pr.colorAttachmentCount = 1;
            pr.pColorAttachmentFormats = &cf;
            pr.depthAttachmentFormat = cur_depth_ ? depth_fmt_ : VK_FORMAT_UNDEFINED;
            pr.stencilAttachmentFormat = cur_depth_ ? depth_fmt_ : VK_FORMAT_UNDEFINED;
            VkGraphicsPipelineCreateInfo gp{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
            gp.pNext = &pr;
            gp.stageCount = 2;
            gp.pStages = st;
            gp.pVertexInputState = &vi;
            gp.pInputAssemblyState = &ia;
            gp.pViewportState = &vp;
            gp.pRasterizationState = &rs;
            gp.pMultisampleState = &ms;
            gp.pDepthStencilState = &ds;
            gp.pColorBlendState = &cbs;
            gp.pDynamicState = &dy;
            gp.layout = clear_layout_;
            VKCHECK(vkCreateGraphicsPipelines(dev_, pcache_, 1, &gp, nullptr, &pipe));
        }
        VkCommandBuffer c = cb();
        vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
        bound_pipe_ = pipe;
        VkViewport vpt{0, 0, float(target_w_), float(target_h_), 0.0f, 1.0f};
        vkCmdSetViewport(c, 0, 1, &vpt);
        VkRect2D sc{{x, y}, {u32(w), u32(h)}};
        vkCmdSetScissor(c, 0, 1, &sc);
        vkCmdPushConstants(c, clear_layout_, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 16, rgba);
        vkCmdDraw(c, 3, 1, 0, 0);
    }

    // Pipeline cache on disk, per device (pipelines are the main first-use cost).
    void init_pipeline_cache() {
        char name[160];
        u64 h = fnv(c_.props.pipelineCacheUUID, VK_UUID_SIZE);
        h = fnv(&c_.props.driverVersion, sizeof c_.props.driverVersion, h);
        std::snprintf(name, sizeof name, "work/shadercache/v4/vk-%016llx.bin", (unsigned long long)h);
        pcache_path_ = name;
        std::vector<u8> data;
        if (std::FILE* f = std::fopen(name, "rb")) {
            std::fseek(f, 0, SEEK_END);
            const long n = std::ftell(f);
            std::fseek(f, 0, SEEK_SET);
            if (n > 0) {
                data.resize(size_t(n));
                if (std::fread(data.data(), 1, data.size(), f) != data.size()) data.clear();
            }
            std::fclose(f);
        }
        VkPipelineCacheCreateInfo ci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
        ci.initialDataSize = data.size();
        ci.pInitialData = data.empty() ? nullptr : data.data();
        if (vkCreatePipelineCache(dev_, &ci, nullptr, &pcache_) != VK_SUCCESS) {
            ci.initialDataSize = 0;
            ci.pInitialData = nullptr;
            VKCHECK(vkCreatePipelineCache(dev_, &ci, nullptr, &pcache_));
        }
        std::fprintf(stderr, "[vk] pipeline cache %s: %zu bytes\n", name, data.size());
    }
    void save_pipeline_cache(bool force) {
        if (!new_pipes_ || (!force && ++frames_since_save_ < 300)) return;
        frames_since_save_ = 0;
        new_pipes_ = 0;
        size_t n = 0;
        if (vkGetPipelineCacheData(dev_, pcache_, &n, nullptr) != VK_SUCCESS || !n) return;
        std::vector<u8> data(n);
        if (vkGetPipelineCacheData(dev_, pcache_, &n, data.data()) != VK_SUCCESS) return;
        std::error_code ec;
        std::filesystem::create_directories("work/shadercache/v4", ec);
        shader_store::write_file(pcache_path_, data.data(), n);
    }

    vk::Context& c_;
    VkDevice dev_;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    Cmd cmd_[kCmdBufs];
    u32 cur_cmd_ = 0;
    bool recording_ = false, rendering_ = false;
    u64 submitted_ = 0, epoch_ = 1;
    std::vector<u64> waits_;  // presenter timeline values the next submission waits for
    std::vector<std::pair<u64, u64>> sub_waits_;  // recent (our value, window value it waited for), for diagnostics
    std::vector<std::function<void()>> later_;
    std::deque<std::pair<u64, std::vector<std::function<void()>>>> pending_;
    Buffer ring_;
    size_t pos_ = 0, chunk_ = 0;
    u64 chunk_value_[kChunks] = {};
    std::unordered_map<u32, Buffer> big_;  // oversized stream allocations by slice id
    u32 next_big_ = kBigBase;
    std::vector<Image> images_;
    std::vector<u32> free_images_;
    std::mutex images_m_;  // images_ changes vs. the window thread's image_of()
    std::vector<Read> reads_;
    std::vector<std::unique_ptr<Prog>> progs_;
    std::unordered_map<u64, u32> programs_by_key_;
    std::unordered_map<u64, std::unique_ptr<Pipe>> pipes_;
    Workers workers_;
    bool async_ = true;
    float max_aniso_ = 1.0f;  // VC3_ANISOTROPIC
    std::vector<Recipe> recipes_;
    std::unordered_set<u64> known_recipes_;
    std::string recipes_path_;
    std::unordered_map<u64, VkSampler> samplers_;
    VkPipeline clear_pipes_[16][2] = {};
    VkShaderModule clear_vs_ = VK_NULL_HANDLE, clear_fs_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE, clear_layout_ = VK_NULL_HANDLE;
    VkPipeline bound_pipe_ = VK_NULL_HANDLE;
    VkPipelineCache pcache_ = VK_NULL_HANDLE;
    std::string pcache_path_;
    u32 new_pipes_ = 0, frames_since_save_ = 0;
    VkFormat depth_fmt_ = VK_FORMAT_D24_UNORM_S8_UINT;
    u32 cur_color_ = 0, cur_depth_ = 0;
    int target_w_ = 0, target_h_ = 0;
    Tex dummy2d_, dummycube_;
    bool attr_ok_[kU32x1 + 1] = {};  // vertex format fetched by the driver (else converted)

    static size_t attr_bytes(u8 f) {
        switch (f) {
            case kF32x1: case kF32x2: case kF32x3: case kF32x4: return size_t(4 * (f - kF32x1 + 1));
            case kUN8x1: case kUN8x2: case kUN8x3: case kUN8x4: return size_t(f - kUN8x1 + 1);
            case kBGRA8: return 4;
            case kSN16x1: case kSN16x2: case kSN16x3: case kSN16x4: return size_t(2 * (f - kSN16x1 + 1));
            case kS16x1: case kS16x2: case kS16x3: case kS16x4: return size_t(2 * (f - kS16x1 + 1));
            default: return 4;
        }
    }
    // One attribute -> (x, y, z, w) floats, missing components 0, 0, 0, 1 (as GL fetches them).
    static void decode_attr(u8 f, const u8* p, float* o) {
        o[0] = o[1] = o[2] = 0.0f;
        o[3] = 1.0f;
        int16_t sh[4];
        switch (f) {
            case kF32x1: case kF32x2: case kF32x3: case kF32x4:
                std::memcpy(o, p, size_t(4 * (f - kF32x1 + 1)));
                break;
            case kUN8x1: case kUN8x2: case kUN8x3: case kUN8x4:
                for (int c = 0; c <= f - kUN8x1; ++c) o[c] = float(p[c]) / 255.0f;
                break;
            case kBGRA8:
                o[0] = float(p[2]) / 255.0f;
                o[1] = float(p[1]) / 255.0f;
                o[2] = float(p[0]) / 255.0f;
                o[3] = float(p[3]) / 255.0f;
                break;
            case kSN16x1: case kSN16x2: case kSN16x3: case kSN16x4:
                std::memcpy(sh, p, size_t(2 * (f - kSN16x1 + 1)));
                for (int c = 0; c <= f - kSN16x1; ++c) o[c] = std::max(float(sh[c]) / 32767.0f, -1.0f);
                break;
            case kS16x1: case kS16x2: case kS16x3: case kS16x4:
                std::memcpy(sh, p, size_t(2 * (f - kS16x1 + 1)));
                for (int c = 0; c <= f - kS16x1; ++c) o[c] = float(sh[c]);
                break;
            default: break;
        }
    }
    shaderc_compiler_t compiler_ = nullptr;
    bool quiet_ = false;  // warm-up: failures are expected (stale sources)

public:
    VkImage image(Tex t) {
        std::lock_guard lk(images_m_);
        return t && t.id <= images_.size() ? images_[t.id - 1].img : VK_NULL_HANDLE;
    }
};

VulkanDevice* g_vk = nullptr;
}  // namespace

namespace vk {
VkImage image_of(Tex t) { return g_vk ? g_vk->image(t) : VK_NULL_HANDLE; }
}  // namespace vk

Device* create_vulkan_device() {
    if (!vk::init(nullptr)) return nullptr;
    g_vk = new VulkanDevice();
    return g_vk;
}

}  // namespace vc3::gpu::dev
