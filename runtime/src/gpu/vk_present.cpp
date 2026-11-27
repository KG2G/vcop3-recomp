// Vulkan presentation: the window thread keeps its own copy of the current picture (so
// the GPU thread's display slot can be reused at once) and blits it, letterboxed, into a
// swapchain image at every present.
#include "gpu/vk_present.h"
#include "gpu/vk_device.h"

#include <SDL3/SDL.h>

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace vc3::gpu::vkp {
namespace {
using dev::u64;

#define VKCHECK(x)                                                                                  \
    do {                                                                                            \
        const VkResult r_ = (x);                                                                    \
        if (r_ != VK_SUCCESS) {                                                                     \
            std::fprintf(stderr, "[vkp] %s failed: %d (%s:%d)\n", #x, int(r_), __FILE__, __LINE__); \
            std::abort();                                                                           \
        }                                                                                           \
    } while (0)

struct Img {
    VkImage img = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    int w = 0, h = 0;
};
struct Buf {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    void* map = nullptr;
    size_t size = 0;
};

constexpr int kFrames = 2;
SDL_Window* g_window = nullptr;
bool g_active = false, g_block = true;
VkDevice g_dev = VK_NULL_HANDLE;
VkSwapchainKHR g_sc = VK_NULL_HANDLE;
VkFormat g_sc_fmt = VK_FORMAT_B8G8R8A8_UNORM;
VkExtent2D g_sc_ext{};
std::vector<VkImage> g_sc_images;
std::vector<VkSemaphore> g_done_sem;  // per swapchain image: rendering finished
VkSemaphore g_acquire_sem[kFrames] = {};
VkCommandPool g_pool = VK_NULL_HANDLE;
VkCommandBuffer g_cb[kFrames] = {};
u64 g_cb_value[kFrames] = {};
int g_frame = 0;
u64 g_value = 0;  // presenter timeline: last value submitted
bool g_need_rebuild = false;
VkPresentModeKHR g_mode = VK_PRESENT_MODE_FIFO_KHR;

Img g_pic, g_ram;  // current picture (from the GPU), guest RAM picture
bool g_show_ram = true;
Buf g_staging[kFrames];  // guest RAM uploads
Buf g_readback;
Buf g_overlay[kFrames];  // overlay colour strips
Buf g_shot;              // VC3_WINDOW_SHOT readback
std::string g_shot_path;  // a capture's window shot, waiting for the next present
bool g_sc_readable = false;

// Pending for the next present: copy a device texture into g_pic.
struct Take { dev::Tex t; int w = 0, h = 0; dev::Sync ready = 0; bool on = false; } g_take;
bool g_ram_pending = false;
int g_ram_w = 0, g_ram_h = 0;

void destroy_img(Img& i) {
    if (i.img) vkDestroyImage(g_dev, i.img, nullptr);
    if (i.mem) vkFreeMemory(g_dev, i.mem, nullptr);
    i = Img{};
}
Img make_img(int w, int h) {
    Img i;
    i.w = w;
    i.h = h;
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = VK_FORMAT_B8G8R8A8_UNORM;
    ci.extent = {uint32_t(w), uint32_t(h), 1};
    ci.mipLevels = ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    VKCHECK(vkCreateImage(g_dev, &ci, nullptr, &i.img));
    VkMemoryRequirements mr;
    vkGetImageMemoryRequirements(g_dev, i.img, &mr);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = dev::vk::memory_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VKCHECK(vkAllocateMemory(g_dev, &ai, nullptr, &i.mem));
    VKCHECK(vkBindImageMemory(g_dev, i.img, i.mem, 0));
    return i;
}
Buf make_buf(size_t size, VkBufferUsageFlags usage) {
    Buf b;
    b.size = size;
    VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    ci.size = size;
    ci.usage = usage;
    VKCHECK(vkCreateBuffer(g_dev, &ci, nullptr, &b.buf));
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(g_dev, b.buf, &mr);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = dev::vk::memory_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VKCHECK(vkAllocateMemory(g_dev, &ai, nullptr, &b.mem));
    VKCHECK(vkBindBufferMemory(g_dev, b.buf, b.mem, 0));
    VKCHECK(vkMapMemory(g_dev, b.mem, 0, VK_WHOLE_SIZE, 0, &b.map));
    return b;
}
void destroy_buf(Buf& b) {
    if (b.buf) vkDestroyBuffer(g_dev, b.buf, nullptr);
    if (b.mem) vkFreeMemory(g_dev, b.mem, nullptr);
    b = Buf{};
}

// Everything submitted to the queue is done. (Not vkDeviceWaitIdle: that needs every queue
// externally synchronised, but the GPU thread submits whenever it likes; it takes the queue
// mutex for each submission, so holding that makes this wait valid.)
void queue_idle() {
    auto& c = dev::vk::ctx();
    std::lock_guard lk(c.queue_m);
    vkQueueWaitIdle(c.queue);
}

void wait_presenter(u64 v) {
    if (!v) return;
    VkSemaphore s = dev::vk::presenter_timeline();
    VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    wi.semaphoreCount = 1;
    wi.pSemaphores = &s;
    wi.pValues = &v;
    vkWaitSemaphores(g_dev, &wi, 5000000000ull);
}

void barrier(VkCommandBuffer cb, VkImage img, VkImageLayout from, VkImageLayout to) {
    VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    b.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    b.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
    b.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    b.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = img;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.imageMemoryBarrierCount = 1;
    di.pImageMemoryBarriers = &b;
    vkCmdPipelineBarrier2(cb, &di);
}
void global_barrier(VkCommandBuffer cb) {
    VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    mb.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    mb.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
    mb.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    mb.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.memoryBarrierCount = 1;
    di.pMemoryBarriers = &mb;
    vkCmdPipelineBarrier2(cb, &di);
}

void destroy_swapchain() {
    for (VkSemaphore s : g_done_sem) vkDestroySemaphore(g_dev, s, nullptr);
    g_done_sem.clear();
    g_sc_images.clear();
    if (g_sc) vkDestroySwapchainKHR(g_dev, g_sc, nullptr);
    g_sc = VK_NULL_HANDLE;
}

bool build_swapchain() {
    auto& c = dev::vk::ctx();
    queue_idle();
    VkSwapchainKHR old = g_sc;
    for (VkSemaphore s : g_done_sem) vkDestroySemaphore(g_dev, s, nullptr);
    g_done_sem.clear();
    VkSurfaceCapabilitiesKHR caps;
    VKCHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(c.phys, c.surface, &caps));
    int w = 0, h = 0;
    SDL_GetWindowSizeInPixels(g_window, &w, &h);
    if (caps.currentExtent.width != UINT32_MAX) {
        g_sc_ext = caps.currentExtent;
    } else {
        g_sc_ext = {uint32_t(std::clamp(w, int(caps.minImageExtent.width), int(caps.maxImageExtent.width))),
                    uint32_t(std::clamp(h, int(caps.minImageExtent.height), int(caps.maxImageExtent.height)))};
    }
    if (!g_sc_ext.width || !g_sc_ext.height) return false;  // minimised
    uint32_t n = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(c.phys, c.surface, &n, nullptr);
    std::vector<VkSurfaceFormatKHR> fmts(n);
    vkGetPhysicalDeviceSurfaceFormatsKHR(c.phys, c.surface, &n, fmts.data());
    VkSurfaceFormatKHR fmt = fmts.empty() ? VkSurfaceFormatKHR{VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR} : fmts[0];
    for (const auto& f : fmts)  // UNORM like the GL window (no sRGB conversion of the picture)
        if ((f.format == VK_FORMAT_B8G8R8A8_UNORM || f.format == VK_FORMAT_R8G8B8A8_UNORM) &&
            f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            fmt = f;
            break;
        }
    g_sc_fmt = fmt.format;
    g_sc_readable = (caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) != 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(c.phys, c.surface, &n, nullptr);
    std::vector<VkPresentModeKHR> modes(n);
    vkGetPhysicalDeviceSurfacePresentModesKHR(c.phys, c.surface, &n, modes.data());
    g_mode = VK_PRESENT_MODE_FIFO_KHR;
    if (!g_block) {
        for (VkPresentModeKHR m : modes)
            if (m == VK_PRESENT_MODE_MAILBOX_KHR) g_mode = m;
        if (g_mode == VK_PRESENT_MODE_FIFO_KHR)
            for (VkPresentModeKHR m : modes)
                if (m == VK_PRESENT_MODE_IMMEDIATE_KHR) g_mode = m;
    }
    VkSwapchainCreateInfoKHR ci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    ci.surface = c.surface;
    ci.minImageCount = std::max(caps.minImageCount, 3u);
#ifdef _WIN32
    // Windows, vsync (FIFO): two images, so at most one finished frame waits for the display.
    // With three, the driver queued two frames ahead (presents go through the desktop compositor
    // there): about two refreshes of extra aim lag and an uneven present rhythm (choppy).
    if (g_mode == VK_PRESENT_MODE_FIFO_KHR) ci.minImageCount = std::max(caps.minImageCount, 2u);
#endif
    if (caps.maxImageCount) ci.minImageCount = std::min(ci.minImageCount, caps.maxImageCount);
    ci.imageFormat = fmt.format;
    ci.imageColorSpace = fmt.colorSpace;
    ci.imageExtent = g_sc_ext;
    ci.imageArrayLayers = 1;
    ci.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                    (caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT);  // window shots
    ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.preTransform = caps.currentTransform;
    ci.compositeAlpha = (caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR) ? VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR
                                                                                        : VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;
    ci.presentMode = g_mode;
    ci.clipped = VK_TRUE;
    ci.oldSwapchain = old;
    VKCHECK(vkCreateSwapchainKHR(g_dev, &ci, nullptr, &g_sc));
    if (old) vkDestroySwapchainKHR(g_dev, old, nullptr);
    vkGetSwapchainImagesKHR(g_dev, g_sc, &n, nullptr);
    g_sc_images.resize(n);
    vkGetSwapchainImagesKHR(g_dev, g_sc, &n, g_sc_images.data());
    VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    g_done_sem.resize(n);
    for (auto& s : g_done_sem) VKCHECK(vkCreateSemaphore(g_dev, &si, nullptr, &s));
    std::fprintf(stderr, "[present] Vulkan swapchain %ux%u, %u images, %s\n", g_sc_ext.width, g_sc_ext.height, n,
                 g_mode == VK_PRESENT_MODE_FIFO_KHR ? "vsync (FIFO)" : g_mode == VK_PRESENT_MODE_MAILBOX_KHR ? "mailbox" : "immediate");
    g_need_rebuild = false;
    return true;
}
}  // namespace

bool init(SDL_Window* window) {
    auto& c = dev::vk::ctx();
    if (!dev::vk::init(window) || !c.surface || !c.swapchain) return false;
    g_window = window;
    g_dev = c.device;
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = c.queue_family;
    VKCHECK(vkCreateCommandPool(g_dev, &pci, nullptr, &g_pool));
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = g_pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = kFrames;
    VKCHECK(vkAllocateCommandBuffers(g_dev, &ai, g_cb));
    VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    for (auto& s : g_acquire_sem) VKCHECK(vkCreateSemaphore(g_dev, &si, nullptr, &s));
    for (auto& b : g_staging) b = make_buf(640 * 480 * 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    g_ram = make_img(640, 480);
    g_show_ram = true;
    if (!build_swapchain()) g_need_rebuild = true;
    g_active = true;
    return true;
}

bool active() { return g_active; }

void set_vsync(bool block) {
    if (block == g_block) return;
    g_block = block;
    g_need_rebuild = true;
}

dev::Sync take_texture(dev::Tex t, int w, int h, dev::Sync ready) {
    g_take = {t, w, h, ready, true};
    g_show_ram = false;
    return dev::Sync((g_value + 1) | dev::vk::kPresenterBit);  // the next present's submission
}

void take_ram(const void* bgra, int w, int h) {
    wait_presenter(g_cb_value[g_frame]);  // the staging buffer's last upload is done
    std::memcpy(g_staging[g_frame].map, bgra, size_t(w) * size_t(h) * 4);
    g_ram_pending = true;
    g_ram_w = w;
    g_ram_h = h;
    g_show_ram = true;
}

// Copies a picture the GPU thread handed over (take_texture) and signals the presenter
// timeline value promised for it, without a swapchain image: used when a present can't
// happen (swapchain being rebuilt, window minimised). The GPU thread waits for that value
// before reusing the display slot, so it must always come.
void submit_without_present() {
    if (!g_take.on && !g_ram_pending) return;
    const int f = g_frame;
    wait_presenter(g_cb_value[f]);
    VkCommandBuffer cb = g_cb[f];
    VKCHECK(vkResetCommandBuffer(cb, 0));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VKCHECK(vkBeginCommandBuffer(cb, &bi));
    global_barrier(cb);
    std::vector<VkSemaphoreSubmitInfo> waits;
    if (g_ram_pending) {
        if (!g_ram.img || g_ram.w != g_ram_w || g_ram.h != g_ram_h) {
            queue_idle();
            destroy_img(g_ram);
            g_ram = make_img(g_ram_w, g_ram_h);
        }
        barrier(cb, g_ram.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        VkBufferImageCopy c{};
        c.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        c.imageExtent = {uint32_t(g_ram_w), uint32_t(g_ram_h), 1};
        vkCmdCopyBufferToImage(cb, g_staging[f].buf, g_ram.img, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
        g_ram_pending = false;
    }
    if (g_take.on) {
        if (!g_pic.img || g_pic.w != g_take.w || g_pic.h != g_take.h) {
            queue_idle();
            destroy_img(g_pic);
            g_pic = make_img(g_take.w, g_take.h);
        }
        barrier(cb, g_pic.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        VkImageCopy c{};
        c.srcSubresource = c.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        c.extent = {uint32_t(g_take.w), uint32_t(g_take.h), 1};
        vkCmdCopyImage(cb, dev::vk::image_of(g_take.t), VK_IMAGE_LAYOUT_GENERAL, g_pic.img, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
        if (g_take.ready && !(g_take.ready & dev::vk::kPresenterBit)) {
            VkSemaphoreSubmitInfo gw{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
            gw.semaphore = dev::vk::gpu_timeline();
            gw.value = g_take.ready;
            gw.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            waits.push_back(gw);
        }
        g_take.on = false;
    }
    VKCHECK(vkEndCommandBuffer(cb));
    const u64 value = ++g_value;
    g_cb_value[f] = value;
    VkSemaphoreSubmitInfo sig{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    sig.semaphore = dev::vk::presenter_timeline();
    sig.value = value;
    sig.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    VkCommandBufferSubmitInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    cbi.commandBuffer = cb;
    VkSubmitInfo2 si{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    si.waitSemaphoreInfoCount = uint32_t(waits.size());
    si.pWaitSemaphoreInfos = waits.data();
    si.commandBufferInfoCount = 1;
    si.pCommandBufferInfos = &cbi;
    si.signalSemaphoreInfoCount = 1;
    si.pSignalSemaphoreInfos = &sig;
    {
        auto& c = dev::vk::ctx();
        std::lock_guard lk(c.queue_m);
        VKCHECK(vkQueueSubmit2(c.queue, 1, &si, VK_NULL_HANDLE));
    }
    dev::vk::presenter_submitted(value);
    g_frame = (g_frame + 1) % kFrames;
}

void request_shot(const std::string& path) { g_shot_path = path; }

void present(float x, float y, float pw, float ph, const std::vector<dev::OverlayRect>& overlay) {
    if (!g_active) return;
    int ww = 0, wh = 0;
    SDL_GetWindowSizeInPixels(g_window, &ww, &wh);
    if (!g_sc || g_need_rebuild || int(g_sc_ext.width) != ww || int(g_sc_ext.height) != wh) {
        if (!build_swapchain()) return submit_without_present();
    }
    const int f = g_frame;
    wait_presenter(g_cb_value[f]);  // this frame's command buffer is free again
    uint32_t idx = 0;
    VkResult r = vkAcquireNextImageKHR(g_dev, g_sc, 1000000000ull, g_acquire_sem[f], VK_NULL_HANDLE, &idx);
    if (r == VK_ERROR_OUT_OF_DATE_KHR) {
        g_need_rebuild = true;
        return submit_without_present();
    }
    if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR) return submit_without_present();
    VkCommandBuffer cb = g_cb[f];
    VKCHECK(vkResetCommandBuffer(cb, 0));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VKCHECK(vkBeginCommandBuffer(cb, &bi));
    global_barrier(cb);  // after everything submitted before (the GPU thread's frame included)
    // New pictures into our own images.
    if (g_ram_pending) {
        if (!g_ram.img || g_ram.w != g_ram_w || g_ram.h != g_ram_h) {
            queue_idle();
            destroy_img(g_ram);
            g_ram = make_img(g_ram_w, g_ram_h);
        }
        barrier(cb, g_ram.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        VkBufferImageCopy c{};
        c.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        c.imageExtent = {uint32_t(g_ram_w), uint32_t(g_ram_h), 1};
        vkCmdCopyBufferToImage(cb, g_staging[f].buf, g_ram.img, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
        g_ram_pending = false;
    }
    std::vector<VkSemaphoreSubmitInfo> waits;
    VkSemaphoreSubmitInfo aw{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    aw.semaphore = g_acquire_sem[f];
    aw.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    waits.push_back(aw);
    if (g_take.on) {
        if (!g_pic.img || g_pic.w != g_take.w || g_pic.h != g_take.h) {
            queue_idle();
            destroy_img(g_pic);
            g_pic = make_img(g_take.w, g_take.h);
        }
        barrier(cb, g_pic.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        VkImageCopy c{};
        c.srcSubresource = c.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        c.extent = {uint32_t(g_take.w), uint32_t(g_take.h), 1};
        vkCmdCopyImage(cb, dev::vk::image_of(g_take.t), VK_IMAGE_LAYOUT_GENERAL, g_pic.img, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
        if (g_take.ready && !(g_take.ready & dev::vk::kPresenterBit)) {
            VkSemaphoreSubmitInfo gw{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
            gw.semaphore = dev::vk::gpu_timeline();
            gw.value = g_take.ready;
            gw.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            waits.push_back(gw);
        }
        g_take.on = false;
    }
    global_barrier(cb);
    // Into the swapchain image: black, then the picture letterboxed.
    VkImage sc = g_sc_images[idx];
    barrier(cb, sc, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkClearColorValue black{};
    black.float32[3] = 1.0f;
    VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdClearColorImage(cb, sc, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &range);
    const Img& src = g_show_ram ? g_ram : g_pic;
    if (src.img) {
        global_barrier(cb);
        VkImageBlit b{};
        b.srcSubresource = b.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        b.srcOffsets[1] = {src.w, src.h, 1};
        b.dstOffsets[0] = {int(x), int(y), 0};
        b.dstOffsets[1] = {int(x + pw), int(y + ph), 1};
        vkCmdBlitImage(cb, src.img, VK_IMAGE_LAYOUT_GENERAL, sc, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &b, VK_FILTER_LINEAR);
    }
    // Overlay (crosshairs, Sinden border): each rectangle copied from a strip of its colour,
    // as many rows at a time as the strip holds.
    if (!overlay.empty()) {
        global_barrier(cb);
        constexpr uint32_t kPx = 16384;  // pixels per colour strip (>= any window width)
        const size_t need = overlay.size() * kPx * 4;
        if (g_overlay[f].size < need) {
            destroy_buf(g_overlay[f]);
            g_overlay[f] = make_buf(need, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        }
        const bool rgba = g_sc_fmt == VK_FORMAT_R8G8B8A8_UNORM || g_sc_fmt == VK_FORMAT_R8G8B8A8_SRGB;
        auto* px = static_cast<uint32_t*>(g_overlay[f].map);
        std::vector<VkBufferImageCopy> regions;
        for (size_t i = 0; i < overlay.size(); ++i) {
            const dev::OverlayRect& r = overlay[i];
            const int x0 = std::max(0, r.x), y0 = std::max(0, r.y);
            const int x1 = std::min(int(g_sc_ext.width), r.x + r.w), y1 = std::min(int(g_sc_ext.height), r.y + r.h);
            if (x1 <= x0 || y1 <= y0 || uint32_t(x1 - x0) > kPx) continue;
            const uint32_t c = rgba ? (0xFF000000u | uint32_t(r.b) << 16 | uint32_t(r.g) << 8 | r.r)
                                    : (0xFF000000u | uint32_t(r.r) << 16 | uint32_t(r.g) << 8 | r.b);
            std::fill_n(px + i * kPx, kPx, c);
            const int rows = int(kPx / uint32_t(x1 - x0));
            for (int yy = y0; yy < y1; yy += rows) {
                VkBufferImageCopy rc{};
                rc.bufferOffset = i * kPx * 4;
                rc.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                rc.imageOffset = {x0, yy, 0};
                rc.imageExtent = {uint32_t(x1 - x0), uint32_t(std::min(rows, y1 - yy)), 1};
                regions.push_back(rc);
            }
        }
        if (!regions.empty())
            vkCmdCopyBufferToImage(cb, g_overlay[f].buf, sc, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, uint32_t(regions.size()),
                                   regions.data());
    }
    // VC3_WINDOW_SHOT=N (testing): what the window shows at the N-th present, to
    // work/window_shot.ppm (the GL path's equivalent counts guest vblanks).
    static const long shot_at = [] {
        const char* e = std::getenv("VC3_WINDOW_SHOT");
        return e && !std::strchr(e, ':') ? std::atol(e) : -1;
    }();
    static long presents = 0;
    const bool shot = ((shot_at >= 0 && presents++ == shot_at) || !g_shot_path.empty()) && g_sc_readable;
    const std::string shot_path = g_shot_path.empty() ? "work/window_shot.ppm" : g_shot_path;
    g_shot_path.clear();
    if (shot) {
        destroy_buf(g_shot);
        g_shot = make_buf(size_t(g_sc_ext.width) * g_sc_ext.height * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        barrier(cb, sc, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        VkBufferImageCopy rc{};
        rc.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        rc.imageExtent = {g_sc_ext.width, g_sc_ext.height, 1};
        vkCmdCopyImageToBuffer(cb, sc, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, g_shot.buf, 1, &rc);
        barrier(cb, sc, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    } else
        barrier(cb, sc, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    VKCHECK(vkEndCommandBuffer(cb));
    const u64 value = ++g_value;
    g_cb_value[f] = value;
    VkSemaphoreSubmitInfo sig[2]{};
    sig[0].sType = sig[1].sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    sig[0].semaphore = g_done_sem[idx];
    sig[0].stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    sig[1].semaphore = dev::vk::presenter_timeline();
    sig[1].value = value;
    sig[1].stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    VkCommandBufferSubmitInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    cbi.commandBuffer = cb;
    VkSubmitInfo2 si{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    si.waitSemaphoreInfoCount = uint32_t(waits.size());
    si.pWaitSemaphoreInfos = waits.data();
    si.commandBufferInfoCount = 1;
    si.pCommandBufferInfos = &cbi;
    si.signalSemaphoreInfoCount = 2;
    si.pSignalSemaphoreInfos = sig;
    VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores = &g_done_sem[idx];
    pi.swapchainCount = 1;
    pi.pSwapchains = &g_sc;
    pi.pImageIndices = &idx;
    {
        auto& c = dev::vk::ctx();
        std::lock_guard lk(c.queue_m);
        VKCHECK(vkQueueSubmit2(c.queue, 1, &si, VK_NULL_HANDLE));
        r = vkQueuePresentKHR(c.queue, &pi);
    }
    dev::vk::presenter_submitted(value);
    if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR) g_need_rebuild = true;
    g_frame = (g_frame + 1) % kFrames;
    if (shot) {
        wait_presenter(value);
        const bool rgba = g_sc_fmt == VK_FORMAT_R8G8B8A8_UNORM || g_sc_fmt == VK_FORMAT_R8G8B8A8_SRGB;
        const auto* p = static_cast<const uint8_t*>(g_shot.map);
        if (std::FILE* fp = std::fopen(shot_path.c_str(), "wb")) {
            std::fprintf(fp, "P6\n%u %u\n255\n", g_sc_ext.width, g_sc_ext.height);
            for (size_t i = 0; i < size_t(g_sc_ext.width) * g_sc_ext.height; ++i) {
                const uint8_t rgb[3] = {p[i * 4 + (rgba ? 0 : 2)], p[i * 4 + 1], p[i * 4 + (rgba ? 2 : 0)]};
                std::fwrite(rgb, 1, 3, fp);
            }
            std::fclose(fp);
        }
        std::fprintf(stderr, "[present] wrote %s (%ux%u, Vulkan)\n", shot_path.c_str(), g_sc_ext.width, g_sc_ext.height);
    }
}

bool read_picture(std::vector<uint8_t>& bgra, int& w, int& h) {
    const Img& src = g_show_ram ? g_ram : g_pic;
    if (!src.img) return false;
    w = src.w;
    h = src.h;
    const size_t bytes = size_t(w) * size_t(h) * 4;
    if (g_readback.size < bytes) {
        destroy_buf(g_readback);
        g_readback = make_buf(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    }
    queue_idle();
    VkCommandBuffer cb = g_cb[g_frame];
    VKCHECK(vkResetCommandBuffer(cb, 0));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VKCHECK(vkBeginCommandBuffer(cb, &bi));
    global_barrier(cb);
    VkBufferImageCopy c{};
    c.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    c.imageExtent = {uint32_t(w), uint32_t(h), 1};
    vkCmdCopyImageToBuffer(cb, src.img, VK_IMAGE_LAYOUT_GENERAL, g_readback.buf, 1, &c);
    VKCHECK(vkEndCommandBuffer(cb));
    const u64 value = ++g_value;
    g_cb_value[g_frame] = value;
    VkSemaphoreSubmitInfo sig{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    sig.semaphore = dev::vk::presenter_timeline();
    sig.value = value;
    sig.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    VkCommandBufferSubmitInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    cbi.commandBuffer = cb;
    VkSubmitInfo2 si{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    si.commandBufferInfoCount = 1;
    si.pCommandBufferInfos = &cbi;
    si.signalSemaphoreInfoCount = 1;
    si.pSignalSemaphoreInfos = &sig;
    {
        auto& ctx = dev::vk::ctx();
        std::lock_guard lk(ctx.queue_m);
        VKCHECK(vkQueueSubmit2(ctx.queue, 1, &si, VK_NULL_HANDLE));
    }
    dev::vk::presenter_submitted(value);
    wait_presenter(value);
    bgra.resize(bytes);
    std::memcpy(bgra.data(), g_readback.map, bytes);
    return true;
}

void shutdown() {
    if (!g_active) return;
    queue_idle();
    destroy_swapchain();
    destroy_img(g_pic);
    destroy_img(g_ram);
    for (auto& b : g_staging) destroy_buf(b);
    destroy_buf(g_readback);
    for (auto& b : g_overlay) destroy_buf(b);
    destroy_buf(g_shot);
    g_active = false;
}

}  // namespace vc3::gpu::vkp
