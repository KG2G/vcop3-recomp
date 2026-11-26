// Vulkan context shared by the GPU thread's device (vk_device.cpp) and the window's
// presenter (vk_present.cpp).
#pragma once
#include <vulkan/vulkan.h>

#include <cstdint>
#include <mutex>

#include "gpu/device.h"

struct SDL_Window;

namespace vc3::gpu::dev::vk {

struct Context {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    uint32_t queue_family = 0;
    VkQueue queue = VK_NULL_HANDLE;
    std::mutex queue_m;  // vkQueueSubmit/vkQueuePresent from more than one thread
    VkPhysicalDeviceProperties props{};
    VkPhysicalDeviceMemoryProperties mem{};
    bool bc = false;                  // BC1-3 textures
    bool custom_border = false;       // VK_EXT_custom_border_color
    bool aniso = false;               // samplerAnisotropy enabled
    bool swapchain = false;           // VK_KHR_swapchain enabled
    PFN_vkCmdPushDescriptorSetKHR push_descriptor = nullptr;
    VkSurfaceKHR surface = VK_NULL_HANDLE;  // the window's (if any)
};
Context& ctx();

// Creates instance + device (on the main thread when a window exists, so the window's
// surface can be checked for presentation). Idempotent; false if Vulkan 1.3 is missing.
bool init(SDL_Window* window);

// Memory type with `flags` among `bits` (UINT32_MAX if none).
uint32_t memory_type(uint32_t bits, VkMemoryPropertyFlags flags);

// The device's image for a texture handle and the GPU thread's timeline semaphore.
VkImage image_of(Tex t);
// Timeline semaphores: device Syncs carry the semaphore in the top bit (0: GPU thread's,
// 1: the presenter's) and the value below it.
constexpr uint64_t kPresenterBit = 1ull << 63;
VkSemaphore gpu_timeline();
VkSemaphore presenter_timeline();
// The presenter has submitted the work that signals presenter-timeline value v (called by it
// after each queue submission). The GPU thread waits for that before queueing work that
// waits for v: both share one queue, which runs submissions in order, so a wait queued
// ahead of its signal would block the queue for good.
void presenter_submitted(uint64_t v);

}  // namespace vc3::gpu::dev::vk
