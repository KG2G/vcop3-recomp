// Guest physical memory management.
//
// One allocator owns the 128 MiB of guest RAM (4 KiB pages). User virtual
// addresses equal physical addresses; the kernel's contiguous view is the
// 0x80000000 mirror (see vc3/memory.h), so MmGetPhysicalAddress just strips
// the mirror bits.
//
// Fixed layout:
//   0x00000000 - 0x0000FFFF   reserved (null page)
//   0x00010000 - image end    XBE
//   0x07600000 - 0x07DFFFFF   kernel pool (ExAllocatePool, kernel objects)
//   0x07E00000 - 0x07EFFFFF   kernel data exports
//   0x07FF0000 - 0x07FFFFFF   NV2A instance memory (MmClaimGpuInstanceMemory)
#pragma once
#include <cstdint>

namespace vc3::kernel::mm {

constexpr uint32_t kPage = 0x1000;
constexpr uint32_t kPoolBase = 0x07600000, kPoolEnd = 0x07E00000;
constexpr uint32_t kGpuInstanceBase = 0x07FF0000;

void init(uint32_t image_end);

// Page-granular physical allocation; returns 0 on failure. Memory is zeroed.
uint32_t alloc_pages(uint32_t bytes, uint32_t lowest = 0, uint32_t highest = 0xFFFFFFFF, uint32_t align = kPage,
                     bool top_down = false);
void free_pages(uint32_t phys, uint32_t bytes);

// Kernel pool (byte-granular, 16-byte aligned).
uint32_t pool_alloc(uint32_t bytes);
void pool_free(uint32_t addr);
uint32_t pool_size(uint32_t addr);

inline uint32_t to_phys(uint32_t va) {
    if (va >= 0xF0000000u) return va - 0xF0000000u;
    if (va >= 0x80000000u) return va - 0x80000000u;
    return va;
}

}  // namespace vc3::kernel::mm
