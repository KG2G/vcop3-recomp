#include "mm.h"

#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <vector>

#include "kernel.h"
#include "vc3/memory.h"

namespace vc3::kernel::mm {

namespace {

constexpr u32 kPages = mem::kRamSize / kPage;
std::recursive_mutex g_lock;
std::vector<bool> g_used(kPages, false);

void mark(u32 phys, u32 bytes, bool used) {
    for (u32 p = phys / kPage; p < (phys + bytes + kPage - 1) / kPage && p < kPages; ++p) g_used[p] = used;
}
u32 round_page(u32 v) { return (v + kPage - 1) & ~(kPage - 1); }

// --- pool: first-fit with coalescing ---
std::map<u32, u32> g_pool_free;  // addr -> size
std::map<u32, u32> g_pool_used;  // addr -> size

// --- regions handed out through the Mm*/Nt* APIs ---
struct Region {
    u32 size;
    u32 protect;
    enum Kind { Virtual, Contiguous, System, Image } kind;
};
std::map<u32, Region> g_regions;  // physical base -> region

const Region* region_containing(u32 phys, u32* base) {
    auto it = g_regions.upper_bound(phys);
    if (it == g_regions.begin()) return nullptr;
    --it;
    if (phys >= it->first + it->second.size) return nullptr;
    if (base) *base = it->first;
    return &it->second;
}

constexpr u32 PAGE_READWRITE = 0x04;
constexpr u32 MEM_COMMIT = 0x1000, MEM_RESERVE = 0x2000, MEM_DECOMMIT = 0x4000, MEM_RELEASE = 0x8000, MEM_FREE = 0x10000;
constexpr u32 MEM_PRIVATE = 0x20000, MEM_IMAGE = 0x1000000, MEM_TOP_DOWN = 0x100000;

}  // namespace

void init(u32 image_end) {
    std::lock_guard lk(g_lock);
    mark(0, 0x10000, true);
    mark(0x10000, round_page(image_end) - 0x10000, true);
    g_regions[0x10000] = Region{round_page(image_end) - 0x10000, PAGE_READWRITE, Region::Image};
    mark(kPoolBase, kPoolEnd - kPoolBase, true);
    mark(kKernelDataBase, mem::kRamSize - kKernelDataBase, true);  // kernel data, spare, GPU instance memory
    g_pool_free[kPoolBase] = kPoolEnd - kPoolBase;
}

u32 alloc_pages(u32 bytes, u32 lowest, u32 highest, u32 align, bool top_down) {
    std::lock_guard lk(g_lock);
    const u32 n = round_page(bytes) / kPage;
    if (n == 0) return 0;
    align = std::max(align, kPage);
    const u32 lo = std::max(lowest, 0u) / kPage;
    const u32 hi = std::min<u64>(u64(highest) + 1, mem::kRamSize) / kPage;  // exclusive
    if (hi < lo + n) return 0;
    auto fits = [&](u32 start) {
        for (u32 i = 0; i < n; ++i)
            if (g_used[start + i]) return false;
        return true;
    };
    const u32 step = align / kPage;
    if (!top_down) {
        for (u32 s = (lo + step - 1) / step * step; s + n <= hi; s += step)
            if (fits(s)) { mark(s * kPage, n * kPage, true); std::memset(host_ptr(s * kPage), 0, n * kPage); return s * kPage; }
    } else {
        for (s64 s = s64((hi - n) / step * step); s >= s64(lo); s -= step)
            if (fits(u32(s))) { mark(u32(s) * kPage, n * kPage, true); std::memset(host_ptr(u32(s) * kPage), 0, n * kPage); return u32(s) * kPage; }
    }
    return 0;
}

void free_pages(u32 phys, u32 bytes) {
    std::lock_guard lk(g_lock);
    mark(phys, round_page(bytes), false);
}

u32 pool_alloc(u32 bytes) {
    std::lock_guard lk(g_lock);
    bytes = std::max<u32>((bytes + 15) & ~15u, 16);
    for (auto it = g_pool_free.begin(); it != g_pool_free.end(); ++it) {
        if (it->second < bytes) continue;
        const u32 addr = it->first, rest = it->second - bytes;
        g_pool_free.erase(it);
        if (rest) g_pool_free[addr + bytes] = rest;
        g_pool_used[addr] = bytes;
        std::memset(host_ptr(addr), 0, bytes);
        return addr;
    }
    std::fprintf(stderr, "[mm] kernel pool exhausted (%u bytes)\n", bytes);
    return 0;
}

void pool_free(u32 addr) {
    std::lock_guard lk(g_lock);
    auto it = g_pool_used.find(addr);
    if (it == g_pool_used.end()) return;
    u32 a = addr, size = it->second;
    g_pool_used.erase(it);
    auto next = g_pool_free.find(a + size);
    if (next != g_pool_free.end()) { size += next->second; g_pool_free.erase(next); }
    auto prev = g_pool_free.lower_bound(a);
    if (prev != g_pool_free.begin()) {
        --prev;
        if (prev->first + prev->second == a) { a = prev->first; size += prev->second; g_pool_free.erase(prev); }
    }
    g_pool_free[a] = size;
}

u32 pool_size(u32 addr) {
    std::lock_guard lk(g_lock);
    auto it = g_pool_used.find(addr);
    return it == g_pool_used.end() ? 0 : it->second;
}

// ================================================================ exports

KEXPORT(ExAllocatePool) { return pool_alloc(a[0]); }
KEXPORT(ExAllocatePoolWithTag) { return pool_alloc(a[0]); }
KEXPORT(ExFreePool) { pool_free(a[0]); return 0; }
KEXPORT(ExQueryPoolBlockSize) { return pool_size(a[0]); }

// NtAllocateVirtualMemory(PVOID* BaseAddress, ULONG ZeroBits, PSIZE_T RegionSize, ULONG AllocationType, ULONG Protect)
KEXPORT(NtAllocateVirtualMemory) {
    std::lock_guard lk(g_lock);
    const u32 pbase = a[0], psize = a[2], type = a[3], protect = a[4];
    u32 base = rd<u32>(pbase);
    u32 size = rd<u32>(psize);
    if (size == 0) return STATUS_INVALID_PARAMETER;

    if (base) {
        u32 rbase;
        if (const Region* r = region_containing(base, &rbase)) {
            // Commit inside an existing reservation: nothing to do, pages are always backed.
            if (base + size > rbase + r->size) return STATUS_CONFLICTING_ADDRESSES;
            const u32 b = base & ~(kPage - 1);
            wr<u32>(pbase, b);
            wr<u32>(psize, round_page(base + size) - b);
            if (!(type & 0x800000)) {}  // MEM_NOZERO: committed pages keep their contents
            return STATUS_SUCCESS;
        }
        const u32 b = base & ~0xFFFFu;
        const u32 len = round_page(base + size) - b;
        for (u32 p = b; p < b + len; p += kPage)
            if (p >= mem::kRamSize || g_used[p / kPage]) return STATUS_CONFLICTING_ADDRESSES;
        mark(b, len, true);
        std::memset(host_ptr(b), 0, len);
        g_regions[b] = Region{len, protect, Region::Virtual};
        wr<u32>(pbase, b);
        wr<u32>(psize, len);
        return STATUS_SUCCESS;
    }
    const u32 len = round_page(size);
    const u32 b = alloc_pages(len, 0x10000, 0xFFFFFFFF, 0x10000, (type & MEM_TOP_DOWN) != 0);
    if (!b) return STATUS_NO_MEMORY;
    g_regions[b] = Region{len, protect, Region::Virtual};
    wr<u32>(pbase, b);
    wr<u32>(psize, len);
    return STATUS_SUCCESS;
}

// NtFreeVirtualMemory(PVOID* BaseAddress, PSIZE_T RegionSize, ULONG FreeType)
KEXPORT(NtFreeVirtualMemory) {
    std::lock_guard lk(g_lock);
    const u32 base = rd<u32>(a[0]);
    u32 rbase;
    const Region* r = region_containing(base, &rbase);
    if (!r || r->kind != Region::Virtual) return STATUS_MEMORY_NOT_ALLOCATED;
    if (a[2] & MEM_RELEASE) {
        wr<u32>(a[0], rbase);
        wr<u32>(a[1], r->size);
        mark(rbase, r->size, false);
        g_regions.erase(rbase);
    }
    // MEM_DECOMMIT: pages stay reserved and backed.
    return STATUS_SUCCESS;
}

// NtQueryVirtualMemory(PVOID BaseAddress, PMEMORY_BASIC_INFORMATION Buffer)
KEXPORT(NtQueryVirtualMemory) {
    std::lock_guard lk(g_lock);
    const u32 va = a[0] & ~(kPage - 1), out = a[1];
    u32 rbase;
    const Region* r = region_containing(mm::to_phys(va), &rbase);
    u32 alloc_base = 0, size, state, type = 0, protect = 0;
    if (r) {
        alloc_base = rbase;
        size = rbase + r->size - va;
        state = MEM_COMMIT;
        type = r->kind == Region::Image ? MEM_IMAGE : MEM_PRIVATE;
        protect = r->protect ? r->protect : PAGE_READWRITE;
    } else {
        auto it = g_regions.upper_bound(va);
        const u32 end = it == g_regions.end() ? mem::kRamSize : it->first;
        size = end > va ? end - va : kPage;
        state = MEM_FREE;
    }
    wr<u32>(out + 0x00, va);
    wr<u32>(out + 0x04, alloc_base);
    wr<u32>(out + 0x08, protect);
    wr<u32>(out + 0x0C, size);
    wr<u32>(out + 0x10, state);
    wr<u32>(out + 0x14, protect);
    wr<u32>(out + 0x18, type);
    return STATUS_SUCCESS;
}

KEXPORT(NtProtectVirtualMemory) {
    if (a[3]) wr<u32>(a[3], PAGE_READWRITE);
    return STATUS_SUCCESS;
}

// MmAllocateContiguousMemoryEx(SIZE NumberOfBytes, ULONG_PTR Lowest, ULONG_PTR Highest, ULONG_PTR Alignment, ULONG Protect)
static u32 alloc_contiguous(u32 bytes, u32 lowest, u32 highest, u32 align, u32 protect) {
    const u32 p = alloc_pages(bytes, lowest, highest, align ? align : kPage, true);
    if (!p) return 0;
    std::lock_guard lk(g_lock);
    g_regions[p] = Region{round_page(bytes), protect, Region::Contiguous};
    return 0x80000000u | p;
}
KEXPORT(MmAllocateContiguousMemory) { return alloc_contiguous(a[0], 0, 0xFFFFFFFF, 0, PAGE_READWRITE); }
KEXPORT(MmAllocateContiguousMemoryEx) { return alloc_contiguous(a[0], a[1], a[2], a[3], a[4]); }

KEXPORT(MmFreeContiguousMemory) {
    std::lock_guard lk(g_lock);
    const u32 p = to_phys(a[0]);
    auto it = g_regions.find(p);
    if (it == g_regions.end()) return 0;
    mark(p, it->second.size, false);
    g_regions.erase(it);
    return 0;
}

KEXPORT(MmAllocateSystemMemory) {
    const u32 p = alloc_pages(a[0]);
    if (!p) return 0;
    std::lock_guard lk(g_lock);
    g_regions[p] = Region{round_page(a[0]), a[1], Region::System};
    return p;
}
KEXPORT(MmFreeSystemMemory) {
    std::lock_guard lk(g_lock);
    auto it = g_regions.find(to_phys(a[0]));
    if (it == g_regions.end()) return 0;
    const u32 pages = it->second.size / kPage;
    mark(it->first, it->second.size, false);
    g_regions.erase(it);
    return pages;
}

KEXPORT(MmQueryAllocationSize) {
    std::lock_guard lk(g_lock);
    u32 base;
    const Region* r = region_containing(to_phys(a[0]), &base);
    return r ? r->size : 0;
}
KEXPORT(MmGetPhysicalAddress) { return to_phys(a[0]); }
KEXPORT(MmPersistContiguousMemory) { return 0; }
KEXPORT(MmQueryAddressProtect) { return PAGE_READWRITE; }
KEXPORT(MmSetAddressProtect) { return 0; }
KEXPORT(MmLockUnlockBufferPages) { return 0; }
KEXPORT(MmLockUnlockPhysicalPage) { return 0; }

// MmClaimGpuInstanceMemory(SIZE_T NumberOfBytes, SIZE_T* NumberOfPaddingBytes)
// Chihiro: no padding; returns the end of the contiguous physical map.
KEXPORT(MmClaimGpuInstanceMemory) {
    if (a[1]) wr<u32>(a[1], 0);
    return 0x80000000u + mem::kRamSize;
}

// MmQueryStatistics(PMM_STATISTICS) — Length must be 0x24.
KEXPORT(MmQueryStatistics) {
    const u32 s = a[0];
    if (!s || rd<u32>(s) != 0x24) return STATUS_INVALID_PARAMETER;
    std::lock_guard lk(g_lock);
    u32 free_pages = 0;
    for (bool u : g_used) free_pages += !u;
    wr<u32>(s + 0x04, kPages);
    wr<u32>(s + 0x08, free_pages);
    for (u32 o = 0x0C; o < 0x24; o += 4) wr<u32>(s + o, 0);
    return STATUS_SUCCESS;
}

}  // namespace vc3::kernel::mm
