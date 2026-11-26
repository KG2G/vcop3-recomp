#include "vc3/memory.h"

#include <cerrno>
#include <cstring>
#include <vector>

#include "vc3/recomp.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace vc3 {
uint8_t* g_mem = nullptr;
}

namespace vc3::mem {

static constexpr size_t kGuestSpace = size_t(1) << 32;

#ifdef _WIN32
// Windows: the 4 GiB guest space is one placeholder reservation (VirtualAlloc2), split
// where RAM views go; RAM is a pagefile-backed section mapped into those placeholders
// (MapViewOfFile3), so every mirror shares the same pages. What stays a placeholder faults
// on access, which is where the MMIO handler (mmio.cpp) comes in. Needs Windows 10 1803+.
namespace {
HANDLE g_section = nullptr;
using VirtualAlloc2Fn = PVOID(WINAPI*)(HANDLE, PVOID, SIZE_T, ULONG, ULONG, void*, ULONG);
using MapViewOfFile3Fn = PVOID(WINAPI*)(HANDLE, HANDLE, PVOID, ULONG64, SIZE_T, ULONG, ULONG, void*, ULONG);
using UnmapViewOfFile2Fn = BOOL(WINAPI*)(HANDLE, PVOID, ULONG);
VirtualAlloc2Fn p_VirtualAlloc2 = nullptr;
MapViewOfFile3Fn p_MapViewOfFile3 = nullptr;
UnmapViewOfFile2Fn p_UnmapViewOfFile2 = nullptr;
constexpr ULONG kReservePlaceholder = 0x00040000;  // MEM_RESERVE_PLACEHOLDER
constexpr ULONG kReplacePlaceholder = 0x00004000;  // MEM_REPLACE_PLACEHOLDER
constexpr ULONG kPreservePlaceholder = 0x00000002; // MEM_PRESERVE_PLACEHOLDER
std::vector<uint8_t*> g_views;

std::string last_error(const char* what) {
    return std::string(what) + " (Windows error " + std::to_string(GetLastError()) + ")";
}

// Map [guest, guest+size) of the placeholder to the section at `offset`: split the
// placeholder around it first.
bool map_view(uint32_t guest, uint32_t offset, uint32_t size) {
    uint8_t* at = g_mem + guest;
    if (!VirtualFree(at, size, MEM_RELEASE | kPreservePlaceholder)) return false;
    void* v = p_MapViewOfFile3(g_section, GetCurrentProcess(), at, offset, size, kReplacePlaceholder, PAGE_READWRITE,
                               nullptr, 0);
    if (!v) return false;
    g_views.push_back(at);
    return true;
}
}  // namespace

bool init(std::string& err) {
    HMODULE kb = GetModuleHandleA("kernelbase.dll");
    if (kb) {
        p_VirtualAlloc2 = reinterpret_cast<VirtualAlloc2Fn>(reinterpret_cast<void*>(GetProcAddress(kb, "VirtualAlloc2")));
        p_MapViewOfFile3 = reinterpret_cast<MapViewOfFile3Fn>(reinterpret_cast<void*>(GetProcAddress(kb, "MapViewOfFile3")));
        p_UnmapViewOfFile2 = reinterpret_cast<UnmapViewOfFile2Fn>(reinterpret_cast<void*>(GetProcAddress(kb, "UnmapViewOfFile2")));
    }
    if (!p_VirtualAlloc2 || !p_MapViewOfFile3) {
        err = "Windows 10 version 1803 or newer is needed (VirtualAlloc2 / MapViewOfFile3)";
        return false;
    }
    void* base = p_VirtualAlloc2(nullptr, nullptr, kGuestSpace, MEM_RESERVE | kReservePlaceholder, PAGE_NOACCESS, nullptr, 0);
    if (!base) {
        err = last_error("reserve 4 GiB guest space");
        return false;
    }
    g_mem = static_cast<uint8_t*>(base);
    g_section = CreateFileMappingA(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, kRamSize, nullptr);
    if (!g_section) {
        err = last_error("guest RAM section");
        return false;
    }
    for (uint32_t m : kRamMirrors)
        if (!map_view(m, 0, kRamSize)) {
            err = last_error("map RAM mirror");
            return false;
        }
    return true;
}

bool map_ram_alias(uint32_t guest, uint32_t ram_offset, uint32_t size) {
    return map_view(guest, ram_offset, size);
}

void shutdown() {
    if (p_UnmapViewOfFile2)
        for (uint8_t* v : g_views) p_UnmapViewOfFile2(GetCurrentProcess(), v, kPreservePlaceholder);
    g_views.clear();
    if (g_mem) VirtualFree(g_mem, 0, MEM_RELEASE);
    if (g_section) CloseHandle(g_section);
    g_mem = nullptr;
    g_section = nullptr;
}

#else

static int g_ram_fd = -1;

bool init(std::string& err) {
    void* base = mmap(nullptr, kGuestSpace, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (base == MAP_FAILED) {
        err = std::string("reserve 4 GiB guest space: ") + std::strerror(errno);
        return false;
    }
    g_mem = static_cast<uint8_t*>(base);

    g_ram_fd = memfd_create("vc3-guest-ram", MFD_CLOEXEC);
    if (g_ram_fd < 0 || ftruncate(g_ram_fd, kRamSize) != 0) {
        err = std::string("memfd for guest RAM: ") + std::strerror(errno);
        return false;
    }
    for (uint32_t m : kRamMirrors) {
        void* p = mmap(g_mem + m, kRamSize, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, g_ram_fd, 0);
        if (p == MAP_FAILED) {
            err = std::string("map RAM mirror: ") + std::strerror(errno);
            return false;
        }
    }
    return true;
}

bool map_ram_alias(uint32_t guest, uint32_t ram_offset, uint32_t size) {
    void* p = mmap(g_mem + guest, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, g_ram_fd, ram_offset);
    return p != MAP_FAILED;
}

void shutdown() {
    if (g_mem) munmap(g_mem, kGuestSpace);
    if (g_ram_fd >= 0) close(g_ram_fd);
    g_mem = nullptr;
    g_ram_fd = -1;
}

#endif

}  // namespace vc3::mem
