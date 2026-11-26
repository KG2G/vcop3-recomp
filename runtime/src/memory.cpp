#include "vc3/memory.h"

#include <sys/mman.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>

#include "vc3/recomp.h"

namespace vc3 {
uint8_t* g_mem = nullptr;
}

namespace vc3::mem {

static constexpr size_t kGuestSpace = size_t(1) << 32;
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

}  // namespace vc3::mem
