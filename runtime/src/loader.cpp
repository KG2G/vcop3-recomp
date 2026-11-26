#include "loader.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "kernel/kernel.h"
#include "kernel/mm.h"
#include "vc3/memory.h"
#include "vc3/recomp.h"

namespace vc3 {

bool load_xbe(const Xbe& xbe, std::string& err) {
    const auto& raw = xbe.raw();
    if (xbe.base + xbe.size_image > mem::kRamSize) {
        err = "XBE image does not fit in guest RAM";
        return false;
    }
    kernel::mm::init(xbe.base + xbe.size_image);
    std::memcpy(host_ptr(xbe.base), raw.data(), std::min<size_t>(xbe.size_headers, raw.size()));
    for (const auto& s : xbe.sections) {
        if (s.raw + s.rawsize > raw.size()) {
            err = "section " + s.name + " extends past end of file";
            return false;
        }
        std::memset(host_ptr(s.vaddr), 0, s.vsize);
        std::memcpy(host_ptr(s.vaddr), raw.data() + s.raw, std::min(s.rawsize, s.vsize));
    }

    size_t data_exports = 0;
    for (size_t i = 0; i < xbe.kernel_ordinals.size(); ++i) {
        uint32_t ord = xbe.kernel_ordinals[i];
        uint32_t addr = kernel::data_export_addr(ord);
        if (addr) ++data_exports;
        else addr = kKernelThunkBase + ord;
        wr<uint32_t>(xbe.kernel_thunk + uint32_t(i) * 4, addr);
    }
    // VC3_TEST_MODE=1: boot the way SegaBoot's "GAME TEST MODE" relaunches the game, with a
    // LaunchDataPage of type LDT_TITLE whose first launch-data dword is 1.
    if (const char* tm = std::getenv("VC3_TEST_MODE"); tm && std::atoi(tm)) {
        const u32 page = kernel::mm::alloc_pages(0x1000);
        const u32 cert = rd<u32>(xbe.base + 0x118);
        wr<u32>(page + 0x000, 0);                      // dwLaunchDataType = LDT_TITLE
        wr<u32>(page + 0x004, rd<u32>(cert + 0x08));   // dwTitleId (must match the XBE)
        wr<u32>(page + 0x400, 1);                      // LaunchData[0] = 1: test mode
        wr<u32>(kernel::data_export_addr("LaunchDataPage"), 0x80000000u | page);
        std::printf("[loader] launching into test mode\n");
    }
    std::printf("[loader] %zu sections, %zu kernel imports (%zu data exports)\n", xbe.sections.size(),
                xbe.kernel_ordinals.size(), data_exports);
    return true;
}

}  // namespace vc3
