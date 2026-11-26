#pragma once
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace vc3 {

struct XbeSection {
    std::string name;
    uint32_t flags = 0;
    uint32_t vaddr = 0, vsize = 0;
    uint32_t raw = 0, rawsize = 0;

    static constexpr uint32_t kWritable = 0x1;
    static constexpr uint32_t kExecutable = 0x4;  // XBE "executable" bit
    bool contains(uint32_t va) const { return va >= vaddr && va < vaddr + vsize; }
};

struct XbeLibrary {
    std::string name;
    uint16_t major, minor, build, flags;
};

// Minimal XBE parser. Mirrors tools/xbe_info.py.
class Xbe {
public:
    static std::optional<Xbe> load(const std::string& path, std::string& err);
    // Wrap a raw code blob as a one-section ".text" image (used by the lifter tests).
    static std::optional<Xbe> load_raw(const std::string& path, uint32_t base, std::string& err);

    uint32_t base = 0, size_image = 0, size_headers = 0;
    uint32_t entry = 0, kernel_thunk = 0, stack_size = 0;
    std::string platform;  // "retail" / "debug" / "chihiro"
    std::vector<XbeSection> sections;
    std::vector<XbeLibrary> libraries;
    std::vector<uint32_t> kernel_ordinals;  // in thunk-table order

    const XbeSection* section_at(uint32_t va) const;
    // Pointer to file bytes backing `va` (nullptr if BSS / unmapped); `avail` = bytes readable.
    const uint8_t* ptr(uint32_t va, size_t* avail = nullptr) const;
    uint32_t read32(uint32_t va) const;

    const std::vector<uint8_t>& raw() const { return data_; }

private:
    std::vector<uint8_t> data_;
};

}  // namespace vc3
