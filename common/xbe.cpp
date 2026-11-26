#include "xbe.h"

#include <cstring>
#include <fstream>
#include <iterator>

namespace vc3 {

namespace {
template <typename T>
T rd(const std::vector<uint8_t>& d, size_t off) {
    T v{};
    if (off + sizeof(T) <= d.size()) std::memcpy(&v, d.data() + off, sizeof(T));
    return v;
}

struct XorKey { const char* name; uint32_t entry, thunk; };
constexpr XorKey kXorKeys[] = {
    {"retail",  0xA8FC57AB, 0x5B6D40B6},
    {"debug",   0x94859D4B, 0xEFB1F152},
    {"chihiro", 0x40B5C16E, 0x2290059D},
};
}  // namespace

std::optional<Xbe> Xbe::load(const std::string& path, std::string& err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { err = "cannot open " + path; return std::nullopt; }
    Xbe x;
    x.data_.assign(std::istreambuf_iterator<char>(f), {});
    const auto& d = x.data_;
    if (d.size() < 0x178 || std::memcmp(d.data(), "XBEH", 4) != 0) { err = "not an XBE"; return std::nullopt; }

    x.base         = rd<uint32_t>(d, 0x104);
    x.size_headers = rd<uint32_t>(d, 0x108);
    x.size_image   = rd<uint32_t>(d, 0x10C);
    const uint32_t nsec    = rd<uint32_t>(d, 0x11C);
    const uint32_t sec_va  = rd<uint32_t>(d, 0x120);
    const uint32_t entry_e = rd<uint32_t>(d, 0x128);
    x.stack_size           = rd<uint32_t>(d, 0x130);
    const uint32_t thunk_e = rd<uint32_t>(d, 0x158);
    const uint32_t nlibs   = rd<uint32_t>(d, 0x160);
    const uint32_t libs_va = rd<uint32_t>(d, 0x164);

    auto hdr = [&](uint32_t va) { return size_t(va - x.base); };
    auto cstr = [&](uint32_t va) {
        std::string s;
        for (size_t o = hdr(va); o < d.size() && d[o]; ++o) s += char(d[o]);
        return s;
    };

    for (uint32_t i = 0; i < nsec; ++i) {
        size_t o = hdr(sec_va) + i * 56;
        XbeSection s;
        s.flags   = rd<uint32_t>(d, o + 0);
        s.vaddr   = rd<uint32_t>(d, o + 4);
        s.vsize   = rd<uint32_t>(d, o + 8);
        s.raw     = rd<uint32_t>(d, o + 12);
        s.rawsize = rd<uint32_t>(d, o + 16);
        s.name    = cstr(rd<uint32_t>(d, o + 20));
        x.sections.push_back(std::move(s));
    }
    for (uint32_t i = 0; i < nlibs; ++i) {
        size_t o = hdr(libs_va) + i * 16;
        XbeLibrary l;
        l.name.assign(reinterpret_cast<const char*>(d.data() + o), strnlen(reinterpret_cast<const char*>(d.data() + o), 8));
        l.major = rd<uint16_t>(d, o + 8);  l.minor = rd<uint16_t>(d, o + 10);
        l.build = rd<uint16_t>(d, o + 12); l.flags = rd<uint16_t>(d, o + 14);
        x.libraries.push_back(std::move(l));
    }

    auto in_image = [&](uint32_t va) { return va >= x.base && va < x.base + x.size_image; };
    for (const auto& k : kXorKeys) {
        if (in_image(entry_e ^ k.entry) && in_image(thunk_e ^ k.thunk)) {
            x.platform = k.name;
            x.entry = entry_e ^ k.entry;
            x.kernel_thunk = thunk_e ^ k.thunk;
            break;
        }
    }
    if (x.platform.empty()) { err = "could not decode entry point / kernel thunk"; return std::nullopt; }

    for (uint32_t va = x.kernel_thunk;; va += 4) {
        uint32_t v = x.read32(va);
        if (v == 0) break;
        x.kernel_ordinals.push_back(v & 0x7FFFFFFF);
    }
    return x;
}

std::optional<Xbe> Xbe::load_raw(const std::string& path, uint32_t base, std::string& err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { err = "cannot open " + path; return std::nullopt; }
    Xbe x;
    x.data_.assign(std::istreambuf_iterator<char>(f), {});
    x.base = base;
    x.size_headers = 0;
    x.size_image = uint32_t(x.data_.size());
    x.entry = base;
    x.platform = "raw";
    XbeSection s;
    s.name = ".text";
    s.flags = XbeSection::kExecutable;
    s.vaddr = base;
    s.vsize = s.rawsize = uint32_t(x.data_.size());
    s.raw = 0;
    x.sections.push_back(s);
    return x;
}

const XbeSection* Xbe::section_at(uint32_t va) const {
    for (const auto& s : sections)
        if (s.contains(va)) return &s;
    return nullptr;
}

const uint8_t* Xbe::ptr(uint32_t va, size_t* avail) const {
    size_t off, end;
    if (va >= base && va < base + size_headers) {
        off = va - base; end = size_headers;
    } else if (const auto* s = section_at(va); s && va < s->vaddr + s->rawsize) {
        off = s->raw + (va - s->vaddr); end = s->raw + s->rawsize;
    } else {
        return nullptr;
    }
    if (end > data_.size() || off >= end) return nullptr;
    if (avail) *avail = end - off;
    return data_.data() + off;
}

uint32_t Xbe::read32(uint32_t va) const {
    size_t n = 0;
    const uint8_t* p = ptr(va, &n);
    uint32_t v = 0;
    if (p && n >= 4) std::memcpy(&v, p, 4);
    return v;
}

}  // namespace vc3
