// Av*, Hal*, Ex settings, Dbg*, Fsc*, Xe*, Xc* and other small exports.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <vector>

#include "../log.h"
#include "gpu/nv2a.h"
#include "kernel.h"
#include "mm.h"

namespace vc3::kernel {

// ---------------------------------------------------------------- AV
// Chihiro drives a 31 kHz VGA monitor; report a VGA pack, NTSC-M timing, 60 Hz, 480p.
constexpr u32 AV_PACK_VGA = 0x5, AV_STANDARD_NTSC_M = 0x100, AV_FLAGS_HDTV_480p = 0x80000, AV_FLAGS_60Hz = 0x400000;
constexpr u32 kAvCapabilities = AV_PACK_VGA | AV_STANDARD_NTSC_M | AV_FLAGS_60Hz | AV_FLAGS_HDTV_480p;
static u32 g_av_saved_data = 0;

KEXPORT(AvGetSavedDataAddress) { return g_av_saved_data; }
KEXPORT(AvSetSavedDataAddress) { g_av_saved_data = a[0]; return 0; }
// AvSendTVEncoderOption(PVOID RegisterBase, ULONG Option, ULONG Param, PULONG Result)
KEXPORT(AvSendTVEncoderOption) {
    if (a[3]) wr<u32>(a[3], a[1] == 6 /*AV_QUERY_AV_CAPABILITIES*/ ? kAvCapabilities : 0);
    return 0;
}
// AvSetDisplayMode(PVOID RegisterBase, ULONG Step, ULONG Mode, ULONG Format, ULONG Pitch, ULONG FrameBuffer)
KEXPORT(AvSetDisplayMode) {
    if (trace_enabled())
        std::fprintf(stderr, "[av] AvSetDisplayMode(step=%u mode=%08x format=%08x pitch=%u fb=%08x)\n", a[1], a[2], a[3],
                     a[4], a[5]);
    if (a[5]) gpu::set_scanout(a[5]);  // the real kernel programs NV_PCRTC_START
    return 0;
}

// ---------------------------------------------------------------- EEPROM settings
// ExQueryNonVolatileSetting(ULONG ValueIndex, PULONG Type, PVOID Value, ULONG ValueLength, PULONG ResultLength)
KEXPORT(ExQueryNonVolatileSetting) {
    u32 value = 0;
    switch (a[0]) {
        case 7: value = 1; break;                                       // XC_LANGUAGE: English
        case 8: value = AV_FLAGS_60Hz | AV_FLAGS_HDTV_480p; break;      // XC_VIDEO
        case 0x103: value = AV_STANDARD_NTSC_M | AV_FLAGS_60Hz; break;  // XC_FACTORY_AV_REGION
        case 0x104: value = 1; break;                                   // XC_FACTORY_GAME_REGION: North America
        default: break;
    }
    if (a[1]) wr<u32>(a[1], 4);  // REG_DWORD
    if (a[2] && a[3] >= 4) wr<u32>(a[2], value);
    if (a[4]) wr<u32>(a[4], 4);
    if (trace_enabled()) std::fprintf(stderr, "[ex] ExQueryNonVolatileSetting(%#x) -> %#x\n", a[0], value);
    return STATUS_SUCCESS;
}

// ---------------------------------------------------------------- HAL
KEXPORT(HalRegisterShutdownNotification) { return 0; }
KEXPORT(HalReturnToFirmware) {
    std::fprintf(stderr, "[hal] HalReturnToFirmware(%u): guest requested reboot/shutdown\n", a[0]);
    // Quit without running global destructors: the renderer and audio threads are still
    // going (std::exit tore their state down under them: a crash on the way out).
    log::end("the game asked to restart; closed");
    std::_Exit(0);
}
KEXPORT(HalInitiateShutdown) {
    std::fprintf(stderr, "[hal] HalInitiateShutdown\n");
    log::end("the game shut down");
    std::_Exit(0);
}
// HalReadWritePCISpace(BusNumber, SlotNumber, RegisterNumber, Buffer, Length, WritePCISpace)
KEXPORT(HalReadWritePCISpace) {
    std::fprintf(stderr, "[hal] PCI %s bus=%u slot=%u reg=%#x len=%u\n", (a[5] & 0xFF) ? "write" : "read", a[0], a[1],
                 a[2], a[4]);
    if (!(a[5] & 0xFF)) std::memset(host_ptr(a[3]), 0, a[4]);
    return 0;
}

// ---------------------------------------------------------------- debug
static std::string format_guest(const std::string& fmt, u32 va_args) {
    std::string out;
    for (size_t i = 0; i < fmt.size(); ++i) {
        if (fmt[i] != '%' || i + 1 >= fmt.size()) { out += fmt[i]; continue; }
        std::string spec = "%";
        size_t j = i + 1;
        while (j < fmt.size() && std::strchr("-+ #0123456789.lhI", fmt[j])) spec += fmt[j++];
        if (j >= fmt.size()) break;
        const char conv = fmt[j];
        char buf[512];
        if (conv == '%') { out += '%'; }
        else if (conv == 's') { std::snprintf(buf, sizeof buf, "%s", read_cstr(rd<u32>(va_args)).c_str()); out += buf; va_args += 4; }
        else if (conv == 'c') { out += char(rd<u32>(va_args)); va_args += 4; }
        else if (std::strchr("diuxXp", conv)) {
            std::string s = spec;
            std::erase(s, 'l'); std::erase(s, 'h'); std::erase(s, 'I');
            s += conv == 'p' ? 'x' : conv;
            std::snprintf(buf, sizeof buf, s.c_str(), rd<u32>(va_args));
            out += buf;
            va_args += 4;
        } else if (std::strchr("fgeE", conv)) {
            std::snprintf(buf, sizeof buf, (spec + conv).c_str(), rd<double>(va_args));
            out += buf;
            va_args += 8;
        } else {
            out += spec + conv;
        }
        i = j;
    }
    return out;
}
// DbgPrint(PCHAR Format, ...)  [cdecl]
KEXPORT(DbgPrint) {
    std::string s = format_guest(read_cstr(a[0]), c.esp + 8);
    std::fprintf(stderr, "[guest] %s%s", s.c_str(), (!s.empty() && s.back() == '\n') ? "" : "\n");
    return STATUS_SUCCESS;
}
KEXPORT(DbgBreakPoint) {
    std::fprintf(stderr, "[dbg] DbgBreakPoint from %08x\n", rd<u32>(c.esp));
    return 0;
}

// ---------------------------------------------------------------- file system cache
static u32 g_fsc_pages = 0x100;
KEXPORT(FscGetCacheSize) { return g_fsc_pages; }
KEXPORT(FscSetCacheSize) { g_fsc_pages = a[0]; return STATUS_SUCCESS; }

// ---------------------------------------------------------------- XBE sections
// The loader maps every section up front, so (un)loading only maintains the
// reference count (XBE section header +0x18).
KEXPORT(XeLoadSection) {
    wr<u32>(a[0] + 0x18, rd<u32>(a[0] + 0x18) + 1);
    return STATUS_SUCCESS;
}
KEXPORT(XeUnloadSection) {
    const u32 n = rd<u32>(a[0] + 0x18);
    if (n) wr<u32>(a[0] + 0x18, n - 1);
    return STATUS_SUCCESS;
}

// ---------------------------------------------------------------- crypto (SHA-1)
namespace {
struct Sha1 {
    u32 h[5] = {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0};
    u64 len = 0;
    std::vector<u8> buf;

    void block(const u8* p) {
        u32 w[80];
        for (int i = 0; i < 16; ++i) w[i] = u32(p[4 * i]) << 24 | u32(p[4 * i + 1]) << 16 | u32(p[4 * i + 2]) << 8 | p[4 * i + 3];
        for (int i = 16; i < 80; ++i) w[i] = std::rotl(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
        u32 a = h[0], b = h[1], c = h[2], d = h[3], e = h[4];
        for (int i = 0; i < 80; ++i) {
            u32 f, k;
            if (i < 20) { f = (b & c) | (~b & d); k = 0x5A827999; }
            else if (i < 40) { f = b ^ c ^ d; k = 0x6ED9EBA1; }
            else if (i < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8F1BBCDC; }
            else { f = b ^ c ^ d; k = 0xCA62C1D6; }
            const u32 t = std::rotl(a, 5) + f + e + k + w[i];
            e = d; d = c; c = std::rotl(b, 30); b = a; a = t;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e;
    }
    void update(const u8* p, size_t n) {
        len += n;
        buf.insert(buf.end(), p, p + n);
        size_t off = 0;
        for (; off + 64 <= buf.size(); off += 64) block(buf.data() + off);
        buf.erase(buf.begin(), buf.begin() + long(off));
    }
    void final(u8 out[20]) {
        const u64 bits = len * 8;
        u8 pad = 0x80;
        update(&pad, 1);
        pad = 0;
        while (buf.size() != 56) update(&pad, 1);
        u8 l[8];
        for (int i = 0; i < 8; ++i) l[i] = u8(bits >> (56 - 8 * i));
        update(l, 8);
        for (int i = 0; i < 20; ++i) out[i] = u8(h[i / 4] >> (24 - 8 * (i % 4)));
    }
};
std::mutex g_sha_m;
std::map<u32, Sha1> g_sha;  // keyed by guest context address
}  // namespace

KEXPORT(XcSHAInit) { std::lock_guard lk(g_sha_m); g_sha[a[0]] = Sha1{}; return 0; }
KEXPORT(XcSHAUpdate) {
    std::lock_guard lk(g_sha_m);
    g_sha[a[0]].update(host_ptr(a[1]), a[2]);
    return 0;
}
KEXPORT(XcSHAFinal) {
    std::lock_guard lk(g_sha_m);
    g_sha[a[0]].final(host_ptr(a[1]));
    g_sha.erase(a[0]);
    return 0;
}
// XcHMAC(PBYTE Key, ULONG KeyLen, PBYTE Data, ULONG DataLen, PBYTE Data2, ULONG Data2Len, PBYTE Digest)
KEXPORT(XcHMAC) {
    u8 key[64] = {};
    if (a[1] > 64) { Sha1 s; s.update(host_ptr(a[0]), a[1]); s.final(key); }
    else std::memcpy(key, host_ptr(a[0]), a[1]);
    u8 ipad[64], opad[64], inner[20];
    for (int i = 0; i < 64; ++i) { ipad[i] = key[i] ^ 0x36; opad[i] = key[i] ^ 0x5C; }
    Sha1 in;
    in.update(ipad, 64);
    if (a[2]) in.update(host_ptr(a[2]), a[3]);
    if (a[4]) in.update(host_ptr(a[4]), a[5]);
    in.final(inner);
    Sha1 out;
    out.update(opad, 64);
    out.update(inner, 20);
    out.final(host_ptr(a[6]));
    return 0;
}
KEXPORT(XcVerifyPKCS1Signature) { return 1; }  // signatures are not checked

}  // namespace vc3::kernel
