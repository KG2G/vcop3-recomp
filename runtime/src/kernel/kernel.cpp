#include "kernel.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string_view>
#include <unordered_map>

namespace vc3::kernel {

namespace {

struct ExportInfo {
    uint32_t ordinal;
    const char* name;
    bool is_variable;
    Conv conv;
    int nargs;
};

constexpr ExportInfo kExports[] = {
#define X(ord, nm, var, conv, nargs) {ord, #nm, var != 0, Conv(conv), nargs},
#include "exports.inc"
#undef X
};

const ExportInfo* find_export(uint32_t ordinal) {
    for (const auto& e : kExports)
        if (e.ordinal == ordinal) return &e;
    return nullptr;
}
const ExportInfo* find_export(std::string_view name) {
    for (const auto& e : kExports)
        if (name == e.name) return &e;
    return nullptr;
}

std::map<std::string, Handler>& pending_registrations() {
    static std::map<std::string, Handler> m;
    return m;
}

Handler g_handlers[512] = {};
std::unordered_map<uint32_t, uint32_t> g_data_addr;
bool g_initialized = false;

// Sizes of the data exports (bytes). Anything not listed gets 0x100.
uint32_t data_export_size(std::string_view n) {
    if (n == "XboxHardwareInfo" || n == "XboxKrnlVersion" || n == "XeImageFileName") return 8;
    if (n == "XboxHDKey" || n == "XboxSignatureKey") return 16;
    if (n == "XboxAlternateSignatureKeys") return 16 * 16;
    if (n == "XePublicKeyData") return 284;
    if (n.find("ObjectType") != std::string_view::npos) return 0x20;  // OBJECT_TYPE
    return 0x100;
}

}  // namespace

Registrar::Registrar(const char* name, Handler h) { pending_registrations()[name] = h; }

bool trace_enabled() {
    static const bool on = std::getenv("VC3_TRACE_KERNEL") != nullptr;
    return on;
}

void init() {
    if (g_initialized) return;
    g_initialized = true;

    for (const auto& [name, h] : pending_registrations()) {
        const auto* e = find_export(name);
        if (!e || e->is_variable) {
            std::fprintf(stderr, "[kernel] handler for unknown export %s\n", name.c_str());
            std::abort();
        }
        g_handlers[e->ordinal] = h;
    }

    uint32_t p = kKernelDataBase;
    for (const auto& e : kExports) {
        if (!e.is_variable) continue;
        g_data_addr[e.ordinal] = p;
        std::memset(host_ptr(p), 0, data_export_size(e.name));
        p += (data_export_size(e.name) + 15) & ~15u;
    }
    if (p > kKernelDataBase + kKernelDataSize) std::abort();

    // KeTimeIncrement: 1 ms in 100 ns units.
    wr<u32>(data_export_addr("KeTimeIncrement"), 10000);
    // XboxKrnlVersion. TODO: use the version of the actual Chihiro BIOS kernel.
    const u32 kv = data_export_addr("XboxKrnlVersion");
    wr<u16>(kv + 0, 1); wr<u16>(kv + 2, 0); wr<u16>(kv + 4, 5838); wr<u16>(kv + 6, 1);
    // XboxHardwareInfo: Flags (8 = arcade/Chihiro), GpuRevision, McpRevision.
    const u32 hw = data_export_addr("XboxHardwareInfo");
    wr<u32>(hw, 0xC0000030 | 0x8);
    wr<u8>(hw + 4, 0xD3);
    // XeImageFileName: ANSI_STRING pointing at a path buffer after it.
    const u32 xe = data_export_addr("XeImageFileName");
    static const char kImage[] = "\\Device\\CdRom0\\vc3.xbe";
    const u32 buf = kKernelDataBase + kKernelDataSize - 0x100;
    std::memcpy(host_ptr(buf), kImage, sizeof kImage);
    wr<u16>(xe, sizeof kImage - 1); wr<u16>(xe + 2, sizeof kImage); wr<u32>(xe + 4, buf);
    // LaunchDataPage stays NULL (cold boot). HalDiskCachePartitionCount: 3.
    wr<u32>(data_export_addr("HalDiskCachePartitionCount"), 3);
}

uint32_t data_export_addr(uint32_t ordinal) {
    if (!g_initialized) init();
    auto it = g_data_addr.find(ordinal);
    return it == g_data_addr.end() ? 0 : it->second;
}
uint32_t data_export_addr(const char* name) {
    const auto* e = find_export(name);
    return e ? data_export_addr(e->ordinal) : 0;
}

const char* name(uint32_t ordinal) {
    const auto* e = find_export(ordinal);
    return e ? e->name : "?";
}

void call(Cpu& c, uint32_t ordinal) {
    const auto* e = find_export(ordinal);
    Handler h = ordinal < 512 ? g_handlers[ordinal] : nullptr;
    if (!e || !h) {
        std::fprintf(stderr, "[kernel] unimplemented export #%u %s (ret=%08x)\n", ordinal, name(ordinal),
                     rd<uint32_t>(c.esp));
        std::abort();
    }
    const Args a{c, e->conv};
    if (trace_enabled()) {
        std::fprintf(stderr, "[kernel] %s(", e->name);
        for (int i = 0; i < e->nargs; ++i) std::fprintf(stderr, "%s%08x", i ? ", " : "", a[unsigned(i)]);
        std::fprintf(stderr, ") from %08x\n", rd<u32>(c.esp));
    }
    const u32 ret = h(c, a);
    c.eax = ret;
    // Pop the return address, then stack arguments for callee-cleaned conventions.
    u32 pop = 4;
    if (e->conv == kStdcall) pop += 4 * u32(std::max(e->nargs, 0));
    else if (e->conv == kFastcall) pop += 4 * u32(std::max(e->nargs - 2, 0));
    c.esp += pop;
    if (trace_enabled()) std::fprintf(stderr, "[kernel]   %s -> %08x\n", e->name, ret);
}

// ---------------------------------------------------------------- helpers
std::string read_cstr(u32 addr, size_t max) {
    std::string s;
    for (size_t i = 0; i < max; ++i) {
        const char ch = char(rd<u8>(addr + u32(i)));
        if (!ch) break;
        s += ch;
    }
    return s;
}

std::string read_ansi_string(u32 str) {
    if (!str) return {};
    const u16 len = rd<u16>(str);
    const u32 buf = rd<u32>(str + 4);
    std::string s(len, '\0');
    if (buf) std::memcpy(s.data(), host_ptr(buf), len);
    return s;
}

s64 now_100ns() {
    using namespace std::chrono;
    return duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count() / 100;
}

s64 system_time_100ns() {
    using namespace std::chrono;
    const s64 unix_100ns = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count() / 100;
    return unix_100ns + 116444736000000000LL;
}

}  // namespace vc3::kernel
