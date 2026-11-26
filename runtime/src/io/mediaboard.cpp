#include "mediaboard.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <mutex>
#include <string>
#include <vector>

#include "kernel/io.h"
#include "kernel/ke.h"
#include "mmio.h"

namespace vc3::io::mediaboard {

namespace {

uint8_t g_regs[0x100] = {};
std::mutex g_m;
uint8_t g_cmd_reply[0x20] = {}, g_cmd[0x20] = {};
std::vector<uint8_t> g_flash;  // media board flash (word-swapped like MAME's region)

constexpr uint32_t kIrqVector = 0x3A;  // IRQ 10
constexpr uint64_t kReplySector = 0x4800, kCommandSector = 0x4801;

void put32(uint8_t* p, uint32_t v) { std::memcpy(p, &v, 4); }
void put16(uint8_t* p, uint16_t v) { std::memcpy(p, &v, 2); }

// Sector command: {u16 tag, u16 command, args...}. Reply echoes the tag with
// command 0x8001 and command-specific data (see MAME baseboard_ide_event).
void run_command() {
    if (g_cmd[0] == 0 && g_cmd[1] == 0) return;
    std::memset(g_cmd_reply, 0, sizeof g_cmd_reply);
    g_cmd_reply[0] = g_cmd[0];
    g_cmd_reply[1] = g_cmd[1];
    g_cmd_reply[2] = 0x01;
    g_cmd_reply[3] = 0x80;
    const uint16_t cmd = uint16_t(g_cmd[2] | g_cmd[3] << 8);
    switch (cmd) {
        case 0x0001: put32(g_cmd_reply + 4, 0x00F00000); break;
        case 0x0100: put32(g_cmd_reply + 4, 5); put32(g_cmd_reply + 8, 0); break;  // loading phase: done, 0%
        case 0x0101: put16(g_cmd_reply + 4, 0x1234); put16(g_cmd_reply + 6, 0x4567); break;  // DIMM firmware 12.34
        case 0x0102: put32(g_cmd_reply + 4, 0); break;  // bit 16: development mode
        case 0x0103: std::memcpy(g_cmd_reply + 4, "-abc-abc12345678", 16); break;  // DIMM serial
        default: std::fprintf(stderr, "[mediaboard] unknown sector command %04x\n", cmd); break;
    }
    std::memset(g_cmd, 0, 4);
    g_regs[0xE1] = 1;
    kernel::ke::raise_interrupt(kIrqVector);
}

long mbcom_read(uint64_t off, uint8_t* buf, uint32_t len) {
    std::lock_guard lk(g_m);
    std::memset(buf, 0, len);
    for (uint32_t i = 0; i < len; i += 512) {
        const uint64_t lba = (off + i) / 512;
        if (lba == kReplySector) std::memcpy(buf + i, g_cmd_reply, std::min<uint32_t>(0x20, len - i));
        else if (lba == kCommandSector) std::memcpy(buf + i, g_cmd, std::min<uint32_t>(0x20, len - i));
    }
    return long(len);
}

long mbcom_write(uint64_t off, uint8_t* buf, uint32_t len) {
    std::lock_guard lk(g_m);
    static const bool tr = std::getenv("VC3_TRACE_USB") != nullptr;
    for (uint32_t i = 0; i < len; i += 512) {
        const uint64_t lba = (off + i) / 512;
        if (lba == kReplySector) std::memcpy(g_cmd_reply, buf + i, std::min<uint32_t>(0x20, len - i));
        if (lba == kCommandSector) {
            std::memcpy(g_cmd, buf + i, std::min<uint32_t>(0x20, len - i));
            if (tr) std::fprintf(stderr, "[mediaboard] sector command %02x%02x %02x%02x\n", g_cmd[1], g_cmd[0], g_cmd[3], g_cmd[2]);
            run_command();
        }
    }
    return long(len);
}

long mbrom_read(uint64_t off, uint8_t* buf, uint32_t len) {
    for (uint32_t i = 0; i < len; ++i) {
        const uint64_t o = (off + i) & 0xFFFFF;  // MAME: (lba & 0x7ff) * 512
        buf[i] = o < g_flash.size() ? g_flash[o] : 0xFF;
    }
    return long(len);
}

uint32_t read(void*, uint32_t port, unsigned size) {
    const uint32_t off = port & 0xFF;
    uint32_t v = 0;
    std::memcpy(&v, g_regs + off, std::min<unsigned>(size, 0x100 - off));
    static const bool tr = std::getenv("VC3_TRACE_USB") != nullptr;
    if (tr) std::fprintf(stderr, "[mediaboard] in  %#x (%u) -> %#x\n", port, size, v);
    return v;
}

void write(void*, uint32_t port, uint32_t v, unsigned size) {
    static const bool tr = std::getenv("VC3_TRACE_USB") != nullptr;
    if (tr) std::fprintf(stderr, "[mediaboard] out %#x (%u) = %#x\n", port, size, v);
    const uint32_t off = port & 0xFF;
    // 0xE0 (high byte, i.e. port 0x40E1) acknowledges IRQ 10.
    if (off == 0xE1 || (off == 0xE0 && size >= 2)) g_regs[0xE1] = 0;
}

void set16(uint32_t off, uint16_t v) { std::memcpy(g_regs + off, &v, 2); }

}  // namespace

void init() {
    set16(0x1E, 0x1000);
    set16(0x20, 0x00A0);
    set16(0x22, 0x4258);  // "XB"
    set16(0x24, 0x4D41);  // "AM"
    set16(0xF0, 0x0000);  // 0 = media board present
    set16(0xF4, 0x0002);  // DIMM size: 512 MB
    mmio::register_ports(0x4000, 0x100, "mediaboard", read, write);

    const char* dir = std::getenv("VC3_BIOS_DIR");
    const std::string path = std::string(dir ? dir : "work/bios") + "/fpr21042_m29w160et.bin";
    if (std::FILE* f = std::fopen(path.c_str(), "rb")) {
        g_flash.resize(0x200000);
        g_flash.resize(std::fread(g_flash.data(), 1, g_flash.size(), f));
        std::fclose(f);
        for (size_t i = 0; i + 1 < g_flash.size(); i += 2) std::swap(g_flash[i], g_flash[i + 1]);
    }
    // Partitions the Chihiro kernel exposes for the media board (see MAME's ide_baseboard notes).
    kernel::io::add_symlink("\\??\\mbfs:", "\\Device\\CdRom0");
    kernel::io::register_raw_device("\\Device\\MbCom", 0x8000ull * 512, mbcom_read, mbcom_write, "\\??\\mbcom:");
    kernel::io::register_raw_device("\\Device\\MbRom0", 0x800ull * 512, mbrom_read, nullptr, "\\??\\mbrom0:");
    kernel::io::register_raw_device("\\Device\\MbRom1", 0x800ull * 512, mbrom_read, nullptr, "\\??\\mbrom1:");
}

}  // namespace vc3::io::mediaboard
