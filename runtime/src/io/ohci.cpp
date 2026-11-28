// OHCI USB host controller (Xbox MCPX has two: 0xFED00000 IRQ 1, 0xFED08000 IRQ 9).
//
// Registers are served through the MMIO layer; a frame thread runs the
// schedule every millisecond like the hardware does: periodic (interrupt)
// EDs from the HCCA table, then the control and bulk lists. Completed TDs go
// to the done queue, which is written back to the HCCA with a WDH interrupt.
// Descriptors live in guest physical memory (== guest address in low RAM).
#include "platform.h"
#include "ohci.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>

#include "kernel/ke.h"
#include "log.h"
#include "vc3/memory.h"
#include "mmio.h"
#include "scheduler.h"
#include "vc3/helpers.h"

namespace vc3::io::ohci {

namespace {

constexpr int kPorts = 4;

// The controllers' recent history, written into the log only if the program crashes (a rare crash on
// Windows + OpenGL in the XDK USB driver's completion code: it followed a descriptor whose next pointer
// read 0xFFFFFFFx). Kinds: R retired TD (ed, td, next, cc/done, new head), X head changed by the driver
// meanwhile, D done queue written to the HCCA, G register written by the driver (offset, value).
struct Event { double t; char kind; u32 a, b, c, d, e; };
constexpr u32 kEvents = 256;
Event g_events[kEvents];
std::atomic<u32> g_event_n{0};
void note(char kind, u32 a, u32 b = 0, u32 c = 0, u32 d = 0, u32 e = 0) {
    g_events[g_event_n.fetch_add(1, std::memory_order_relaxed) % kEvents] = Event{log::uptime(), kind, a, b, c, d, e};
}
void dump_events() {
    const u32 n = g_event_n.load(), first = n > kEvents ? n - kEvents : 0;
    std::fprintf(stderr, "[ohci] last %u controller events (R ed td next cc|done new-head; X ed old-head now-head; D hcca done-head; G reg value):\n", n - first);
    u32 tds[16] = {};
    int ntd = 0;
    for (u32 i = first; i < n; ++i) {
        const Event& e = g_events[i % kEvents];
        std::fprintf(stderr, "  %9.4f %c %08x %08x %08x %08x %08x\n", e.t, e.kind, e.a, e.b, e.c, e.d, e.e);
        if (e.kind == 'R') tds[ntd++ % 16] = e.b;
    }
    std::fprintf(stderr, "[ohci] the last retired TDs now (ctrl cbp next be | driver words):\n");
    for (int i = 0; i < std::min(ntd, 16); ++i) {
        const u32 td = tds[i];
        if (!td || td + 32 > mem::kRamSize) continue;
        std::fprintf(stderr, "  %08x: %08x %08x %08x %08x | %08x %08x %08x %08x\n", td, rd<u32>(td), rd<u32>(td + 4),
                     rd<u32>(td + 8), rd<u32>(td + 12), rd<u32>(td + 16), rd<u32>(td + 20), rd<u32>(td + 24), rd<u32>(td + 28));
    }
}

// Address range of the descriptors the drivers gave the controllers (see overlaps_descriptors).
std::atomic<u32> g_desc_lo{~0u}, g_desc_hi{0};
void note_descriptor(u32 addr, u32 size) {
    u32 v = g_desc_lo.load(std::memory_order_relaxed);
    while (addr < v && !g_desc_lo.compare_exchange_weak(v, addr)) {}
    v = g_desc_hi.load(std::memory_order_relaxed);
    while (addr + size > v && !g_desc_hi.compare_exchange_weak(v, addr + size)) {}
}

// HcControl
constexpr u32 PLE = 1u << 2, CLE = 1u << 4, BLE = 1u << 5, HCFS_MASK = 3u << 6, HCFS_OPERATIONAL = 2u << 6;
// HcCommandStatus
constexpr u32 HCR = 1u << 0, CLF = 1u << 1, BLF = 1u << 2;
// Interrupts
constexpr u32 INT_WDH = 1u << 1, INT_SF = 1u << 2, INT_RHSC = 1u << 6, INT_MIE = 1u << 31;
// Port status
constexpr u32 CCS = 1u << 0, PES = 1u << 1, PRS = 1u << 4, PPS = 1u << 8, CSC = 1u << 16, PESC = 1u << 17,
              PRSC = 1u << 20;
// TD condition codes
constexpr u32 CC_NOERROR = 0, CC_STALL = 4, CC_DATAUNDERRUN = 9, CC_NOT_ACCESSED = 15;

struct Controller {
    const char* name;
    u32 vector;
    std::recursive_mutex m;

    u32 control = 0, command = 0, int_status = 0, int_enable = 0, hcca = 0;
    u32 control_head = 0, control_cur = 0, bulk_head = 0, bulk_cur = 0, done_head = 0;
    u32 fm_interval = 0x2EDF, periodic_start = 0, ls_threshold = 0x628;
    u32 rh_desc_a = kPorts | 0x200 /*NPS*/, rh_desc_b = 0, rh_status = 0;
    u32 port[kPorts] = {};
    usb::Device* device[kPorts] = {};
    u16 frame = 0;
    bool irq_pending = false;

    void reset() {
        control = command = int_status = int_enable = hcca = 0;
        control_head = control_cur = bulk_head = bulk_cur = done_head = 0;
        fm_interval = 0x2EDF;
        for (int p = 0; p < kPorts; ++p) port[p] = PPS | (device[p] ? (CCS | CSC) : 0);
    }

    usb::Device* find(u32 addr) {
        for (int p = 0; p < kPorts; ++p)
            if (device[p] && (port[p] & PES) && device[p]->address == addr) return device[p];
        return nullptr;
    }

    // Process all pending TDs of one ED. Returns true if any TD was retired.
    bool process_ed(u32 ed, bool isochronous_ok) {
        const u32 d0 = rd<u32>(ed);
        if (d0 & (1u << 14)) return false;  // sKip
        if (d0 & (1u << 15)) return false;  // isochronous: not supported (no iso devices)
        (void)isochronous_ok;
        bool worked = false;
        for (int guard = 0; guard < 64; ++guard) {
            if (guard && (rd<u32>(ed) & (1u << 14))) break;  // sKip set meanwhile (the driver is editing the list)
            const u32 tail = rd<u32>(ed + 4) & ~0xFu;
            u32 head = rd<u32>(ed + 8);
            if (head & 1) break;  // Halted
            const u32 td = head & ~0xFu;
            if (td == tail || !td) break;
            note_descriptor(ed, 16);
            note_descriptor(td, 32);  // 16 bytes for the controller + the driver's own fields after them

            const u32 fa = d0 & 0x7F, en = (d0 >> 7) & 0xF, ed_dir = (d0 >> 11) & 3, mps = (d0 >> 16) & 0x7FF;
            u32 t0 = rd<u32>(td);
            const u32 cbp = rd<u32>(td + 4), next = rd<u32>(td + 8), be = rd<u32>(td + 12);
            const u32 dir = (ed_dir == 1 || ed_dir == 2) ? ed_dir : (t0 >> 19) & 3;  // 0 setup, 1 out, 2 in
            const usb::Pid pid = dir == 0 ? usb::Pid::Setup : dir == 1 ? usb::Pid::Out : usb::Pid::In;

            // Buffer may span two 4 KiB pages: [cbp, page end) then [page(be), be].
            u32 len = 0;
            if (cbp) len = ((be & ~0xFFFu) == (cbp & ~0xFFFu)) ? be - cbp + 1 : (0x1000 - (cbp & 0xFFF)) + (be & 0xFFF) + 1;
            auto addr_at = [&](u32 off) {
                const u32 first = 0x1000 - (cbp & 0xFFF);
                return off < first ? cbp + off : (be & ~0xFFFu) + (off - first);
            };

            usb::Device* dev = find(fa);
            u32 cc = CC_NOERROR, done = 0;
            if (!dev) {
                cc = 5;  // DeviceNotResponding
            } else {
                // Packetise by max packet size; a short packet ends the transfer.
                std::vector<u8> tmp;
                do {
                    const u32 chunk = std::min(len - done, mps ? mps : 8u);
                    tmp.assign(chunk, 0);
                    if (pid != usb::Pid::In)
                        for (u32 i = 0; i < chunk; ++i) tmp[i] = rd<u8>(addr_at(done + i));
                    const int r = dev->transfer(int(en), pid, tmp.data(), int(chunk));
                    if (r == usb::kNak) { if (done == 0) return worked; break; }
                    if (r == usb::kStall) { cc = CC_STALL; break; }
                    if (pid == usb::Pid::In)
                        for (int i = 0; i < r; ++i) wr<u8>(addr_at(done + u32(i)), tmp[size_t(i)]);
                    done += u32(r);
                    if (u32(r) < chunk) break;
                } while (done < len);
                if (cc == CC_NOERROR && done < len && pid == usb::Pid::In && !(t0 & (1u << 18))) cc = CC_DATAUNDERRUN;
            }

            {
                static const bool tr = std::getenv("VC3_TRACE_USB") != nullptr;
                if (tr) std::fprintf(stderr, "[ohci] retire ed %08x td %08x (tail %08x next %08x) dev %u ep %u dir %u %u/%u cc %u sw[%08x %08x %08x %08x] frame %u\n",
                                     ed, td, tail, next, fa, en, dir, done, len, cc, rd<u32>(td + 16), rd<u32>(td + 20),
                                     rd<u32>(td + 24), rd<u32>(td + 28), frame);
            }
            if (cc != CC_NOERROR) {
                static const bool tr = std::getenv("VC3_TRACE_USB") != nullptr;
                if (tr || cc != CC_DATAUNDERRUN)
                    std::fprintf(stderr, "[ohci] %s TD %08x ep %u/%u dir %u: cc %u (%u of %u bytes, R=%u)\n", name, td, fa, en,
                                 dir, cc, done, len, (t0 >> 18) & 1);
            }
            // Retire the TD. The ED's head is advanced first, and only if the driver didn't change it
            // while the transfer ran: this thread runs beside the guest (no guest lock), and a driver
            // that cancelled the transfer meanwhile (a timeout: its clean-up edits HeadP and frees the
            // TDs) must not get the old head written back over its own -- it then walked TDs it had
            // already freed (NextTD 0xFFFFFFFF) and crashed in its completion code (fn_001ac5a4).
            const u32 toggle = ((t0 >> 25) & 1) ? ((t0 >> 24) & 1) ^ 1 : ((head >> 1) & 1) ^ 1;
            const u32 new_head = (next & ~0xFu) | (toggle << 1) | (cc != CC_NOERROR ? 1u : 0u);
            u32 expected = head;
            if (!__atomic_compare_exchange_n(reinterpret_cast<u32*>(host_ptr(ed + 8)), &expected, new_head, false,
                                             __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {
                note('X', ed, head, expected);
                static int logged = 0;
                if (logged++ < 10)
                    std::fprintf(stderr, "[ohci] %s: ED %08x changed by the driver during a transfer (head %08x -> %08x): "
                                         "transfer dropped\n", name, ed, head, expected);
                break;
            }
            head = new_head;
            note('R', ed, td, next, cc << 16 | (done & 0xFFFF), new_head);
            t0 = (t0 & 0x03FFFFFFu & ~(3u << 24)) | (cc << 28) | (2u << 24) | (toggle << 24);
            wr<u32>(td, t0);
            wr<u32>(td + 4, done >= len ? 0 : addr_at(done));
            wr<u32>(td + 8, done_head);
            done_head = td;
            worked = true;
            if (cc != CC_NOERROR) break;
        }
        return worked;
    }

    bool process_list(u32 first) {
        bool worked = false;
        for (u32 ed = first, guard = 0; ed && guard < 256; ed = rd<u32>(ed + 12) & ~0xFu, ++guard)
            worked |= process_ed(ed, false);
        return worked;
    }

    void run_frame() {
        std::lock_guard lk(m);
        if ((control & HCFS_MASK) != HCFS_OPERATIONAL) return;
        ++frame;
        if (hcca) wr<u16>(hcca + 0x80, frame);
        int_status |= INT_SF;

        if ((control & PLE) && hcca) process_list(rd<u32>(hcca + 4 * (frame % 32)) & ~0xFu);
        if ((control & CLE) && (command & CLF)) {
            if (!process_list(control_head)) command &= ~CLF;
        }
        if ((control & BLE) && (command & BLF)) {
            if (!process_list(bulk_head)) command &= ~BLF;
        }
        if (done_head && !(int_status & INT_WDH) && hcca) {
            wr<u32>(hcca + 0x84, done_head);
            note('D', hcca, done_head);
            done_head = 0;
            int_status |= INT_WDH;
        }
        update_irq();
    }

    void update_irq() {
        // Level-style: deliver once while an enabled cause is pending.
        // Level-triggered (deliveries are deduplicated by the kernel).
        const bool active = (int_enable & INT_MIE) && (int_status & int_enable & ~INT_MIE);
        if (active) kernel::ke::raise_interrupt(vector);
        irq_pending = active;
    }
};

Controller g_hc[2] = {{"usb0", 0x31}, {"usb1", 0x39}};

u32 read(void* ctx, u32 addr, unsigned size) {
    auto& h = *static_cast<Controller*>(ctx);
    std::lock_guard lk(h.m);
    const u32 off = addr & 0xFFF;
    u32 v = 0;
    switch (off & ~3u) {
        case 0x00: v = 0x10; break;  // HcRevision 1.0
        case 0x04: v = h.control; break;
        case 0x08: v = h.command; break;
        case 0x0C: v = h.int_status; break;
        case 0x10: case 0x14: v = h.int_enable; break;
        case 0x18: v = h.hcca; break;
        case 0x1C: v = 0; break;
        case 0x20: v = h.control_head; break;
        case 0x24: v = h.control_cur; break;
        case 0x28: v = h.bulk_head; break;
        case 0x2C: v = h.bulk_cur; break;
        case 0x30: v = h.done_head; break;
        case 0x34: v = h.fm_interval; break;
        case 0x38: v = 0x2000; break;
        case 0x3C: v = h.frame; break;
        case 0x40: v = h.periodic_start; break;
        case 0x44: v = h.ls_threshold; break;
        case 0x48: v = h.rh_desc_a; break;
        case 0x4C: v = h.rh_desc_b; break;
        case 0x50: v = h.rh_status; break;
        default:
            if (off >= 0x54 && off < 0x54 + 4 * kPorts) v = h.port[(off - 0x54) / 4];
            else std::fprintf(stderr, "[ohci] %s read of unknown register %#x\n", h.name, off);
    }
    if (size < 4) v >>= 8 * (off & 3);
    return v;
}

void write(void* ctx, u32 addr, u32 v, unsigned size) {
    auto& h = *static_cast<Controller*>(ctx);
    std::lock_guard lk(h.m);
    const u32 off = addr & 0xFFF;
    note('G', off, v, h.vector);
    static const bool tr = std::getenv("VC3_TRACE_USB") != nullptr;
    if (tr) std::fprintf(stderr, "[ohci] %s write %#x = %08x\n", h.name, off, v);
    if (size != 4) std::fprintf(stderr, "[ohci] %s %u-byte write to %#x\n", h.name, size, off);
    switch (off) {
        case 0x04: h.control = v; break;
        case 0x08:
            if (v & HCR) h.reset();
            h.command |= v & (CLF | BLF);
            break;
        case 0x0C: h.int_status &= ~v; h.update_irq(); break;
        case 0x10: h.int_enable |= v; h.update_irq(); break;
        case 0x14: h.int_enable &= ~v; break;
        case 0x18: h.hcca = v; break;
        case 0x1C: break;
        case 0x20: h.control_head = v; break;
        case 0x24: h.control_cur = v; break;
        case 0x28: h.bulk_head = v; break;
        case 0x2C: h.bulk_cur = v; break;
        case 0x30: h.done_head = v; break;
        case 0x34: h.fm_interval = v; break;
        case 0x40: h.periodic_start = v; break;
        case 0x44: h.ls_threshold = v; break;
        case 0x48: h.rh_desc_a = (h.rh_desc_a & 0xFF) | (v & ~0xFFu); break;
        case 0x4C: h.rh_desc_b = v; break;
        case 0x50: h.rh_status &= ~(v & 0x20000u); break;
        default:
            if (off >= 0x54 && off < 0x54 + 4 * kPorts) {
                const int p = int(off - 0x54) / 4;
                u32& s = h.port[p];
                s &= ~(v & 0x1F0000u);                       // write-1-to-clear change bits
                if ((v & 1) && (s & PES)) { s &= ~PES; s |= PESC; }  // ClearPortEnable
                if ((v & 2) && (s & CCS)) s |= PES;          // SetPortEnable
                if (v & PPS) s |= PPS;                       // SetPortPower
                if ((v & PRS) && (s & CCS)) {                // SetPortReset: completes immediately
                    if (h.device[p]) h.device[p]->reset();
                    s |= PES | PRSC;
                    h.int_status |= INT_RHSC;
                    h.update_irq();
                }
            } else {
                std::fprintf(stderr, "[ohci] %s write of unknown register %#x = %#x\n", h.name, off, v);
            }
    }
}

}  // namespace

bool overlaps_descriptors(uint32_t lo, uint32_t hi) {
    const u32 dlo = g_desc_lo.load(std::memory_order_relaxed), dhi = g_desc_hi.load(std::memory_order_relaxed);
    return dlo < dhi && lo < dhi && dlo < hi;
}
void descriptor_range(uint32_t& lo, uint32_t& hi) {
    lo = g_desc_lo.load(std::memory_order_relaxed);
    hi = g_desc_hi.load(std::memory_order_relaxed);
}

void attach(int controller, int port, usb::Device* dev) {
    auto& h = g_hc[controller];
    std::lock_guard lk(h.m);
    h.device[port] = dev;
    h.port[port] |= CCS | CSC;
    h.int_status |= INT_RHSC;
}

void init() {
    log::on_crash(dump_events);
    for (auto& h : g_hc) h.reset();
    mmio::register_range(0xFED00000, 0x1000, "usb0", read, write, &g_hc[0]);
    mmio::register_range(0xFED08000, 0x1000, "usb1", read, write, &g_hc[1]);
    std::thread([] {
        // The game's USB driver expects a frame every millisecond; a starved controller thread makes
        // its transfers time out, and the clean-up then races the controller (rare crash on Windows).
        platform::set_thread_priority(platform::Priority::kTimeCritical);
        auto next = std::chrono::steady_clock::now();
        for (;;) {
            next += std::chrono::milliseconds(1);
            platform::sleep_until(next);
            // Like real hardware, the controller runs independently of the CPU
            // (it must not need the guest lock: drivers spin on it at raised IRQL).
            for (auto& h : g_hc) h.run_frame();
        }
    }).detach();
}

}  // namespace vc3::io::ohci
