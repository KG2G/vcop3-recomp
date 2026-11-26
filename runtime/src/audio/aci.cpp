#include "aci.h"

#include <cstdio>
#include <cstdlib>
#include <mutex>

#include "kernel/ke.h"
#include "mmio.h"
#include "vc3/recomp.h"

namespace vc3::audio::aci {

namespace {

constexpr uint32_t kBase = 0xFEC00000;
constexpr uint32_t kIrqVector = 0x30 + 6;

// Status register bits.
constexpr uint16_t SR_DCH = 1, SR_CELV = 2, SR_LVBCI = 4, SR_BCIS = 8, SR_FIFOE = 16;
constexpr uint16_t SR_W1C = SR_LVBCI | SR_BCIS | SR_FIFOE;
// Control register bits.
constexpr uint8_t CR_RPBM = 1, CR_RR = 2, CR_LVBIE = 4, CR_FEIE = 8, CR_IOCE = 16;

struct Box {
    uint32_t bdbar = 0;
    uint8_t civ = 0, lvi = 0, piv = 0, cr = 0;
    uint16_t sr = SR_DCH, picb = 0;
    uint16_t bd_ctl = 0;  // control word of the current descriptor
    void reset() { *this = Box{}; }
};

// Box offsets within NABM: PCM in, PCM out, mic, S/PDIF out (nForce).
constexpr uint32_t kBoxOff[] = {0x00, 0x10, 0x20, 0x70};
constexpr int kBoxes = 4;
constexpr uint8_t kGlobStaBoxBit[] = {5, 6, 7, 24};  // PIINT, POINT, MINT, (S/PDIF)

std::mutex g_m;
Box g_box[kBoxes];
uint16_t g_nam[0x80] = {};
uint32_t g_glob_cnt = 0;
bool g_trace = false;

int box_at(uint32_t nabm_off) {
    for (int i = 0; i < kBoxes; ++i)
        if (nabm_off >= kBoxOff[i] && nabm_off < kBoxOff[i] + 0x10) return i;
    return -1;
}

bool box_irq(const Box& b) {
    return ((b.sr & SR_BCIS) && (b.cr & CR_IOCE)) || ((b.sr & SR_LVBCI) && (b.cr & CR_LVBIE)) ||
           ((b.sr & SR_FIFOE) && (b.cr & CR_FEIE));
}

uint32_t glob_sta() {
    uint32_t v = 0x300;  // primary + secondary codec ready
    for (int i = 0; i < kBoxes; ++i)
        if (box_irq(g_box[i])) v |= 1u << kGlobStaBoxBit[i];
    return v;
}

void update_irq() {
    for (const auto& b : g_box)
        if (box_irq(b)) {
            kernel::ke::raise_interrupt(kIrqVector);
            return;
        }
}

void fetch_bd(Box& b) {
    const uint32_t bd = (b.bdbar & ~7u) + 8u * b.civ;
    const uint32_t w1 = rd<uint32_t>(bd + 4);
    b.picb = uint16_t(w1);
    b.bd_ctl = uint16_t(w1 >> 16);
    b.piv = uint8_t((b.civ + 1) & 31);
}

// Run state: RPBM set and not halted (DCH clear).
bool running(const Box& b) { return (b.cr & CR_RPBM) && !(b.sr & SR_DCH); }

void start(Box& b) {
    b.sr &= uint16_t(~(SR_DCH | SR_CELV));
    fetch_bd(b);
}

// Current buffer finished: raise its status and move to the next one.
void buffer_done(Box& b) {
    if (b.bd_ctl & 0x8000) b.sr |= SR_BCIS;  // IOC
    if (b.civ == b.lvi) {
        b.sr |= SR_LVBCI | SR_CELV | SR_DCH;  // ran out of valid buffers: halt
        return;
    }
    b.civ = b.piv;
    fetch_bd(b);
}

uint32_t box_read(const Box& b, uint32_t r, unsigned size) {
    // Byte image of the 16-byte box, then extract the access.
    uint8_t img[16] = {};
    for (int i = 0; i < 4; ++i) img[i] = uint8_t(b.bdbar >> (8 * i));
    img[4] = b.civ;
    img[5] = b.lvi;
    img[6] = uint8_t(b.sr);
    img[7] = uint8_t(b.sr >> 8);
    img[8] = uint8_t(b.picb);
    img[9] = uint8_t(b.picb >> 8);
    img[10] = b.piv;
    img[11] = b.cr;
    uint32_t v = 0;
    for (unsigned i = 0; i < size && r + i < 16; ++i) v |= uint32_t(img[r + i]) << (8 * i);
    return v;
}

void box_write_byte(Box& b, uint32_t r, uint8_t v) {
    switch (r) {
        case 0: case 1: case 2: case 3: {
            const uint32_t sh = 8 * r;
            b.bdbar = (b.bdbar & ~(0xFFu << sh)) | (uint32_t(v) << sh);
            break;
        }
        case 5:  // LVI
            b.lvi = v & 31;
            // A halted engine resumes when software appends buffers.
            if ((b.cr & CR_RPBM) && (b.sr & SR_CELV)) {
                b.sr &= uint16_t(~(SR_DCH | SR_CELV));
                b.civ = b.piv;
                fetch_bd(b);
            }
            break;
        case 6: b.sr &= uint16_t(~(v & SR_W1C)); break;
        case 7: break;
        case 11:  // CR
            if (v & CR_RR) {
                b.reset();  // RR self-clears
                break;
            }
            if ((v & CR_RPBM) && !(b.cr & CR_RPBM)) {
                b.cr = v;
                start(b);
            } else {
                if (!(v & CR_RPBM)) b.sr |= SR_DCH;
                b.cr = v;
            }
            break;
        default: break;  // CIV, PICB, PIV are read-only
    }
}

uint32_t read(void*, uint32_t addr, unsigned size) {
    const uint32_t off = addr - kBase;
    std::lock_guard lk(g_m);
    uint32_t v = 0;
    if (off < 0x100) {
        const uint32_t r = (off & 0x7F) >> 1;
        v = g_nam[r] | (size == 4 ? uint32_t(g_nam[(r + 1) & 0x3F]) << 16 : 0);
        if (off & 1) v >>= 8;
    } else {
        const uint32_t n = off - 0x100;
        const int bi = box_at(n);
        if (bi >= 0) v = box_read(g_box[bi], n - kBoxOff[bi], size);
        else if ((n & ~3u) == 0x2C) v = g_glob_cnt >> (8 * (n & 3));
        else if ((n & ~3u) == 0x30) v = glob_sta() >> (8 * (n & 3));
        else if ((n & ~3u) == 0x34) v = 0;  // CAS: codec access semaphore always free
    }
    if (size < 4) v &= (1u << (8 * size)) - 1;
    if (g_trace) std::fprintf(stderr, "[aci] read  %03x/%u -> %x\n", off, size, v);
    return v;
}

void write(void*, uint32_t addr, uint32_t v, unsigned size) {
    const uint32_t off = addr - kBase;
    std::lock_guard lk(g_m);
    if (g_trace) std::fprintf(stderr, "[aci] write %03x/%u = %x\n", off, size, v);
    if (off < 0x100) {
        // Codec registers are 16-bit; a 32-bit access covers two.
        for (unsigned i = 0; i < size; i += 2) {
            const uint32_t r = ((off + i) & 0x7F) >> 1;
            const uint16_t w = uint16_t(v >> (8 * i));
            if (r == 0) {  // codec reset
                for (auto& x : g_nam) x = 0;
                continue;
            }
            g_nam[r] = w;
        }
        g_nam[0x26 >> 1] = (g_nam[0x26 >> 1] & 0xFF00) | 0x000F;  // ADC/DAC/analog/ref ready
        return;
    }
    const uint32_t n = off - 0x100;
    const int bi = box_at(n);
    if (bi >= 0) {
        Box& b = g_box[bi];
        const uint32_t r = n - kBoxOff[bi];
        for (unsigned i = 0; i < size; ++i) box_write_byte(b, r + i, uint8_t(v >> (8 * i)));
        update_irq();
    } else if ((n & ~3u) == 0x2C) {
        const uint32_t sh = 8 * (n & 3), mask = (size == 4 ? 0xFFFFFFFFu : ((1u << (8 * size)) - 1)) << sh;
        g_glob_cnt = (g_glob_cnt & ~mask) | ((v << sh) & mask);
    }
}

}  // namespace

void tick(uint32_t frames) {
    std::lock_guard lk(g_m);
    bool any = false;
    for (int i = 0; i < kBoxes; ++i) {
        Box& b = g_box[i];
        if (!running(b) || (b.bdbar & ~7u) == 0) continue;
        uint32_t words = frames * 2;  // 16-bit stereo: two words per frame
        int guard = 64;
        while (words && running(b) && guard--) {
            if (b.picb == 0) {
                buffer_done(b);
                continue;
            }
            const uint32_t take = words < b.picb ? words : b.picb;
            b.picb = uint16_t(b.picb - take);
            words -= take;
            if (b.picb == 0) buffer_done(b);
        }
        any = true;
    }
    if (any) update_irq();
}

void init() {
    g_trace = std::getenv("VC3_TRACE_ACI") != nullptr;
    g_nam[0x26 >> 1] = 0x000F;
    mmio::register_range(kBase, 0x1000, "aci", read, write);
}

}  // namespace vc3::audio::aci
