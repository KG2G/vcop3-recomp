#include "platform.h"
#include "apu.h"

#include <SDL3/SDL.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <deque>
#include <filesystem>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "audio/aci.h"
#include "kernel/ke.h"
#include "mmio.h"
#include "vc3/recomp.h"

namespace vc3::audio::apu {

namespace {

using Clock = std::chrono::steady_clock;

constexpr uint32_t kBase = 0xFE800000, kSize = 0x80000;
constexpr uint32_t kIrqVector = 0x30 + 5;
constexpr int kFrame = 32;          // samples per VP frame (48 kHz)
constexpr int kFramesPerChunk = 8;  // processed and output together (256 samples, 5.33 ms)
constexpr int kMixbins = 32;
constexpr int kMaxVoices = 256, kMax3dVoices = 64;
constexpr uint16_t kNoVoice = 0xFFFF;

// ---- registers (byte offsets in the APU BAR)
enum : uint32_t {
    ISTS = 0x1000, IEN = 0x1004,
    FECTL = 0x1100, FECV = 0x1110, FEAV = 0x1118, FENADDR = 0x115C,
    FEDECMETH = 0x1300, FEDECPARAM = 0x1304, FEMEMADDR = 0x1324, FEMEMDATA = 0x1334,
    FETFORCE1 = 0x1504,
    SECTL = 0x2000, XGSCNT = 0x200C,
    VPVADDR = 0x202C, VPSGEADDR = 0x2030, VPSSLADDR = 0x2034, GPSADDR = 0x2040,
    TVL2D = 0x2054, CVL2D = 0x2058, NVL2D = 0x205C,
};
constexpr uint32_t ISTS_GINT = 1, ISTS_FETINT = 1 << 4, ISTS_FENINT = 1 << 5, ISTS_FEVINT = 1 << 6;
constexpr uint32_t FECTL_METHMODE = 0xE0, FECTL_HALTED = 0x80, FECTL_TRAPPED = 0xE0;
constexpr uint32_t FECTL_TRAPREASON = 0xF00, FECTL_TRAP_REQUESTED = 0xF00;
constexpr uint32_t FETFORCE1_IDLE_VOICE = 1 << 15;
constexpr uint32_t SE2FE_IDLE_VOICE = 0x8000;

// ---- voice structure (0x80 bytes each at VPVADDR)
enum : uint32_t {
    V_VBIN = 0x00, V_FMT = 0x04, V_ENV0 = 0x08, V_ENVA = 0x0C, V_ENV1 = 0x10, V_ENVF = 0x14,
    V_MISC = 0x18, V_HRTF = 0x1C, V_BA = 0x20, V_LBO = 0x24, V_ECNT = 0x34, V_STATE = 0x54,
    V_OFFSET = 0x58, V_NEXT = 0x5C, V_VOLA = 0x60, V_VOLB = 0x64, V_VOLC = 0x68, V_LFO_ENV = 0x6C,
    V_FCA = 0x74, V_FCB = 0x78, V_PITCH_LINK = 0x7C,
};
// V_FMT fields
constexpr uint32_t FMT_MULTIPASS = 1 << 21, FMT_PERSIST = 1 << 23, FMT_STREAM = 1 << 24,
                   FMT_LOOP = 1 << 25, FMT_CLEAR_MIX = 1 << 26, FMT_STEREO = 1 << 27;
// V_STATE fields
constexpr uint32_t ST_PAUSED = 1 << 18, ST_NEW = 1 << 20, ST_ACTIVE = 1 << 21;
enum Env { E_OFF, E_DELAY, E_ATTACK, E_HOLD, E_DECAY, E_SUSTAIN, E_RELEASE, E_FORCE_RELEASE };

// Notifiers: 16 bytes each at FENADDR, 4 per voice after 2 global ones.
constexpr int kNotifierSslaDone = 0;

struct VoiceHost {
    // Linear interpolation between source samples a (position 0) and b (position 1).
    float a[2] = {}, b[2] = {};
    double frac = 0;
    bool primed = false;
    // Stream voices: the two segment lists and the play position in them.
    uint32_t ssl_base[2] = {};
    uint8_t ssl_count[2] = {};
    int ssl_index = 0, ssl_seg = 0;
    // Decoded ADPCM block cache.
    uint32_t adpcm_key = ~0u;
    int16_t adpcm[65 * 2] = {};
    bool locked = false;
};

std::mutex g_m;  // registers, voice memory updates, VP state
std::condition_variable g_cv;
std::vector<uint32_t> g_regs(0x20000 / 4);
std::unordered_map<uint32_t, uint32_t> g_other;  // VP/GP/EP windows (DSP memories etc.)
VoiceHost g_vh[kMaxVoices];
uint32_t g_ssl_base_page = 0;
uint8_t g_submix_headroom[kMixbins] = {};
uint8_t g_hrtf_headroom = 0;
uint8_t g_hrtf_submix[4] = {};
uint32_t g_sge_handle = 0;
bool g_trace = false;
std::set<uint32_t> g_seen;
struct { uint32_t frames, idle_frames, voices, traps, fetches, locked_skips, underruns, lock_waits, idle_trapped, idle_halted; } g_stat{};
bool g_mute[kMaxVoices] = {};  // VC3_APU_MUTE=v,v,... / VC3_APU_SOLO (debugging)
double g_bin_energy[kMixbins] = {};  // per-mixbin signal energy since the last status line
// VC3_TRACE_APU: how long front-end traps stay open (raised -> DSOUND resumes the FE), in ns.
std::atomic<int64_t> g_trap_raised_ns{0}, g_trap_open_sum_ns{0}, g_trap_open_max_ns{0}, g_traps_closed{0};
int64_t now_ns() { return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count(); }

const Clock::time_point g_epoch = Clock::now();

uint32_t& reg(uint32_t off) { return g_regs[(off & 0x1FFFF) >> 2]; }

// ---------------------------------------------------------------- interrupts
void update_irq() {
    uint32_t& ists = reg(ISTS);
    if ((reg(FECTL) & FECTL_METHMODE) == FECTL_TRAPPED) ists |= ISTS_FETINT;
    const uint32_t ien = reg(IEN);
    if ((ien & ISTS_GINT) && (ists & ~ISTS_GINT & ien)) {
        ists |= ISTS_GINT;
        kernel::ke::raise_interrupt(kIrqVector);
    } else {
        ists &= ~ISTS_GINT;
    }
}

// ---------------------------------------------------------------- voice memory
uint32_t vaddr(uint16_t v, uint32_t off) { return (reg(VPVADDR) + v * 0x80u + off) & 0x7FFFFFF; }
uint32_t vget(uint16_t v, uint32_t off) { return rd<uint32_t>(vaddr(v, off)); }
void vput(uint16_t v, uint32_t off, uint32_t val) { wr<uint32_t>(vaddr(v, off), val); }
uint32_t vfield(uint16_t v, uint32_t off, uint32_t mask) {
    return (vget(v, off) & mask) >> __builtin_ctz(mask);
}
void vset(uint16_t v, uint32_t off, uint32_t mask, uint32_t val) {
    vput(v, off, (vget(v, off) & ~mask) | ((val << __builtin_ctz(mask)) & mask));
}
uint16_t next_voice(uint16_t v) { return uint16_t(vfield(v, V_PITCH_LINK, 0xFFFF)); }

void notify(uint16_t v, int notifier) {
    const uint32_t addr = (reg(FENADDR) + 16u * (2 + v * 4u + notifier) + 15) & 0x7FFFFFF;
    wr<uint8_t>(addr, 0x01);      // NV1BA0_NOTIFICATION_STATUS_DONE_SUCCESS
    wr<uint8_t>(addr - 1, 0x01);  // envelope state (not tracked)
    reg(ISTS) |= ISTS_FEVINT | ISTS_FENINT;
    update_irq();
}

// A voice starting: its notifiers read "in progress" again. Without this a reused voice
// still showed the "done" its previous sound left, so DSOUND took a buffer it had just
// started for already finished: the game then started it again on a fresh voice every
// frame (the subway train: thousands of restarts, copies never stopped piling up into a
// repeating noise, gunshots and lines left without a voice).
void notifiers_in_progress(uint16_t v) {
    if (!reg(FENADDR)) return;
    for (int n = 0; n < 2; ++n) {
        const uint32_t addr = (reg(FENADDR) + 16u * (2 + v * 4u + n) + 15) & 0x7FFFFFF;
        wr<uint8_t>(addr, 0x80);  // NV1BA0_NOTIFICATION_STATUS_IN_PROGRESS
    }
}

void voice_off(uint16_t v) {
    vset(v, V_STATE, ST_ACTIVE, 0);
    int n = kNotifierSslaDone;
    if (vget(v, V_FMT) & FMT_STREAM) n += g_vh[v].ssl_index & 1;
    notify(v, n);
}

// VC3_TRACE_APU: why a voice stopped (with its buffer, to match the VOICE_ON line).
void trace_stop(uint16_t v, const char* why) {
    if (!g_trace) return;
    std::fprintf(stderr, "[apu] voice %u stop: %s (ba %06x cbo %06x, state %08x)\n", v, why, vget(v, V_BA) & 0xFFFFFF,
                 vget(v, V_OFFSET) & 0xFFFFFF, vget(v, V_STATE));
}

void reset_host(uint16_t v) {
    auto& h = g_vh[v];
    h.primed = false;
    h.frac = 0;
    h.ssl_index = h.ssl_seg = 0;
    h.adpcm_key = ~0u;
}

uint32_t sge_addr(uint32_t linear);

// ---------------------------------------------------------------- FE methods
void voice_on(uint32_t arg) {
    const uint16_t v = uint16_t(arg);
    if (v >= kMaxVoices) return;
    if (g_trace && (vget(v, V_STATE) & ST_ACTIVE))
        std::fprintf(stderr, "[apu] voice %u started again while still playing\n", v);
    const uint32_t list = (reg(FEAV) >> 16) & 3;
    if (list != 0) {  // insert at the top of the 2D / 3D / MP list
        uint32_t& top = reg(TVL2D + 12 * (list - 1));
        vset(v, V_PITCH_LINK, 0xFFFF, top & 0xFFFF);
        top = v;
    } else {          // insert after the antecedent voice
        const uint16_t ante = uint16_t(reg(FEAV));
        if (ante < kMaxVoices) {
            vset(v, V_PITCH_LINK, 0xFFFF, next_voice(ante));
            vset(ante, V_PITCH_LINK, 0xFFFF, v);
        }
    }
    vset(v, V_OFFSET, 0xFFFFFF, 0);  // CBO
    reset_host(v);
    notifiers_in_progress(v);
    // Envelope start states and their initial counts.
    const uint32_t ea = arg >> 28, ef = (arg >> 24) & 0xF;
    vset(v, V_STATE, 0xF0000000, ea);
    vset(v, V_STATE, 0x0F000000, ef);
    auto start_count = [&](uint32_t st, uint32_t reg0, uint32_t rega) -> uint32_t {
        if (st == E_DELAY) return vfield(v, reg0, 0xFFF000) * 16;
        if (st == E_HOLD) return vfield(v, rega, 0xFFF000) * 16;
        return 0;
    };
    vset(v, V_ECNT, 0x0000FFFF, start_count(ea, V_ENV0, V_ENVA));
    vset(v, V_ECNT, 0xFFFF0000, start_count(ef, V_ENV1, V_ENVF));
    vset(v, V_STATE, ST_ACTIVE | ST_NEW, 3);
    vset(v, V_STATE, ST_PAUSED, 0);
    if (g_trace) {
        const uint32_t fmt = vget(v, V_FMT);
        std::fprintf(stderr,
                     "[apu] voice %u on list %u: fmt %08x (%s %s cs%u ss%u spb%u%s%s) ba %06x lbo %06x ebo %06x "
                     "pitch %d vbin %08x vol %08x %08x %08x ssl %x/%u %x/%u env %x misc %08x fca %08x fcb %08x\n",
                     v, list, fmt, (fmt & FMT_STREAM) ? "stream" : "buffer", (fmt & FMT_STEREO) ? "stereo" : "mono",
                     fmt >> 30, (fmt >> 28) & 3, ((fmt >> 16) & 31) + 1, (fmt & FMT_LOOP) ? " loop" : "",
                     (fmt & FMT_MULTIPASS) ? " multipass" : "", vget(v, V_BA) & 0xFFFFFF, vget(v, V_LBO) & 0xFFFFFF,
                     vget(v, V_NEXT) & 0xFFFFFF, int16_t(vget(v, V_PITCH_LINK) >> 16), vget(v, V_VBIN),
                     vget(v, V_VOLA), vget(v, V_VOLB), vget(v, V_VOLC), g_vh[v].ssl_base[0], g_vh[v].ssl_count[0],
                     g_vh[v].ssl_base[1], g_vh[v].ssl_count[1], arg >> 24, vget(v, V_MISC), vget(v, V_FCA), vget(v, V_FCB));
        if (!(fmt & FMT_STREAM) && (fmt >> 30) == 0) {  // 8-bit: raw bytes (signedness check)
            const uint32_t ba = vget(v, V_BA) & 0xFFFFFF;
            std::fprintf(stderr, "[apu]   8-bit data:");
            for (int i = 0; i < 48; ++i) std::fprintf(stderr, " %02x", rd<uint8_t>(sge_addr(ba + 64 + i)));
            const uint32_t n = std::min<uint32_t>(8192, vget(v, V_NEXT) & 0xFFFFFF);
            double du = 0, ds = 0, mu = 0;
            for (uint32_t i = 1; i < n; ++i) {
                const int a = rd<uint8_t>(sge_addr(ba + i - 1)), b = rd<uint8_t>(sge_addr(ba + i));
                du += std::abs(b - a);
                ds += std::abs(int(int8_t(b)) - int(int8_t(a)));
                mu += b;
            }
            if (n > 1)
                std::fprintf(stderr, " | %u bytes: mean %.1f, mean step unsigned %.1f signed %.1f\n", n, mu / (n - 1), du / (n - 1), ds / (n - 1));
            else std::fprintf(stderr, "\n");
        }
        if (!(fmt & FMT_STREAM) && (fmt >> 30) == 1) {
            const uint32_t ba = vget(v, V_BA) & 0xFFFFFF;
            std::fprintf(stderr, "[apu]   sge[%u]=%08x data:", ba >> 12,
                         rd<uint32_t>((reg(VPSGEADDR) + (ba >> 12) * 8) & 0x7FFFFFF));
            for (int i = 0; i < 24; ++i) std::fprintf(stderr, " %d", rd<int16_t>(sge_addr(ba + i * 2)));
            std::fprintf(stderr, "\n");
        }
    }
}

void voice_release(uint16_t v) {
    if (v >= kMaxVoices) return;
    vset(v, V_ECNT, 0x0000FFFF, vfield(v, V_LFO_ENV, 0xFFF) * 16);
    vset(v, V_STATE, 0xF0000000, E_RELEASE);
    vset(v, V_ECNT, 0xFFFF0000, vfield(v, V_MISC, 0xFFF) * 16);
    vset(v, V_STATE, 0x0F000000, E_RELEASE);
}

struct FieldMethod { uint32_t method, off, mask; };
// Methods that store (part of) their argument into a field of the current voice.
constexpr FieldMethod kFieldMethods[] = {
    {0x300, V_VBIN, ~0u},   {0x304, V_FMT, ~0u},    {0x308, V_ENV0, ~0u},
    {0x30C, V_ENVA, ~0u},   {0x310, V_ENV1, ~0u},   {0x314, V_ENVF, ~0u},
    {0x318, V_MISC, ~0u},   {0x31C, V_HRTF, 0xFFFF}, {0x360, V_VOLA, ~0u},
    {0x364, V_VOLB, ~0u},   {0x368, V_VOLC, ~0u},   {0x36C, V_LFO_ENV, ~0u},
    {0x374, V_FCA, ~0u},    {0x378, V_FCB, ~0u},    {0x37C, V_PITCH_LINK, 0xFFFF0000},
    {0x3A0, V_BA, 0xFFFFFF},
    {0x3A4, V_LBO, 0xFFFFFF}, {0x3D8, V_OFFSET, 0xFFFFFF}, {0x3DC, V_NEXT, 0xFFFFFF},
};

int g_trace_voice = -1;  // VC3_TRACE_APU_VOICE=v: log every method touching voice v

// VC3_TRACE_APU: the last APU accesses (FE methods, register reads and writes, idle-voice
// traps), dumped when a looping voice is released before it has played anything -- to see
// what the game did and read in between.
struct TraceEvent { char kind; uint32_t a, b, c; };
TraceEvent g_ring[64];
unsigned g_ring_pos = 0;
void ring(char kind, uint32_t a, uint32_t b, uint32_t c = 0) {
    if (!g_trace) return;
    g_ring[g_ring_pos++ % 64] = {kind, a, b, c};
}
void dump_ring(const char* why) {
    static int dumps = 0;
    if (!g_trace || ++dumps > 20) return;
    std::fprintf(stderr, "[apu] --- last APU accesses before %s (oldest first):\n", why);
    for (unsigned i = 0; i < 64; ++i) {
        const TraceEvent& e = g_ring[(g_ring_pos + i) % 64];
        if (!e.kind) continue;
        switch (e.kind) {
            case 'M': std::fprintf(stderr, "[apu]   method %#05x arg %#010x (current voice %u)\n", e.a, e.b, e.c); break;
            case 'R': std::fprintf(stderr, "[apu]   read  %#07x -> %#010x\n", e.a, e.b); break;
            case 'W': std::fprintf(stderr, "[apu]   write %#07x = %#010x\n", e.a, e.b); break;
            case 'T': std::fprintf(stderr, "[apu]   idle-voice trap raised for voice %u\n", e.a); break;
            case 'I': std::fprintf(stderr, "[apu]   voice %u went idle (%s)\n", e.a, e.b ? "envelope" : "other"); break;
        }
    }
    std::fprintf(stderr, "[apu] ---\n");
}

void fe_method(uint32_t m, uint32_t arg) {
    ring('M', m, arg, reg(FECV));
    if (g_trace_voice >= 0 &&
        ((m != 0x2F8 && reg(FECV) == uint32_t(g_trace_voice)) || (m >= 0x120 && m <= 0x140 && (arg & 0xFFFF) == uint32_t(g_trace_voice)) ||
         (m == 0x2F8 && arg == uint32_t(g_trace_voice)))) {
        static const auto t0 = Clock::now();
        std::fprintf(stderr, "[apu] %.3f method %#x arg %#x (cbo %#x)\n",
                     std::chrono::duration<double>(Clock::now() - t0).count(), m, arg,
                     reg(VPVADDR) ? vget(uint16_t(g_trace_voice), V_OFFSET) & 0xFFFFFF : 0);
    }
    reg(FEDECMETH) = m;
    reg(FEDECPARAM) = arg;
    const uint16_t cv = uint16_t(reg(FECV));
    for (const auto& f : kFieldMethods)
        if (f.method == m) {
            if (cv < kMaxVoices) vput(cv, f.off, (vget(cv, f.off) & ~f.mask) | (arg & f.mask));
            return;
        }
    switch (m) {
        case 0x120: reg(FEAV) = arg; break;                     // SET_ANTECEDENT_VOICE
        case 0x124: voice_on(arg); break;                       // VOICE_ON
        case 0x128:                                             // VOICE_OFF
            if ((arg & 0xFFFF) < kMaxVoices) {
                trace_stop(uint16_t(arg), "VOICE_OFF from the game");
                voice_off(uint16_t(arg));
            }
            break;
        case 0x12C:                                             // VOICE_RELEASE
            if ((arg & 0xFFFF) < kMaxVoices) {
                trace_stop(uint16_t(arg), "VOICE_RELEASE from the game");
                const uint16_t rv = uint16_t(arg);
                if (g_trace && reg(VPVADDR) && (vget(rv, V_FMT) & FMT_LOOP) && (vget(rv, V_OFFSET) & 0xFFFFFF) == 0)
                    dump_ring("a looping voice was released before playing");
            }
            voice_release(uint16_t(arg));
            break;
        case 0x140:                                             // VOICE_PAUSE
            if ((arg & 0xFFFF) < kMaxVoices) vset(uint16_t(arg), V_STATE, ST_PAUSED, (arg >> 18) & 1);
            break;
        case 0x160: break;                                      // SET_CURRENT_HRTF_ENTRY
        case 0x190: g_ssl_base_page = arg; break;               // SET_CURRENT_SSL
        case 0x280: g_hrtf_headroom = arg & 7; break;
        case 0x2C0:
            for (int i = 0; i < 4; ++i) g_hrtf_submix[i] = (arg >> (8 * i)) & 0x1F;
            break;
        case 0x2F8: reg(FECV) = arg; break;                     // SET_CURRENT_VOICE
        case 0x2FC:                                             // VOICE_LOCK
            if (cv < kMaxVoices) g_vh[cv].locked = arg & 1;
            break;
        case 0x320: case 0x35C:                                 // SET_VOICE_SSL_A / _B
            if (cv < kMaxVoices) {
                const int i = m == 0x320 ? 0 : 1;
                g_vh[cv].ssl_base[i] = arg >> 8;
                g_vh[cv].ssl_count[i] = uint8_t(arg);
            }
            break;
        case 0x804: g_sge_handle = arg; break;                  // SET_CURRENT_INBUF_SGE
        case 0x808:                                             // SET_CURRENT_INBUF_SGE_OFFSET
            wr<uint32_t>((reg(VPSGEADDR) + g_sge_handle * 8) & 0x7FFFFFF, arg & 0xFFFFF000);
            break;
        case 0x1800: g_sge_handle = arg; break;                 // SET_CURRENT_OUTBUF_SGE
        case 0x1808:
            wr<uint32_t>((reg(VPSGEADDR) + g_sge_handle * 8) & 0x7FFFFFF, arg & 0xFFFFF000);
            break;
        default:
            if (m >= 0x200 && m < 0x280) {                      // SET_SUBMIX_HEADROOM[bin]
                g_submix_headroom[(m - 0x200) / 4] = arg & 7;
            } else if (m >= 0x600 && m < 0x800) {               // SSL segment offset/length pairs
                wr<uint32_t>((reg(VPSSLADDR) + g_ssl_base_page * 8 + (m - 0x600)) & 0x7FFFFFF, arg);
            } else if (g_trace && g_seen.insert(0x80000000 | m).second) {
                std::fprintf(stderr, "[apu] unhandled FE method %#x (%#x)\n", m, arg);
            }
            break;
    }
}

// ---------------------------------------------------------------- envelopes
// One step per VP frame. Returns the envelope value 0..1 and updates the
// voice's state, counter and level fields in place.
struct EnvRegs { uint32_t reg0, rega, rr_off, lvl_off, cnt_mask, cur_mask; };
constexpr EnvRegs kEnvA{V_ENV0, V_ENVA, V_LFO_ENV, V_OFFSET, 0x0000FFFF, 0xF0000000};
constexpr EnvRegs kEnvF{V_ENV1, V_ENVF, V_MISC, V_NEXT, 0xFFFF0000, 0x0F000000};

float step_envelope(uint16_t v, const EnvRegs& e) {
    uint32_t cur = vfield(v, V_STATE, e.cur_mask);
    uint32_t count = vfield(v, V_ECNT, e.cnt_mask);
    const uint32_t lvl_mask = 0xFF000000;
    auto set_state = [&](uint32_t s) { vset(v, V_STATE, e.cur_mask, s); };
    auto set_count = [&](uint32_t c) { vset(v, V_ECNT, e.cnt_mask, c); };
    auto set_level = [&](uint32_t l) { vset(v, e.lvl_off, lvl_mask, l); };
    switch (cur) {
        case E_OFF:
            set_count(0);
            set_level(0xFF);
            return 1.0f;
        case E_DELAY:
            set_level(0);
            if (count == 0) set_state(E_ATTACK);
            else set_count(count - 1);
            return 0.0f;
        case E_ATTACK: {
            const uint32_t rate = vfield(v, e.reg0, 0xFFF) * 16;
            const float value = rate == 0 || count >= rate ? 255.0f : float(count * 255 / rate);
            set_level(uint32_t(value));
            if (count >= rate) {
                set_state(E_HOLD);
                set_count(vfield(v, e.rega, 0xFFF000) * 16);
            } else {
                set_count(count + 1);
            }
            return value / 255.0f;
        }
        case E_HOLD:
            set_level(0xFF);
            if (count == 0) {
                set_state(E_DECAY);
                set_count(vfield(v, e.rega, 0xFFF) * 16);
            } else {
                set_count(count - 1);
            }
            return 1.0f;
        case E_DECAY: {
            const uint32_t rate = vfield(v, e.rega, 0xFFF);
            const uint32_t sustain = vfield(v, e.rega, 0xFF000000);
            // Exponential fall from full scale towards the sustain level.
            const float value = rate == 0 ? 0.0f
                : 255.0f * std::pow(0.99988799f, float(rate * 16 - std::min(count, rate * 16)) * 4096.0f / float(rate));
            if (value <= sustain + 0.2f || value > 255.0f) {
                set_state(E_SUSTAIN);
                return sustain / 255.0f;
            }
            set_count(count ? count - 1 : 0);
            set_level(uint32_t(value));
            return value / 255.0f;
        }
        case E_SUSTAIN: {
            const uint32_t sustain = vfield(v, e.rega, 0xFF000000);
            set_count(0);
            set_level(sustain);
            return sustain / 255.0f;
        }
        case E_RELEASE: {
            const uint32_t rate = vfield(v, e.rr_off, 0xFFF);
            if (rate == 0) count = 0;
            if (count == 0) {
                set_state(E_FORCE_RELEASE);
                return 0.0f;
            }
            // Decay of the current level to -60 dB over the release time.
            const float pos = std::clamp(1.0f - float(count) / float(rate * 16), 0.0f, 1.0f);
            const float value = std::exp(-6.91f * pos) * float(vfield(v, e.lvl_off, lvl_mask));
            set_count(count - 1);
            return value / 255.0f;
        }
        case E_FORCE_RELEASE:
            if (e.cnt_mask == kEnvA.cnt_mask) {
                if (vget(v, V_STATE) & ST_ACTIVE) trace_stop(v, "amplitude envelope ended");
                voice_off(v);
            }
            return 0.0f;
        default:
            return 0.0f;
    }
}

// ---------------------------------------------------------------- sample fetch
float s16(int16_t x) { return x / 32768.0f; }

// Xbox ADPCM (IMA): per channel a 4-byte header (initial sample, step index) and
// 32 bytes = 64 nibbles; channels interleave in 4-byte words. 64 samples/block.
void adpcm_decode(const uint8_t* in, int channels, int16_t* out) {
    static const int16_t kStep[89] = {
        7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31, 34, 37, 41, 45, 50, 55, 60, 66,
        73, 80, 88, 97, 107, 118, 130, 143, 157, 173, 190, 209, 230, 253, 279, 307, 337, 371, 408,
        449, 494, 544, 598, 658, 724, 796, 876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066,
        2272, 2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358, 5894, 6484, 7132, 7845, 8630,
        9493, 10442, 11487, 12635, 13899, 15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767};
    static const int8_t kIndex[8] = {-1, -1, -1, -1, 2, 4, 6, 8};
    int pred[2], idx[2];
    for (int ch = 0; ch < channels; ++ch) {
        pred[ch] = int16_t(in[0] | in[1] << 8);
        idx[ch] = std::clamp<int>(in[2], 0, 88);
        out[ch] = int16_t(pred[ch]);
        in += 4;
    }
    // 8 chunks of 4 bytes (8 samples) per channel.
    for (int chunk = 0; chunk < 8; ++chunk) {
        for (int ch = 0; ch < channels; ++ch) {
            for (int i = 0; i < 8; ++i) {
                const int nib = (in[i / 2] >> (4 * (i & 1))) & 0xF;
                const int step = kStep[idx[ch]];
                int delta = step >> 3;
                if (nib & 1) delta += step >> 2;
                if (nib & 2) delta += step >> 1;
                if (nib & 4) delta += step;
                if (nib & 8) delta = -delta;
                pred[ch] = std::clamp(pred[ch] + delta, -32768, 32767);
                idx[ch] = std::clamp(idx[ch] + kIndex[nib & 7], 0, 88);
                out[(1 + chunk * 8 + i) * channels + ch] = int16_t(pred[ch]);
            }
            in += 4;
        }
    }
}

// Buffer voices address their data through the scatter-gather table (4 KiB pages).
uint32_t sge_addr(uint32_t linear) {
    const uint32_t page = rd<uint32_t>((reg(VPSGEADDR) + (linear >> 12) * 8) & 0x7FFFFFF);
    return (page + (linear & 0xFFF)) & 0x7FFFFFF;
}

// Read the voice's next source sample into out[0..1] and advance its position.
// Returns false when the voice has no data (ended, stopped or starved).
bool fetch(uint16_t v, float out[2]) {
    VoiceHost& h = g_vh[v];
    const uint32_t fmt = vget(v, V_FMT);
    if (!(vget(v, V_STATE) & ST_ACTIVE)) return false;
    const bool stream = fmt & FMT_STREAM;
    bool stereo = fmt & FMT_STEREO;
    uint32_t cs = fmt >> 30;               // container: B8, B16, ADPCM, B32
    uint32_t ss = (fmt >> 28) & 3;         // sample size: U8, S16, S24, S32
    uint32_t spb = ((fmt >> 16) & 0x1F) + 1;
    uint32_t ebo, base;
    uint32_t cbo = vget(v, V_OFFSET) & 0xFFFFFF;

    if (stream) {
        const int idx = h.ssl_index & 1;
        const int count = h.ssl_count[idx];
        if (count == 0) {  // nothing queued
            vset(v, V_OFFSET, 0xFFFFFF, 0);
            h.ssl_seg = 0;
            if (!(fmt & FMT_PERSIST)) {
                h.ssl_index = 0;
                voice_off(v);
            }
            return false;
        }
        const uint32_t page = h.ssl_base[idx] + h.ssl_seg;
        const uint32_t ent = (reg(VPSSLADDR) + page * 8) & 0x7FFFFFF;
        base = rd<uint32_t>(ent) & 0x7FFFFFF;
        const uint32_t len = rd<uint32_t>(ent + 4);
        if ((len & 0xFFFF) == 0) return false;
        ebo = (len & 0xFFFF) - 1;
        cs = (len >> 16) & 3;
        spb = ((len >> 18) & 0x1F) + 1;
        stereo = (len >> 23) & 1;
        if (cs == 2) ss = 2;
    } else {
        ebo = vget(v, V_NEXT) & 0xFFFFFF;
        base = vget(v, V_BA) & 0xFFFFFF;
        if (cbo > ebo) {
            if (fmt & FMT_LOOP) cbo = vget(v, V_LBO) & 0xFFFFFF;
            else {
                voice_off(v);
                return false;
            }
        }
    }
    const int ch = stereo ? 2 : 1;

    if (cs == 2) {  // ADPCM
        const uint32_t block_bytes = 36 * spb;
        const uint32_t block = cbo / 64, pos = cbo % 64;
        const uint32_t key = stream ? base + block * block_bytes : 0x80000000u | (base + block * block_bytes);
        if (h.adpcm_key != key) {
            uint8_t raw[36 * 2];
            const uint32_t n = std::min<uint32_t>(block_bytes, sizeof raw);
            for (uint32_t i = 0; i < n; i += 4) {
                const uint32_t a = stream ? base + block * block_bytes + i : sge_addr(base + block * block_bytes + i);
                const uint32_t w = rd<uint32_t>(a & 0x7FFFFFF);
                std::memcpy(raw + i, &w, 4);
            }
            adpcm_decode(raw, ch, h.adpcm);
            h.adpcm_key = key;
        }
        out[0] = s16(h.adpcm[pos * ch]);
        out[1] = s16(h.adpcm[pos * ch + (ch - 1)]);
    } else {
        static const uint32_t kContainer[4] = {1, 2, 0, 4};
        const uint32_t csize = kContainer[cs];
        const uint32_t lin = base + cbo * csize * spb;
        for (int c = 0; c < ch; ++c) {
            const uint32_t a = stream ? (lin + c * csize) & 0x7FFFFFF : sge_addr(lin + c * csize);
            float f;
            switch (ss) {
                case 0: f = (int(rd<uint8_t>(a)) - 128) / 128.0f; break;
                case 1: f = s16(rd<int16_t>(a)); break;
                case 2: f = float(int32_t(rd<uint32_t>(a) << 8)) / 2147483648.0f; break;
                default: f = float(int32_t(rd<uint32_t>(a))) / 2147483648.0f; break;
            }
            out[c] = f;
        }
        if (ch == 1) out[1] = out[0];
    }

    // Advance.
    ++cbo;
    if (cbo > ebo) {
        if (stream) {
            cbo = 0;
            if (++h.ssl_seg >= h.ssl_count[h.ssl_index & 1]) {
                const int done = h.ssl_index & 1;
                h.ssl_index = done ^ 1;
                h.ssl_seg = 0;
                notify(v, kNotifierSslaDone + done);
            }
        } else if (fmt & FMT_LOOP) {
            cbo = vget(v, V_LBO) & 0xFFFFFF;
        } else {
            vset(v, V_OFFSET, 0xFFFFFF, ebo);
            trace_stop(v, "end of a one-shot buffer");
            voice_off(v);
            return true;  // this last sample still plays
        }
    }
    vset(v, V_OFFSET, 0xFFFFFF, cbo);
    return true;
}

// ---------------------------------------------------------------- voice processing
float attenuate(uint32_t vol) {
    vol &= 0xFFF;
    return vol == 0xFFF ? 0.0f : std::pow(10.0f, float(vol) / (64.0f * -20.0f));
}

void process_voice(uint16_t v, float (*mix)[kFrame]) {
    const uint32_t state = vget(v, V_STATE);
    if (state & ST_PAUSED) return;
    vset(v, V_STATE, ST_NEW, 0);
    const uint32_t fmt = vget(v, V_FMT);

    // Non-persistent streams only run once their envelope has been released.
    if ((fmt & FMT_STREAM) && !(fmt & FMT_PERSIST) && vfield(v, V_STATE, 0xF0000000) < E_RELEASE) {
        voice_off(v);
        return;
    }

    const float ef = step_envelope(v, kEnvF);
    const int16_t pitch = int16_t(vget(v, V_PITCH_LINK) >> 16);
    const int8_t pscale = int8_t(vget(v, V_ENV0) >> 24);
    const double step = std::exp2((pitch + pscale * 32 * ef) / 4096.0);
    const float ea = step_envelope(v, kEnvA);
    if (!(vget(v, V_STATE) & ST_ACTIVE)) return;

    float smp[kFrame][2] = {};
    const bool stereo = fmt & FMT_STEREO;
    if (fmt & FMT_MULTIPASS) {
        const int bin = (fmt >> 16) & 0x1F;
        for (int i = 0; i < kFrame; ++i) smp[i][0] = smp[i][1] = mix[bin][i];
        if (fmt & FMT_CLEAR_MIX) std::memset(mix[bin], 0, sizeof mix[bin]);
    } else {
        VoiceHost& h = g_vh[v];
        bool live = true;
        if (!h.primed) {
            live = fetch(v, h.a);
            if (live && !fetch(v, h.b)) h.b[0] = h.b[1] = 0;
            h.frac = 0;
            h.primed = true;
        }
        for (int i = 0; i < kFrame && live; ++i) {
            const float t = float(h.frac);
            smp[i][0] = h.a[0] + (h.b[0] - h.a[0]) * t;
            smp[i][1] = h.a[1] + (h.b[1] - h.a[1]) * t;
            h.frac += step;
            while (h.frac >= 1.0) {
                h.frac -= 1.0;
                h.a[0] = h.b[0];
                h.a[1] = h.b[1];
                if (!fetch(v, h.b)) {
                    h.b[0] = h.b[1] = 0;
                    if (!(vget(v, V_STATE) & ST_ACTIVE)) live = false;
                    break;
                }
            }
        }
    }

    // Destination mixbins and volumes (12-bit attenuation in 1/64 dB).
    const uint32_t vbin = vget(v, V_VBIN), vola = vget(v, V_VOLA), volb = vget(v, V_VOLB), volc = vget(v, V_VOLC);
    int bin[8] = {int(vbin & 31), int((vbin >> 5) & 31), int((vbin >> 10) & 31), int((vbin >> 16) & 31),
                  int((vbin >> 21) & 31), int((vbin >> 26) & 31), int(fmt & 31), int((fmt >> 5) & 31)};
    const uint32_t vol[8] = {
        (vola >> 4) & 0xFFF, vola >> 20, (volb >> 4) & 0xFFF, volb >> 20, (volc >> 4) & 0xFFF, volc >> 20,
        ((volc & 0xF) << 8) | ((volb & 0xF) << 4) | (vola & 0xF),
        (((volc >> 16) & 0xF) << 8) | (((volb >> 16) & 0xF) << 4) | ((vola >> 16) & 0xF)};
    const bool is3d = v < kMax3dVoices;
    if (is3d)
        for (int i = 0; i < 4; ++i) bin[i] = g_hrtf_submix[i];
    const int ch = stereo ? 2 : 1;
    for (int b = 0; b < 8; ++b) {
        const float att = attenuate(vol[b]);
        if (att == 0.0f) continue;
        const float hr = float(1 << ((is3d && b < 4) ? g_hrtf_headroom : g_submix_headroom[bin[b]]));
        const float g = ea * att / hr;
        float* dst = mix[bin[b]];
        for (int i = 0; i < kFrame; ++i) dst[i] += g * smp[i][b % ch];
    }
}

// ---------------------------------------------------------------- reverb
// DSOUND's effects program on the GP DSP isn't emulated. The only effect VC3 feeds
// is the I3DL2 room reverb (mixbin 10, the I3DL2 send), so that bin goes through a
// stand-in: a Freeverb-style stereo reverb (Jezar's public-domain design: 8 damped
// comb filters + 4 all-pass filters per channel). The game's per-voice send levels
// still decide how much of each sound is reverberated; the room character is generic.
// VC3_REVERB=0 disables it, VC3_REVERB=x scales the wet level (default 1).
class Reverb {
public:
    void init(float room, float damp) {
        // Freeverb tunings (44.1 kHz) scaled to 48 kHz; right channel offset by 23.
        static const int kComb[8] = {1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617};
        static const int kAllpass[4] = {556, 441, 341, 225};
        for (int ch = 0; ch < 2; ++ch) {
            for (int i = 0; i < 8; ++i) comb_[ch][i].init(scale(kComb[i] + 23 * ch), room * 0.28f + 0.7f, damp * 0.4f);
            for (int i = 0; i < 4; ++i) allpass_[ch][i].init(scale(kAllpass[i] + 23 * ch));
        }
    }
    // in: mono send; adds the wet stereo signal to out (interleaved L/R).
    void process(const float* in, float* out, int n, float wet) {
        for (int i = 0; i < n; ++i) {
            const float x = in[i] * 0.015f;  // Freeverb's fixed input gain
            for (int ch = 0; ch < 2; ++ch) {
                float y = 0;
                for (auto& c : comb_[ch]) y += c.run(x);
                for (auto& a : allpass_[ch]) y = a.run(y);
                out[2 * i + ch] += y * wet * kWetScale;
            }
        }
    }

private:
    static int scale(int n) { return n * 48000 / 44100; }
    // Freeverb's stock wet scale. Measured in-game (VC3_TRACE_APU "reverb rms" vs "send
    // rms"), it gives roughly unity gain from the I3DL2 send to the reverb output.
    static constexpr float kWetScale = 3.0f;
    struct Comb {
        std::vector<float> buf;
        size_t pos = 0;
        float feedback = 0, damp = 0, store = 0;
        void init(int n, float fb, float d) { buf.assign(size_t(n), 0.0f); feedback = fb; damp = d; }
        float run(float x) {
            const float y = buf[pos];
            store = y * (1 - damp) + store * damp;
            buf[pos] = x + store * feedback;
            if (++pos == buf.size()) pos = 0;
            return y;
        }
    };
    struct Allpass {
        std::vector<float> buf;
        size_t pos = 0;
        void init(int n) { buf.assign(size_t(n), 0.0f); }
        float run(float x) {
            const float b = buf[pos];
            buf[pos] = x + b * 0.5f;
            if (++pos == buf.size()) pos = 0;
            return b - x;
        }
    };
    Comb comb_[2][8];
    Allpass allpass_[2][4];
};
Reverb g_reverb;
double g_dry_energy = 0, g_wet_energy = 0;
float g_reverb_wet = 1.0f;
constexpr int kI3dl2Bin = 10;

// Stereo downmix of the DSOUND speaker mixbins: 0/1 front L/R, 2 centre, 3 LFE,
// 4/5 back L/R, 6-9 the 3D (crosstalk) front/back L/R. Effect sends (I3DL2
// reverb, FX) are processed by the GP DSP on hardware and are dropped here.
void downmix(const float (*mix)[kFrame], float* out /* kFrame*2 */) {
    constexpr float k = 0.7071f;
    for (int i = 0; i < kFrame; ++i) {
        const float c = mix[2][i] * k + mix[3][i] * 0.5f;
        out[2 * i] = mix[0][i] + c + mix[4][i] * k + mix[6][i] + mix[8][i] * k;
        out[2 * i + 1] = mix[1][i] + c + mix[5][i] * k + mix[7][i] + mix[9][i] * k;
    }
    if (g_reverb_wet > 0) {
        float dry[kFrame * 2];
        if (g_trace) std::memcpy(dry, out, sizeof dry);
        // Keep the stand-in at a steady gain from the send. On hardware the game's I3DL2 room
        // settings (applied by the DSP, which isn't emulated) set the reverb level; the
        // stand-in's own gain varies with its input -- with a sustained loud send it rang up
        // to 2.4x the send. Some scenes (the subway at the Brand fight, the final boss's
        // hall, the ending) send most of the sound to the reverb: unregulated it came out
        // twice as loud as everything else and clipped into static, while holding it under
        // the (quiet) dry mix made the whole scene go quiet. So the output is regulated to a
        // fixed share of the send (measured over ~0.3 s): the game's send levels decide how
        // much reverb there is, as on hardware; the output limiter catches peaks.
        float wet[kFrame * 2] = {};
        g_reverb.process(mix[kI3dl2Bin], wet, kFrame, g_reverb_wet);
        double se = 0, we = 0;
        for (int i = 0; i < kFrame; ++i) se += double(mix[kI3dl2Bin][i]) * mix[kI3dl2Bin][i];
        for (int i = 0; i < kFrame * 2; ++i) we += double(wet[i]) * wet[i];
        se /= kFrame;
        we /= kFrame * 2;
        static double send_env = 0, wet_env = 0, cap_s = 1.0;
        send_env += (se - send_env) * 0.05;
        wet_env += (we - wet_env) * 0.05;
        constexpr double kWetToSend = 0.7;  // about -3 dB
        const double cap = wet_env > 1e-12 ? std::min(1.0, kWetToSend * std::sqrt(send_env / wet_env)) : 1.0;
        cap_s += (cap - cap_s) * (cap < cap_s ? 0.2 : 0.02);  // down quickly, back up gently
        for (int i = 0; i < kFrame * 2; ++i) out[i] += wet[i] * float(cap_s);
        if (g_trace)
            for (int i = 0; i < kFrame * 2; ++i) {
                g_dry_energy += double(dry[i]) * dry[i];
                g_wet_energy += double(out[i] - dry[i]) * (out[i] - dry[i]);
            }
    }
}

// VC3_APU_STEMS=ba,ba,... (debugging): every voice playing one of those buffers (base address, hex,
// as VC3_TRACE_APU prints it) is also written on its own to work/stems/<ba>.wav (dry, stereo downmix,
// time-aligned with VC3_WAV), from VC3_APU_STEMS_FROM seconds of audio on.
struct Stem { uint32_t ba; FILE* f = nullptr; uint32_t bytes = 0; float buf[kFrame * 2] = {}; bool used = false; };
std::deque<Stem> g_stems;  // (deque: pointers stay valid as stems are added)
bool g_stems_auto = false;  // VC3_APU_STEMS=loop: every looping or streaming voice
uint64_t g_frames_total = 0, g_stems_from = 0;
void stems_init() {
    if (const char* e = std::getenv("VC3_APU_STEMS"); e && !std::strcmp(e, "loop")) g_stems_auto = true;
    else if (e)
        for (const char* p = e; *p;) {
            char* end;
            Stem st;
            st.ba = uint32_t(std::strtoul(p, &end, 16));
            if (end == p) break;
            g_stems.push_back(st);
            p = *end ? end + 1 : end;
        }
    if (const char* e = std::getenv("VC3_APU_STEMS_FROM")) g_stems_from = uint64_t(std::atof(e) * 48000 / kFrame);
}
Stem* stem_for(uint16_t v) {
    if (g_stems.empty() && !g_stems_auto) return nullptr;
    const uint32_t ba = vget(v, V_BA) & 0xFFFFFF;
    for (Stem& st : g_stems) if (st.ba == ba) return &st;
    if (!g_stems_auto || g_frames_total < g_stems_from || !(vget(v, V_FMT) & (FMT_LOOP | FMT_STREAM))) return nullptr;
    g_stems.emplace_back();
    g_stems.back().ba = ba;
    return &g_stems.back();
}
void stems_write() {
    ++g_frames_total;
    if (g_frames_total < g_stems_from) { for (Stem& st : g_stems) st.used = false; return; }
    for (Stem& st : g_stems) {
        if (!st.f) {
            std::filesystem::create_directories("work/stems");
            char name[64];
            std::snprintf(name, sizeof name, "work/stems/%06x.wav", st.ba);
            st.f = std::fopen(name, "wb");
            if (!st.f) continue;
            uint8_t h[44] = {'R','I','F','F',0,0,0,0,'W','A','V','E','f','m','t',' ',16,0,0,0,1,0,2,0,
                             0x80,0xBB,0,0,0x00,0xEE,0x02,0,4,0,16,0,'d','a','t','a',0,0,0,0};
            std::fwrite(h, 1, 44, st.f);
            // silence up to now, so every stem lines up with the others
            const int16_t zero[kFrame * 2] = {};
            for (uint64_t k = g_stems_from; k + 1 < g_frames_total; ++k) { std::fwrite(zero, 2, kFrame * 2, st.f); st.bytes += sizeof zero; }
        }
        int16_t pcm[kFrame * 2];
        for (int i = 0; i < kFrame * 2; ++i)
            pcm[i] = int16_t(std::clamp(st.used ? st.buf[i] : 0.0f, -1.0f, 1.0f) * 32767.0f);
        std::fwrite(pcm, 2, kFrame * 2, st.f);
        st.bytes += sizeof pcm;
        if ((st.bytes & 0xFFFFF) < sizeof pcm) {  // keep the header current (about every MiB)
            const long pos = std::ftell(st.f);
            uint32_t riff = 36 + st.bytes;
            std::fseek(st.f, 4, SEEK_SET); std::fwrite(&riff, 4, 1, st.f);
            std::fseek(st.f, 40, SEEK_SET); std::fwrite(&st.bytes, 4, 1, st.f);
            std::fseek(st.f, pos, SEEK_SET);
        }
        st.used = false;
    }
}

// One VP frame (32 samples) of all voice lists into `out`. Returns false when the
// front end isn't running (sample counter off, or trapped/halted).
bool run_frame(float* out) {
    // A trapped or halted front end only stops method processing; the voice
    // processor keeps mixing (DSOUND handles a trap from its interrupt, which can
    // take a while: silencing the output meanwhile left gaps of up to 20 ms).
    const uint32_t mode = reg(FECTL) & FECTL_METHMODE;
    if (((reg(SECTL) >> 3) & 3) == 0 || (reg(VPVADDR) & 0x7FFFFFF) == 0) {
        ++g_stat.idle_frames;
        return false;
    }
    if (mode == FECTL_TRAPPED) ++g_stat.idle_trapped;
    else if (mode == FECTL_HALTED) ++g_stat.idle_halted;
    ++g_stat.frames;
    float mix[kMixbins][kFrame] = {};
    bool trapped = mode == FECTL_TRAPPED || mode == FECTL_HALTED;  // one trap at a time
    for (int list = 0; list < 3; ++list) {
        uint16_t v = uint16_t(reg(TVL2D + 12 * list));
        for (int guard = 0; v != kNoVoice && v < kMaxVoices && guard < kMaxVoices; ++guard) {
            reg(CVL2D + 12 * list) = v;
            const uint16_t nxt = next_voice(v);
            reg(NVL2D + 12 * list) = nxt;
            if (!(vget(v, V_STATE) & ST_ACTIVE)) {
                // Inactive voice still linked: ask DSOUND to unlink it (trap the FE).
                if (!trapped && (reg(FETFORCE1) & FETFORCE1_IDLE_VOICE)) {
                    reg(FEDECMETH) = SE2FE_IDLE_VOICE;
                    reg(FEDECPARAM) = v;
                    ring('T', v, 0);
                    reg(FECTL) = (reg(FECTL) & ~(FECTL_METHMODE | FECTL_TRAPREASON)) | FECTL_TRAPPED |
                                 FECTL_TRAP_REQUESTED;
                    trapped = true;
                    ++g_stat.traps;
                    g_trap_raised_ns.store(now_ns(), std::memory_order_relaxed);
                    update_irq();
                }
            } else if (g_vh[v].locked) {
                ++g_stat.locked_skips;
            } else if (!g_mute[v]) {
                ++g_stat.voices;
                if (Stem* st = stem_for(v)) {  // VC3_APU_STEMS: this voice on its own as well
                    float solo[kMixbins][kFrame] = {};
                    process_voice(v, solo);
                    for (int b = 0; b < kMixbins; ++b)
                        for (int i = 0; i < kFrame; ++i) mix[b][i] += solo[b][i];
                    float o[kFrame * 2];
                    downmix(solo, o);
                    if (!st->used) std::fill(std::begin(st->buf), std::end(st->buf), 0.0f);
                    for (int i = 0; i < kFrame * 2; ++i) st->buf[i] += o[i];
                    st->used = true;
                } else {
                    process_voice(v, mix);
                }
            }
            v = nxt;
        }
    }
    if (g_trace)
        for (int b = 0; b < kMixbins; ++b)
            for (int i = 0; i < kFrame; ++i) g_bin_energy[b] += double(mix[b][i]) * mix[b][i];
    downmix(mix, out);
    if (!g_stems.empty() || g_stems_auto) stems_write();
    return true;
}

// ---------------------------------------------------------------- GP DSP stand-in
// DSOUND talks to its effects program on the GP DSP through a command word at
// offset 0x810 of the GP scratch memory (pages listed in the GPSADDR table):
// it writes a command and spins (at raised IRQL) until the DSP clears it. The
// DSP isn't emulated, so acknowledge commands immediately.
void gp_mailbox() {
    const uint32_t sge = reg(GPSADDR) & 0x7FFFFFF;
    if (!sge) return;
    const uint32_t page = rd<uint32_t>(sge) & 0x7FFFF000;
    if (!page) return;
    const uint32_t cmd_addr = page + 0x810;
    const uint32_t cmd = rd<uint32_t>(cmd_addr);
    if (cmd == 0) return;
    if (g_trace) {
        std::fprintf(stderr, "[apu] GP command %#x acknowledged; scratch pages:", cmd);
        for (uint32_t i = 0; i < 64; ++i) {
            const uint32_t pg = rd<uint32_t>(sge + 8 * i) & 0x7FFFF000;
            if (!pg) break;
            std::fprintf(stderr, " %x", pg);
        }
        std::fprintf(stderr, "\n");
    }
    wr<uint32_t>(cmd_addr, 0);
}

// ---------------------------------------------------------------- output
SDL_AudioStream* g_stream = nullptr;
FILE* g_wav = nullptr;
uint32_t g_wav_bytes = 0;
float g_volume = 1.0f;

void wav_header() {
    if (!g_wav) return;
    auto u32le = [](uint8_t* p, uint32_t v) { for (int i = 0; i < 4; ++i) p[i] = uint8_t(v >> (8 * i)); };
    uint8_t h[44] = {'R', 'I', 'F', 'F', 0, 0, 0, 0, 'W', 'A', 'V', 'E', 'f', 'm', 't', ' ', 16, 0, 0, 0, 1, 0, 2, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 16, 0, 'd', 'a', 't', 'a', 0, 0, 0, 0};
    u32le(h + 4, 36 + g_wav_bytes);
    u32le(h + 24, 48000);
    u32le(h + 28, 48000 * 4);
    u32le(h + 40, g_wav_bytes);
    const long pos = std::ftell(g_wav);
    std::fseek(g_wav, 0, SEEK_SET);
    std::fwrite(h, 1, sizeof h, g_wav);
    std::fseek(g_wav, pos < 44 ? 44 : pos, SEEK_SET);
    std::fflush(g_wav);
}

void output(const float* buf, int frames) {
    int16_t pcm[kFrame * kFramesPerChunk * 2];
    // Safety limiter: peaks over the ceiling turn the level down briefly (fast attack,
    // ~100 ms release) instead of being clipped, which crackles like static.
    static float gain = 1.0f;
    constexpr float kCeiling = 0.95f, kRelease = 1.0f / (0.1f * 48000.0f);
    for (int i = 0; i < frames; ++i) {
        const float l = buf[2 * i] * g_volume, r = buf[2 * i + 1] * g_volume;
        const float peak = std::max(std::fabs(l), std::fabs(r));
        if (peak * gain > kCeiling) gain = kCeiling / peak;
        else gain = std::min(1.0f, gain + kRelease);
        pcm[2 * i] = int16_t(std::lrint(std::clamp(l * gain, -1.0f, 1.0f) * 32767.0f));
        pcm[2 * i + 1] = int16_t(std::lrint(std::clamp(r * gain, -1.0f, 1.0f) * 32767.0f));
    }
    if (g_stream) SDL_PutAudioStreamData(g_stream, pcm, frames * 4);
    if (g_wav) {
        std::fwrite(pcm, 4, frames, g_wav);
        g_wav_bytes += frames * 4;
        static int n = 0;
        if (++n % 64 == 0) wav_header();
    }
}

std::atomic<bool> g_quit{false}, g_thread_done{false};
void frame_thread() {
    constexpr auto kChunk = std::chrono::microseconds(1000000 * kFrame * kFramesPerChunk / 48000);
    auto next = Clock::now();
    float buf[kFrame * kFramesPerChunk * 2];
    // With a sound device, production follows the device's own clock: keep its queue
    // topped up to ~50 ms. (Pacing by the host clock instead lets the two drift apart;
    // the queue then runs dry and the output pops.) Without one, pace by the host clock.
    constexpr int kTargetQueued = 48000 * 4 * 50 / 1000;
    for (;;) {
        if (g_quit.load()) {
            g_thread_done = true;
            return;
        }
        if (g_stream) {
            if (SDL_GetAudioStreamQueued(g_stream) >= kTargetQueued) {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue;
            }
        } else {
            next += kChunk;
            const auto now = Clock::now();
            if (now > next + std::chrono::milliseconds(100)) next = now;  // fell far behind: resync
            platform::sleep_until(next);
        }

        bool any = false;
        {
            std::unique_lock lk(g_m);
            // Give a voice that DSOUND has locked for an update a moment to finish.
            for (int tries = 0; tries < 20; ++tries) {
                bool locked = false;
                for (const auto& h : g_vh) locked |= h.locked;
                if (!locked) break;
                ++g_stat.lock_waits;
                lk.unlock();
                std::this_thread::sleep_for(std::chrono::microseconds(100));
                lk.lock();
            }
            gp_mailbox();
            for (int f = 0; f < kFramesPerChunk; ++f) {
                if (run_frame(buf + f * kFrame * 2)) any = true;
                else std::memset(buf + f * kFrame * 2, 0, sizeof(float) * kFrame * 2);
            }
        }
        aci::tick(kFrame * kFramesPerChunk);
        (void)any;
        if (g_trace) {
            static auto last = Clock::now();
            if (Clock::now() - last > std::chrono::seconds(5)) {
                last = Clock::now();
                std::lock_guard lk(g_m);
                std::fprintf(stderr, "[apu] underruns %u, locked-voice skips %u, lock waits %u, frames while trapped %u halted %u, queued %d bytes\n",
                             g_stat.underruns, g_stat.locked_skips, g_stat.lock_waits, g_stat.idle_trapped, g_stat.idle_halted,
                             g_stream ? SDL_GetAudioStreamQueued(g_stream) : -1);
                std::fprintf(stderr, "[apu] frames %u idle %u voice-frames %u traps %u fectl %#x sectl %#x ists %#x ien %#x\n",
                             g_stat.frames, g_stat.idle_frames, g_stat.voices, g_stat.traps, reg(FECTL), reg(SECTL),
                             reg(ISTS), reg(IEN));
                if (const int64_t n = g_traps_closed.exchange(0))
                    std::fprintf(stderr, "[apu] traps open avg %.2f ms, max %.2f ms (%lld)\n",
                                 double(g_trap_open_sum_ns.exchange(0)) / double(n) / 1e6,
                                 double(g_trap_open_max_ns.exchange(0)) / 1e6, (long long)n);
                g_stat = {};
                std::fprintf(stderr, "[apu] dry rms %.4f, reverb rms %.4f, send rms %.4f\n",
                             std::sqrt(g_dry_energy / (96000.0 * 5)), std::sqrt(g_wet_energy / (96000.0 * 5)),
                             std::sqrt(g_bin_energy[kI3dl2Bin] / (48000.0 * 5)));
                g_dry_energy = g_wet_energy = 0;
                std::string e = "[apu] mixbin rms:";
                for (int b = 0; b < kMixbins; ++b) {
                    char t[24];
                    std::snprintf(t, sizeof t, " %d:%.4f", b, std::sqrt(g_bin_energy[b] / (48000.0 * 5)));
                    e += t;
                    g_bin_energy[b] = 0;
                }
                std::fprintf(stderr, "%s\n", e.c_str());
            }
        }
        if (g_stream && SDL_GetAudioStreamQueued(g_stream) == 0) ++g_stat.underruns;
        output(buf, kFrame * kFramesPerChunk);
    }
}

// ---------------------------------------------------------------- MMIO
uint32_t read(void*, uint32_t addr, unsigned size) {
    const uint32_t off = addr - kBase;
    if ((off & ~3u) == XGSCNT) {
        const auto us = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - g_epoch).count();
        return uint32_t(us * 48 / 1000);
    }
    if ((off & ~3u) == 0x20010 || (off & ~3u) == 0x30010) return 0x80;  // FIFO free space
    std::lock_guard lk(g_m);
    uint32_t v;
    if (off < 0x20000) v = reg(off & ~3u) >> (8 * (off & 3));
    else v = g_other[off & ~3u] >> (8 * (off & 3));
    if (g_trace && g_seen.insert(off & ~3u).second) std::fprintf(stderr, "[apu] first read %#x -> %#x\n", off, v);
    ring('R', off, v);
    return size == 4 ? v : v & ((1u << (8 * size)) - 1);
}

void write(void*, uint32_t addr, uint32_t v, unsigned size) {
    const uint32_t off = addr - kBase;
    std::lock_guard lk(g_m);
    if (g_trace && g_seen.insert(off & ~3u).second) std::fprintf(stderr, "[apu] first write %#x = %#x\n", off, v);
    if (off < 0x20000 || off >= 0x30000) ring('W', off, v);
    if (off >= 0x20000 && off < 0x30000) {  // VP: FE methods
        fe_method(off - 0x20000, v);
        return;
    }
    if (off >= 0x20000) {
        if (g_trace) {  // GP/EP DSP memory and registers: log changed words (first 400, then every 100th)
            static unsigned n = 0;
            const uint32_t old = g_other.count(off & ~3u) ? g_other[off & ~3u] : 0;
            if (old != v && (++n <= 400 || n % 100 == 0))
                std::fprintf(stderr, "[apu] dsp write %#x: %#x -> %#x (#%u)\n", off, old, v, n);
        }
        uint32_t& r = g_other[off & ~3u];
        if (size == 4) r = v;
        else {
            const uint32_t sh = 8 * (off & 3), mask = ((1u << (8 * size)) - 1) << sh;
            r = (r & ~mask) | ((v << sh) & mask);
        }
        return;
    }
    switch (off) {
        case ISTS:  // write 1 to clear
            reg(ISTS) &= ~v;
            update_irq();
            break;
        case IEN:
            reg(IEN) = v;
            update_irq();
            break;
        case FECTL:
        case SECTL:
            if (off == FECTL && (reg(FECTL) & FECTL_METHMODE) == FECTL_TRAPPED && (v & FECTL_METHMODE) != FECTL_TRAPPED) {
                if (const int64_t t = g_trap_raised_ns.exchange(0, std::memory_order_relaxed)) {
                    const int64_t d = now_ns() - t;
                    g_trap_open_sum_ns += d;
                    ++g_traps_closed;
                    if (d > g_trap_open_max_ns.load(std::memory_order_relaxed)) g_trap_open_max_ns.store(d, std::memory_order_relaxed);
                }
            }
            reg(off) = v;
            update_irq();
            g_cv.notify_all();
            break;
        case FEMEMDATA:  // the FE stores this at FEMEMADDR (DSOUND uses it as a fence)
            reg(off) = v;
            wr<uint32_t>(reg(FEMEMADDR) & 0x7FFFFFF, v);
            break;
        default:
            if (size == 4) reg(off) = v;
            else {
                uint32_t& r = reg(off & ~3u);
                const uint32_t sh = 8 * (off & 3), mask = ((1u << (8 * size)) - 1) << sh;
                r = (r & ~mask) | ((v << sh) & mask);
            }
            break;
    }
}

}  // namespace

void init(bool play) {
    g_trace = std::getenv("VC3_TRACE_APU") != nullptr;
    if (const char* s = std::getenv("VC3_VOLUME")) g_volume = std::strtof(s, nullptr);
    if (const char* s = std::getenv("VC3_APU_MUTE"))
        for (char* e; *s; s = *e ? e + 1 : e) {
            const long v = std::strtol(s, &e, 0);
            if (e == s) break;
            if (v >= 0 && v < kMaxVoices) g_mute[v] = true;
        }
    if (const char* s = std::getenv("VC3_TRACE_APU_VOICE")) g_trace_voice = std::atoi(s);
    if (const char* s = std::getenv("VC3_APU_SOLO")) {  // mute every voice except these
        bool keep[kMaxVoices] = {};
        for (char* e; *s; s = *e ? e + 1 : e) {
            const long v = std::strtol(s, &e, 0);
            if (e == s) break;
            if (v >= 0 && v < kMaxVoices) keep[v] = true;
        }
        for (int v = 0; v < kMaxVoices; ++v) g_mute[v] = !keep[v];
    }
    for (uint32_t r = TVL2D; r <= 0x2074; r += 4) reg(r) = kNoVoice;
    if (const char* r = std::getenv("VC3_REVERB")) g_reverb_wet = std::strtof(r, nullptr);
    stems_init();
    g_reverb.init(0.5f, 0.5f);
    mmio::register_range(kBase, kSize, "apu", read, write);
    aci::init();

    if (const char* path = std::getenv("VC3_WAV")) {
        g_wav = std::fopen(path, "wb");
        if (g_wav) wav_header();
        else std::fprintf(stderr, "[apu] can't write %s\n", path);
    }
    if (const char* a = std::getenv("VC3_AUDIO")) play = std::strcmp(a, "0") != 0;
    if (play) {
        if (!SDL_InitSubSystem(SDL_INIT_AUDIO)) {
            std::fprintf(stderr, "[apu] SDL audio init failed: %s\n", SDL_GetError());
        } else {
            const SDL_AudioSpec spec{SDL_AUDIO_S16LE, 2, 48000};
            g_stream = SDL_OpenAudioDeviceStream(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &spec, nullptr, nullptr);
            if (!g_stream) std::fprintf(stderr, "[apu] no audio device: %s\n", SDL_GetError());
            else SDL_ResumeAudioStreamDevice(g_stream);
        }
    }
    std::thread(frame_thread).detach();
}

void shutdown() {
    g_quit = true;
    for (int i = 0; i < 200 && !g_thread_done.load(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    if (!g_thread_done.load()) std::_Exit(0);  // still busy: exit now rather than pull the device from under it
    if (g_stream) {
        SDL_DestroyAudioStream(g_stream);
        g_stream = nullptr;
    }
    if (g_wav) {
        wav_header();
        std::fclose(g_wav);
        g_wav = nullptr;
    }
}

}  // namespace vc3::audio::apu
