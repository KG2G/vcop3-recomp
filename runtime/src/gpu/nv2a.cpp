#ifdef VC3_HAVE_VULKAN
#include "gpu/vk_present.h"
#endif
#include "nv2a.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <mutex>
#include <set>
#include <thread>
#include <unordered_map>
#include <vector>

#include <epoxy/gl.h>

#include "gpu/gl_backend.h"
#include "gpu/pgraph.h"
#include "gpu/device.h"
#include "io/input.h"
#include "io/jvs.h"
#include "kernel/ke.h"
#include "mmio.h"
#include "profiler.h"
#include "scheduler.h"
#include "vc3/helpers.h"
#include "vc3/memory.h"

namespace vc3::gpu {

namespace {

using Clock = std::chrono::steady_clock;

constexpr u32 kBase = 0xFD000000;
constexpr u32 kRaminSize = 0x100000;
constexpr u32 kRaminRamOffset = mem::kRamSize - kRaminSize;  // PRAMIN = top 1 MiB of RAM
constexpr u32 kIrqVector = 0x33;                              // IRQ 3

// Block offsets
constexpr u32 PMC = 0x000000, PBUS = 0x001000, PFIFO = 0x002000, PTIMER = 0x009000, PFB = 0x100000,
              PGRAPH = 0x400000, PCRTC = 0x600000, PRMCIO = 0x601000, PRAMDAC = 0x680000, PRAMIN = 0x700000,
              USER = 0x800000;

bool trace() {
    static const bool on = std::getenv("VC3_TRACE_GPU") != nullptr;
    return on;
}

struct Subchannel {
    u32 handle = 0, instance = 0, cls = 0;
};

struct State {
    std::recursive_mutex m;
    std::condition_variable_any cv;
    std::unordered_map<u32, u32> regs;  // generic register file, keyed by offset from kBase

    // PMC / PCRTC / PGRAPH interrupt state
    u32 pmc_intr = 0, pmc_intr_en = 0;
    u32 pcrtc_intr = 0, pcrtc_intr_en = 0, pcrtc_start = 0;
    u32 pgraph_intr = 0, pgraph_intr_en = 0;
    // PGRAPH FIFO access (0x400720 bit 0). D3D's interrupt handler clears it on entry
    // and sets it again when it's done with a trapped method; the pusher stays
    // stalled until then, not just until the interrupt bit is acknowledged.
    bool fifo_access = true;
    bool irq_raised = false;

    // Channel 0 (the only one D3D uses)
    // Push buffer pointers: shared between the CPU (registers) and the GPU thread,
    // which renders without holding `m` so register access never waits on rendering.
    std::atomic<u32> dma_put{0}, dma_get{0}, ref{0};
    // CACHE1_DMA_SUBROUTINE: return offset | 1 while a push-buffer `call` is active.
    // D3D reads it to locate the GPU when DMA_GET points into a called sub-buffer.
    std::atomic<u32> subroutine{0};
    std::array<Subchannel, 8> sub{};
    std::vector<u32> call_stack;

    // Kelvin (NV097) method state, captured for the renderer
    std::array<u32, 0x2000 / 4> kelvin{};
    u32 semaphore_offset = 0, report_value = 0;
    u32 flip_read = 0, flip_write = 0, flip_modulo = 0;

    std::mutex counts_m;                                 // guards method_counts (trace only)
    std::map<std::pair<u32, u32>, u64> method_counts;  // (class, method) -> count
    u64 frames = 0;
    std::atomic<u64> flips{0};
} g;

const Clock::time_point g_epoch = Clock::now();

u32 ram32(u32 phys) { return rd<u32>(phys & (mem::kRamSize - 1)); }
u32 ramin32(u32 off) { return rd<u32>(kRaminRamOffset + (off & (kRaminSize - 1))); }

// ------------------------------------------------------------------ interrupts
void update_irq() {
    if (g.pcrtc_intr & g.pcrtc_intr_en) g.pmc_intr |= 1u << 24; else g.pmc_intr &= ~(1u << 24);
    if (g.pgraph_intr & g.pgraph_intr_en) g.pmc_intr |= 1u << 12; else g.pmc_intr &= ~(1u << 12);
    // Level-triggered: deliver while asserted (deliveries are deduplicated), so a
    // cause that appears while the ISR is running is not lost.
    const bool active = (g.pmc_intr & 0x7FFFFFFF) && (g.pmc_intr_en & 1);
    if (active) kernel::ke::raise_interrupt(kIrqVector);
    g.irq_raised = active;
}

// ------------------------------------------------------------------ objects
// DMA object in RAMIN: w0 = class/flags (adjust in bits 20-31), w1 = limit, w2 = frame | flags.
u32 dma_base(u32 instance_off) {
    if (!instance_off) return 0;
    const u32 w0 = ramin32(instance_off), w2 = ramin32(instance_off + 8);
    return (w2 & 0xFFFFF000u) + ((w0 >> 20) & 0xFFF);
}

// RAMHT lookup (PFIFO_RAMHT at 0x2210: base in bits 4-8 (x4 KiB), size in bits 16-17 (4 KiB << n)).
u32 ramht_lookup(u32 handle) {
    const u32 reg = g.regs[PFIFO + 0x210];
    const u32 base = ((reg >> 4) & 0x1F) << 12;
    const u32 size = 0x1000u << ((reg >> 16) & 3);
    const u32 bits = __builtin_ctz(size) - 1;  // entries are 8 bytes
    u32 hash = 0;
    for (u32 h = handle; h; h >>= bits) hash ^= h & ((1u << bits) - 1);
    // channel 0: no channel term
    u32 e = base + hash * 8;
    if (ramin32(e) != handle) {
        for (u32 o = base; o < base + size; o += 8)
            if (ramin32(o) == handle) { e = o; break; }
    }
    if (ramin32(e) != handle) {
        std::fprintf(stderr, "[nv2a] RAMHT: handle %08x not found\n", handle);
        return 0;
    }
    return (ramin32(e + 4) & 0xFFFF) << 4;
}
// VC3_TRACE_HITCH=1: frame times as the display sees them (vblanks at which the
// picture changed: a new display buffer, or a new frame written into it). Logs every frame longer than
// 20 ms with what happened meanwhile, and a summary every 5 s.
extern std::atomic<u64> g_gpu_busy_total_ns;
void note_frame() {
    static const bool on = std::getenv("VC3_TRACE_HITCH") != nullptr;
    if (!on) return;
    static auto last = Clock::now(), report = Clock::now();
    static u64 last_busy = 0, last_sn = 0, last_sns = 0, n = 0, over20 = 0, over33 = 0;
    static double worst = 0, sum = 0;
    const auto now = Clock::now();
    const double ms = std::chrono::duration<double, std::milli>(now - last).count();
    last = now;
    u64 sn, sns;
    pgraph::glb::shader_stats(sn, sns);
    const u64 busy = g_gpu_busy_total_ns.load();
    static u64 last_io = 0;
    const u64 io = sched::g_io_ns.load();
    u64 hold;
    int hold_prio;
    sched::take_max_hold(hold, hold_prio);
    if (ms > 20.0)
        std::fprintf(stderr, "[hitch] frame at vblank %llu (flip %llu): %.1f ms (gpu thread busy %.1f ms, %llu shaders %.1f ms, file reads %.1f ms, longest lock hold %.1f ms by p%d)\n",
                     (unsigned long long)g.frames, (unsigned long long)g.flips.load(), ms, double(busy - last_busy) / 1e6,
                     (unsigned long long)(sn - last_sn), double(sns - last_sns) / 1e6, double(io - last_io) / 1e6,
                     double(hold) / 1e6, hold_prio);
    last_io = io;
    last_busy = busy;
    last_sn = sn;
    last_sns = sns;
    ++n;
    sum += ms;
    worst = std::max(worst, ms);
    over20 += ms > 20.0;
    over33 += ms > 33.0;
    if (now - report > std::chrono::seconds(5)) {
        std::fprintf(stderr, "[hitch] %llu frames, avg %.2f ms, worst %.1f ms, >20 ms: %llu, >33 ms: %llu\n",
                     (unsigned long long)n, sum / double(n), worst, (unsigned long long)over20, (unsigned long long)over33);
        report = now;
        n = over20 = over33 = 0;
        worst = sum = 0;
    }
}


// ------------------------------------------------------------------ methods
// VC3_FLIP_STALL=0: old behaviour (never stall on a full flip queue).
const bool g_flip_stall = !(std::getenv("VC3_FLIP_STALL") && !std::strcmp(std::getenv("VC3_FLIP_STALL"), "0"));
void kelvin_method(u32 method, u32 data) {
    g.kelvin[method / 4] = data;
    pgraph::method(method, data);
    switch (method) {
        case 0x0110: break;  // WAIT_FOR_IDLE
        // Flip queue (NV_PGRAPH_SURFACE READ_3D/WRITE_3D/MODULO_3D). Present queues a
        // buffer (INCREMENT_WRITE) and FLIP_STALL makes the GPU wait while the queue is
        // full (read == write) until D3D's vblank handler has shown one and advanced
        // READ_3D through NV_PGRAPH_INCREMENT. Without the stall the GPU could finish
        // two frames inside one vblank, and D3D would only ever display the newer one.
        case 0x0120: { std::lock_guard lk(g.m); g.flip_read = data; break; }
        case 0x0124: { std::lock_guard lk(g.m); g.flip_write = data; break; }
        case 0x0128: { std::lock_guard lk(g.m); g.flip_modulo = data; break; }
        case 0x012C: {  // FLIP_INCREMENT_WRITE
            std::lock_guard lk(g.m);
            if (g.flip_modulo) g.flip_write = (g.flip_write + 1) % g.flip_modulo;
            break;
        }
        case 0x0130: {  // FLIP_STALL
            std::unique_lock lk(g.m);
            if (g.flip_modulo && g_flip_stall) {
                const bool ok = g.cv.wait_for(lk, std::chrono::milliseconds(100), [] { return g.flip_read != g.flip_write; });
                if (!ok) {
                    static int n = 0;
                    if (n++ < 3) std::fprintf(stderr, "[nv2a] FLIP_STALL timed out (read %u write %u modulo %u)\n", g.flip_read, g.flip_write, g.flip_modulo);
                }
            }
            ++g.flips;
            // VC3_SCRIPT_BY_FLIP: the input script counts game frames instead of vblanks
            // (identical input timing for renderers of any speed).
            if (static const bool by_flip = std::getenv("VC3_SCRIPT_BY_FLIP") != nullptr; by_flip) io::script_tick(g.flips.load());
            break;
        }
        case 0x17C8: g.report_value = 0; break;  // CLEAR_REPORT_VALUE
        case 0x17CC: break;                      // SET_ZPASS_PIXEL_COUNT_ENABLE
        case 0x17D0: {                           // GET_REPORT(offset | type << 24): {u64 time, u32 value, u32 status=0}
            const u32 addr = (dma_base(g.kelvin[0x01A8 / 4]) + (data & 0xFFFFFF)) & (mem::kRamSize - 1);
            const u64 ns = u64(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - g_epoch).count());
            wr<u64>(addr, ns);
            wr<u32>(addr + 8, g.report_value);  // z-pass pixel count (no rasteriser yet: 0)
            wr<u32>(addr + 12, 0);
            if (trace()) std::fprintf(stderr, "[nv2a] GET_REPORT %08x -> dma inst %05x base %08x addr %08x\n", data, g.kelvin[0x01A8 / 4], dma_base(g.kelvin[0x01A8 / 4]), addr);
            break;
        }
        case 0x1D6C: g.semaphore_offset = data; break;
        case 0x1D70: {       // BACK_END_WRITE_SEMAPHORE_RELEASE
            const u32 dma = g.kelvin[0x01A4 / 4];  // SET_CONTEXT_DMA_SEMAPHORE (instance after binding)
            const u32 addr = dma_base(dma) + g.semaphore_offset;
            if (trace()) std::fprintf(stderr, "[nv2a] SEMAPHORE_RELEASE %08x -> %08x\n", data, addr);
            wr<u32>(addr & (mem::kRamSize - 1), data);
            break;
        }
        default: break;
    }
}

void method(int subch, u32 m, u32 data) {
    auto& s = g.sub[size_t(subch)];
    if (m == 0) {  // SET_OBJECT
        s.handle = data;
        s.instance = ramht_lookup(data);
        s.cls = ramin32(s.instance) & 0xFF;
        if (trace()) std::fprintf(stderr, "[nv2a] subch %d <- object %08x class %02x (inst %05x)\n", subch, data, s.cls, s.instance);
        return;
    }
    if (trace()) {
        std::lock_guard lk(g.counts_m);
        ++g.method_counts[{s.cls, m}];
    }
    // VC3_GPU_DUMP=N: write the method stream of the Nth frame (FLIP_STALL to FLIP_STALL) to work/gpu_frame.txt.
    static const long dump_frame = std::getenv("VC3_GPU_DUMP") ? std::atol(std::getenv("VC3_GPU_DUMP")) : -1;
    static long flips = 0;
    static std::FILE* dump = nullptr;
    if (dump_frame >= 0) {
        if (flips == dump_frame && !dump) dump = std::fopen("work/gpu_frame.txt", "w");
        if (dump && flips == dump_frame) std::fprintf(dump, "%d %02x %04x %08x\n", subch, s.cls, m, data);
        if (s.cls == 0x97 && m == 0x130 && ++flips > dump_frame && dump) { std::fclose(dump); dump = nullptr; }
    }
    if (m == 0x50) { g.ref = data; return; }  // SET_REFERENCE (any object)
    if (s.cls == 0x97 && m == 0x100 && data) {
        // NO_OPERATION with a parameter is D3D's software method: trap it to the CPU
        // (PGRAPH error interrupt) and stall until the ISR acknowledges.
        std::unique_lock lk(g.m);
        g.regs[PGRAPH + 0x704] = (u32(subch) << 16) | m;  // TRAPPED_ADDR
        g.regs[PGRAPH + 0x708] = data;                     // TRAPPED_DATA_LOW
        g.regs[PGRAPH + 0x108] = 1;                        // NSOURCE: notification
        g.pgraph_intr |= 1u << 20;                         // INTR_ERROR
        update_irq();
        g.cv.wait(lk, [] { return !(g.pgraph_intr & (1u << 20)) && g.fifo_access; });
        return;
    }
    // Context DMA handles are resolved to instances when set (methods 0x180-0x1FC).
    if (m >= 0x180 && m < 0x200 && data) data = ramht_lookup(data);
    if (s.cls == 0x97) kelvin_method(m, data);
}

// ------------------------------------------------------------------ pusher
// Runs on the GPU thread without g.m held. A command is fully executed before
// DMA_GET moves past it, so the CPU never sees progress that hasn't happened.
void run_pusher(u32 base) {
    int guard = 0;
    u32 get = g.dma_get.load();
    u32 published = get;  // last value we stored into DMA_GET
    while (get != g.dma_put.load() && guard++ < 1 << 22) {
        // The guest may write DMA_GET itself (D3D does when it resets the push
        // buffer). That write wins: continue from there rather than overwriting it.
        if (const u32 cur = g.dma_get.load(); cur != published) {
            get = published = cur;
            g.call_stack.clear();
            g.subroutine.store(0);
            continue;
        }
        const u32 cmd = ram32(base + get);
        get += 4;
        if ((cmd & 3) == 1) {  // jump
            get = cmd & ~3u;
        } else if ((cmd & 3) == 2) {  // call
            g.call_stack.push_back(get);
            g.subroutine.store(get | 1);
            get = cmd & ~3u;
        } else if (cmd == 0x00020000) {  // return
            if (!g.call_stack.empty()) { get = g.call_stack.back(); g.call_stack.pop_back(); }
            g.subroutine.store(g.call_stack.empty() ? 0 : (g.call_stack.back() | 1));
        } else if ((cmd & 0xE0000003) == 0x20000000) {  // old jump
            get = cmd & 0x1FFFFFFC;
        } else if ((cmd & 0xE0030003) == 0 || (cmd & 0xE0030003) == 0x40000000) {  // (non-)increasing methods
            const bool inc = (cmd & 0xE0030003) == 0;
            u32 m = cmd & 0x1FFC;
            const int subch = int((cmd >> 13) & 7);
            const u32 count = (cmd >> 18) & 0x7FF;
            for (u32 i = 0; i < count; ++i) {
                method(subch, m, ram32(base + get));
                get += 4;
                if (inc) m += 4;
            }
        } else if (cmd == 0) {
            // nop
        } else {
            std::fprintf(stderr, "[nv2a] unknown push buffer command %08x at %08x\n", cmd, get - 4);
            get = g.dma_put.load();
        }
        // Publish progress unless the guest moved DMA_GET meanwhile (checked above next time).
        if (g.dma_get.compare_exchange_strong(published, get)) published = get;
    }
    // The guard only bounds one pass (the rest is processed next time); hitting it on a
    // big batch is normal, so it's reported only when tracing.
    if (guard >= 1 << 22 && trace()) {
        static int n = 0;
        if (n++ < 5) {
            std::fprintf(stderr, "[nv2a] pusher pass limit reached: get=%08x put=%08x base=%08x cmd@get=%08x", get, g.dma_put.load(), base, ram32(base + get));
            for (int i = -4; i < 4; ++i) std::fprintf(stderr, " %08x", ram32(base + get + 4 * i));
            std::fprintf(stderr, " subroutine=%08x callstack=%zu\n", g.subroutine.load(), g.call_stack.size());
        }
    }
}

std::atomic<u64> g_gpu_busy_ns{0};  // time the pusher spent processing (VC3_FPS)
std::atomic<u64> g_gpu_busy_total_ns{0};  // same, never reset (VC3_TRACE_HITCH)
std::atomic<bool> g_pusher_busy{false};
u64 g_vb_missed_busy = 0, g_vb_missed_idle = 0;  // vblanks without a new frame, by GPU thread state


void gpu_thread() {
    for (;;) {
        u32 base;
        {
            std::unique_lock lk(g.m);
            // Wake at least every millisecond to finish pending surface write-backs.
            while (!g.cv.wait_for(lk, std::chrono::milliseconds(1), [] { return g.dma_get.load() != g.dma_put.load(); })) {
                lk.unlock();
                pgraph::glb::poll();
                lk.lock();
            }
            base = dma_base((g.regs[PFIFO + 0x122C] & 0xFFFF) << 4);  // CACHE1_DMA_INSTANCE
        }
        const auto t0 = Clock::now();
        g_pusher_busy = true;
        run_pusher(base);
        g_pusher_busy = false;
        const u64 busy_ns = u64(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t0).count());
        g_gpu_busy_ns += busy_ns;
        g_gpu_busy_total_ns += busy_ns;
    }
}

// VC3_SHOT_EVERY=N: write the scan-out buffer to work/shots/frame_NNNNN.ppm every N frames.
// VC3_FLIPSHOT=N: save the displayed picture once the guest has flipped N+2 times
// (work/flipshot.ppm) -- the same game frame whatever the renderer's speed.
void maybe_flipshot() {
    static const long at = std::getenv("VC3_FLIPSHOT") ? std::atol(std::getenv("VC3_FLIPSHOT")) : -1;
    static bool done = false;
    if (at < 0 || done || long(g.flips.load()) < at + 2) return;
    done = true;
    const u32 start = g.pcrtc_start & (mem::kRamSize - 1);
    std::FILE* f = std::fopen("work/flipshot.ppm", "wb");
    if (!f) return;
    std::fprintf(f, "P6\n640 480\n255\n");
    for (u32 y = 0; y < 480; ++y)
        for (u32 x = 0; x < 640; ++x) {
            const u8* p = host_ptr(start + y * 2560 + x * 4);
            const u8 rgb[3] = {p[2], p[1], p[0]};
            std::fwrite(rgb, 1, 3, f);
        }
    std::fclose(f);
    std::fprintf(stderr, "[flipshot] wrote work/flipshot.ppm at flip %llu\n", (unsigned long long)g.flips.load());
}

void maybe_screenshot() {
    maybe_flipshot();
    static const long every = std::getenv("VC3_SHOT_EVERY") ? std::atol(std::getenv("VC3_SHOT_EVERY")) : 0;
    if (every <= 0 || g.frames % u64(every) != 0) return;
    // Optional capture window (vblank numbers), e.g. for recording a video.
    static const u64 from = std::getenv("VC3_SHOT_FROM") ? std::strtoull(std::getenv("VC3_SHOT_FROM"), nullptr, 10) : 0;
    static const u64 to = std::getenv("VC3_SHOT_TO") ? std::strtoull(std::getenv("VC3_SHOT_TO"), nullptr, 10) : ~0ull;
    if (g.frames < from || g.frames > to) return;
    const u32 start = g.pcrtc_start & (mem::kRamSize - 1);
    if (!start) return;
    std::filesystem::create_directories("work/shots");
    char name[64];
    std::snprintf(name, sizeof name, "work/shots/frame_%05llu.ppm", (unsigned long long)g.frames);
    std::FILE* f = std::fopen(name, "wb");
    if (!f) return;
    std::fprintf(f, "P6\n640 480\n255\n");
    for (u32 y = 0; y < 480; ++y)
        for (u32 x = 0; x < 640; ++x) {
            const u8* p = host_ptr(start + y * 2560 + x * 4);
            const u8 rgb[3] = {p[2], p[1], p[0]};
            std::fwrite(rgb, 1, 3, f);
        }
    std::fclose(f);
}

// ------------------------------------------------------------------ display timing
// The guest's vblank comes from a steady 60 Hz host timer; nothing the display
// does can delay it (a compositor hiccup must never cost a game frame). When the
// window's display refreshes at a whole multiple of 60 Hz (60/120/180/240) and
// presents measurably keep pace with it, the timer is phase-locked to the display:
// every k-th refresh reports its time, and the timer's phase and period are nudged
// toward it, so each game frame lands on its own refresh without drift. Missing
// reports just leave the timer free-running. VC3_VSYNC=0 / VC3_VSYNC_LOCK=0 turn
// this off.
std::atomic<bool> g_vs_locked{false};
std::atomic<s64> g_vs_report_ns{0};  // time of the latest locked refresh (steady clock, ns)
std::atomic<u64> g_vs_reports{0};
// Like the CRTC, the picture shown during a refresh is latched at vblank: the
// window presents the buffer that was being scanned out at the last guest vblank.
std::atomic<u64> g_display_seq{0};
u32 g_display_start = 0;             // under g.m

s64 to_ns(Clock::time_point t) { return std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count(); }

void vblank_thread() {
    constexpr double kPeriod = 1e9 / 60.0;
    double period = kPeriod;
    auto next = Clock::now();
    u64 seen_reports = 0;
    for (;;) {
        // Phase-locked loop: steer the tick to fall ~1 ms after a display refresh.
        if (g_vs_locked.load(std::memory_order_relaxed)) {
            const u64 r = g_vs_reports.load(std::memory_order_acquire);
            if (r != seen_reports) {
                seen_reports = r;
                const double target = double(g_vs_report_ns.load(std::memory_order_relaxed)) + 1e6;
                double err = std::fmod(target - double(to_ns(next)), period);  // where the next tick should move
                if (err > period / 2) err -= period;
                if (err < -period / 2) err += period;
                next += std::chrono::nanoseconds(s64(err * 0.1));
                period = std::clamp(period + err * 0.005, kPeriod * 0.99, kPeriod * 1.01);
            }
        } else {
            period += (kPeriod - period) * 0.01;  // relax back to the nominal rate
        }
        next += std::chrono::nanoseconds(s64(period));
        const auto now = Clock::now();
        if (now > next + std::chrono::milliseconds(50)) next = now;  // fell far behind: resync
        std::this_thread::sleep_until(next);
        std::lock_guard lk(g.m);
        ++g.frames;
        g_display_start = g.pcrtc_start;
        g_display_seq.fetch_add(1, std::memory_order_release);
        {
            // VC3_TRACE_HITCH: the picture changed this vblank if the game switched
            // display buffers or a frame was written into one (copy-swap scenes).
            static u32 last_start = 0;
            static u64 last_writes = 0;
            const u64 w = pgraph::glb::ram_writes();
            if (g.pcrtc_start != last_start || w != last_writes) note_frame();
            last_start = g.pcrtc_start;
            last_writes = w;
        }
        {
            static u64 last_flips = 0;
            const u64 f = g.flips.load();
            if (f == last_flips) ++(g_pusher_busy ? g_vb_missed_busy : g_vb_missed_idle);
            last_flips = f;
        }
        static const bool by_flip = std::getenv("VC3_SCRIPT_BY_FLIP") != nullptr;
        if (!by_flip) io::script_tick(g.frames);
        vc3::per_vblank();
        profiler::vblank(long(g.frames));
        maybe_screenshot();
        static const bool fps = std::getenv("VC3_FPS") != nullptr;
        if (fps && g.frames % 300 == 0) {
            static u64 last = 0;
            const u64 f = g.flips.load();
            std::fprintf(stderr, "[nv2a] %.1f fps, gpu thread busy %.0f%%, missed vblanks: gpu busy %llu / idle %llu (%s per 5 s) put=%08x get=%08x pmc=%08x/%08x pgraph=%08x/%08x pcrtc=%x/%x\n",
                         double(f - last) / 5.0, double(g_gpu_busy_ns.exchange(0)) / 5e7, (unsigned long long)g_vb_missed_busy,
                         (unsigned long long)g_vb_missed_idle, pgraph::profile_report().c_str(), g.dma_put.load(), g.dma_get.load(),
                         g.pmc_intr, g.pmc_intr_en, g.pgraph_intr, g.pgraph_intr_en, g.pcrtc_intr, g.pcrtc_intr_en);
            last = f;
            g_vb_missed_busy = g_vb_missed_idle = 0;
        }
        g.pcrtc_intr |= 1;
        update_irq();
        if (trace() && g.frames % 3000 == 0) {
            std::fprintf(stderr, "[nv2a] frame %llu: put=%08x get=%08x methods:", (unsigned long long)g.frames, g.dma_put.load(), g.dma_get.load());
            std::lock_guard clk(g.counts_m);
            for (const auto& [k, v] : g.method_counts)
                std::fprintf(stderr, " %02x:%04x=%llu", k.first, k.second, (unsigned long long)v);
            std::fprintf(stderr, "\n");
        }
    }
}

// ------------------------------------------------------------------ registers
u32 scanline() {
    const auto us = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - g_epoch).count();
    return u32((us % 16667) * 525 / 16667);  // 525 lines per frame, vblank after line 480
}

u32 reg_read(void*, u32 addr, unsigned size) {
    std::lock_guard lk(g.m);
    const u32 off = addr - kBase;
    const u32 r = off & ~3u;
    u32 v = 0;
    switch (r) {
        case PMC + 0x000: v = 0x02A000A2; break;  // BOOT_0: NV2A A2
        case PMC + 0x100: v = g.pmc_intr; break;
        case PMC + 0x140: v = g.pmc_intr_en; break;
        case PFIFO + 0x400: v = 0x10; break;   // RUNOUT_STATUS: empty
        case PFIFO + 0x1214: v = 0x10; break;  // CACHE1_STATUS: empty
        case PFIFO + 0x1240: v = g.dma_put; break;
        case PFIFO + 0x1244: v = g.dma_get; break;
        case PFIFO + 0x124C: v = g.subroutine.load(); break;  // CACHE1_DMA_SUBROUTINE
        case PTIMER + 0x400: case PTIMER + 0x410: {
            const u64 ns = u64(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - g_epoch).count());
            v = r == PTIMER + 0x400 ? u32(ns) & ~0x1Fu : u32(ns >> 32) & 0x1FFFFFFF;
            break;
        }
        case PFB + 0x200: v = 3; break;               // CFG0
        case PFB + 0x20C: v = mem::kRamSize; break;  // CSTATUS: memory size
        case PGRAPH + 0x100: v = g.pgraph_intr; break;
        case PGRAPH + 0x140: v = g.pgraph_intr_en; break;
        case PGRAPH + 0x700: v = 0; break;  // STATUS: idle
        case PGRAPH + 0x710:  // SURFACE: flip queue
            v = (g.regs[r] & 0x888FFFFFu) | (g.flip_write & 7) << 20 | (g.flip_read & 7) << 24 | (g.flip_modulo & 7) << 28;
            break;
        case PCRTC + 0x100: v = g.pcrtc_intr; break;
        case PCRTC + 0x140: v = g.pcrtc_intr_en; break;
        case PCRTC + 0x800: v = g.pcrtc_start; break;
        case PCRTC + 0x808: {  // RASTER: line + vblank flag
            const u32 line = scanline();
            v = line | (line >= 480 ? 0x10000 : 0);
            break;
        }
        case USER + 0x40: v = g.dma_put; break;
        case USER + 0x44: v = g.dma_get; break;
        case USER + 0x48: v = g.ref.load(); break;
        default:
            if (r == PRMCIO + 0x3D8) {  // VGA input status 1 (0x3DA): bit 3 vblank, bit 0 display disabled
                const u32 line = scanline();
                v = (line >= 480 ? 0x08080808u : 0) | 0x01010101u;
            } else {
                v = g.regs[r];
            }
    }
    if (trace()) std::fprintf(stderr, "[nv2a] r %06x -> %08x\n", off, v);
    if (size < 4) v = (v >> (8 * (off & 3))) & ((1u << (8 * size)) - 1);
    return v;
}

void reg_write(void*, u32 addr, u32 v, unsigned size) {
    std::lock_guard lk(g.m);
    const u32 off = addr - kBase;
    const u32 r = off & ~3u;
    if (size < 4) {  // merge partial writes into the register file
        const u32 shift = 8 * (off & 3), mask = ((1u << (8 * size)) - 1) << shift;
        v = (g.regs[r] & ~mask) | ((v << shift) & mask);
    }
    if (trace()) std::fprintf(stderr, "[nv2a] w %06x = %08x\n", off, v);
    g.regs[r] = v;
    switch (r) {
        case PMC + 0x100: g.pmc_intr &= ~v; update_irq(); break;
        case PMC + 0x140: g.pmc_intr_en = v; update_irq(); break;
        case PMC + 0x200:  // ENABLE: resetting PFIFO/PGRAPH clears the get pointer
            if (v & 0x1100) g.dma_get = 0;
            break;
        case PFIFO + 0x1240: g.dma_put = v; g.cv.notify_all(); break;
        case PFIFO + 0x1244: g.dma_get = v; break;
        case PGRAPH + 0x100: g.pgraph_intr &= ~v; update_irq(); g.cv.notify_all(); break;
        case PGRAPH + 0x140: g.pgraph_intr_en = v; update_irq(); break;
        case PGRAPH + 0x720: g.fifo_access = v & 1; g.cv.notify_all(); break;
        case PGRAPH + 0x710:  // SURFACE
            g.flip_write = (v >> 20) & 7;
            g.flip_read = (v >> 24) & 7;
            g.flip_modulo = (v >> 28) & 7;
            g.cv.notify_all();
            break;
        case PGRAPH + 0x71C:  // INCREMENT: bit 1 advances READ_3D (D3D's vblank handler, after a flip)
            if ((v & 2) && g.flip_modulo) g.flip_read = (g.flip_read + 1) % g.flip_modulo;
            g.cv.notify_all();
            break;
        case PCRTC + 0x100: g.pcrtc_intr &= ~v; update_irq(); break;
        case PCRTC + 0x140: g.pcrtc_intr_en = v; update_irq(); break;
        case PCRTC + 0x800: g.pcrtc_start = v; pgraph::glb::note_scanout(v); break;
        case PFB + 0x410: g.regs[r] = v & ~0x10000u; break;  // WBC: write-back cache flush completes at once
        case USER + 0x40: g.dma_put = v; g.cv.notify_all(); break;
        case USER + 0x44: g.dma_get = v; break;
        default: break;
    }
}

// ------------------------------------------------------------------ display
// The window draws with its own GL context, which shares textures with the GPU
// thread's: rendered frames go from GPU memory to the window without a round trip
// through guest RAM (CPU-drawn frames are uploaded from RAM).
SDL_GLContext g_glctx = nullptr;
bool g_vulkan = false;  // the window presents through Vulkan (vk_present.cpp)
GLuint g_ram_tex = 0, g_cur_tex = 0, g_read_fbo = 0;
int g_cur_w = 0, g_cur_h = 0;
GLuint g_src_tex = 0;  // picture shown by the next present
int g_src_w = 640, g_src_h = 480;
bool g_src_from_ram = true;
// The current picture as RGBA bytes, g_src_w x g_src_h (screenshots).
void read_picture(std::vector<u8>& rgba) {
#ifdef VC3_HAVE_VULKAN
    if (g_vulkan) {
        int w = 0, h = 0;
        if (!vkp::read_picture(rgba, w, h)) rgba.assign(size_t(g_src_w) * size_t(g_src_h) * 4, 0);
        for (size_t i = 0; i + 3 < rgba.size(); i += 4) std::swap(rgba[i], rgba[i + 2]);  // BGRA -> RGBA
        return;
    }
#endif
    rgba.resize(size_t(g_src_w) * size_t(g_src_h) * 4);
    glGetTextureImage(g_src_tex, 0, GL_RGBA, GL_UNSIGNED_BYTE, GLsizei(rgba.size()), rgba.data());
}

}  // namespace

void init_hardware() {
    if (!mem::map_ram_alias(kBase + PRAMIN, kRaminRamOffset, kRaminSize))
        std::fprintf(stderr, "[nv2a] failed to map PRAMIN\n");
    // Everything except PRAMIN goes through the trap handler.
    mmio::register_range(kBase, PRAMIN, "nv2a", reg_read, reg_write);
    mmio::register_range(kBase + USER, 0x01000000 - USER, "nv2a-user", reg_read, reg_write);
    pgraph::init(pgraph::Host{g.kelvin.data(), [](u32 inst) { return dma_base(inst); },
                              [](u32 inst) { return inst ? ramin32(inst + 4) + 1 : 0u; }});
    std::thread(gpu_thread).detach();
    std::thread(vblank_thread).detach();
}

void set_scanout(u32 addr) {
    std::lock_guard lk(g.m);
    g.pcrtc_start = addr;
    pgraph::glb::note_scanout(addr);
}

namespace {
// Display refresh state, updated about once a second from the window's display.
SDL_Window* g_window = nullptr;
bool g_vsync = true;
double g_refresh = 0;  // Hz
int g_vs_k = 0;        // refreshes per guest vblank if lockable, else 0

void update_display_mode() {
    const SDL_DisplayMode* m = g_window ? SDL_GetCurrentDisplayMode(SDL_GetDisplayForWindow(g_window)) : nullptr;
    const double r = m ? double(m->refresh_rate) : 0;
    if (r == g_refresh) return;
    g_refresh = r;
    const long k = std::lround(r / 60.0);
    // A multiple of 60 Hz (+-1 Hz): one guest vblank every k refreshes (the game runs at
    // the display's rate, e.g. 59.94). Any faster display: guest vblanks fall on
    // refreshes picked by an accumulator, averaging exactly 60 Hz (e.g. 144 Hz: 2,3,2,3,2...),
    // so the cadence is fixed instead of depending on how two free clocks line up.
    static const bool no_lock = std::getenv("VC3_VSYNC_LOCK") && !std::strcmp(std::getenv("VC3_VSYNC_LOCK"), "0");
    g_vs_k = g_vsync && !no_lock && k >= 1 && k <= 4 && std::fabs(r / double(k) - 60.0) < 1.0 ? int(k) : 0;
    // Presents wait for vsync only where that is needed: to measure the refresh for
    // the lock, or on platforms whose windows could tear. A Wayland compositor never
    // shows a torn window, so there a non-blocking present simply hands it the newest
    // picture, and a busy compositor can't hold the present loop up.
    const bool wayland = !std::strcmp(SDL_GetCurrentVideoDriver(), "wayland");
    const bool block = g_vsync && (g_vs_k || !wayland);
#ifdef VC3_HAVE_VULKAN
    if (g_vulkan) vkp::set_vsync(block);
    else
#endif
        SDL_GL_SetSwapInterval(block ? 1 : 0);
    std::fprintf(stderr, "[present] display %dx%d @ %.2f Hz: %s, %s presents\n", m ? m->w : 0, m ? m->h : 0, r,
                 g_vs_k ? "game timer phase-locked to vsync when presents keep pace" : "game on its own 60 Hz timer",
                 block ? "vsync-paced" : "non-blocking");
    if (!g_vs_k) g_vs_locked = false;
}

// Called after each present while lockable: measure the real refresh interval and
// lock the guest vblank to it only while presents keep pace with the display.
void on_refresh(Clock::time_point now) {
    static Clock::time_point last{};
    static double ema = 0;
    static int good = 0, bad = 0;
    static u64 count = 0;
    const double expected = 1000.0 / g_refresh;
    const double dt = std::chrono::duration<double, std::milli>(now - last).count();
    last = now;
    if (dt > 100.0) { ema = expected; good = bad = 0; return; }
    ema = ema * 0.95 + dt * 0.05;
    const bool ok = std::fabs(ema - expected) < 0.05 * expected;
    good = ok ? good + 1 : 0;
    bad = ok ? 0 : bad + 1;
    if (!g_vs_locked && good > 120) {
        g_vs_locked = true;
        std::fprintf(stderr, "[present] phase-locked to vsync (%.3f ms refresh)\n", ema);
    } else if (g_vs_locked && bad > 60) {
        g_vs_locked = false;
        std::fprintf(stderr, "[present] vsync lock released (%.3f ms refresh)\n", ema);
    }
    // Only refreshes that came on time are reported (a late one has no useful phase).
    if (g_vs_locked && ++count % u64(g_vs_k) == 0 && dt < 1.5 * expected) {
        g_vs_report_ns.store(to_ns(now), std::memory_order_relaxed);
        g_vs_reports.fetch_add(1, std::memory_order_release);
    }
}

}  // namespace

bool wants_vulkan_window() {
#ifdef VC3_HAVE_VULKAN
    const char* r = std::getenv("VC3_RENDERER");
    return r && !std::strcmp(r, "vulkan");
#else
    return false;
#endif
}

bool init(SDL_Window* window) {
    g_window = window;
#ifdef VC3_HAVE_VULKAN
    if (wants_vulkan_window()) {
        const char* vs = std::getenv("VC3_VSYNC");
        g_vsync = !(vs && !std::strcmp(vs, "0"));
        if (!vkp::init(window)) {
            std::fprintf(stderr, "[present] Vulkan presentation unavailable\n");
            return false;
        }
        g_vulkan = true;
        update_display_mode();
        return true;
    }
#endif
    g_glctx = SDL_GL_CreateContext(window);
    if (!g_glctx) return false;
    SDL_GL_MakeCurrent(window, g_glctx);
    const char* vs = std::getenv("VC3_VSYNC");
    g_vsync = !(vs && !std::strcmp(vs, "0"));
    glCreateTextures(GL_TEXTURE_2D, 1, &g_ram_tex);
    glTextureStorage2D(g_ram_tex, 1, GL_RGBA8, 640, 480);
    glCreateFramebuffers(1, &g_read_fbo);
    g_src_tex = g_ram_tex;
    std::fprintf(stderr, "[present] OpenGL %s on %s, vsync %s\n", reinterpret_cast<const char*>(glGetString(GL_RENDERER)),
                 SDL_GetCurrentVideoDriver(), g_vsync ? "on" : "off");
    // Let the GPU thread share this context's objects (it creates its context later).
    if (std::getenv("VC3_NO_SHARE") == nullptr)
        pgraph::glb::set_share(SDL_EGL_GetCurrentDisplay(), SDL_EGL_GetCurrentConfig(), g_glctx);
    update_display_mode();
    return true;
}

namespace {
// The picture's rectangle in the window (its own aspect ratio, letterboxed).
void picture_rect(float w, float h, float& x, float& y, float& pw, float& ph) {
    const float aspect = float(g_src_w) / float(std::max(1, g_src_h));
    const float sc = std::min(w / aspect, h);
    pw = aspect * sc;
    ph = sc;
    x = (w - pw) / 2;
    y = (h - ph) / 2;
}
}  // namespace

void to_screen_coords(SDL_Event& e) {
    if (!g_window) return;
    int w = 0, h = 0;
    SDL_GetWindowSize(g_window, &w, &h);
    float x0, y0, pw, ph;
    picture_rect(float(w), float(h), x0, y0, pw, ph);
    // The game's 640x480 screen is the middle 4:3 part of a widescreen picture.
    const float gw = std::min(pw, ph * 4.0f / 3.0f), gx = x0 + (pw - gw) / 2;
    auto map = [&](float& x, float& y) {
        x = (x - gx) * 640.0f / gw;
        y = (y - y0) * 480.0f / ph;
    };
    if (e.type == SDL_EVENT_MOUSE_MOTION) map(e.motion.x, e.motion.y);
    else if (e.type == SDL_EVENT_MOUSE_BUTTON_DOWN || e.type == SDL_EVENT_MOUSE_BUTTON_UP) map(e.button.x, e.button.y);
}

void window_to_game(float nx, float ny, float& gx, float& gy) {
    int w = 640, h = 480;
    if (g_window) SDL_GetWindowSize(g_window, &w, &h);
    float x0, y0, pw, ph;
    picture_rect(float(w), float(h), x0, y0, pw, ph);
    const float gw = std::min(pw, ph * 4.0f / 3.0f), gx0 = x0 + (pw - gw) / 2;
    gx = (nx * float(w) - gx0) * 640.0f / gw;
    gy = (ny * float(h) - y0) * 480.0f / ph;
}

void toggle_fullscreen() {
    if (!g_window) return;
    if (SDL_GetWindowFlags(g_window) & SDL_WINDOW_FULLSCREEN) {
        SDL_SetWindowFullscreen(g_window, false);
        return;
    }
    // Borderless at the desktop mode if its refresh suits the game (a multiple of
    // 60 Hz); otherwise try an exclusive mode at the same resolution that does,
    // preferring the highest such rate. (Wayland compositors don't let programs
    // change the refresh rate; there, adaptive sync is what smooths 60 fps.)
    const SDL_DisplayID d = SDL_GetDisplayForWindow(g_window);
    const SDL_DisplayMode* desk = SDL_GetDesktopDisplayMode(d);
    auto lockable = [](float r) {
        const long k = std::lround(r / 60.0);
        return k >= 1 && k <= 4 && std::fabs(double(r) / double(k) - 60.0) < 1.0;
    };
    const SDL_DisplayMode* best = nullptr;
    int n = 0;
    SDL_DisplayMode** modes = nullptr;
    if (desk && !lockable(desk->refresh_rate)) {
        modes = SDL_GetFullscreenDisplayModes(d, &n);
        for (int i = 0; i < n; ++i)
            if (modes[i]->w == desk->w && modes[i]->h == desk->h && lockable(modes[i]->refresh_rate) &&
                (!best || modes[i]->refresh_rate > best->refresh_rate))
                best = modes[i];
    }
    SDL_SetWindowFullscreenMode(g_window, best);
    SDL_SetWindowFullscreen(g_window, true);
    if (best) std::fprintf(stderr, "[present] fullscreen mode %dx%d @ %.2f Hz\n", best->w, best->h, double(best->refresh_rate));
    SDL_free(modes);
}

namespace {
// Pick the picture for scan-out address `start`: the GPU's published frame if there
// is one and the CPU hasn't overwritten that memory since; else guest RAM.
u64 g_pics_gpu = 0, g_pics_ram = 0, g_pics_cpu_wrote = 0, g_pics_none = 0;
void select_picture(u32 start) {
    pgraph::glb::DisplayFrame f;
#ifdef VC3_HAVE_VULKAN
    if (g_vulkan) {
        if (pgraph::glb::display_frame(start, f)) {
            const bool cpu_wrote = f.ram_sparse && pgraph::glb::sparse_ram_hash(start, f.bytes) != f.ram_sparse;
            if (!cpu_wrote) {
                pgraph::glb::display_done(f.slot, vkp::take_texture(f.tex, f.w, f.h, f.fence));
                g_src_w = f.w;
                g_src_h = f.h;
                g_src_from_ram = false;
                ++g_pics_gpu;
                return;
            }
            pgraph::glb::display_done(f.slot, 0);
            ++g_pics_cpu_wrote;
        } else {
            ++g_pics_none;
        }
        if (start && start + 640 * 480 * 4 <= mem::kRamSize) {
            std::lock_guard lk(pgraph::glb::ram_write_mutex());
            vkp::take_ram(host_ptr(start), 640, 480);
        }
        ++g_pics_ram;
        g_src_w = 640;
        g_src_h = 480;
        g_src_from_ram = true;
        return;
    }
#endif
    if (pgraph::glb::display_frame(start, f)) {
        if (f.fence) {  // the GL device's syncs are GLsync objects
            GLsync fence = reinterpret_cast<GLsync>(uintptr_t(f.fence));
            glWaitSync(fence, 0, GL_TIMEOUT_IGNORED);
            glDeleteSync(fence);
        }
        const bool cpu_wrote = f.ram_sparse && pgraph::glb::sparse_ram_hash(start, f.bytes) != f.ram_sparse;
        if (!cpu_wrote) {
            if (g_cur_w != f.w || g_cur_h != f.h) {
                if (g_cur_tex) glDeleteTextures(1, &g_cur_tex);
                glCreateTextures(GL_TEXTURE_2D, 1, &g_cur_tex);
                glTextureStorage2D(g_cur_tex, 1, GL_RGBA8, f.w, f.h);
                g_cur_w = f.w;
                g_cur_h = f.h;
            }
            glCopyImageSubData(f.tex.id, GL_TEXTURE_2D, 0, 0, 0, 0, g_cur_tex, GL_TEXTURE_2D, 0, 0, 0, 0, f.w, f.h, 1);
            GLsync done = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
            glFlush();
            pgraph::glb::display_done(f.slot, dev::Sync(reinterpret_cast<uintptr_t>(done)));
            g_src_tex = g_cur_tex;
            g_src_w = f.w;
            g_src_h = f.h;
            g_src_from_ram = false;
            ++g_pics_gpu;
            return;
        }
        pgraph::glb::display_done(f.slot, 0);
        ++g_pics_cpu_wrote;
    } else {
        ++g_pics_none;
    }
    if (start && start + 640 * 480 * 4 <= mem::kRamSize) {
        std::lock_guard lk(pgraph::glb::ram_write_mutex());
        glTextureSubImage2D(g_ram_tex, 0, 0, 0, 640, 480, GL_BGRA, GL_UNSIGNED_BYTE, host_ptr(start));
    }
    ++g_pics_ram;
    g_src_tex = g_ram_tex;
    g_src_w = 640;
    g_src_h = 480;
    g_src_from_ram = true;
}

// Crosshairs over the picture (VC3_CROSSHAIR=on|off), for players in a game only: rectangles in window pixels,
// top-left origin, drawn after the picture (outline first).
using dev::OverlayRect;
std::vector<OverlayRect> crosshair_rects(int wh, float x, float y, float pw, float ph) {
    std::vector<OverlayRect> out;
    static const std::string mode = std::getenv("VC3_CROSSHAIR") ? std::getenv("VC3_CROSSHAIR") : "on";
    if (mode == "off") return out;
    const float gw = std::min(pw, ph * 4.0f / 3.0f), gx0 = x + (pw - gw) / 2;
    // VC3_CROSSHAIR_SIZE: scale (1 = 100%).
    static const float size = [] {
        const char* e = std::getenv("VC3_CROSSHAIR_SIZE");
        const float v = e ? float(std::atof(e)) : 1.0f;
        return std::clamp(v > 0 ? v : 1.0f, 0.25f, 4.0f);
    }();
    // 100%: 5% of the window height across (the screen's height when fullscreen).
    const float s = float(wh) * size;
    const int arm = std::max(3, int(s * 0.0183f + 0.5f)), th = std::max(2, int(s * 0.0045f + 0.5f)),
              gap = std::max(2, int(s * 0.0067f + 0.5f));
    for (int p = 0; p < 2; ++p) {
        float ax, ay;
        if (!io::input::crosshair(p, ax, ay) || !vc3::player_in_game(p)) continue;
        const int cx = int(gx0 + ax * gw / 640.0f), cy = int(y + ay * ph / 480.0f);
        const uint8_t r = p == 0 ? 255 : 40, g = p == 0 ? 40 : 140, b = p == 0 ? 40 : 255;  // P1 red, P2 blue
        const int arms[4][4] = {{cx - gap - arm, cy - th / 2, arm, th}, {cx + gap, cy - th / 2, arm, th},
                                {cx - th / 2, cy - gap - arm, th, arm}, {cx - th / 2, cy + gap, th, arm}};
        for (const auto& a : arms) out.push_back({a[0] - 1, a[1] - 1, a[2] + 2, a[3] + 2, 0, 0, 0});  // outline
        for (const auto& a : arms) out.push_back({a[0], a[1], a[2], a[3], r, g, b});
        const int d = std::max(2, th);
        out.push_back({cx - d / 2, cy - d / 2, d, d, r, g, b});  // centre dot
    }
    return out;
}

// Everything drawn over the picture in a w x h window: crosshairs, then the Sinden lightgun
// border (VC3_SINDEN_BORDER=percent, 0 off): a white frame at the window's edges, that
// percent of the window height thick. The gun's camera finds the screen by it; the gun then
// moves the system mouse over the whole screen, which maps to the game like any mouse.
std::vector<OverlayRect> overlay_rects(int w, int h, float x, float y, float pw, float ph) {
    std::vector<OverlayRect> out = crosshair_rects(h, x, y, pw, ph);
    static const float pct = [] {
        const char* e = std::getenv("VC3_SINDEN_BORDER");
        const float v = e ? float(std::atof(e)) : 0.0f;
        return v > 0 ? std::clamp(v, 0.5f, 10.0f) : 0.0f;
    }();
    if (pct > 0) {
        const int t = std::max(2, int(float(h) * pct / 100.0f + 0.5f));
        out.push_back({0, 0, w, t, 255, 255, 255});
        out.push_back({0, h - t, w, t, 255, 255, 255});
        out.push_back({0, t, t, h - 2 * t, 255, 255, 255});
        out.push_back({w - t, t, t, h - 2 * t, 255, 255, 255});
    }
    return out;
}

void draw_picture() {
    if (g_vulkan) return;  // vkp::present draws and presents in one go
    int w = 0, h = 0;
    SDL_GetWindowSizeInPixels(g_window, &w, &h);
    float x, y, pw, ph;
    picture_rect(float(w), float(h), x, y, pw, ph);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, w, h);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glNamedFramebufferTexture(g_read_fbo, GL_COLOR_ATTACHMENT0, g_src_tex, 0);
    // Texture row 0 is the top line; the window's row 0 is the bottom.
    glBlitNamedFramebuffer(g_read_fbo, 0, 0, 0, g_src_w, g_src_h, GLint(x), GLint(y + ph), GLint(x + pw), GLint(y),
                           GL_COLOR_BUFFER_BIT, GL_LINEAR);
    // Crosshairs, Sinden border: solid rectangles (scissored clears; GL's window rows run bottom-up).
    glEnable(GL_SCISSOR_TEST);
    for (const OverlayRect& r : overlay_rects(w, h, x, y, pw, ph)) {
        glScissor(r.x, h - r.y - r.h, r.w, r.h);
        glClearColor(r.r / 255.0f, r.g / 255.0f, r.b / 255.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
    }
    glDisable(GL_SCISSOR_TEST);
}
}  // namespace

// Show the latched scan-out buffer (assumed 640x480 X8R8G8B8). Without the vsync
// lock, presents happen once per guest vblank (so adaptive-sync displays follow
// the game's 60 Hz); on a lockable display, every refresh is presented (the
// refreshes themselves drive the guest vblank). Returns false if nothing was shown.
std::string g_capture_shot;  // GL: where the next present saves the window (F9 capture)
void request_window_shot(const std::string& path) {
#ifdef VC3_HAVE_VULKAN
    if (g_vulkan) return vkp::request_shot(path);
#endif
    g_capture_shot = path;
}

bool present() {
    if (!g_glctx && !g_vulkan) return false;
    static u64 shown_seq = 0;
    static Clock::time_point last_present{}, last_mode_check{};
    const auto now = Clock::now();
    if (now - last_mode_check > std::chrono::seconds(1)) {
        last_mode_check = now;
        update_display_mode();
    }
    const u64 seq = g_display_seq.load(std::memory_order_acquire);
    const bool fresh = seq != shown_seq;
    if (!g_vs_k && !fresh && now - last_present < std::chrono::milliseconds(100)) return false;
    last_present = now;
    static u64 uploads = 0, skipped = 0;
    static const bool tp = std::getenv("VC3_TRACE_PRESENT") != nullptr;
    static double last_upload_ms = 0, last_present_ms = 0;
    static Clock::time_point last_done{};
    if (fresh && shown_seq && seq - shown_seq > 1) {
        skipped += seq - shown_seq - 1;
        if (tp)
            std::fprintf(stderr, "[present] skipped %llu picture(s) at vblank %llu: %.1f ms since the last present returned "
                                 "(it took: upload %.1f ms, present %.1f ms)\n",
                         (unsigned long long)(seq - shown_seq - 1), (unsigned long long)seq,
                         std::chrono::duration<double, std::milli>(now - last_done).count(), last_upload_ms, last_present_ms);
    }
    const auto t_up = Clock::now();
    if (fresh) {
        shown_seq = seq;
        u32 start;
        {
            std::lock_guard lk(g.m);
            start = g_display_start & (mem::kRamSize - 1);
        }
        select_picture(start);
        ++uploads;
    }
    const auto t_pr = Clock::now();
    draw_picture();
    // VC3_WINDOW_SHOT=from:to:step: save the picture (render resolution) every `step`
    // vblanks to work/wshots/VVVVV_fFFFFF.ppm (vblank, guest flip) -- for surveying a run.
    if (static const char* ws = std::getenv("VC3_WINDOW_SHOT"); ws && std::strchr(ws, ':') && fresh) {
        static long from = std::atol(ws), to = std::atol(std::strchr(ws, ':') + 1),
                    step = std::strchr(std::strchr(ws, ':') + 1, ':') ? std::atol(std::strchr(std::strchr(ws, ':') + 1, ':') + 1) : 60;
        static long next = from;
        // VC3_SHOT_BY_FLIP=1: from/to/step count guest flips instead of vblanks (the same
        // game frames whatever the host speed; tests/render/compare.py).
        static const bool by_flip = std::getenv("VC3_SHOT_BY_FLIP") != nullptr;
        const long at = by_flip ? long(g.flips.load()) : long(seq);
        if (at >= next && at <= to) {
            next = at + std::max(1L, step);
            std::filesystem::create_directories("work/wshots");
            std::vector<u8> src;
            read_picture(src);
            char name[64];
            std::snprintf(name, sizeof name, "work/wshots/%05ld_f%05llu.ppm", long(seq), (unsigned long long)g.flips.load());
            if (std::FILE* f = std::fopen(name, "wb")) {
                std::fprintf(f, "P6\n%d %d\n255\n", g_src_w, g_src_h);
                for (size_t i = 0; i < size_t(g_src_w) * size_t(g_src_h); ++i) std::fwrite(&src[i * 4], 1, 3, f);
                std::fclose(f);
            }
        }
    }
    // A capture (F9): the window as shown, on GL here (Vulkan: in its presenter).
    if (!g_capture_shot.empty() && !g_vulkan) {
        int w = 0, h = 0;
        SDL_GetWindowSizeInPixels(g_window, &w, &h);
        std::vector<u8> px(size_t(w) * size_t(h) * 3);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, px.data());
        if (std::FILE* f = std::fopen(g_capture_shot.c_str(), "wb")) {
            std::fprintf(f, "P6\n%d %d\n255\n", w, h);
            for (int y = h - 1; y >= 0; --y) std::fwrite(&px[size_t(y) * size_t(w) * 3], 1, size_t(w) * 3, f);
            std::fclose(f);
        }
        g_capture_shot.clear();
    }
    // VC3_WINDOW_SHOT=N: save what the window shows at guest vblank N (work/window_shot.ppm,
    // window resolution) -- checks the display path, unlike VC3_SHOT_EVERY (guest RAM).
    if (static const long shot_at = std::getenv("VC3_WINDOW_SHOT") && !std::strchr(std::getenv("VC3_WINDOW_SHOT"), ':')
                                        ? std::atol(std::getenv("VC3_WINDOW_SHOT")) : -1;
        shot_at >= 0 && fresh && seq >= u64(shot_at)) {
        static bool done = false;
        if (!done && !g_vulkan) {
            done = true;
            int w = 0, h = 0;
            SDL_GetWindowSizeInPixels(g_window, &w, &h);
            std::vector<u8> px(size_t(w) * size_t(h) * 3);
            glPixelStorei(GL_PACK_ALIGNMENT, 1);
            glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, px.data());
            if (std::FILE* f = std::fopen("work/window_shot.ppm", "wb")) {
                std::fprintf(f, "P6\n%d %d\n255\n", w, h);
                for (int y = h - 1; y >= 0; --y) std::fwrite(&px[size_t(y) * size_t(w) * 3], 1, size_t(w) * 3, f);
                std::fclose(f);
            }
            // ...and the picture itself at render resolution (work/window_src.ppm).
            std::vector<u8> src;
            read_picture(src);
            if (std::FILE* f = std::fopen("work/window_src.ppm", "wb")) {
                std::fprintf(f, "P6\n%d %d\n255\n", g_src_w, g_src_h);
                for (size_t i = 0; i < size_t(g_src_w) * size_t(g_src_h); ++i) std::fwrite(&src[i * 4], 1, 3, f);
                std::fclose(f);
            }
            std::fprintf(stderr, "[present] (guest flip %llu at this vblank)\n", (unsigned long long)g.flips.load());
            std::fprintf(stderr, "[present] wrote work/window_shot.ppm (%dx%d, source %dx%d %s)\n", w, h, g_src_w, g_src_h,
                         g_src_from_ram ? "from guest RAM" : "from GPU");
        }
    }
#ifdef VC3_HAVE_VULKAN
    if (g_vulkan) {
        int w = 0, h = 0;
        SDL_GetWindowSizeInPixels(g_window, &w, &h);
        float x, y, pw, ph;
        picture_rect(float(w), float(h), x, y, pw, ph);
        vkp::present(x, y, pw, ph, overlay_rects(w, h, x, y, pw, ph));
    } else
#endif
        SDL_GL_SwapWindow(g_window);
    last_done = Clock::now();
    last_upload_ms = std::chrono::duration<double, std::milli>(t_pr - t_up).count();
    last_present_ms = std::chrono::duration<double, std::milli>(last_done - t_pr).count();
    if (g_vs_k) on_refresh(last_done);
    if (tp) {
        static u64 n = 0, base_frames = 0;
        static auto t0 = now;
        ++n;
        if (now - t0 > std::chrono::seconds(5)) {
            const u64 fr = g.frames;
            std::fprintf(stderr, "[present] 5 s: %llu presents, %llu new pictures, %llu skipped, %llu guest vblanks, %s; "
                                 "pictures from GPU %llu, from RAM %llu (no GPU frame %llu, CPU-drawn %llu)\n",
                         (unsigned long long)n, (unsigned long long)uploads, (unsigned long long)skipped,
                         (unsigned long long)(fr - base_frames), g_vs_locked ? "phase-locked" : "free timer",
                         (unsigned long long)g_pics_gpu, (unsigned long long)g_pics_ram, (unsigned long long)g_pics_none,
                         (unsigned long long)g_pics_cpu_wrote);
            g_pics_gpu = g_pics_ram = g_pics_none = g_pics_cpu_wrote = 0;
            n = uploads = skipped = 0;
            base_frames = fr;
            t0 = now;
        }
    }
    return true;
}

void shutdown() {
#ifdef VC3_HAVE_VULKAN
    vkp::shutdown();
#endif
    if (g_glctx) SDL_GL_DestroyContext(g_glctx);
    g_glctx = nullptr;
}

}  // namespace vc3::gpu
