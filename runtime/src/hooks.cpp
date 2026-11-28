// Hooks into lifted guest functions (see config/vc3_hooks.txt).
#include <atomic>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <map>
#include <set>
#include <string>
#include <cstring>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "log.h"
#include "vc3/helpers.h"
#include "vc3/recomp.h"
#include "vc3/memory.h"
#include "io/jvs.h"

#include "gpu/gl_backend.h"

namespace vc3 {

namespace gpu { extern std::atomic<bool> g_fast_stop; }

// Free play (VC3_FREEPLAY=1). The game's credit library keeps its coin settings in a
// 32-byte block at 0x55FB18; byte 7 = 1 is free play (checked when starting and continuing;
// the cabinet's coin settings normally set it). The game copies the block to 0x3A6550,
// whose byte 7 picks "FREEPLAY" over "CREDIT(S)" on screen. Set both every vblank so a
// settings reload can't undo it. (Addresses of the vc3.xbe the code is recompiled from;
// the installer checks its SHA-1.)
// ---- Cheats (VC3_CHEAT_AMMO, VC3_CHEAT_LIVES, VC3_CHEAT_WEAPON), for players in a game.
// Game data (vc3.xbe this build is recompiled from):
//   weapon block per player at 0x3DE3D8 + p * 0xD8: +0x04 flags, +0x58 weapon type,
//     +0x5A magazine, +0x5C reserve (u16, 1000 = unlimited), +0x8C/+0x8D/+0x8E the other
//     weapon (type, magazine, reserve), +0x84 special weapons held in reserve (+0x64/+0x65/
//     +0x66 type, magazine, reserve), +0x90/+0x91/+0x92 a pending pickup
//   weapon table at 0x1E61C0 + type * 16: +0 capacity, +2 starting reserve (u16), +4 fire rate
//   lives at 0x3F3F74 + p * 0xB0 (next to the player state byte, see player_in_game)
//   E.S. (slow motion, the pedal) gauge: s32 at +0x150 of the player's E.S. data, whose
//     address is at 0x3F3FEC + p * 0xB0 (full 4500; drained by fn_0009ae80 while E.S. is on)
namespace {
constexpr uint32_t kWeaponBlock = 0x003DE3D8, kWeaponStride = 0xD8, kWeaponTable = 0x001E61C0;
constexpr uint32_t kLives = 0x003F3F74, kStateStride = 0xB0, kEsPtr = 0x003F3FEC;
constexpr int8_t kHandgun = 7;

int cheat_setting(const char* env) {
    const char* e = std::getenv(env);
    return e ? std::atoi(e) : 0;
}

// The weapon cheat (VC3_CHEAT_WEAPON = weapon type + 1: 2 G36C, 3 MP7, 4 M4 shotgun,
// 5 M629 magnum, 9 SAM-92A missile launcher, 10 LMG249 (6: a 99-round handgun, not offered); 7 is
// the handgun, 0, 1 and 8 unused; 99 = all weapons). Runs in the player's update (hook at
// fn_0007ed30, the game's thread). Once the player has had only the handgun for a moment
// (the mission's set-up done, or the special weapon lost to a hit or used up), it hands over
// an item pickup the way the game's items do: the type in +0x94 and flag 0x4000000; the
// update (at 0x7EE1C) runs the pickup routine fn_0007c310 when the player can act and equips
// it, the handgun becoming the held one.
// All weapons: Change weapon (which the game then doesn't see, io/baseboard.cpp) steps
// through handgun, G36C, MP7, M4, M629, LMG249, SAM-92A: the next one replaces the one in
// hand, full, through the game's queued weapon switch (+0x90 type, +0x91 magazine, +0x92
// reserve, -1 = the weapon's own; flag 0x1000000; applied by the update at 0x7EEB5 with the
// HUD's weapon change), nothing held besides. A hit takes it away; it comes back as above.
constexpr int kWeaponDelay = 30;  // frames with only the handgun first
constexpr int8_t kCycle[] = {kHandgun, 2, 3, 4, 5, 10, 9};
void weapon_cheat(Cpu& c) {
    static const int weapon = cheat_setting("VC3_CHEAT_WEAPON");
    if (weapon <= 0) return;
    const uint32_t w = rd<uint32_t>(c.esp + 4);  // the player's weapon block
    if (w < kWeaponBlock || (w - kWeaponBlock) % kWeaponStride || (w - kWeaponBlock) / kWeaponStride > 1) return;
    const int p = int((w - kWeaponBlock) / kWeaponStride);
    const uint32_t flags = rd<uint32_t>(w + 0x04);
    auto switch_to = [&](int8_t type) {
        wr<int32_t>(w + 0x84, 0);  // nothing held besides
        wr<int8_t>(w + 0x90, type);
        wr<int8_t>(w + 0x91, -1);
        wr<int16_t>(w + 0x92, -1);
        wr<uint32_t>(w + 0x04, rd<uint32_t>(w + 0x04) | 0x1000000u);
    };
    static int sel[2] = {1, 1};  // all weapons: index in kCycle of the chosen one (G36C first)
    static bool was_down[2] = {false, false};
    if (weapon == 99) {
        const bool down = io::state().gun[p].change;
        if (down && !was_down[p] && player_in_game(p) && !(flags & 0x1000000u)) {
            sel[p] = (sel[p] + 1) % int(sizeof kCycle);
            switch_to(kCycle[sel[p]]);
        }
        was_down[p] = down;
        if (!player_in_game(p)) sel[p] = 1;
    }
    static int handgun_only[2] = {0, 0};
    // Only the handgun: it's the current weapon, none is held (+0x84 counts those; a hit's
    // drop routine fn_0007c480 sets it to 0) and no pickup or switch is waiting.
    const bool alone = player_in_game(p) && rd<int8_t>(w + 0x58) == kHandgun && rd<int32_t>(w + 0x84) == 0 &&
                       !(flags & 0x5000000u);
    handgun_only[p] = alone ? handgun_only[p] + 1 : 0;
    if (handgun_only[p] < kWeaponDelay) return;
    handgun_only[p] = 0;
    if (weapon == 99) {
        if (kCycle[sel[p]] != kHandgun) switch_to(kCycle[sel[p]]);  // the chosen one back
        return;
    }
    wr<int8_t>(w + 0x94, int8_t(weapon - 1));
    wr<uint32_t>(w + 0x04, flags | 0x4000000u);
}

void cheats() {
    static const bool ammo = cheat_setting("VC3_CHEAT_AMMO") != 0, lives = cheat_setting("VC3_CHEAT_LIVES") != 0,
                      slowmo = cheat_setting("VC3_CHEAT_SLOWMO") != 0;
    if (!ammo && !lives && !slowmo) return;
    static int32_t es_full[2] = {0, 0};
    if (slowmo)
        for (int p = 0; p < 2; ++p) {
            const uint32_t es = rd<uint32_t>(kEsPtr + uint32_t(p) * kStateStride);
            if (!player_in_game(p) || es < 0x10000 || es >= mem::kRamSize - 0x200) {
                es_full[p] = 0;
                continue;
            }
            const int32_t g = rd<int32_t>(es + 0x150);
            if (g > es_full[p] && g <= 100000) es_full[p] = g;  // the gauge full (4500)
            else if (g < es_full[p]) wr<int32_t>(es + 0x150, es_full[p]);
        }
    static uint8_t start_lives[2] = {0, 0};
    for (int p = 0; p < 2; ++p) {
        if (!player_in_game(p)) {
            start_lives[p] = 0;
            continue;
        }
        const uint32_t w = kWeaponBlock + uint32_t(p) * kWeaponStride;
        const int8_t type = rd<int8_t>(w + 0x58);
        if (type < 0 || type > 10) continue;  // not set up yet
        if (ammo) {
            const uint32_t t = kWeaponTable + uint32_t(rd<int8_t>(w + 0x58)) * 16;
            const uint8_t cap = rd<uint8_t>(t);
            if (rd<uint8_t>(w + 0x5A) < cap) wr<uint8_t>(w + 0x5A, cap);
            const uint16_t res = rd<uint16_t>(t + 2);
            if (rd<uint16_t>(w + 0x5C) < res) wr<uint16_t>(w + 0x5C, res);
        }
        if (lives) {
            const uint32_t l = kLives + uint32_t(p) * kStateStride;
            const uint8_t n = rd<uint8_t>(l);
            if (!start_lives[p]) start_lives[p] = n;
            if (n < start_lives[p]) wr<uint8_t>(l, start_lives[p]);
        }
    }
}
}  // namespace

// Section skipping (testing) is not done yet: requesting the game's GameSceneClear state
// mid-section left it in an empty room without HUD. request_scene_skip() is a no-op for now.
void request_scene_skip() {}

std::atomic<unsigned long> g_vblanks{0};  // per_vblank calls (for logs)

void per_vblank() {
    g_vblanks.fetch_add(1, std::memory_order_relaxed);
    cheats();
    // VC3_FAST_STOP_FRAME=N (testing): end VC3_FAST_UNTIL's fast-forward once the game has
    // been N frames in the current stage ([0x3DDC60]) with someone playing.
    static const long fast_stop_frame = cheat_setting("VC3_FAST_STOP_FRAME");
    if (fast_stop_frame > 0 && (player_in_game(0) || player_in_game(1)) &&
        rd<int32_t>(0x003DDC60) >= fast_stop_frame && !gpu::g_fast_stop.exchange(true))
        std::fprintf(stderr, "[dev] %.0f s: stage frame %d, normal speed from here\n", log::uptime(), rd<int32_t>(0x003DDC60));
    // VC3_RAM_DUMP=N,N,...: save guest RAM 0-16 MiB at those vblanks to work/ramdump/N.bin (dev).
    static const char* dump = std::getenv("VC3_RAM_DUMP");
    static unsigned long vb = 0;
    ++vb;
    // VC3_RAM_POKE=ADDR=VAL@VBLANK,... (hex address, decimal value): write a guest byte at that
    // vblank, or every vblank from it with @VBLANK+ (dev: e.g. hand a player a weapon:
    // 3DE430 = P1's weapon type, see cheats()).
    struct Poke { uint32_t addr; uint8_t val; unsigned long from; bool keep; };
    static const std::vector<Poke> pokes = [] {
        std::vector<Poke> v;
        if (const char* e = std::getenv("VC3_RAM_POKE"))
            for (const char* p = e; *p;) {
                char* end;
                Poke k{uint32_t(std::strtoul(p, &end, 16)), 0, 0, false};
                if (*end == '=') k.val = uint8_t(std::strtoul(end + 1, &end, 10));
                if (*end == '@') k.from = std::strtoul(end + 1, &end, 10);
                if (*end == '+') k.keep = true, ++end;
                v.push_back(k);
                p = *end ? end + 1 : end;
            }
        return v;
    }();
    for (const Poke& k : pokes)
        if (vb == k.from || (k.keep && vb > k.from)) wr<uint8_t>(k.addr, k.val);
    // VC3_TRACE_TASKS=1 (dev): log the game's task list (linked from 0x3D6A60: +0x04 id,
    // +0x08 flags, +0x38 next) whenever it changes.
    static const bool trace_tasks = std::getenv("VC3_TRACE_TASKS") != nullptr;
    if (trace_tasks) {
        static std::string last;
        std::string cur;
        int guard = 0;
        for (uint32_t t = 0x003D6A60; t && t < mem::kRamSize && guard < 64; t = rd<uint32_t>(t + 0x38), ++guard) {
            char b[96];
            std::snprintf(b, sizeof b, " [id %d fl %x cb %x %x %x %x]", rd<int32_t>(t + 0x04), rd<uint32_t>(t + 0x08),
                          rd<uint32_t>(t + 0x18), rd<uint32_t>(t + 0x1C), rd<uint32_t>(t + 0x20), rd<uint32_t>(t + 0x24));
            cur += b;
        }
        if (cur != last) {
            std::fprintf(stderr, "[tasks] vblank %lu:%s\n", vb, cur.c_str());
            last = cur;
        }
    }
    // VC3_RAM_WATCH=ADDR,ADDR,... (hex): log changes of those guest bytes per vblank (dev).
    static const std::vector<uint32_t> watch = [] {
        std::vector<uint32_t> v;
        if (const char* e = std::getenv("VC3_RAM_WATCH"))
            for (const char* p = e; *p;) {
                char* end;
                v.push_back(uint32_t(std::strtoul(p, &end, 16)));
                p = *end ? end + 1 : end;
            }
        return v;
    }();
    static std::vector<int> last(watch.size(), -1);
    for (size_t i = 0; i < watch.size(); ++i)
        if (const int b = rd<uint8_t>(watch[i]); b != last[i]) {
            std::fprintf(stderr, "[watch] vblank %lu %#x: %d -> %d\n", vb, watch[i], last[i], b);
            last[i] = b;
        }
    if (dump) {
        for (const char* p = dump; *p;) {
            char* end;
            const unsigned long n = std::strtoul(p, &end, 10);
            if (n == vb) {
                std::filesystem::create_directories("work/ramdump");
                if (std::FILE* f = std::fopen(("work/ramdump/" + std::to_string(n) + ".bin").c_str(), "wb")) {
                    std::fwrite(host_ptr(0), 1, (std::getenv("VC3_RAM_DUMP_MB") ? std::atoi(std::getenv("VC3_RAM_DUMP_MB")) : 16) * (1u << 20), f);
                    std::fclose(f);
                }
            }
            p = *end ? end + 1 : end;
        }
    }
    static const bool freeplay = [] {
        const char* e = std::getenv("VC3_FREEPLAY");
        return e && std::strcmp(e, "0");
    }();
    if (!freeplay) return;
    wr<uint8_t>(0x0055FB1F, 1);
    wr<uint8_t>(0x003A6557, 1);
}

// ---- Widescreen shooting (issue #10). The game only knows its 640x480 screen: a gun's raw position
// (centred units, [block+0x1C]/[+0x20]) goes through the GUN ADJUSTMENT calibration -- per side of the
// calibrated centre ([+0xB8] x, [+0xBA] y, s16) a slope ([+0xBC] left, [+0xC0] right, [+0xC8] / [+0xC4]
// for y) -- into the aim, [+0x0C] = screen x - 320, [+0x10] = 240 - screen y (fn_0007be20, called by
// fn_0007de30 each frame; the off-screen bit skips it). Everything after works from the aim with the
// view's projection, which the widescreen picture keeps (the middle is the original, the 3D view
// continues into the sides), so an aim beyond +-320 is a shot into the side areas. When a player
// aims into a side area (io::aim_extent), the raw position is set to the one the game's own
// calibration turns into that aim; inside the 4:3 screen nothing changes (calibration as before).
float rd_f32(uint32_t a) {
    const uint32_t u = rd<uint32_t>(a);
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
void wide_aim(Cpu& c) {
    const uint32_t blk = rd<uint32_t>(c.esp + 4);
    if (blk < kWeaponBlock || blk >= kWeaponBlock + 2 * kWeaponStride || (blk - kWeaponBlock) % kWeaponStride) return;
    if (io::aim_extent() <= 0.0f) return;
    const int p = int((blk - kWeaponBlock) / kWeaponStride);
    float ax, ay;
    io::aim_point(p, ax, ay);
    if (ax >= 0.0f && ax < 640.0f) return;  // on the 4:3 screen: the game's own result
    const int want = int(std::lround(ax)) - 320;
    const int cx = int(rd<int16_t>(blk + 0xB8));
    const float left = rd_f32(blk + 0xBC), right = rd_f32(blk + 0xC0);
    // The game: aim = (int)((raw - cx) * slope), the slope by the side of the centre (truncated).
    auto aim_for = [&](int raw) { return int(double(raw - cx) * double(raw < cx ? left : right)); };
    const float slope = want < 0 ? left : right;
    if (!(std::fabs(slope) > 1e-4f)) return;
    int raw = cx + int(std::lround(double(want) / double(slope)));
    for (int d : {0, -1, 1, -2, 2})
        if (aim_for(raw + d) == want) {
            raw += d;
            break;
        }
    wr<int32_t>(blk + 0x1C, raw);
}

// Each player's state byte (P1 0x3F3F68, P2 0x3F4018, 0xB0 apart): 0 not playing,
// 2 playing (from pressing start, mission select included), 8 continue countdown,
// 16 game over, 128 attract-mode demo.
bool player_in_game(int p) {
    const uint8_t st = rd<uint8_t>(p ? 0x003F4018 : 0x003F3F68);
    return st == 2 || st == 8;
}

// ---- The subway trains in the Brand fight area (mission 1, section 0x0E). Each has its own update:
// fn_00094ff0 (ambience channel 2) and fn_00094e30 (channel 1, from section frame 1570 on). Each frame
// it starts its rumble when its "sound on" flag (0x4000) is clear (fn_000b2a80(ch, 1)), then stops it
// again (fn_000b2a80(ch, 0), flag cleared) if the nearest car (fn_0008d9a0: the car closest to the
// camera, if one is within 1000) is over 127 away. Two things go wrong with that:
// - While the train is far, that is a start and a stop every frame: each pair allocates and frees a
//   DSOUND voice (~0.5 ms of the single guest CPU, 60 times a second), and a pair just before the real
//   start can strand that start's voice (see train_update). The hook marks the sound as on while the
//   train is out of range, so the game skips the start; its own range check then clears the flag and
//   posts a stop that finds nothing playing. In range nothing changes.
// - A train is retired (handler 0x8AE90) the frame it passes x 650 -- fn_00094e30 at once, fn_00094ff0
//   after 3000 section frames -- without clearing the sound flag or stopping the sound. If it is still
//   in range then, nothing ever turns the rumble off: it looped through the whole Brand fight and the
//   ending in every run (a hardware recording has no rumble there). The retired handler's hook sends the
//   game's own stop for that train's channel. (Likewise if the section ends with a train's sound on.)
// VC3_TRAIN_FIX=0 turns all of it off.
namespace {
struct Train { uint32_t obj; int channel; uint32_t entry; };
Train g_trains[2] = {{0, 2, 0x00094ff0}, {0, 1, 0x00094e30}};

bool train_fix_off() {
    static const bool off = std::getenv("VC3_TRAIN_FIX") && std::getenv("VC3_TRAIN_FIX")[0] == '0';
    return off;
}

// The game's "ambience channel off" (fn_000b2a80(channel, 0), cdecl) for a train whose sound is on.
void train_sound_off(Cpu& c, const Train& t, uint32_t ret_addr, const char* why) {
    wr<uint32_t>(t.obj, rd<uint32_t>(t.obj) & ~0x4000u);
    const Cpu saved = c;
    push32(c, 0);
    push32(c, uint32_t(t.channel));
    call_indirect(c, 0x000b2a80, ret_addr);
    c = saved;
    std::fprintf(stderr, "[sound] %.0f s: subway train (channel %d) %s with its rumble on: rumble stopped\n", log::uptime(),
                 t.channel, why);
}

void train_update(Cpu& c, int which) {
    if (train_fix_off()) return;
    Train& t = g_trains[which];
    const uint32_t train = rd<uint32_t>(c.esp + 4), es = rd<uint32_t>(0x003F3FEC);
    if (train < 0x10000 || train >= mem::kRamSize - 0x100 || !es) return;
    t.obj = train;
    const uint32_t flags = rd<uint32_t>(train);
    if (std::getenv("VC3_TRACE_SOUND")) {  // (dev) each train's sound flag as the update sees it
        static uint32_t last[2] = {0, 0};
        if ((flags ^ last[which]) & 0x4000u)
            std::fprintf(stderr, "[train] vblank %lu: train %d (channel %d) sound %s, x %.1f, section frame %d\n",
                         g_vblanks.load(std::memory_order_relaxed), which + 1, t.channel, flags & 0x4000 ? "on" : "off",
                         rd<float>(train + 0x80), rd<int32_t>(es + 0x18));
        last[which] = flags;
    }
    if (rd<int32_t>(es + 0x24) != 0x0E) {  // the update returns at once outside its section
        if (flags & 0x4000) train_sound_off(c, t, t.entry, "left behind");
        return;
    }
    if (flags & 0x4000) return;
    const uint32_t cam = rd<uint32_t>(0x003A5254) + 8;
    const double cx = rd<float>(cam), cy = rd<float>(cam + 4), cz = rd<float>(cam + 8);
    double nearest = 1e30;
    uint32_t node = rd<uint32_t>(rd<uint32_t>(train + 0x6C));
    for (int i = 0, n = rd<uint8_t>(train + 0x58); i < n && node >= 0x10000 && node < mem::kRamSize - 0x100; ++i) {
        const double dx = rd<float>(node + 0xA4) - cx, dy = rd<float>(node + 0xA8) - cy, dz = rd<float>(node + 0xAC) - cz;
        nearest = std::min(nearest, std::sqrt(dx * dx + dy * dy + dz * dz));
        node = rd<uint32_t>(node);
    }
    // Exactly the game's threshold, without a margin: start/stop pairs left in the frames just before
    // the train comes in range made the real start collide with the previous pair's release (DSOUND
    // then took the new voice's buffer for stopped when the old voice went idle, the sound engine
    // freed its slot, and nothing could stop that rumble any more: one stray copy per pass, heard as
    // hiss up to the results screen). Beyond 1000 (where the game's routine returns stale data) the
    // start is skipped too; the game then starts it when a car is really in range.
    if (nearest > 127.0) wr<uint32_t>(train, flags | 0x4000);
}

void train_retired(Cpu& c) {
    if (train_fix_off()) return;
    const uint32_t obj = rd<uint32_t>(c.esp + 4);
    for (const Train& t : g_trains)
        if (t.obj && obj == t.obj && (rd<uint32_t>(obj) & 0x4000)) train_sound_off(c, t, 0x0008ae90, "retired");
}
}  // namespace

void guest_hook(Cpu& c, uint32_t entry) {
    switch (entry) {
        case 0x000398c0: {
            // fn_000398C0(fov, aspect, near, far) builds the four frustum planes the game culls
            // objects against (the projection matrix is built separately). In widescreen,
            // widen the culling aspect to the displayed width so objects in the side areas
            // are drawn instead of popping in at the old 4:3 edge.
            static const bool off = std::getenv("VC3_WIDE_CULL") && std::getenv("VC3_WIDE_CULL")[0] == '0';
            const double k = gpu::pgraph::glb::widescreen_factor();
            if (k < 1.0 && !off) {
                const uint32_t arg = c.esp + 8;  // [esp] return address, +4 fov, +8 aspect
                wr<float>(arg, float(double(rd<float>(arg)) / k));
            }
            break;
        }
        case 0x0007be20: wide_aim(c); break;  // a gun's aim from its raw position (widescreen shooting)
        case 0x0007ed30:  // player update
            weapon_cheat(c);
            break;
        case 0x000a46a0: {  // a shot hits something; its damage (s8 at +0x6278: the hits this
            // frame) is read right after. VC3_TEST_ONEHIT=1 (testing, not a setting): the most a
            // hit can do, 127 -- every enemy at once, bosses in a few bullets.
            static const bool onehit = std::getenv("VC3_TEST_ONEHIT") != nullptr;
            const uint32_t shot = rd<uint32_t>(c.esp + 4);
            if (std::getenv("VC3_TRACE_HITS"))  // (dev) each hit with the players' aims (widescreen shooting)
                std::fprintf(stderr, "[hit] %.2f s: object %08x, aims P1 x %d y %d, P2 x %d y %d\n", log::uptime(), c.esi,
                             rd<int32_t>(kWeaponBlock + 0x0C), rd<int32_t>(kWeaponBlock + 0x10),
                             rd<int32_t>(kWeaponBlock + kWeaponStride + 0x0C), rd<int32_t>(kWeaponBlock + kWeaponStride + 0x10));
            if (std::getenv("VC3_TRACE_ENEMIES")) {  // (dev) which objects get hit
                static std::set<uint32_t> seen;
                if (seen.insert(c.esi).second)
                    std::fprintf(stderr, "[enemy] hit object %08x hp %d flag156 %d shot %08x\n", c.esi,
                                 c.esi < mem::kRamSize - 0x200 ? rd<int32_t>(c.esi + 0x140) : 0,
                                 c.esi < mem::kRamSize - 0x200 ? rd<uint8_t>(c.esi + 0x156) : 0, shot);
            }
            if (onehit && shot) {
                wr<int8_t>(shot + 0x6278, 127);
                // Bosses have more health than that: the enemy being hit is in the caller's esi,
                // its health an s32 at +0x140 (fn_0005ee40 and the other hit routines): left at 1.
                const uint32_t enemy = c.esi;
                if (enemy >= 0x10000 && enemy < mem::kRamSize - 0x200) {
                    const int32_t hp = rd<int32_t>(enemy + 0x140);
                    if (hp > 1 && hp <= 1000000) wr<int32_t>(enemy + 0x140, 1);
                }
            }
            break;
        }
        case 0x000ae7c0: {  // mission select, each frame ([esp+4]: its data)
            // Start at the final stage: no mission select. Its state machine (data +0: 0 fade in,
            // 1 choose, 2/3 selection animation, 4 done -> the task ends the game's way and its
            // end routine fn_000adbf0 sets the stage from the card in +0x10) is put at "done"
            // with a card chosen; the set-stage hook below makes it the final stage.
            static const bool final_stage = cheat_setting("VC3_CHEAT_FINAL") != 0;
            const uint32_t data = rd<uint32_t>(c.esp + 4);
            if (final_stage && data && (player_in_game(0) || player_in_game(1))) {
                wr<int32_t>(data + 0x10, 1);
                wr<int32_t>(data, 4);
            }
            break;
        }
        case 0x00077c20: {  // set the current stage ([esp+4]; kept in [0x3DDC54])
            // VC3_CHEAT_FINAL: the final stage instead of the chosen mission. Stage table rows
            // at 0x1E4DC0: 0, 1, 2 the three missions, 3 the final stage (fld_stage4a); 9 the
            // attract demo.
            static const bool final_stage = cheat_setting("VC3_CHEAT_FINAL") != 0;
            // VC3_DEV_STAGE=n (testing): play stage n (0-3) whichever mission card is picked.
            if (const char* ds = std::getenv("VC3_DEV_STAGE"); ds && *ds) {
                const int32_t st = rd<int32_t>(c.esp + 4);
                if (st >= 0 && st <= 3 && (player_in_game(0) || player_in_game(1))) wr<int32_t>(c.esp + 4, std::atoi(ds));
            }
            const int32_t stage = rd<int32_t>(c.esp + 4);
            if (final_stage && stage >= 0 && stage <= 2 && (player_in_game(0) || player_in_game(1))) {
                wr<int32_t>(c.esp + 4, 3);
                gpu::pgraph::glb::hide_stale_map();  // not the chosen mission's map over its intro
            }
            {  // for the log: what the game was doing when something went wrong
                const int32_t s = rd<int32_t>(c.esp + 4);
                static const char* const kNames[] = {"mission 1", "mission 2", "mission 3", "final stage"};
                std::fprintf(stderr, "[game] %.0f s: %s\n", log::uptime(),
                             s >= 0 && s <= 3 ? kNames[s] : s == 9 ? "attract demo" : "stage");
            }
            break;
        }
        case 0x00094ff0: train_update(c, 0); break;  // the subway trains (Brand fight area), see above
        case 0x00094e30: train_update(c, 1); break;
        case 0x001123b0:  // the sound engine picks a DSP effects image (bank, index) -- VC3_TRACE_SOUND
            if (std::getenv("VC3_TRACE_SOUND"))
                std::fprintf(stderr, "[sound] vblank %lu sf %d: effects image select %u %u\n", g_vblanks.load(std::memory_order_relaxed),
                             rd<int32_t>(0x003DDC60), rd<uint8_t>(c.esp + 4), rd<uint8_t>(c.esp + 8));
            break;
        case 0x0017e0e4: {  // DSOUND DownloadEffectsImage(ds, image, size, loc, desc) -- VC3_TRACE_SOUND saves each image
            if (!std::getenv("VC3_TRACE_SOUND")) break;
            const uint32_t img = rd<uint32_t>(c.esp + 8), size = rd<uint32_t>(c.esp + 12);
            uint64_t h = 1469598103934665603ull;
            for (uint32_t i = 0; i < size && size < (8u << 20); ++i) h = (h ^ rd<uint8_t>(img + i)) * 1099511628211ull;
            std::fprintf(stderr, "[sound] vblank %lu: DownloadEffectsImage %08x size %u hash %016llx\n",
                         g_vblanks.load(std::memory_order_relaxed), img, size, (unsigned long long)h);
            char name[96];
            std::snprintf(name, sizeof name, "work/dsp_images/%016llx.bin", (unsigned long long)h);
            std::filesystem::create_directories("work/dsp_images");
            if (!std::filesystem::exists(name) && size < (8u << 20))
                if (std::FILE* f = std::fopen(name, "wb")) {
                    std::fwrite(host_ptr(img), 1, size, f);
                    std::fclose(f);
                }
            break;
        }
        case 0x0008ae90:  // an object's "retired" update handler ([esp+4]: the object)
            train_retired(c);
            break;
        case 0x0010e130: {  // the sound thread runs a request (cmd word in [esp+4]) -- VC3_TRACE_SOUND
            static const bool trace = std::getenv("VC3_TRACE_SOUND") != nullptr;
            if (!trace) break;
            const uint32_t cmd = rd<uint32_t>(c.esp + 4);
            static std::map<uint32_t, int> seen;
            const int n = ++seen[cmd];
            if (n <= 3 || n % 300 == 0)
                std::fprintf(stderr, "[sound] vblank %lu: run %08x from %06x (x%d)\n", g_vblanks.load(std::memory_order_relaxed),
                             cmd, rd<uint32_t>(c.esp), n);
            break;
        }
        case 0x0010dc10: {  // the sound engine's request post (cmd word in [esp+4]) -- VC3_TRACE_SOUND
            static const bool trace = std::getenv("VC3_TRACE_SOUND") != nullptr;
            if (!trace) break;
            const uint32_t cmd = rd<uint32_t>(c.esp + 4);
            static std::map<std::string, int> seen;
            char b[48];
            std::snprintf(b, sizeof b, "cmd %08x from", cmd);
            std::string key = b;
            for (uint32_t i = 0, n = 0; i < 64 && n < 6; ++i) {  // return addresses on the guest stack
                const uint32_t w = rd<uint32_t>(c.esp + 4 * i);
                if (w >= 0x00011000u && w < 0x001B3000u) {
                    std::snprintf(b, sizeof b, " %06x", w);
                    key += b;
                    ++n;
                }
            }
            const int n = ++seen[key];
            if (n <= 3 || n % 300 == 0)
                std::fprintf(stderr, "[sound] vblank %lu sf %d: %s (x%d)\n", g_vblanks.load(std::memory_order_relaxed), rd<int32_t>(0x003DDC60), key.c_str(), n);
            break;
        }
        case 0x001878c3:    // DSOUND voice release / voice activate: who calls them (sound tracing)
        case 0x00188ac3: {
            static const bool trace = std::getenv("VC3_TRACE_APU") != nullptr;
            if (!trace) break;
            static std::map<std::string, int> seen;
            std::string chain;
            for (uint32_t i = 0, n = 0; i < 96 && n < 10; ++i) {  // return addresses on the guest stack
                const uint32_t w = rd<uint32_t>(c.esp + 4 * i);
                if (w >= 0x00011000u && w < 0x001B3000u) {
                    char b[12];
                    std::snprintf(b, sizeof b, " %06x", w);
                    chain += b;
                    ++n;
                }
            }
            if (++seen[chain] <= 3 || seen[chain] % 500 == 0)
                std::fprintf(stderr, "[dsound] %s called from:%s (x%d)\n", entry == 0x001878c3 ? "release" : "activate",
                             chain.c_str(), seen[chain]);
            break;
        }
        default: break;
    }
}

}  // namespace vc3

