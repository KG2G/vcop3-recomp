// Hooks into lifted guest functions (see config/vc3_hooks.txt).
#include <cstdlib>
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

void per_vblank() {
    cheats();
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

// Each player's state byte (P1 0x3F3F68, P2 0x3F4018, 0xB0 apart): 0 not playing,
// 2 playing (from pressing start, mission select included), 8 continue countdown,
// 16 game over, 128 attract-mode demo.
bool player_in_game(int p) {
    const uint8_t st = rd<uint8_t>(p ? 0x003F4018 : 0x003F3F68);
    return st == 2 || st == 8;
}

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
        case 0x0007ed30:  // player update
            weapon_cheat(c);
            break;
        case 0x000a46a0: {  // a shot hits something; its damage (s8 at +0x6278: the hits this
            // frame) is read right after. VC3_TEST_ONEHIT=1 (testing, not a setting): the most a
            // hit can do, 127 -- every enemy at once, bosses in a few bullets.
            static const bool onehit = std::getenv("VC3_TEST_ONEHIT") != nullptr;
            const uint32_t shot = rd<uint32_t>(c.esp + 4);
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
        default: break;
    }
}

}  // namespace vc3

