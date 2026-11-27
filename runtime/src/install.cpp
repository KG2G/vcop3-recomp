#include "platform.h"
#include "install.h"
#include "privacy.h"
#include "settings.h"

#include <SDL3/SDL.h>
#include <libchdr/chd.h>
#include <miniz.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace vc3::install {
namespace {

using u8 = uint8_t;
using u32 = uint32_t;
using u64 = uint64_t;

// Known-good SHA-1s.
constexpr const char* kPicSha1 = "9f0f4bf6b15f3b6eeea81eaa27b3d25bd94110da";  // 317-0354-com.data
constexpr const char* kChdSha1 = "4268aadb83c880d4f3dab1d43ddd0a3a2f8befa2";  // gdx-0003b (CHD data SHA-1)
constexpr const char* kXbeSha1 = "e22c2b0ecf091a2388d9e3ca7eb9e07ce54d4727";  // vc3.xbe (the recompiled one)
constexpr const char* kPicName = "317-0354-com.data";
constexpr const char* kChdName = "gdx-0003b.chd";

Progress g_progress;

void set_status(const std::string& m, float frac = -1) {
    std::lock_guard lk(g_progress.m);
    g_progress.message = m;
    if (frac >= 0) g_progress.fraction = frac;
    std::fprintf(stderr, "[install] %s\n", privacy::scrub(m).c_str());
}

// ---------------------------------------------------------------- SHA-1
struct Sha1 {
    u32 h[5] = {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0};
    u8 buf[64];
    size_t n = 0;
    u64 total = 0;
    static u32 rol(u32 x, int s) { return (x << s) | (x >> (32 - s)); }
    void block(const u8* p) {
        u32 w[80];
        for (int i = 0; i < 16; ++i) w[i] = u32(p[4 * i]) << 24 | u32(p[4 * i + 1]) << 16 | u32(p[4 * i + 2]) << 8 | p[4 * i + 3];
        for (int i = 16; i < 80; ++i) w[i] = rol(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
        u32 a = h[0], b = h[1], c = h[2], d = h[3], e = h[4];
        for (int i = 0; i < 80; ++i) {
            u32 f, k;
            if (i < 20) { f = (b & c) | (~b & d); k = 0x5A827999; }
            else if (i < 40) { f = b ^ c ^ d; k = 0x6ED9EBA1; }
            else if (i < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8F1BBCDC; }
            else { f = b ^ c ^ d; k = 0xCA62C1D6; }
            const u32 t = rol(a, 5) + f + e + k + w[i];
            e = d; d = c; c = rol(b, 30); b = a; a = t;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e;
    }
    void update(const void* data, size_t len) {
        const u8* p = static_cast<const u8*>(data);
        total += len;
        while (len) {
            const size_t k = std::min(len, 64 - n);
            std::memcpy(buf + n, p, k);
            n += k; p += k; len -= k;
            if (n == 64) { block(buf); n = 0; }
        }
    }
    std::string hex() {
        const u64 bits = total * 8;
        const u8 pad = 0x80;
        update(&pad, 1);
        const u8 z = 0;
        while (n != 56) update(&z, 1);
        u8 lenb[8];
        for (int i = 0; i < 8; ++i) lenb[i] = u8(bits >> (56 - 8 * i));
        update(lenb, 8);
        char out[41];
        for (int i = 0; i < 5; ++i) std::snprintf(out + 8 * i, 9, "%08x", h[i]);
        return out;
    }
};
std::string sha1_of(const void* p, size_t n) { Sha1 s; s.update(p, n); return s.hex(); }
std::string sha1_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return "";
    Sha1 s;
    std::vector<char> buf(1 << 20);
    while (f) {
        f.read(buf.data(), std::streamsize(buf.size()));
        s.update(buf.data(), size_t(f.gcount()));
    }
    return s.hex();
}

// ---------------------------------------------------------------- DES (decrypt)
// Table-driven: IP/FP through per-byte lookup tables, rounds through combined S+P tables.
constexpr u8 kIP[64] = {58, 50, 42, 34, 26, 18, 10, 2, 60, 52, 44, 36, 28, 20, 12, 4, 62, 54, 46, 38, 30, 22,
                        14, 6, 64, 56, 48, 40, 32, 24, 16, 8, 57, 49, 41, 33, 25, 17, 9, 1, 59, 51, 43, 35,
                        27, 19, 11, 3, 61, 53, 45, 37, 29, 21, 13, 5, 63, 55, 47, 39, 31, 23, 15, 7};
constexpr u8 kFP[64] = {40, 8, 48, 16, 56, 24, 64, 32, 39, 7, 47, 15, 55, 23, 63, 31, 38, 6, 46, 14, 54, 22,
                        62, 30, 37, 5, 45, 13, 53, 21, 61, 29, 36, 4, 44, 12, 52, 20, 60, 28, 35, 3, 43, 11,
                        51, 19, 59, 27, 34, 2, 42, 10, 50, 18, 58, 26, 33, 1, 41, 9, 49, 17, 57, 25};
constexpr u8 kP[32] = {16, 7, 20, 21, 29, 12, 28, 17, 1, 15, 23, 26, 5, 18, 31, 10,
                       2, 8, 24, 14, 32, 27, 3, 9, 19, 13, 30, 6, 22, 11, 4, 25};
constexpr u8 kPC1[56] = {57, 49, 41, 33, 25, 17, 9, 1, 58, 50, 42, 34, 26, 18, 10, 2, 59, 51, 43, 35, 27,
                         19, 11, 3, 60, 52, 44, 36, 63, 55, 47, 39, 31, 23, 15, 7, 62, 54, 46, 38, 30, 22,
                         14, 6, 61, 53, 45, 37, 29, 21, 13, 5, 28, 20, 12, 4};
constexpr u8 kPC2[48] = {14, 17, 11, 24, 1, 5, 3, 28, 15, 6, 21, 10, 23, 19, 12, 4, 26, 8, 16, 7, 27, 20, 13, 2,
                         41, 52, 31, 37, 47, 55, 30, 40, 51, 45, 33, 48, 44, 49, 39, 56, 34, 53, 46, 42, 50, 36, 29, 32};
constexpr u8 kShifts[16] = {1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1};
constexpr u8 kS[8][64] = {
    {14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7, 0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8,
     4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0, 15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13},
    {15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10, 3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5,
     0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15, 13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9},
    {10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8, 13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1,
     13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7, 1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12},
    {7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15, 13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9,
     10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4, 3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14},
    {2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9, 14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6,
     4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14, 11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3},
    {12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11, 10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8,
     9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6, 4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13},
    {4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1, 13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6,
     1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2, 6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12},
    {13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7, 1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2,
     7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8, 2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11}};

// out bit i (MSB first, of n) = in bit table[i] (1-based, MSB first, of in_bits)
u64 permute(u64 in, const u8* table, int n, int in_bits) {
    u64 out = 0;
    for (int i = 0; i < n; ++i) out = (out << 1) | ((in >> (in_bits - table[i])) & 1);
    return out;
}

struct Des {
    u64 ip[8][256], fp[8][256];
    u32 sp[8][64];
    u64 sub[16];  // 48-bit round keys
    explicit Des(const u8 key[8]) {
        for (int b = 0; b < 8; ++b)
            for (int v = 0; v < 256; ++v) {
                const u64 in = u64(v) << (56 - 8 * b);
                ip[b][v] = permute(in, kIP, 64, 64);
                fp[b][v] = permute(in, kFP, 64, 64);
            }
        for (int j = 0; j < 8; ++j)
            for (int v = 0; v < 64; ++v) {
                const int row = ((v >> 4) & 2) | (v & 1), col = (v >> 1) & 15;
                sp[j][v] = u32(permute(u64(kS[j][row * 16 + col]) << (28 - 4 * j), kP, 32, 32));
            }
        u64 k = 0;
        for (int i = 0; i < 8; ++i) k = (k << 8) | key[i];
        const u64 cd = permute(k, kPC1, 56, 64);
        u32 c = u32(cd >> 28) & 0xFFFFFFF, d = u32(cd) & 0xFFFFFFF;
        for (int r = 0; r < 16; ++r) {
            for (int s = 0; s < kShifts[r]; ++s) {
                c = ((c << 1) | (c >> 27)) & 0xFFFFFFF;
                d = ((d << 1) | (d >> 27)) & 0xFFFFFFF;
            }
            sub[r] = permute((u64(c) << 28) | d, kPC2, 48, 56);
        }
    }
    u64 perm(const u64 (&t)[8][256], u64 x) const {
        u64 o = 0;
        for (int b = 0; b < 8; ++b) o |= t[b][(x >> (56 - 8 * b)) & 0xFF];
        return o;
    }
    u32 f(u32 r, u64 k) const {
        const u32 rr = (r >> 1) | (r << 31);
        const u64 x = (u64(rr) << 32) | rr;
        u32 out = 0;
        for (int j = 0; j < 8; ++j) out |= sp[j][((x >> (58 - 4 * j)) ^ (k >> (42 - 6 * j))) & 0x3F];
        return out;
    }
    u64 crypt(u64 block, bool decrypt) const {
        const u64 t = perm(ip, block);
        u32 l = u32(t >> 32), r = u32(t);
        for (int i = 0; i < 16; ++i) {
            const u32 nl = r;
            r = l ^ f(r, sub[decrypt ? 15 - i : i]);
            l = nl;
        }
        return perm(fp, (u64(r) << 32) | l);
    }
};

// ---------------------------------------------------------------- zip / PIC
bool read_pic(const std::string& path, std::vector<u8>& out, std::string& err) {
    std::string low = path;
    std::transform(low.begin(), low.end(), low.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    if (low.size() >= 5 && low.substr(low.size() - 5) == ".data") {
        std::ifstream f(path, std::ios::binary);
        out.assign(std::istreambuf_iterator<char>(f), {});
        return !out.empty() || (err = "can't read " + fs::path(path).filename().string(), false);
    }
    mz_zip_archive z{};
    if (!mz_zip_reader_init_file(&z, path.c_str(), 0)) {
        err = fs::path(path).filename().string() + " is not a zip file (7z isn't supported: repack it as zip)";
        return false;
    }
    bool ok = false;
    for (mz_uint i = 0; i < mz_zip_reader_get_num_files(&z); ++i) {
        char name[512];
        mz_zip_reader_get_filename(&z, i, name, sizeof name);
        std::string n = name;
        std::transform(n.begin(), n.end(), n.begin(), [](unsigned char c) { return char(std::tolower(c)); });
        if (n.size() < std::strlen(kPicName) || n.substr(n.size() - std::strlen(kPicName)) != kPicName) continue;
        size_t size = 0;
        if (void* p = mz_zip_reader_extract_to_heap(&z, i, &size, 0)) {
            out.assign(static_cast<u8*>(p), static_cast<u8*>(p) + size);
            mz_free(p);
            ok = true;
        }
        break;
    }
    mz_zip_reader_end(&z);
    if (!ok) err = fs::path(path).filename().string() + " doesn't contain " + kPicName + " (the vcop3 set from MAME)";
    return ok;
}

// ---------------------------------------------------------------- CHD / GD-ROM
struct Disc {
    chd_file* chd = nullptr;
    u32 hunk_bytes = 0, frames_per_hunk = 0;
    u32 hd_chd_frame = 0;  // CHD frame of the high-density area's first sector
    u32 hd_lba = 45000;    // its disc LBA
    std::vector<u8> hunk;
    u32 cur_hunk = ~0u;
    ~Disc() { if (chd) chd_close(chd); }
    // 2048 user bytes of the Mode 1 sector at disc LBA `lba` (high-density area).
    bool sector(u32 lba, u8* out) {
        const u32 frame = hd_chd_frame + (lba - hd_lba);
        const u32 h = frame / frames_per_hunk;
        if (h != cur_hunk) {
            if (chd_read(chd, h, hunk.data()) != CHDERR_NONE) return false;
            cur_hunk = h;
        }
        std::memcpy(out, hunk.data() + (frame % frames_per_hunk) * 2448 + 16, 2048);
        return true;
    }
    bool read(u32 lba, u64 offset, u64 len, u8* out) {
        u8 sec[2048];
        while (len) {
            const u32 s = u32(offset / 2048), o = u32(offset % 2048);
            if (!sector(lba + s, sec)) return false;
            const u64 n = std::min<u64>(len, 2048 - o);
            std::memcpy(out, sec + o, n);
            out += n; offset += n; len -= n;
        }
        return true;
    }
};

bool open_disc(const std::string& path, Disc& d, std::string& err) {
    if (chd_open(path.c_str(), CHD_OPEN_READ, nullptr, &d.chd) != CHDERR_NONE) {
        err = "can't open " + fs::path(path).filename().string() + " as a CHD";
        return false;
    }
    const chd_header* h = chd_get_header(d.chd);
    char sha[41];
    for (int i = 0; i < 20; ++i) std::snprintf(sha + 2 * i, 3, "%02x", h->sha1[i]);
    if (std::string(sha) != kChdSha1) {
        err = fs::path(path).filename().string() + ": SHA-1 " + sha + " is not the Virtua Cop 3 GD-ROM (gdx-0003b, expected " + kChdSha1 + ")";
        return false;
    }
    d.hunk_bytes = h->hunkbytes;
    d.frames_per_hunk = h->hunkbytes / 2448;
    d.hunk.resize(h->hunkbytes);
    // GD-ROM track table: tracks are stored one after another, each padded to 4 frames; the
    // high-density area is the last data track and starts at LBA 45000.
    u32 chd_frame = 0, hd = ~0u;
    for (u32 i = 0;; ++i) {
        char meta[256] = {};
        u32 len = 0, tag = 0;
        u8 flags = 0;
        if (chd_get_metadata(d.chd, GDROM_TRACK_METADATA_TAG, i, meta, sizeof meta - 1, &len, &tag, &flags) != CHDERR_NONE) break;
        int track = 0, frames = 0;
        char type[32] = {};
        std::sscanf(meta, "TRACK:%d TYPE:%31s", &track, type);
        if (const char* f = std::strstr(meta, "FRAMES:")) frames = std::atoi(f + 7);
        if (std::strcmp(type, "AUDIO")) hd = chd_frame;  // data track (the last one wins)
        chd_frame += u32((frames + 3) & ~3);
    }
    if (hd == ~0u) {
        err = fs::path(path).filename().string() + ": no GD-ROM track table";
        return false;
    }
    d.hd_chd_frame = hd;
    u8 pvd[2048];
    if (!d.sector(d.hd_lba + 16, pvd) || std::memcmp(pvd + 1, "CD001", 5)) {
        err = fs::path(path).filename().string() + ": no ISO 9660 volume in the high-density area";
        return false;
    }
    return true;
}

struct IsoFile { u32 lba, size; };
std::map<std::string, IsoFile> iso_root(Disc& d) {
    std::map<std::string, IsoFile> files;
    u8 pvd[2048] = {};
    d.sector(d.hd_lba + 16, pvd);
    u32 root_lba, root_size;
    std::memcpy(&root_lba, pvd + 158, 4);
    std::memcpy(&root_size, pvd + 166, 4);
    std::vector<u8> dir(root_size);
    d.read(root_lba, 0, root_size, dir.data());
    for (size_t o = 0; o < dir.size();) {
        const u8 len = dir[o];
        if (!len) { o = (o / 2048 + 1) * 2048; continue; }
        IsoFile f;
        std::memcpy(&f.lba, &dir[o + 2], 4);
        std::memcpy(&f.size, &dir[o + 10], 4);
        std::string name(reinterpret_cast<const char*>(&dir[o + 33]), dir[o + 32]);
        name = name.substr(0, name.find(';'));
        if (name.size() > 1 || (name[0] != 0 && name[0] != 1)) files[name] = f;
        o += len;
    }
    return files;
}

// ---------------------------------------------------------------- FATX
struct Fatx {
    std::ifstream f;
    u64 size = 0, cluster = 0, fat_off = 0x1000, data_off = 0;
    u32 root = 0, entry = 2, eoc = 0;
    std::vector<u8> fat;
    bool open(const std::string& path) {
        f.open(path, std::ios::binary);
        if (!f) return false;
        f.seekg(0, std::ios::end);
        size = u64(f.tellg());
        u8 sb[16];
        f.seekg(0);
        f.read(reinterpret_cast<char*>(sb), 16);
        if (std::memcmp(sb, "FATX", 4)) return false;
        u32 spc;
        std::memcpy(&spc, sb + 8, 4);
        std::memcpy(&root, sb + 12, 4);
        cluster = u64(spc) * 512;
        // The FAT is sized for the nominal partition, not the image: probe sizes.
        std::vector<u64> cand = {size};
        for (int b = 20; b < 34; ++b) if ((1ull << b) >= size) cand.push_back(1ull << b);
        for (u64 part : cand) {
            const u64 n = part / cluster;
            entry = n < 0xFFF0 ? 2 : 4;
            eoc = entry == 2 ? 0xFFF8 : 0xFFFFFFF8;
            const u64 fat_bytes = (n * entry + 0xFFF) & ~u64(0xFFF);
            data_off = 0x1000 + fat_bytes;
            u8 d[44];
            f.seekg(std::streamoff(data_off + (root - 1) * cluster));
            f.read(reinterpret_cast<char*>(d), 44);
            const u8 nl = d[0], attr = d[1];
            bool ok = nl > 0 && nl <= 42 && (attr == 0 || attr == 1 || attr == 2 || attr == 0x10 || attr == 0x20);
            for (int i = 0; ok && i < nl; ++i) ok = d[2 + i] >= 0x20 && d[2 + i] < 0x7F;
            if (ok) {
                fat.resize(fat_bytes);
                f.seekg(0x1000);
                f.read(reinterpret_cast<char*>(fat.data()), std::streamsize(fat_bytes));
                return true;
            }
        }
        return false;
    }
    u32 next(u32 c) const {
        if (u64(c) * entry + entry > fat.size()) return eoc;
        if (entry == 2) { uint16_t v; std::memcpy(&v, &fat[c * 2], 2); return v; }
        u32 v; std::memcpy(&v, &fat[u64(c) * 4], 4); return v;
    }
    std::vector<u32> chain(u32 c) const {
        std::vector<u32> out;
        while (c > 0 && c < eoc && out.size() < 1000000) { out.push_back(c); c = next(c); }
        return out;
    }
    std::vector<u8> read(u32 first, u64 len) {
        std::vector<u8> out;
        for (u32 c : chain(first)) {
            const size_t at = out.size();
            out.resize(at + cluster);
            f.seekg(std::streamoff(data_off + (c - 1) * cluster));
            f.read(reinterpret_cast<char*>(out.data() + at), std::streamsize(cluster));
            if (len && out.size() >= len) break;
        }
        if (len) out.resize(std::min<u64>(out.size(), len));
        return out;
    }
    // Writes the directory tree under `dst`; counts bytes for progress.
    bool extract(u32 dir, const fs::path& dst, u64& done) {
        const std::vector<u8> raw = read(dir, 0);
        for (size_t o = 0; o + 64 <= raw.size(); o += 64) {
            const u8 nl = raw[o], attr = raw[o + 1];
            if (nl == 0x00 || nl == 0xFF) break;
            if (nl == 0xE5 || nl > 42) continue;
            const std::string name(reinterpret_cast<const char*>(&raw[o + 2]), nl);
            u32 first, size;
            std::memcpy(&first, &raw[o + 0x2C], 4);
            std::memcpy(&size, &raw[o + 0x30], 4);
            const fs::path p = dst / name;
            if (attr & 0x10) {
                fs::create_directories(p);
                if (!extract(first, p, done)) return false;
            } else {
                const std::vector<u8> data = read(first, size);
                std::ofstream out(p, std::ios::binary);
                out.write(reinterpret_cast<const char*>(data.data()), std::streamsize(data.size()));
                if (!out) return false;
                done += size;
            }
        }
        return true;
    }
};

bool fail(const std::string& m) {
    set_status(m);
    g_progress.state = 3;
    return false;
}

std::string data_root() { return fs::current_path().string(); }  // main() picks it (work/ lives here)

fs::path exe_dir() { return platform::program_dir(); }  // (beside the .AppImage on Linux)

}  // namespace

Progress& progress() { return g_progress; }

std::string location() {
    const std::string s = settings::get("install_dir");
    return s.empty() ? default_location() : s;
}
std::string roms_dir() {
    const fs::path r = exe_dir() / "roms";
    std::error_code ec;
    fs::create_directories(r, ec);
    return r.string();
}

std::string g_default_loc;
void set_default_location(const std::string& dir) { g_default_loc = dir; }
std::string default_location() {
    if (!g_default_loc.empty()) return g_default_loc;
    return platform::user_data_dir().string();
}
std::string location();
bool use_location(const std::string& dir) {
    std::error_code ec;
    fs::create_directories(dir, ec);
    fs::current_path(dir, ec);
    if (ec) std::fprintf(stderr, "[install] can't use %s: %s\n", privacy::short_path(dir).c_str(), ec.message().c_str());
    return !ec;
}

Files find() {
    Files out;
    // The roms folder beside the program first; the data folder (and a project tree's
    // vcop3/) as well.
    const std::vector<fs::path> dirs = {fs::path(roms_dir()), fs::path(roms_dir()) / "vcop3", fs::path(data_root()) / "roms",
                                        fs::path(data_root()), fs::path(data_root()) / "vcop3"};
    for (const auto& d : dirs) {
        std::error_code ec;
        if (out.pic.empty()) {
            if (fs::exists(d / "vcop3.zip", ec)) out.pic = (d / "vcop3.zip").string();
            else if (fs::exists(d / kPicName, ec)) out.pic = (d / kPicName).string();
        }
        if (out.chd.empty() && fs::exists(d / kChdName, ec)) out.chd = (d / kChdName).string();
    }
    return out;
}

bool installed(std::string* problem) {
    const std::string xbe = "work/fs/vc3.xbe";
    if (!fs::exists(xbe)) {
        if (problem) *problem = "Game files not installed";
        return false;
    }
    // Hash once per file change (the launcher asks every frame).
    static std::string cached, cached_path;
    static fs::file_time_type cached_time{};
    std::error_code ec;
    const auto t = fs::last_write_time(xbe, ec);
    const std::string abs = fs::absolute(xbe, ec).string();
    if (cached.empty() || t != cached_time || abs != cached_path) {
        cached = sha1_file(xbe);
        cached_time = t;
        cached_path = abs;
    }
    if (cached != kXbeSha1) {
        if (problem) *problem = "work/fs/vc3.xbe is a different version than this build was made from";
        return false;
    }
    return true;
}

bool run(const Files& files) {
    g_progress.state = 1;
    g_progress.fraction = 0;
    // DES self-test (FIPS 46 example).
    {
        const u8 k[8] = {0x13, 0x34, 0x57, 0x79, 0x9B, 0xBC, 0xDF, 0xF1};
        const Des t(k);
        if (t.crypt(0x85E813540F0AB405ull, true) != 0x0123456789ABCDEFull) return fail("internal error: DES self-test failed");
    }
    // 1. PIC data
    set_status("Checking " + fs::path(files.pic).filename().string(), 0.0f);
    std::vector<u8> pic;
    std::string err;
    if (!read_pic(files.pic, pic, err)) return fail(err);
    if (const std::string s = sha1_of(pic.data(), pic.size()); s != kPicSha1)
        return fail(std::string(kPicName) + " has SHA-1 " + s + ", expected " + kPicSha1 + " (wrong or bad dump)");
    // 2. Disc
    set_status("Checking " + fs::path(files.chd).filename().string(), 0.01f);
    Disc disc;
    if (!open_disc(files.chd, disc, err)) return fail(err);
    auto iso = iso_root(disc);
    const std::string key_file(reinterpret_cast<const char*>(&pic[0x21]), strnlen(reinterpret_cast<const char*>(&pic[0x21]), 7));
    u8 key[8];
    {   // DES key = reverse(pic[0x31:0x38] + pic[0x29]) (records '3' and '4' of the PIC)
        u8 k2[8];
        std::memcpy(k2, &pic[0x31], 7);
        k2[7] = pic[0x29];
        for (int i = 0; i < 8; ++i) key[i] = k2[7 - i];
    }
    if (!iso.count(key_file)) return fail("the disc has no key file " + key_file);
    std::vector<u8> kf(iso[key_file].size);
    disc.read(iso[key_file].lba, 0, kf.size(), kf.data());
    const std::string game_file(reinterpret_cast<const char*>(&kf[0xC0]), strnlen(reinterpret_cast<const char*>(&kf[0xC0]), 32));
    if (!iso.count(game_file)) return fail("the disc has no game file " + game_file);
    // 3. Decrypt the game image (DES-ECB on byte-reversed 8-byte blocks)
    fs::create_directories("work");
    const std::string img = "work/vc3.img.new";
    {
        std::ofstream out(img, std::ios::binary);
        if (!out) return fail("can't write " + img);
        const Des des(key);
        const IsoFile gf = iso[game_file];
        constexpr u64 kChunk = 4 << 20;
        std::vector<u8> buf(kChunk);
        for (u64 off = 0; off < gf.size; off += kChunk) {
            const u64 n = std::min<u64>(kChunk, gf.size - off);
            if (!disc.read(gf.lba, off, n, buf.data())) return fail("read error in " + fs::path(files.chd).filename().string());
            for (u64 i = 0; i + 8 <= n; i += 8) {
                u64 v = 0;
                for (int b = 0; b < 8; ++b) v |= u64(buf[i + b]) << (8 * b);  // little-endian block
                v = des.crypt(v, true);
                for (int b = 0; b < 8; ++b) buf[i + b] = u8(v >> (8 * b));
            }
            out.write(reinterpret_cast<const char*>(buf.data()), std::streamsize(n));
            set_status("Building the game (" + std::to_string((off + n) * 100 / gf.size) + "%)",
                       0.02f + 0.68f * float(off + n) / float(gf.size));
        }
        if (!out) return fail("write error: " + img);
    }
    // 4. Unpack FATX into work/fs.new, check the executable, swap in
    set_status("Unpacking files", 0.7f);
    const fs::path tmp = "work/fs.new";
    std::error_code ec;
    fs::remove_all(tmp, ec);
    fs::create_directories(tmp);
    {
        Fatx fx;
        if (!fx.open(img)) return fail("the built game image isn't readable (are these the right files?)");
        u64 done = 0;
        if (!fx.extract(fx.root, tmp, done)) return fail("couldn't write the game files (disk full or no permission?)");
    }
    set_status("Verifying", 0.97f);
    if (const std::string s = sha1_file((tmp / "vc3.xbe").string()); s != kXbeSha1)
        return fail("the extracted vc3.xbe has SHA-1 " + s + ", expected " + kXbeSha1);
    fs::remove_all("work/fs.old", ec);
    if (fs::exists("work/fs")) fs::rename("work/fs", "work/fs.old", ec);
    fs::rename(tmp, "work/fs", ec);
    if (ec) return fail("couldn't move the files into work/fs: " + ec.message());
    fs::remove_all("work/fs.old", ec);
    // The decrypted disc image (~460 MiB) was only needed to unpack it: the game runs from work/fs.
    fs::remove(img, ec);
    fs::remove("work/vc3.img", ec);  // (left by older versions)
    set_status("Game files installed", 1.0f);
    g_progress.state = 2;
    return true;
}

void start(const Files& f) {
    if (g_progress.state == 1) return;
    g_progress.state = 1;
    std::thread([f] { run(f); }).detach();
}

}  // namespace vc3::install
