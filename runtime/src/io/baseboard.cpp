#include "baseboard.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "io/jvs.h"
#include "io/ohci.h"
#include "io/usb.h"

namespace vc3::io::baseboard {

using u8 = uint8_t;

namespace {

bool trace() {
    static const bool on = std::getenv("VC3_TRACE_USB") != nullptr;
    return on;
}

// EEPROM contents as one region (layout of MAME's "others" region, word-swapped):
//   0x0000 ic10_g24lc64.bin (8 KiB: base board firmware data, serial, region at 0x1F10)
//   0x2000 ic11_24lc024.bin (128 B)
//   0x2080 pc20_g24lc64.bin (8 KiB, read by the AN2131SC)
std::vector<uint8_t> g_region(0x4080, 0xFF);

void load_eeproms() {
    const char* dir = std::getenv("VC3_BIOS_DIR");
    const std::string base = dir ? dir : "work/bios";
    struct { const char* file; size_t off, size; } files[] = {
        {"ic10_g24lc64.bin", 0x0000, 0x2000}, {"ic11_24lc024.bin", 0x2000, 0x80}, {"pc20_g24lc64.bin", 0x2080, 0x2000}};
    for (const auto& f : files) {
        const std::string path = base + "/" + f.file;
        std::FILE* fp = std::fopen(path.c_str(), "rb");
        if (!fp) { std::fprintf(stderr, "[baseboard] missing %s (base board EEPROM)\n", path.c_str()); continue; }
        const size_t n = std::fread(&g_region[f.off], 1, f.size, fp);
        std::fclose(fp);
        for (size_t i = 0; i + 1 < n; i += 2) std::swap(g_region[f.off + i], g_region[f.off + i + 1]);  // ROM_LOAD16_WORD_SWAP
    }
}

// ------------------------------------------------------------------ JVS I/O board
// Sega 837-13551. message() takes a packet body (commands, no sync/len/sum) and
// produces the raw reply in MAME's jvs_host format: {0x00, len, status, reports...}.
class Jvs13551 {
public:
    bool address_set() const { return address_ >= 0; }

    std::vector<uint8_t> message(uint8_t dest, const uint8_t* s, size_t n) {
        std::vector<uint8_t> r;
        if (n == 2 && s[0] == 0xF1) {  // set address
            address_ = s[1];
            r = {0x01, 0x01};
        } else if (dest == 0xFF || int(dest) == address_) {
            r.push_back(0x01);  // status: normal
            poll_coins();
            size_t i = 0;
            while (i < n) {
                const int used = command(s + i, n - i, r);
                if (used <= 0) { r.push_back(0x02); break; }  // unknown command
                i += size_t(used);
            }
        } else {
            return {};
        }
        std::vector<uint8_t> out = {0x00, uint8_t(r.size() + 1)};
        out.insert(out.end(), r.begin(), r.end());
        return out;
    }

private:
    int address_ = -1;
    uint16_t coins_[2] = {};
    bool prev_coin_[2] = {};

    void poll_coins() {
        const auto st = state();
        for (int i = 0; i < 2; ++i) {
            if (st.coin[i] && !prev_coin_[i]) coins_[i] = uint16_t((coins_[i] + 1) & 0x3FFF);
            prev_coin_[i] = st.coin[i];
        }
    }

    // Returns bytes consumed (0 = unsupported).
    int command(const uint8_t* s, size_t n, std::vector<uint8_t>& r) {
        auto str = [&](const char* t) { r.insert(r.end(), t, t + std::strlen(t) + 1); };
        switch (s[0]) {
            case 0xF0:  // reset (F0 D9)
                if (n < 2) return 0;
                address_ = -1;
                return 2;
            case 0x10: r.push_back(1); str("SEGA ENTERPRISES,LTD.;I/O BD JVS;837-13551"); return 1;
            case 0x11: r.push_back(1); r.push_back(0x11); return 1;  // command format version
            case 0x12: r.push_back(1); r.push_back(0x20); return 1;  // JVS version
            case 0x13: r.push_back(1); r.push_back(0x10); return 1;  // communication version
            case 0x14:                                               // feature list
                r.push_back(1);
                r.insert(r.end(), {0x01, 2, 13, 0,    // switches: 2 players, 13 buttons
                                   0x02, 2, 0, 0,     // coins: 2 slots
                                   0x03, 8, 16, 0,    // analog: 8 channels, 16 bits
                                   0x12, 6, 0, 0,     // general outputs: 6
                                   0x00});
                return 1;
            case 0x20: {  // switches(players, bytes per player)
                if (n < 3) return 0;
                const auto st = state();
                r.push_back(1);
                r.push_back(uint8_t((st.test ? 0x80 : 0)));
                for (int p = 0; p < s[1]; ++p) {
                    const auto& g = st.gun[p < 2 ? p : 0];
                    uint16_t v = 0;
                    if (p < 2) {
                        // Virtua Cop 3 cabinet wiring (verified in the game's INPUT TEST):
                        if (g.start) v |= 0x8000;
                        if (st.service && p == 0) v |= 0x4000;
                        if (g.trigger) v |= 0x0200;
                        if (g.offscreen) v |= 0x0100;  // gun screen sensor: set = pointing off screen
                        // (The "All weapons" cheat takes the button over: hooks.cpp cycles the weapons.)
                        static const bool cycle = std::getenv("VC3_CHEAT_WEAPON") && std::atoi(std::getenv("VC3_CHEAT_WEAPON")) == 99;
                        if (g.change && !cycle) v |= 0x0080;
                        if (g.pedal) v |= 0x0040;
                    }
                    for (int b = 0; b < s[2]; ++b) r.push_back(uint8_t(v >> (8 * (1 - b))));
                }
                return 3;
            }
            case 0x21:  // coin counters(slots)
                if (n < 2) return 0;
                r.push_back(1);
                for (int i = 0; i < s[1] && i < 2; ++i) { r.push_back(uint8_t(coins_[i] >> 8)); r.push_back(uint8_t(coins_[i])); }
                return 2;
            case 0x22: {  // analog(channels): ch0/1 = P1 X/Y, ch2/3 = P2 X/Y
                if (n < 2) return 0;
                const auto st = state();
                r.push_back(1);
                for (int ch = 0; ch < s[1]; ++ch) {
                    uint16_t v = 0x8000;
                    if (ch < 4) {
                        const auto& g = st.gun[ch / 2];
                        v = g.offscreen ? 0 : (ch & 1 ? g.y : g.x);
                    }
                    r.push_back(uint8_t(v >> 8));
                    r.push_back(uint8_t(v));
                }
                return 2;
            }
            case 0x30: case 0x31: {  // coin decrease / increase(slot, amount)
                if (n < 4) return 0;
                const int slot = s[1] - 1;
                const int amount = (s[2] << 8) | s[3];
                if (slot >= 0 && slot < 2) coins_[slot] = uint16_t((coins_[slot] + (s[0] == 0x30 ? -amount : amount)) & 0x3FFF);
                r.push_back(1);
                return 4;
            }
            case 0x32:  // general output(bytes, data...)
                if (n < 2 || n < size_t(2 + s[1])) return 0;
                r.push_back(1);
                return 2 + s[1];
            case 0x38:  // general output 3
                if (n < 3) return 0;
                r.push_back(1);
                return 3;
            default:
                std::fprintf(stderr, "[jvs] unsupported command %02x\n", s[0]);
                return 0;
        }
    }
};

Jvs13551 g_jvs;
std::mutex g_jvs_m;

// Simple bulk endpoint: a read cursor into a byte buffer (IN) or a write cursor (OUT).
struct Endpoint {
    std::vector<uint8_t> data;
    size_t pos = 0, remain = 0;
};

std::vector<uint8_t> descriptor(std::initializer_list<int> bytes) {
    std::vector<uint8_t> v;
    for (int b : bytes) v.push_back(uint8_t(b));
    return v;
}

std::vector<uint8_t> config(int total, int nendpoints, std::initializer_list<int> endpoints) {
    auto v = descriptor({0x09, 0x02, total & 0xFF, total >> 8, 0x01, 0x01, 0x00, 0x80, 0x96,
                         0x09, 0x04, 0x00, 0x00, nendpoints, 0xFF, 0x00, 0x00, 0x00});
    for (int ep : endpoints) {
        const auto e = descriptor({0x07, 0x05, ep, 0x02, 0x40, 0x00, 0x00});
        v.insert(v.end(), e.begin(), e.end());
    }
    return v;
}

// ------------------------------------------------------------------ AN2131QC: JVS bridge
class An2131qc : public usb::StandardDevice {
public:
    An2131qc() {
        device_desc_ = descriptor({0x12, 0x01, 0x00, 0x01, 0x60, 0x00, 0x00, 0x40, 0xA3, 0x0C, 0x02, 0x00, 0x08, 0x01, 0x01, 0x02, 0x00, 0x01});
        config_desc_ = config(0x58, 10, {0x01, 0x02, 0x03, 0x04, 0x05, 0x81, 0x82, 0x83, 0x84, 0x85});
        strings_ = {descriptor({0x04, 0x03, 0x00, 0x00}), descriptor({0x0A, 0x03, 'S', 0, 'E', 0, 'G', 0, 'A', 0}),
                    descriptor({0x0E, 0x03, 'B', 0, 'A', 0, 'S', 0, 'E', 0, 'B', 0x03, 0xFF, 0x0B})};
    }

protected:
    bool vendor_request(const usb::SetupPacket& s) override {
        if (trace()) std::fprintf(stderr, "[an2131qc] request %02x value=%04x index=%04x len=%u\n", s.bRequest, s.wValue, s.wIndex, s.wLength);
        ctl_.resize(s.wLength);
        for (size_t i = 0; i < ctl_.size(); ++i) ctl_[i] = uint8_t(0x50 ^ i);
        std::lock_guard lk(g_jvs_m);
        const uint8_t sense = g_jvs.address_set() ? 0 : 3;
        auto put = [&](size_t i, uint8_t v) { if (i < ctl_.size()) ctl_[i] = v; };
        // PINSA: bits 0-5 filter board DIP switches, bits 6-7 the filter board TEST/SERVICE
        // buttons (MAME's idle value 0x4B). Pressing a button toggles its bit.
        {
            const auto st = state();
            u8 pinsa = 0x4B;
            // The game also samples these, but mirroring TEST/SERVICE here as well as
            // in the JVS switch report makes the test menu see doubled / phantom
            // presses (e.g. releasing TEST advances the GUN ADJUSTMENT cursor), so
            // they are JVS-only unless VC3_PINSA_BUTTONS is set.
            static const bool pinsa_buttons = std::getenv("VC3_PINSA_BUTTONS") != nullptr;
            if (pinsa_buttons && st.test) pinsa ^= 0x80;     // filter board TEST button
            if (pinsa_buttons && st.service) pinsa ^= 0x40;  // filter board SERVICE button
            put(1, pinsa);
        }
        put(2, 0x52 | sense);  // PINSB: JVS sense line in bits 0-1
        put(3, 0x53);          // OUTB
        uint8_t status = 0;
        switch (s.bRequest) {
            case 0x16: set_in(1, g_region.data() + s.wValue, s.wIndex & 0xFF); break;          // read EEPROM 1 via EP1
            case 0x17: set_in(2, g_region.data() + 0x2000 + s.wValue, s.wIndex & 0xFF); break;  // read EEPROM 2 via EP2
            case 0x18: case 0x1C: case 0x1D: case 0x1E: case 0x1F: case 0x24: break;
            case 0x19:  // fetch JVS replies prepared by 0x20, via EP4
                put(4, uint8_t(jvs_out_.size()));
                put(5, uint8_t(jvs_out_.size() >> 8));
                ep_[4].data = jvs_out_;
                ep_[4].pos = 0;
                ep_[4].remain = jvs_out_.size();
                jvs_out_.clear();
                break;
            case 0x20:  // send JVS packets: wIndex bytes follow on EP4 OUT
                if (jvs_out_.empty()) jvs_out_ = {0, 0};
                jvs_in_.clear();
                jvs_in_expected_ = s.wIndex;
                break;
            case 0x30:
                put(4, (s.wValue & 0xFF) == 0 ? 0 : ((s.wValue & 0xFF) < 128 ? 1 : 0));
                put(5, 0);
                break;
            default: status = 0x99; break;
        }
        put(0, status);
        return true;
    }

    int bulk(int ep, usb::Pid pid, uint8_t* buf, int len) override {
        if (pid == usb::Pid::In && ep < 6) {
            auto& e = ep_[ep];
            const int n = std::min<int>(len, int(e.remain));
            if (n > 0) std::memcpy(buf, e.data.data() + e.pos, size_t(n));
            e.pos += size_t(n);
            e.remain -= size_t(n);
            return n;
        }
        if (pid == usb::Pid::Out && ep == 4) {
            const int n = std::min<int>(len, int(jvs_in_expected_ - jvs_in_.size()));
            jvs_in_.insert(jvs_in_.end(), buf, buf + n);
            if (n > 0 && jvs_in_.size() == jvs_in_expected_) process_jvs();
            return n;
        }
        return len;  // other OUT endpoints: accept and ignore
    }

private:
    Endpoint ep_[6];
    std::vector<uint8_t> jvs_in_, jvs_out_;
    size_t jvs_in_expected_ = 0;

    void set_in(int ep, const uint8_t* p, size_t n) {
        ep_[ep].data.assign(p, p + n);
        ep_[ep].pos = 0;
        ep_[ep].remain = n;
    }

    // Host data: {0, npackets, then per packet: 0, 0xE0, dest, len, body(len-1), sum}.
    // Reply (EP4): {0, npackets, then per packet: node, 0, len16, 0xE0, raw reply..., sum}.
    void process_jvs() {
        const size_t np = jvs_in_.size() > 1 ? jvs_in_[1] : 0;
        size_t p = 2;
        std::lock_guard lk(g_jvs_m);
        for (size_t k = 0; k < np && p + 4 <= jvs_in_.size(); ++k) {
            ++p;  // leading 0
            if (jvs_in_[p] != 0xE0) break;
            const uint8_t dest = jvs_in_[p + 1], len = jvs_in_[p + 2];
            p += 3;
            if (p + len > jvs_in_.size() || len == 0) break;
            int sum = dest + len;
            for (int m = 0; m < len - 1; ++m) sum += jvs_in_[p + size_t(m)];
            if ((sum & 0xFF) != jvs_in_[p + len - 1]) { p += len; continue; }
            if (trace()) {
                std::fprintf(stderr, "[jvs] -> %02x:", dest);
                for (int m = 0; m < len - 1; ++m) std::fprintf(stderr, " %02x", jvs_in_[p + size_t(m)]);
                std::fprintf(stderr, "\n");
            }
            const auto reply = g_jvs.message(dest, &jvs_in_[p], size_t(len - 1));
            p += len;
            if (reply.empty()) continue;
            int rs = 0;
            for (uint8_t b : reply) rs += b;
            const size_t body = reply.size() + 2;  // 0xE0 + reply + sum
            jvs_out_.push_back(dest == 0xFF ? 0 : dest);
            jvs_out_.push_back(0);
            jvs_out_.push_back(uint8_t(body));
            jvs_out_.push_back(uint8_t(body >> 8));
            jvs_out_.push_back(0xE0);
            jvs_out_.insert(jvs_out_.end(), reply.begin(), reply.end());
            jvs_out_.push_back(uint8_t(rs));
            jvs_out_[1] = uint8_t(jvs_out_[1] + 1);
        }
    }
};

// ------------------------------------------------------------------ AN2131SC: serial ports
class An2131sc : public usb::StandardDevice {
public:
    An2131sc() {
        device_desc_ = descriptor({0x12, 0x01, 0x00, 0x01, 0x60, 0x01, 0x00, 0x40, 0xA3, 0x0C, 0x03, 0x00, 0x10, 0x01, 0x01, 0x02, 0x00, 0x01});
        config_desc_ = config(0x3C, 6, {0x01, 0x02, 0x03, 0x81, 0x82, 0x83});
        strings_ = {descriptor({0x04, 0x03, 0x00, 0x00}), descriptor({0x0A, 0x03, 'S', 0, 'E', 0, 'G', 0, 'A', 0}),
                    descriptor({0x0E, 0x03, 'B', 0, 'A', 0, 'S', 0, 'E', 0, 'B', 0, 'D', 0})};
    }

protected:
    bool vendor_request(const usb::SetupPacket& s) override {
        if (trace()) std::fprintf(stderr, "[an2131sc] request %02x value=%04x index=%04x len=%u\n", s.bRequest, s.wValue, s.wIndex, s.wLength);
        ctl_.resize(s.wLength);
        for (size_t i = 0; i < ctl_.size(); ++i) ctl_[i] = uint8_t(0x50 ^ i);
        auto put = [&](size_t i, uint8_t v) { if (i < ctl_.size()) ctl_[i] = v; };
        put(1, 0); put(2, 0x52); put(3, 0x53); put(4, 0); put(5, 0); put(6, 0x56); put(7, 0x57);
        uint8_t status = 0;
        switch (s.bRequest) {
            case 0x16:  // read EEPROM via EP1
                ep1_.data.assign(g_region.begin() + 0x2080 + s.wValue, g_region.begin() + 0x2080 + s.wValue + (s.wIndex & 0xFF));
                ep1_.pos = 0;
                ep1_.remain = ep1_.data.size();
                break;
            case 0x17: case 0x1E: status = 0x90; break;
            case 0x1B: {  // uart1 receive: bytes available via EP3
                const size_t n = (s.wIndex == 0 || s.wIndex > response_.size()) ? response_.size() : s.wIndex;
                put(4, uint8_t(n)); put(5, uint8_t(n >> 8));
                ep3_.data.assign(response_.begin(), response_.begin() + long(n));
                response_.erase(response_.begin(), response_.begin() + long(n));
                ep3_.pos = 0;
                ep3_.remain = s.wIndex;
                break;
            }
            case 0x1D: case 0x23: case 0x25: case 0x26: case 0x27: case 0x31: break;
            case 0x2E: midi_rs232_ = uint8_t(s.wValue); break;
            case 0x2F: put(4, midi_rs232_); break;
            case 0x30: put(4, (s.wValue & 0xFF) == 0 ? 0 : ((s.wValue & 0xFF) < 128 ? 1 : 0)); put(5, 0); break;
            default: status = 0x99; break;
        }
        put(0, status);
        return true;
    }

    int bulk(int ep, usb::Pid pid, uint8_t* buf, int len) override {
        Endpoint* e = ep == 1 ? &ep1_ : ep == 3 ? &ep3_ : nullptr;
        if (pid == usb::Pid::In && e) {
            const int n = std::min<int>(len, int(std::min(e->remain, e->data.size() - e->pos)));
            if (n > 0) std::memcpy(buf, e->data.data() + e->pos, size_t(n));
            e->pos += size_t(n);
            e->remain -= size_t(std::min<int>(n, int(e->remain)));
            return n;
        }
        if (pid == usb::Pid::Out && ep == 3) {
            for (int i = 0; i < len; ++i) uart1_byte(buf[i]);
            return len;
        }
        return pid == usb::Pid::Out ? len : 0;
    }

private:
    Endpoint ep1_, ep3_;
    std::vector<uint8_t> response_;
    uint8_t packet_[4] = {}, midi_rs232_ = 0;
    int step_ = 0;

    // 4-byte packets on uart1 (MIDI-style; drives a cabinet peripheral). Replies per MAME.
    void uart1_byte(uint8_t b) {
        if (step_ == 0 && !(b & 0x80)) return;
        packet_[step_] = b;
        if (++step_ < 4) return;
        step_ = 0;
        uint8_t result = 0;
        switch (packet_[0]) {
            case 0xFF: result = 2; break;
            case 0x81: result = 1; break;
            case 0xFC: result = 3; break;
            default: result = 0; break;
        }
        if (response_.size() < 256) response_.push_back(uint8_t(result | result << 4));
    }
};

An2131qc g_qc;
An2131sc g_sc;

}  // namespace

void init() {
    load_eeproms();
    ohci::attach(0, 0, &g_qc);
    ohci::attach(0, 1, &g_sc);
}

}  // namespace vc3::io::baseboard
