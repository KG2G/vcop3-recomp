#include "updater.h"

#include "log.h"
#include "platform.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <winhttp.h>
#else
#include <spawn.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#endif

namespace vc3::updater {

std::string current_version() {
    if (const char* f = std::getenv("VC3_FAKE_VERSION"); f && *f) return f;  // (testing)
    return VC3_VERSION;
}

namespace {
namespace fs = std::filesystem;
constexpr const char* kRepo = "KG2G/vcop3-recomp";
const std::string kUserAgent = std::string("vcop3-recomp/") + VC3_VERSION;

// ---------------------------------------------------------------- a small JSON reader
struct Json {
    enum Type { kNull, kBool, kNum, kStr, kArr, kObj } type = kNull;
    bool b = false;
    double n = 0;
    std::string s;
    std::vector<Json> a;
    std::vector<std::pair<std::string, Json>> o;
    const Json* get(const char* k) const {
        for (const auto& [key, v] : o)
            if (key == k) return &v;
        return nullptr;
    }
    std::string str(const char* k) const {
        const Json* v = get(k);
        return v && v->type == kStr ? v->s : std::string();
    }
};

struct Parser {
    const std::string& t;
    size_t i = 0;
    bool ok = true;
    void ws() { while (i < t.size() && std::strchr(" \t\r\n", t[i])) ++i; }
    static void utf8(std::string& out, unsigned cp) {
        if (cp < 0x80) out += char(cp);
        else if (cp < 0x800) out += char(0xC0 | cp >> 6), out += char(0x80 | (cp & 0x3F));
        else if (cp < 0x10000) out += char(0xE0 | cp >> 12), out += char(0x80 | ((cp >> 6) & 0x3F)), out += char(0x80 | (cp & 0x3F));
        else out += char(0xF0 | cp >> 18), out += char(0x80 | ((cp >> 12) & 0x3F)), out += char(0x80 | ((cp >> 6) & 0x3F)),
             out += char(0x80 | (cp & 0x3F));
    }
    unsigned hex4() {
        if (i + 4 > t.size()) { ok = false; return 0; }
        const unsigned v = unsigned(std::strtoul(t.substr(i, 4).c_str(), nullptr, 16));
        i += 4;
        return v;
    }
    std::string string() {
        std::string out;
        ++i;  // "
        while (i < t.size() && t[i] != '"') {
            char c = t[i++];
            if (c != '\\') { out += c; continue; }
            if (i >= t.size()) break;
            c = t[i++];
            switch (c) {
                case 'n': out += '\n'; break;
                case 't': out += '\t'; break;
                case 'r': break;
                case 'b': case 'f': break;
                case 'u': {
                    unsigned cp = hex4();
                    if (cp >= 0xD800 && cp < 0xDC00 && i + 6 <= t.size() && t[i] == '\\' && t[i + 1] == 'u') {
                        i += 2;
                        const unsigned lo = hex4();
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    }
                    utf8(out, cp);
                    break;
                }
                default: out += c;  // " \ /
            }
        }
        ++i;
        return out;
    }
    Json value(int depth = 0) {
        Json v;
        ws();
        if (i >= t.size() || depth > 64) { ok = false; return v; }
        const char c = t[i];
        if (c == '{') {
            v.type = Json::kObj;
            ++i;
            ws();
            if (i < t.size() && t[i] == '}') { ++i; return v; }
            while (ok && i < t.size()) {
                ws();
                if (t[i] != '"') { ok = false; break; }
                std::string k = string();
                ws();
                if (i >= t.size() || t[i] != ':') { ok = false; break; }
                ++i;
                v.o.emplace_back(std::move(k), value(depth + 1));
                ws();
                if (i < t.size() && t[i] == ',') { ++i; continue; }
                if (i < t.size() && t[i] == '}') { ++i; break; }
                ok = false;
            }
        } else if (c == '[') {
            v.type = Json::kArr;
            ++i;
            ws();
            if (i < t.size() && t[i] == ']') { ++i; return v; }
            while (ok && i < t.size()) {
                v.a.push_back(value(depth + 1));
                ws();
                if (i < t.size() && t[i] == ',') { ++i; continue; }
                if (i < t.size() && t[i] == ']') { ++i; break; }
                ok = false;
            }
        } else if (c == '"') {
            v.type = Json::kStr;
            v.s = string();
        } else if (!t.compare(i, 4, "true")) {
            v.type = Json::kBool, v.b = true, i += 4;
        } else if (!t.compare(i, 5, "false")) {
            v.type = Json::kBool, i += 5;
        } else if (!t.compare(i, 4, "null")) {
            i += 4;
        } else {
            char* end = nullptr;
            v.type = Json::kNum;
            v.n = std::strtod(t.c_str() + i, &end);
            if (end == t.c_str() + i) ok = false;
            i = size_t(end - t.c_str());
        }
        return v;
    }
};

// ---------------------------------------------------------------- SHA-256
struct Sha256 {
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    uint8_t buf[64];
    uint64_t len = 0;
    size_t fill = 0;
    static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
    void block(const uint8_t* p) {
        static const uint32_t k[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01,
            0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
            0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
            0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08,
            0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
            0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) w[i] = uint32_t(p[4 * i]) << 24 | uint32_t(p[4 * i + 1]) << 16 | uint32_t(p[4 * i + 2]) << 8 | p[4 * i + 3];
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t t1 = hh + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + k[i] + w[i];
            const uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
            hh = g, g = f, f = e, e = d + t1, d = c, c = b, b = a, a = t1 + t2;
        }
        h[0] += a, h[1] += b, h[2] += c, h[3] += d, h[4] += e, h[5] += f, h[6] += g, h[7] += hh;
    }
    void update(const uint8_t* p, size_t n) {
        len += n;
        while (n) {
            const size_t k = std::min(n, 64 - fill);
            std::memcpy(buf + fill, p, k);
            fill += k, p += k, n -= k;
            if (fill == 64) block(buf), fill = 0;
        }
    }
    std::string hex() {
        const uint64_t bits = len * 8;
        const uint8_t one = 0x80, zero = 0;
        update(&one, 1);
        while (fill != 56) update(&zero, 1);
        uint8_t l[8];
        for (int i = 0; i < 8; ++i) l[i] = uint8_t(bits >> (56 - 8 * i));
        update(l, 8);
        char out[65];
        for (int i = 0; i < 8; ++i) std::snprintf(out + 8 * i, 9, "%08x", h[i]);
        return out;
    }
};

std::string sha256_file(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return "";
    Sha256 s;
    std::vector<char> b(1 << 16);
    while (f) {
        f.read(b.data(), std::streamsize(b.size()));
        s.update(reinterpret_cast<const uint8_t*>(b.data()), size_t(f.gcount()));
    }
    return s.hex();
}

// ---------------------------------------------------------------- HTTPS downloads
// Fetches url into the file `to`. expected > 0: the size, for the progress (in *got).
bool fetch(const std::string& url, const fs::path& to, std::atomic<long long>* got, bool api) {
    if (url.rfind("https://", 0) != 0) return false;
#ifdef _WIN32
    auto wide = [](const std::string& s) {
        const int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
        std::wstring w(n > 0 ? size_t(n - 1) : 0, L'\0');
        if (n > 1) MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, w.data(), n);
        return w;
    };
    const std::wstring wurl = wide(url), ua = wide(kUserAgent);
    URL_COMPONENTS uc{};
    uc.dwStructSize = sizeof uc;
    wchar_t host[256] = {}, path[2048] = {};
    uc.lpszHostName = host, uc.dwHostNameLength = 256;
    uc.lpszUrlPath = path, uc.dwUrlPathLength = 2048;
    wchar_t extra[2048] = {};
    uc.lpszExtraInfo = extra, uc.dwExtraInfoLength = 2048;
    if (!WinHttpCrackUrl(wurl.c_str(), 0, 0, &uc)) return false;
    bool ok = false;
    HINTERNET s = WinHttpOpen(ua.c_str(), WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY, WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
    if (!s) s = WinHttpOpen(ua.c_str(), WINHTTP_ACCESS_TYPE_DEFAULT_PROXY, WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
    if (!s) return false;
    WinHttpSetTimeouts(s, 10000, 10000, 15000, 30000);
    if (HINTERNET c = WinHttpConnect(s, host, uc.nPort, 0)) {
        const std::wstring target = std::wstring(path) + extra;
        if (HINTERNET r = WinHttpOpenRequest(c, L"GET", target.c_str(), nullptr, WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES,
                                             WINHTTP_FLAG_SECURE)) {
            if (api) WinHttpAddRequestHeaders(r, L"Accept: application/vnd.github+json", DWORD(-1), WINHTTP_ADDREQ_FLAG_ADD);
            if (WinHttpSendRequest(r, WINHTTP_NO_ADDITIONAL_HEADERS, 0, WINHTTP_NO_REQUEST_DATA, 0, 0, 0) &&
                WinHttpReceiveResponse(r, nullptr)) {
                DWORD status = 0, sz = sizeof status;
                WinHttpQueryHeaders(r, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER, WINHTTP_HEADER_NAME_BY_INDEX, &status, &sz,
                                    WINHTTP_NO_HEADER_INDEX);
                if (status == 200)
                    if (std::FILE* out = _wfopen(to.wstring().c_str(), L"wb")) {
                        std::vector<char> b(1 << 16);
                        ok = true;
                        for (;;) {
                            DWORD n = 0;
                            if (!WinHttpReadData(r, b.data(), DWORD(b.size()), &n)) { ok = false; break; }
                            if (!n) break;
                            if (std::fwrite(b.data(), 1, n, out) != n) { ok = false; break; }
                            if (got) *got += n;
                        }
                        if (std::fclose(out) != 0) ok = false;
                    }
            }
            WinHttpCloseHandle(r);
        }
        WinHttpCloseHandle(c);
    }
    WinHttpCloseHandle(s);
    return ok;
#else
    // The system's curl (or wget), without the AppImage's library path (its bundled libraries
    // aren't the ones the system's tools were built against).
    std::vector<std::string> env;
    for (char** e = environ; *e; ++e)
        if (std::strncmp(*e, "LD_LIBRARY_PATH=", 16) && std::strncmp(*e, "LD_PRELOAD=", 11)) env.emplace_back(*e);
    std::vector<char*> envp;
    for (std::string& e : env) envp.push_back(e.data());
    envp.push_back(nullptr);
    const std::string out = to.string();
    const std::vector<std::vector<std::string>> tools = {
        {"curl", "-fsSL", "--max-time", api ? "20" : "1800", "-A", kUserAgent, "-H",
         api ? "Accept: application/vnd.github+json" : "Accept: */*", "-o", out, url},
        {"wget", "-q", "-T", "30", "-U", kUserAgent, "-O", out, url}};
    for (const auto& args : tools) {
        std::vector<std::string> a = args;
        std::vector<char*> argv;
        for (std::string& s : a) argv.push_back(s.data());
        argv.push_back(nullptr);
        pid_t pid;
        if (posix_spawnp(&pid, argv[0], nullptr, nullptr, argv.data(), envp.data()) != 0) continue;  // not installed
        int st = 0;
        for (;;) {
            const pid_t r = waitpid(pid, &st, WNOHANG);
            if (r == pid || r < 0) break;
            if (got) {
                std::error_code ec;
                const auto n = fs::file_size(to, ec);
                if (!ec) *got = (long long)n;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        if (WIFEXITED(st) && WEXITSTATUS(st) == 0) return true;
        std::error_code ec;
        fs::remove(to, ec);
    }
    return false;
#endif
}

// ---------------------------------------------------------------- state
std::atomic<State> g_state{State::kIdle};
std::atomic<long long> g_got{0}, g_size{0};
std::mutex g_m;
std::string g_latest, g_changelog, g_error, g_url, g_sha256;

std::vector<int> parse_version(std::string v) {
    if (!v.empty() && (v[0] == 'v' || v[0] == 'V')) v.erase(0, 1);
    std::vector<int> out;
    std::stringstream ss(v);
    std::string part;
    while (std::getline(ss, part, '.')) out.push_back(std::atoi(part.c_str()));
    while (out.size() < 3) out.push_back(0);
    return out;
}

// Release notes as plain text: no Markdown emphasis or code marks, and typographic punctuation as
// ASCII (the launcher's font has no glyphs for it).
std::string plain(std::string s) {
    static const std::pair<const char*, const char*> kAscii[] = {
        {"\u2014", "-"}, {"\u2013", "-"}, {"\u2018", "'"}, {"\u2019", "'"}, {"\u201c", "\""}, {"\u201d", "\""},
        {"\u2026", "..."}, {"\u2192", "->"}, {"\u00a0", " "}, {"\u2022", "-"}};
    for (const auto& [from, to] : kAscii)
        for (size_t at; (at = s.find(from)) != std::string::npos;) s.replace(at, std::strlen(from), to);
    std::string out;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '`') continue;
        if (s[i] == '*' && i + 1 < s.size() && s[i + 1] == '*') { ++i; continue; }
        out += s[i];
    }
    while (!out.empty() && (out.back() == '\n' || out.back() == ' ')) out.pop_back();
    return out;
}

const char* asset_suffix() {
#ifdef _WIN32
    return "windows-x64.exe";
#else
    return ".AppImage";
#endif
}

// The file to replace: the AppImage itself (not its temporary mount), or the .exe.
fs::path target() {
#ifdef _WIN32
    return platform::exe_path();
#else
    const char* ai = std::getenv("APPIMAGE");
    return ai && *ai ? fs::path(ai) : fs::path();
#endif
}

bool writable_dir(const fs::path& dir) {
    const fs::path probe = dir / ".vc3-write-test";
    std::error_code ec;
    {
        std::ofstream f(probe);
        if (!f) return false;
    }
    fs::remove(probe, ec);
    return true;
}

void check() {
    const fs::path tmp = fs::temp_directory_path() / ("vc3-releases-" + std::to_string(
#ifdef _WIN32
        GetCurrentProcessId()
#else
        getpid()
#endif
        ) + ".json");
    const std::string url = std::string("https://api.github.com/repos/") + kRepo + "/releases?per_page=20";
    std::fprintf(stderr, "[update] checking for a newer version (have %s)\n", current_version().c_str());
    if (!fetch(url, tmp, nullptr, true)) {
        std::lock_guard lk(g_m);
        g_error = "Couldn't reach GitHub to check for updates.";
        std::fprintf(stderr, "[update] check failed (no connection?)\n");
        g_state = State::kFailed;
        return;
    }
    std::string text;
    {
        std::ifstream f(tmp, std::ios::binary);
        std::stringstream ss;
        ss << f.rdbuf();
        text = ss.str();
    }
    std::error_code ec;
    fs::remove(tmp, ec);
    Parser p{text};
    const Json root = p.value();
    if (!p.ok || root.type != Json::kArr) {
        std::lock_guard lk(g_m);
        g_error = "GitHub's answer couldn't be read.";
        g_state = State::kFailed;
        return;
    }
    const auto have = parse_version(current_version());
    std::vector<const Json*> newer;
    for (const Json& r : root.a) {
        const Json* draft = r.get("draft");
        const Json* pre = r.get("prerelease");
        if ((draft && draft->b) || (pre && pre->b)) continue;
        if (parse_version(r.str("tag_name")) > have) newer.push_back(&r);
    }
    if (newer.empty()) {
        std::fprintf(stderr, "[update] up to date\n");
        g_state = State::kUpToDate;
        return;
    }
    std::sort(newer.begin(), newer.end(),
              [](const Json* a, const Json* b) { return parse_version(a->str("tag_name")) > parse_version(b->str("tag_name")); });
    std::string log;
    for (const Json* r : newer) {
        log += "Version " + r->str("tag_name").substr(r->str("tag_name").rfind('v') == 0 ? 1 : 0) + "\n\n";
        log += plain(r->str("body")) + "\n\n";
    }
    std::string url_dl, sha;
    long long size = 0;
    if (const Json* assets = newer.front()->get("assets"))
        for (const Json& a : assets->a) {
            const std::string name = a.str("name");
            if (name.size() >= std::strlen(asset_suffix()) &&
                !name.compare(name.size() - std::strlen(asset_suffix()), std::string::npos, asset_suffix())) {
                url_dl = a.str("browser_download_url");
                const std::string d = a.str("digest");
                if (d.rfind("sha256:", 0) == 0) sha = d.substr(7);
                if (const Json* s = a.get("size")) size = (long long)s->n;
            }
        }
    {
        std::lock_guard lk(g_m);
        std::string tag = newer.front()->str("tag_name");
        if (!tag.empty() && tag[0] == 'v') tag.erase(0, 1);
        g_latest = tag;
        g_changelog = log;
        g_url = url_dl;
        g_sha256 = sha;
        g_size = size;
    }
    std::fprintf(stderr, "[update] version %s is available\n", g_latest.c_str());
    g_state = State::kAvailable;
}

void install() {
    std::string url, sha;
    {
        std::lock_guard lk(g_m);
        url = g_url;
        sha = g_sha256;
    }
    const fs::path t = target();
    const fs::path dl = fs::path(t.string() + ".download");
    auto fail = [&](const std::string& why) {
        std::error_code ec;
        fs::remove(dl, ec);
        std::lock_guard lk(g_m);
        g_error = why;
        std::fprintf(stderr, "[update] install failed: %s\n", why.c_str());
        g_state = State::kInstallFailed;
    };
    g_got = 0;
    if (!fetch(url, dl, &g_got, false)) return fail("The download didn't complete.");
    std::error_code ec;
    if (g_size > 0 && (long long)fs::file_size(dl, ec) != g_size) return fail("The download is incomplete.");
    if (sha.empty() || sha256_file(dl) != sha) return fail("The download doesn't match the release (checksum).");
#ifdef _WIN32
    const fs::path old = fs::path(t.string() + ".old");
    DeleteFileW(old.wstring().c_str());
    if (!MoveFileExW(t.wstring().c_str(), old.wstring().c_str(), MOVEFILE_REPLACE_EXISTING))
        return fail("The program file couldn't be replaced.");
    if (!MoveFileExW(dl.wstring().c_str(), t.wstring().c_str(), MOVEFILE_REPLACE_EXISTING)) {
        MoveFileExW(old.wstring().c_str(), t.wstring().c_str(), MOVEFILE_REPLACE_EXISTING);
        return fail("The program file couldn't be replaced.");
    }
#else
    chmod(dl.c_str(), 0755);
    fs::rename(dl, t, ec);  // (the running AppImage keeps its old file open until it exits)
    if (ec) return fail("The program file couldn't be replaced.");
#endif
    std::fprintf(stderr, "[update] installed version %s\n", g_latest.c_str());
    g_state = State::kInstalled;
}
}  // namespace

void start_check() {
    State idle = State::kIdle;
    if (!g_state.compare_exchange_strong(idle, State::kChecking)) return;
    std::thread(check).detach();
}

State state() { return g_state.load(); }
std::string latest_version() { std::lock_guard lk(g_m); return g_latest; }
std::string changelog() { std::lock_guard lk(g_m); return g_changelog; }
std::string error() { std::lock_guard lk(g_m); return g_error; }
float progress() {
    const long long s = g_size.load();
    return s > 0 ? std::clamp(float(g_got.load()) / float(s), 0.0f, 1.0f) : 0.0f;
}

bool can_install() {
    const fs::path t = target();
    std::lock_guard lk(g_m);
    return !t.empty() && !g_url.empty() && !g_sha256.empty() && writable_dir(t.parent_path());
}

void start_install() {
    State s = g_state.load();
    if (s != State::kAvailable && s != State::kInstallFailed) return;
    g_state = State::kDownloading;
    std::thread(install).detach();
}

void restart() {
    if (g_state != State::kInstalled) return;
    const fs::path t = target();
    std::vector<std::string> args = platform::args();
    if (args.empty()) args.push_back(t.string());
    args[0] = t.string();
    log::end("updated: restarting the new version");
    std::fflush(stderr);
#ifdef _WIN32
    std::wstring cmd;
    for (const std::string& a : args) {
        const int n = MultiByteToWideChar(CP_UTF8, 0, a.c_str(), -1, nullptr, 0);
        std::wstring w(n > 0 ? size_t(n - 1) : 0, L'\0');
        if (n > 1) MultiByteToWideChar(CP_UTF8, 0, a.c_str(), -1, w.data(), n);
        if (!cmd.empty()) cmd += L' ';
        cmd += L'"' + w + L'"';
    }
    STARTUPINFOW si{};
    si.cb = sizeof si;
    PROCESS_INFORMATION pi{};
    if (CreateProcessW(t.wstring().c_str(), cmd.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr, &si, &pi)) {
        CloseHandle(pi.hThread);
        CloseHandle(pi.hProcess);
        std::_Exit(0);
    }
#else
    // A new AppImage process of its own (it mounts the new file); this one then ends, and the
    // AppImage it ran from unmounts.
    if (fork() == 0) {
        setsid();
        std::vector<char*> argv;
        for (std::string& a : args) argv.push_back(a.data());
        argv.push_back(nullptr);
        execv(t.c_str(), argv.data());
        std::_Exit(127);
    }
    std::_Exit(0);
#endif
}

std::string releases_page() { return std::string("https://github.com/") + kRepo + "/releases/latest"; }

void cleanup() {
    const fs::path t = target();
    if (t.empty()) return;
    std::error_code ec;
#ifdef _WIN32
    fs::remove(fs::path(t.string() + ".old"), ec);
#endif
    fs::remove(fs::path(t.string() + ".download"), ec);
}

}  // namespace vc3::updater
