// Nt*File / Io* : the guest file system, backed by host directories.
//
// Object paths ("\??\D:\media\x.bin", "\Device\Harddisk0\Partition1\vc3.xbe")
// are resolved through a symbolic-link table to a device, and devices map to
// host directories. Lookups are case-insensitive like FATX.
#include <chrono>
#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef _WIN32
#include <io.h>
#include <mutex>
// Windows: files open in binary mode (no newline translation), and the POSIX calls the
// MSVC runtime lacks, done with seek + read/write under a lock (guest threads share fds).
#define VC3_OBIN O_BINARY
namespace {
std::mutex g_pio_m;
ssize_t pread(int fd, void* buf, size_t n, long long off) {
    std::lock_guard lk(g_pio_m);
    if (_lseeki64(fd, off, SEEK_SET) < 0) return -1;
    return _read(fd, buf, unsigned(n));
}
ssize_t pwrite(int fd, const void* buf, size_t n, long long off) {
    std::lock_guard lk(g_pio_m);
    if (_lseeki64(fd, off, SEEK_SET) < 0) return -1;
    return _write(fd, buf, unsigned(n));
}
int ftruncate(int fd, long long size) { return _chsize_s(fd, size) == 0 ? 0 : -1; }
}  // namespace
#else
#define VC3_OBIN 0
#endif

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <mutex>
#include <vector>

#include "scheduler.h"
#include "io.h"
#include "kernel.h"
#include "objects.h"

namespace vc3::kernel {

namespace fs = std::filesystem;

namespace {

std::mutex g_io_m;
std::map<std::string, std::string> g_symlinks;  // lower-case "\??\d:" -> device path

struct Device {
    fs::path dir;  // file-system view (empty: raw only)
    fs::path raw;  // raw volume image (empty: no raw access)
    u64 size = 0;  // raw size in bytes
    u32 partition = 0;
    io::RawFn rd, wr;  // raw access handled by a device model instead of an image file
};
std::map<std::string, Device> g_devices;  // lower-case device path -> device

std::string lower(std::string s) {
    for (auto& ch : s) ch = char(std::tolower(u8(ch)));
    return s;
}

// Xbox hard-disk layout (bytes). Partition 0 is the whole disk; its first 512 KiB
// hold the config area (XAPI's utility-drive cache database lives at 0x800).
constexpr u64 kPartitionSize[] = {0x1DD1B0000ull /*0*/, 0x1312D6000ull, 0x1F400000ull, 0x2EE00000ull, 0x2EE00000ull,
                                  0x2EE00000ull, 0x2EE00000ull};

void init_namespace() {
    static bool done = false;
    if (done) return;
    done = true;
    const char* root = std::getenv("VC3_GAME_DIR");
    const fs::path game = root ? root : "work/fs";
    const fs::path hdd = "work/hdd";
    fs::create_directories(hdd);
    // The game image is read-only media, exposed the way a disc would be.
    g_devices["\\device\\cdrom0"] = Device{game, {}, 0, 0};
    for (u32 p = 0; p <= 6; ++p) {
        Device d;
        d.partition = p;
        d.size = kPartitionSize[p];
        d.raw = hdd / ("partition" + std::to_string(p) + ".img");
        if (p) {
            d.dir = hdd / ("partition" + std::to_string(p));
            fs::create_directories(d.dir);
        }
        if (!fs::exists(d.raw) || fs::file_size(d.raw) != d.size) {
            // Sparse image of the full partition size: unwritten sectors read as zeros.
            const int fd = ::open(d.raw.string().c_str(), O_CREAT | O_RDWR | VC3_OBIN, 0644);
            if (fd >= 0) { if (ftruncate(fd, int64_t(d.size))) {} ::close(fd); }
        }
        g_devices["\\device\\harddisk0\\partition" + std::to_string(p)] = d;
    }
    g_symlinks["\\??\\d:"] = "\\device\\cdrom0";
}

// Resolve `p` case-insensitively under `base`. Missing trailing components are
// returned as-is (for create).
fs::path resolve_ci(const fs::path& base, const std::string& rel) {
    fs::path cur = base;
    size_t i = 0;
    while (i < rel.size()) {
        size_t j = rel.find('\\', i);
        if (j == std::string::npos) j = rel.size();
        const std::string comp = rel.substr(i, j - i);
        i = j + 1;
        if (comp.empty() || comp == ".") continue;
        fs::path next = cur / comp;
        if (!fs::exists(next)) {
            std::error_code ec;
            for (const auto& e : fs::directory_iterator(cur, ec))
                if (lower(e.path().filename().string()) == lower(comp)) { next = e.path(); break; }
        }
        cur = next;
    }
    return cur;
}

// Object path -> device + remaining path (after following symbolic links).
const Device* to_device(std::string path, std::string& rest) {
    init_namespace();
    std::string l = lower(path);
    for (int depth = 0; depth < 8; ++depth) {
        bool changed = false;
        for (const auto& [link, target] : g_symlinks) {
            if (l.rfind(link, 0) == 0 && (l.size() == link.size() || l[link.size()] == '\\')) {
                path = target + path.substr(link.size());
                l = lower(path);
                changed = true;
                break;
            }
        }
        if (!changed) break;
    }
    for (const auto& [dev, d] : g_devices) {
        if (l.rfind(dev, 0) == 0 && (l.size() == dev.size() || l[dev.size()] == '\\')) {
            rest = path.substr(dev.size());
            return &d;
        }
    }
    return nullptr;
}

bool to_host(const std::string& path, fs::path& out) {
    std::string rest;
    const Device* d = to_device(path, rest);
    if (!d || d->dir.empty()) return false;
    out = resolve_ci(d->dir, rest);
    return true;
}

struct File : HostObject {
    int fd = -1;
    bool is_dir = false;
    const Device* raw_device = nullptr;  // non-null: raw volume handle
    u64 pos = 0;
    fs::path host;
    std::vector<std::string> dir_entries;  // for NtQueryDirectoryFile
    size_t dir_index = 0;
    ~File() override { if (fd >= 0) ::close(fd); }
};

std::shared_ptr<File> get_file(u32 h) { return std::dynamic_pointer_cast<File>(handle_host(h)); }

// OBJECT_ATTRIBUTES { HANDLE RootDirectory; POBJECT_STRING ObjectName; ULONG Attributes }
std::string object_path(u32 oa) {
    std::string name = read_ansi_string(rd<u32>(oa + 4));
    const u32 root = rd<u32>(oa);
    if (root == 0xFFFFFFFD) return "\\??\\" + name;                     // ObDosDevicesDirectory()
    if (root == 0xFFFFFFFC) return "\\Win32NamedObjects\\" + name;      // ObWin32NamedObjectsDirectory()
    if (root) {
        if (auto f = get_file(root)) {
            // Relative to an open directory: rebuild an absolute device path is not needed,
            // keep host-relative by tagging.
            return "\\<host>" + f->host.string() + "\\" + name;
        }
    }
    return name;
}

bool resolve_object_path(const std::string& p, fs::path& out) {
    if (p.rfind("\\<host>", 0) == 0) {
        const std::string rest = p.substr(7);
        const size_t cut = rest.find('\\');
        out = resolve_ci(rest.substr(0, cut), cut == std::string::npos ? "" : rest.substr(cut + 1));
        return true;
    }
    return to_host(p, out);
}

void write_iosb(u32 iosb, u32 status, u32 info) {
    if (!iosb) return;
    wr<u32>(iosb, status);
    wr<u32>(iosb + 4, info);
}

s64 to_filetime(const struct stat& st) {
#ifdef _WIN32
    return s64(st.st_mtime) * 10000000 + 116444736000000000LL;
#else
    return s64(st.st_mtim.tv_sec) * 10000000 + st.st_mtim.tv_nsec / 100 + 116444736000000000LL;
#endif
}
u32 attributes(const struct stat& st) { return S_ISDIR(st.st_mode) ? 0x10 : 0x80; }  // DIRECTORY / NORMAL

// FILE_NETWORK_OPEN_INFORMATION (56 bytes)
void write_network_open_info(u32 out, const struct stat& st) {
    const s64 t = to_filetime(st);
    for (u32 o = 0; o < 32; o += 8) wr<s64>(out + o, t);
    wr<s64>(out + 32, (s64(st.st_size) + 0xFFF) & ~0xFFFLL);
    wr<s64>(out + 40, st.st_size);
    wr<u32>(out + 48, attributes(st));
    wr<u32>(out + 52, 0);
}

u32 open_common(u32 phandle, u32 oa, u32 iosb, u32 disposition, u32 options, bool write_access) {
    const std::string path = object_path(oa);
    {
        // Opening a device itself (no trailing path) is raw volume access.
        std::string rest;
        const Device* d = path.rfind("\\<host>", 0) == 0 ? nullptr : to_device(path, rest);
        if (d && rest.empty() && (!d->raw.empty() || d->rd)) {
            auto f = std::make_shared<File>();
            f->raw_device = d;
            f->host = d->raw.empty() ? fs::path(path) : d->raw;
            if (!d->rd) {
                f->fd = ::open(d->raw.string().c_str(), O_RDWR | VC3_OBIN);
                if (f->fd < 0) { write_iosb(iosb, STATUS_ACCESS_DENIED, 0); return STATUS_ACCESS_DENIED; }
            }
            wr<u32>(phandle, create_handle(alloc_object(0x40, data_export_addr("IoFileObjectType")), f));
            write_iosb(iosb, STATUS_SUCCESS, 1);
            if (trace_enabled()) std::fprintf(stderr, "[io] open %s -> raw volume %s\n", path.c_str(), d->raw.string().c_str());
            return STATUS_SUCCESS;
        }
    }
    fs::path host;
    if (!resolve_object_path(path, host)) {
        std::fprintf(stderr, "[io] open %s: no device\n", path.c_str());
        write_iosb(iosb, STATUS_OBJECT_PATH_NOT_FOUND, 0);
        return STATUS_OBJECT_PATH_NOT_FOUND;
    }
    struct stat st{};
    const bool exists = ::stat(host.string().c_str(), &st) == 0;
    const bool want_dir = options & 0x1;       // FILE_DIRECTORY_FILE
    const bool want_file = options & 0x40;     // FILE_NON_DIRECTORY_FILE
    u32 info = 1;                              // FILE_OPENED

    if (!exists) {
        if (disposition == 1 /*OPEN*/ || disposition == 4 /*OVERWRITE*/) {
            if (trace_enabled()) std::fprintf(stderr, "[io] open %s -> %s: not found\n", path.c_str(), host.string().c_str());
            write_iosb(iosb, STATUS_OBJECT_NAME_NOT_FOUND, 5);
            return STATUS_OBJECT_NAME_NOT_FOUND;
        }
        if (!fs::exists(host.parent_path())) {
            write_iosb(iosb, STATUS_OBJECT_PATH_NOT_FOUND, 0);
            return STATUS_OBJECT_PATH_NOT_FOUND;
        }
        if (want_dir) fs::create_directory(host);
        else { const int fd = ::open(host.string().c_str(), O_CREAT | O_RDWR | VC3_OBIN, 0644); if (fd >= 0) ::close(fd); }
        ::stat(host.string().c_str(), &st);
        info = 2;  // FILE_CREATED
    } else if (disposition == 2 /*CREATE*/) {
        write_iosb(iosb, STATUS_OBJECT_NAME_COLLISION, 4);
        return STATUS_OBJECT_NAME_COLLISION;
    }

    const bool is_dir = S_ISDIR(st.st_mode);
    if (is_dir && want_file) { write_iosb(iosb, STATUS_FILE_IS_A_DIRECTORY, 0); return STATUS_FILE_IS_A_DIRECTORY; }
    if (!is_dir && want_dir) { write_iosb(iosb, STATUS_NOT_A_DIRECTORY, 0); return STATUS_NOT_A_DIRECTORY; }

    auto f = std::make_shared<File>();
    f->host = host;
    f->is_dir = is_dir;
    if (!is_dir) {
        int flags = write_access ? O_RDWR : O_RDONLY;
        if (exists && (disposition == 0 || disposition == 4 || disposition == 5)) { flags |= O_TRUNC; info = 3; }
        f->fd = ::open(host.string().c_str(), flags | VC3_OBIN);
        if (f->fd < 0 && write_access) f->fd = ::open(host.string().c_str(), O_RDONLY | VC3_OBIN);  // read-only game media
        if (f->fd < 0) { write_iosb(iosb, STATUS_ACCESS_DENIED, 0); return STATUS_ACCESS_DENIED; }
    }
    const u32 body = alloc_object(0x40, data_export_addr("IoFileObjectType"));
    wr<u32>(phandle, create_handle(body, f));
    write_iosb(iosb, STATUS_SUCCESS, info);
    if (trace_enabled()) std::fprintf(stderr, "[io] open %s -> %s\n", path.c_str(), host.string().c_str());
    return STATUS_SUCCESS;
}

constexpr u32 kGenericWrite = 0x40000000, kFileWriteData = 0x2, kFileAppendData = 0x4;

}  // namespace

// NtOpenFile(PHANDLE, ACCESS_MASK, POBJECT_ATTRIBUTES, PIO_STATUS_BLOCK, ULONG ShareAccess, ULONG OpenOptions)
KEXPORT(NtOpenFile) {
    std::lock_guard lk(g_io_m);
    return open_common(a[0], a[2], a[3], 1, a[5], a[1] & (kGenericWrite | kFileWriteData | kFileAppendData));
}
// NtCreateFile(PHANDLE, ACCESS_MASK, POBJECT_ATTRIBUTES, PIO_STATUS_BLOCK, PLARGE_INTEGER AllocationSize,
//              ULONG FileAttributes, ULONG ShareAccess, ULONG CreateDisposition, ULONG CreateOptions)
KEXPORT(NtCreateFile) {
    std::lock_guard lk(g_io_m);
    return open_common(a[0], a[2], a[3], a[7], a[8], a[1] & (kGenericWrite | kFileWriteData | kFileAppendData));
}

// NtReadFile(HANDLE, HANDLE Event, PIO_APC_ROUTINE, PVOID ApcContext, PIO_STATUS_BLOCK, PVOID Buffer, ULONG Length,
//            PLARGE_INTEGER ByteOffset)
KEXPORT(NtReadFile) {
    auto f = get_file(a[0]);
    if (!f || f->is_dir) return STATUS_INVALID_HANDLE;
    if (a[2]) std::fprintf(stderr, "[io] NtReadFile with APC routine %08x: completion APC not delivered\n", a[2]);
    u64 off = f->pos;
    if (a[7]) {
        const s64 o = rd<s64>(a[7]);
        if (o >= 0) off = u64(o);  // -2 (FILE_USE_FILE_POINTER_POSITION) keeps the current position
    }
    const auto t0 = std::chrono::steady_clock::now();
    const ssize_t n = (f->raw_device && f->raw_device->rd) ? f->raw_device->rd(off, host_ptr(a[5]), a[6])
                                                           : pread(f->fd, host_ptr(a[5]), a[6], int64_t(off));
    sched::g_io_ns += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
    const u32 status = n < 0 ? STATUS_UNSUCCESSFUL : (n == 0 && a[6] ? STATUS_END_OF_FILE : STATUS_SUCCESS);
    if (n > 0) f->pos = off + u64(n);
    write_iosb(a[4], status, n > 0 ? u32(n) : 0);
    if (a[1]) if (const u32 ev = handle_to_object(a[1])) set_event(ev);
    return status;
}
// NtWriteFile: same parameters as NtReadFile.
KEXPORT(NtWriteFile) {
    auto f = get_file(a[0]);
    if (!f || f->is_dir) return STATUS_INVALID_HANDLE;
    u64 off = f->pos;
    if (a[7]) {
        const s64 o = rd<s64>(a[7]);
        if (o >= 0) off = u64(o);
    }
    const ssize_t n = (f->raw_device && f->raw_device->wr) ? f->raw_device->wr(off, host_ptr(a[5]), a[6])
                                                            : pwrite(f->fd, host_ptr(a[5]), a[6], int64_t(off));
    const u32 status = n < 0 ? STATUS_ACCESS_DENIED : STATUS_SUCCESS;
    if (n > 0) f->pos = off + u64(n);
    write_iosb(a[4], status, n > 0 ? u32(n) : 0);
    if (a[1]) if (const u32 ev = handle_to_object(a[1])) set_event(ev);
    return status;
}

// NtQueryInformationFile(HANDLE, PIO_STATUS_BLOCK, PVOID FileInformation, ULONG Length, FILE_INFORMATION_CLASS)
KEXPORT(NtQueryInformationFile) {
    auto f = get_file(a[0]);
    if (!f) return STATUS_INVALID_HANDLE;
    struct stat st{};
    if (f->fd >= 0) ::fstat(f->fd, &st); else ::stat(f->host.string().c_str(), &st);
    const u32 out = a[2];
    u32 len = 0;
    switch (a[4]) {
        case 4:  // FileBasicInformation
            for (u32 o = 0; o < 32; o += 8) wr<s64>(out + o, to_filetime(st));
            wr<u32>(out + 32, attributes(st));
            len = 40;
            break;
        case 5:  // FileStandardInformation
            wr<s64>(out, (s64(st.st_size) + 0xFFF) & ~0xFFFLL);
            wr<s64>(out + 8, st.st_size);
            wr<u32>(out + 16, 1);
            wr<u8>(out + 20, 0);
            wr<u8>(out + 21, f->is_dir);
            len = 24;
            break;
        case 14: wr<u64>(out, f->pos); len = 8; break;  // FilePositionInformation
        case 34: write_network_open_info(out, st); len = 56; break;  // FileNetworkOpenInformation
        default:
            std::fprintf(stderr, "[io] NtQueryInformationFile class %u not supported\n", a[4]);
            return STATUS_INVALID_PARAMETER;
    }
    write_iosb(a[1], STATUS_SUCCESS, len);
    return STATUS_SUCCESS;
}

// NtSetInformationFile(HANDLE, PIO_STATUS_BLOCK, PVOID FileInformation, ULONG Length, FILE_INFORMATION_CLASS)
KEXPORT(NtSetInformationFile) {
    auto f = get_file(a[0]);
    if (!f) return STATUS_INVALID_HANDLE;
    switch (a[4]) {
        case 14: f->pos = rd<u64>(a[2]); break;                                     // FilePositionInformation
        case 20: if (f->fd >= 0 && ftruncate(f->fd, int64_t(rd<u64>(a[2])))) {} break;  // FileEndOfFileInformation
        case 19: break;                                                              // FileAllocationInformation
        case 4: break;                                                               // FileBasicInformation
        case 13:                                                                     // FileDispositionInformation
            if (rd<u8>(a[2])) { std::error_code ec; fs::remove(f->host, ec); }
            break;
        default:
            std::fprintf(stderr, "[io] NtSetInformationFile class %u not supported\n", a[4]);
            return STATUS_INVALID_PARAMETER;
    }
    write_iosb(a[1], STATUS_SUCCESS, 0);
    return STATUS_SUCCESS;
}

// NtQueryFullAttributesFile(POBJECT_ATTRIBUTES, PFILE_NETWORK_OPEN_INFORMATION)
KEXPORT(NtQueryFullAttributesFile) {
    std::lock_guard lk(g_io_m);
    fs::path host;
    struct stat st{};
    if (!resolve_object_path(object_path(a[0]), host) || ::stat(host.string().c_str(), &st) != 0)
        return STATUS_OBJECT_NAME_NOT_FOUND;
    write_network_open_info(a[1], st);
    return STATUS_SUCCESS;
}

// NtQueryVolumeInformationFile(HANDLE, PIO_STATUS_BLOCK, PVOID FsInformation, ULONG Length, FS_INFORMATION_CLASS)
KEXPORT(NtQueryVolumeInformationFile) {
    const u32 out = a[2];
    switch (a[4]) {
        case 3:  // FileFsSizeInformation
            wr<s64>(out, 0x100000);     // TotalAllocationUnits
            wr<s64>(out + 8, 0x80000);  // AvailableAllocationUnits
            wr<u32>(out + 16, 32);      // SectorsPerAllocationUnit (16 KiB clusters)
            wr<u32>(out + 20, 512);     // BytesPerSector
            write_iosb(a[1], STATUS_SUCCESS, 24);
            return STATUS_SUCCESS;
        case 4:  // FileFsDeviceInformation { DeviceType, Characteristics }
            wr<u32>(out, 7);  // FILE_DEVICE_DISK
            wr<u32>(out + 4, 0);
            write_iosb(a[1], STATUS_SUCCESS, 8);
            return STATUS_SUCCESS;
        default:
            std::fprintf(stderr, "[io] NtQueryVolumeInformationFile class %u not supported\n", a[4]);
            return STATUS_INVALID_PARAMETER;
    }
}

// NtQueryDirectoryFile(HANDLE, HANDLE Event, PIO_APC_ROUTINE, PVOID ApcContext, PIO_STATUS_BLOCK, PVOID FileInformation,
//                      ULONG Length, FILE_INFORMATION_CLASS, POBJECT_STRING FileMask, BOOLEAN RestartScan)
KEXPORT(NtQueryDirectoryFile) {
    auto f = get_file(a[0]);
    if (!f || !f->is_dir) return STATUS_INVALID_HANDLE;
    if (f->dir_entries.empty() || (a[9] & 0xFF)) {
        f->dir_entries.clear();
        f->dir_index = 0;
        const std::string mask = lower(a[8] ? read_ansi_string(a[8]) : "*");
        std::error_code ec;
        for (const auto& e : fs::directory_iterator(f->host, ec)) {
            const std::string n = e.path().filename().string();
            if (mask == "*" || mask == "*.*" || lower(n) == mask ||
                (mask.front() == '*' && lower(n).ends_with(mask.substr(1))))
                f->dir_entries.push_back(n);
        }
        std::sort(f->dir_entries.begin(), f->dir_entries.end());
    }
    if (f->dir_index >= f->dir_entries.size()) {
        write_iosb(a[4], STATUS_NO_MORE_FILES, 0);
        return STATUS_NO_MORE_FILES;
    }
    // FILE_DIRECTORY_INFORMATION (one entry per call, ANSI name)
    const std::string n = f->dir_entries[f->dir_index++];
    struct stat st{};
    ::stat((f->host / n).string().c_str(), &st);
    const u32 out = a[5];
    if (a[6] < 0x40 + n.size()) return STATUS_BUFFER_TOO_SMALL;
    wr<u32>(out + 0x00, 0);  // NextEntryOffset
    wr<u32>(out + 0x04, 0);  // FileIndex
    for (u32 o = 0x08; o < 0x28; o += 8) wr<s64>(out + o, to_filetime(st));
    wr<s64>(out + 0x28, st.st_size);
    wr<s64>(out + 0x30, (s64(st.st_size) + 0xFFF) & ~0xFFFLL);
    wr<u32>(out + 0x38, attributes(st));
    wr<u32>(out + 0x3C, u32(n.size()));
    std::memcpy(host_ptr(out + 0x40), n.data(), n.size());
    write_iosb(a[4], STATUS_SUCCESS, u32(0x40 + n.size()));
    return STATUS_SUCCESS;
}

KEXPORT(NtFlushBuffersFile) {
    auto f = get_file(a[0]);
#ifdef _WIN32
    if (f && f->fd >= 0) _commit(f->fd);
#else
    if (f && f->fd >= 0) ::fsync(f->fd);
#endif
    write_iosb(a[1], STATUS_SUCCESS, 0);
    return STATUS_SUCCESS;
}

// NtDeviceIoControlFile(HANDLE, HANDLE Event, PIO_APC_ROUTINE, PVOID ApcContext, PIO_STATUS_BLOCK, ULONG IoControlCode,
//                       PVOID InputBuffer, ULONG InputLength, PVOID OutputBuffer, ULONG OutputLength)
KEXPORT(NtDeviceIoControlFile) {
    auto f = get_file(a[0]);
    if (!f) return STATUS_INVALID_HANDLE;
    const u32 code = a[5], out = a[8];
    const Device* d = f->raw_device;
    u32 info = 0;
    switch (code) {
        case 0x70000:  // IOCTL_DISK_GET_DRIVE_GEOMETRY
            if (!d) return STATUS_INVALID_PARAMETER;
            wr<s64>(out, s64(d->size / (255 * 63 * 512)));  // Cylinders
            wr<u32>(out + 8, 12);                             // FixedMedia
            wr<u32>(out + 12, 255);                           // TracksPerCylinder
            wr<u32>(out + 16, 63);                            // SectorsPerTrack
            wr<u32>(out + 20, 512);                           // BytesPerSector
            info = 24;
            break;
        case 0x74004:  // IOCTL_DISK_GET_PARTITION_INFO
            if (!d) return STATUS_INVALID_PARAMETER;
            wr<s64>(out, 0x80000);                 // StartingOffset (after the config area)
            wr<s64>(out + 8, s64(d->size));        // PartitionLength
            wr<u32>(out + 16, 0);                  // HiddenSectors
            wr<u32>(out + 20, d->partition);       // PartitionNumber
            wr<u8>(out + 24, 0x0E);                // PartitionType
            wr<u8>(out + 25, 0); wr<u8>(out + 26, 1); wr<u8>(out + 27, 0);
            info = 32;
            break;
        default:
            std::fprintf(stderr, "[io] NtDeviceIoControlFile(%s, code=%#x) not supported\n", f->host.string().c_str(), code);
            write_iosb(a[4], STATUS_INVALID_PARAMETER, 0);
            return STATUS_INVALID_PARAMETER;
    }
    write_iosb(a[4], STATUS_SUCCESS, info);
    if (a[1]) if (const u32 ev = handle_to_object(a[1])) set_event(ev);
    return STATUS_SUCCESS;
}

// NtFsControlFile: same parameters, FsControlCode.
KEXPORT(NtFsControlFile) {
    auto f = get_file(a[0]);
    if (!f) return STATUS_INVALID_HANDLE;
    switch (a[5]) {
        case 0x90018:  // FSCTL_LOCK_VOLUME
        case 0x9001C:  // FSCTL_UNLOCK_VOLUME
        case 0x90020:  // FSCTL_DISMOUNT_VOLUME
            write_iosb(a[4], STATUS_SUCCESS, 0);
            return STATUS_SUCCESS;
        default:
            std::fprintf(stderr, "[io] NtFsControlFile(%s, code=%#x) not supported\n", f->host.string().c_str(), a[5]);
            return STATUS_INVALID_PARAMETER;
    }
}

// --- symbolic links
// IoCreateSymbolicLink(POBJECT_STRING SymbolicLinkName, POBJECT_STRING DeviceName)
KEXPORT(IoCreateSymbolicLink) {
    std::lock_guard lk(g_io_m);
    init_namespace();
    const std::string link = lower(read_ansi_string(a[0])), dev = read_ansi_string(a[1]);
    g_symlinks[link] = lower(dev);
    if (trace_enabled()) std::fprintf(stderr, "[io] symlink %s -> %s\n", link.c_str(), dev.c_str());
    return STATUS_SUCCESS;
}
KEXPORT(IoDeleteSymbolicLink) {
    std::lock_guard lk(g_io_m);
    g_symlinks.erase(lower(read_ansi_string(a[0])));
    return STATUS_SUCCESS;
}

struct Symlink : HostObject {
    std::string target;
};
// NtOpenSymbolicLinkObject(PHANDLE LinkHandle, POBJECT_ATTRIBUTES)
KEXPORT(NtOpenSymbolicLinkObject) {
    std::lock_guard lk(g_io_m);
    init_namespace();
    auto it = g_symlinks.find(lower(object_path(a[1])));
    if (it == g_symlinks.end()) return STATUS_OBJECT_NAME_NOT_FOUND;
    auto s = std::make_shared<Symlink>();
    s->target = it->second;
    wr<u32>(a[0], create_handle(alloc_object(0x10, 0), s));
    return STATUS_SUCCESS;
}
// NtQuerySymbolicLinkObject(HANDLE, POBJECT_STRING LinkTarget, PULONG ReturnedLength)
KEXPORT(NtQuerySymbolicLinkObject) {
    auto s = std::dynamic_pointer_cast<Symlink>(handle_host(a[0]));
    if (!s) return STATUS_INVALID_HANDLE;
    const u32 max = rd<u16>(a[1] + 2), buf = rd<u32>(a[1] + 4);
    if (a[2]) wr<u32>(a[2], u32(s->target.size()));
    if (s->target.size() > max) return STATUS_BUFFER_TOO_SMALL;
    std::memcpy(host_ptr(buf), s->target.data(), s->target.size());
    wr<u16>(a[1], u16(s->target.size()));
    return STATUS_SUCCESS;
}

namespace io {

void register_raw_device(const std::string& device, uint64_t size, RawFn read, RawFn write, const std::string& dos_link) {
    std::lock_guard lk(g_io_m);
    init_namespace();
    Device d;
    d.size = size;
    d.rd = std::move(read);
    d.wr = std::move(write);
    g_devices[lower(device)] = d;
    if (!dos_link.empty()) g_symlinks[lower(dos_link)] = lower(device);
}

void add_symlink(const std::string& link, const std::string& target) {
    std::lock_guard lk(g_io_m);
    init_namespace();
    g_symlinks[lower(link)] = lower(target);
}

}  // namespace io

}  // namespace vc3::kernel
