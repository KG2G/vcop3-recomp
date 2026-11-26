// xboxkrnl.exe high-level emulation.
//
// Guest code calls kernel imports through the thunk table; the loader fills
// function imports with kKernelThunkBase + ordinal and data imports with a
// real guest address (data_export_addr). call() routes to the handler
// registered for that ordinal and applies the export's calling convention
// (stdcall / fastcall / cdecl, from exports.inc).
//
// Handlers are defined with KEXPORT(Name) { ... return eax; } and read their
// arguments with a[i] regardless of convention.
#pragma once
#include <cstdint>
#include <string>

#include "vc3/helpers.h"

namespace vc3::kernel {

enum Conv : int { kStdcall = 0, kFastcall = 1, kCdecl = 2 };

struct Args {
    const Cpu& c;
    Conv conv;
    u32 operator[](unsigned i) const {
        if (conv == kFastcall) {
            if (i == 0) return c.ecx;
            if (i == 1) return c.edx;
            i -= 2;
        }
        return rd<u32>(c.esp + 4 + 4 * i);
    }
};

using Handler = u32 (*)(Cpu&, const Args&);

struct Registrar {
    Registrar(const char* name, Handler h);
};

#define KEXPORT(name)                                                     \
    static ::vc3::u32 k_##name(::vc3::Cpu& c, const ::vc3::kernel::Args& a); \
    static ::vc3::kernel::Registrar reg_##name(#name, k_##name);          \
    static ::vc3::u32 k_##name([[maybe_unused]] ::vc3::Cpu& c, [[maybe_unused]] const ::vc3::kernel::Args& a)

void init();
uint32_t data_export_addr(uint32_t ordinal);  // 0 if `ordinal` is a function
uint32_t data_export_addr(const char* name);
void call(Cpu& c, uint32_t ordinal);          // return address already pushed
const char* name(uint32_t ordinal);

// Guest RAM reserved for kernel-side data exports (top 1 MiB below GPU instance memory).
constexpr uint32_t kKernelDataBase = 0x07E00000;
constexpr uint32_t kKernelDataSize = 0x00100000;

// NTSTATUS
constexpr u32 STATUS_SUCCESS = 0x00000000;
constexpr u32 STATUS_ABANDONED = 0x00000080;
constexpr u32 STATUS_USER_APC = 0x000000C0;
constexpr u32 STATUS_ALERTED = 0x00000101;
constexpr u32 STATUS_TIMEOUT = 0x00000102;
constexpr u32 STATUS_PENDING = 0x00000103;
constexpr u32 STATUS_NO_MORE_FILES = 0x80000006;
constexpr u32 STATUS_UNSUCCESSFUL = 0xC0000001;
constexpr u32 STATUS_NOT_IMPLEMENTED = 0xC0000002;
constexpr u32 STATUS_INVALID_HANDLE = 0xC0000008;
constexpr u32 STATUS_INVALID_PARAMETER = 0xC000000D;
constexpr u32 STATUS_END_OF_FILE = 0xC0000011;
constexpr u32 STATUS_NO_MEMORY = 0xC0000017;
constexpr u32 STATUS_CONFLICTING_ADDRESSES = 0xC0000018;
constexpr u32 STATUS_ACCESS_DENIED = 0xC0000022;
constexpr u32 STATUS_BUFFER_TOO_SMALL = 0xC0000023;
constexpr u32 STATUS_OBJECT_TYPE_MISMATCH = 0xC0000024;
constexpr u32 STATUS_OBJECT_NAME_NOT_FOUND = 0xC0000034;
constexpr u32 STATUS_OBJECT_NAME_COLLISION = 0xC0000035;
constexpr u32 STATUS_OBJECT_PATH_NOT_FOUND = 0xC000003A;
constexpr u32 STATUS_MUTANT_NOT_OWNED = 0xC0000046;
constexpr u32 STATUS_SEMAPHORE_LIMIT_EXCEEDED = 0xC0000047;
constexpr u32 STATUS_FILE_IS_A_DIRECTORY = 0xC00000BA;
constexpr u32 STATUS_NOT_A_DIRECTORY = 0xC0000103;
constexpr u32 STATUS_MEMORY_NOT_ALLOCATED = 0xC00000A0;

// Small guest-memory helpers shared by the kernel modules.
std::string read_cstr(u32 addr, size_t max = 4096);
std::string read_ansi_string(u32 string_struct);  // ANSI_STRING/OBJECT_STRING {u16 Length, u16 Max, char* Buffer}
s64 now_100ns();                                  // host monotonic clock in 100 ns units
s64 system_time_100ns();                          // FILETIME (since 1601)

bool trace_enabled();

}  // namespace vc3::kernel
