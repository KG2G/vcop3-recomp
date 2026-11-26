// Rtl*: critical sections, strings, memory and time helpers.
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "kernel.h"
#include "mm.h"
#include "objects.h"
#include "ps.h"

namespace vc3::kernel {

// ---------------------------------------------------------------- critical sections
// RTL_CRITICAL_SECTION { DISPATCHER_HEADER Event; LONG LockCount@0x10; LONG RecursionCount@0x14; HANDLE OwningThread@0x18 }
// All guest code runs under the guest lock, so these read-modify-writes are atomic
// with respect to other guest threads.
constexpr u32 kCsLock = 0x10, kCsRecursion = 0x14, kCsOwner = 0x18;

KEXPORT(RtlInitializeCriticalSection) {
    const u32 cs = a[0];
    init_dispatcher_header(cs, kEventSynchronization, 4, 0);
    wr<s32>(cs + kCsLock, -1);
    wr<s32>(cs + kCsRecursion, 0);
    wr<u32>(cs + kCsOwner, 0);
    return 0;
}

KEXPORT(RtlEnterCriticalSection) {
    const u32 cs = a[0], me = ps::current_kthread();
    const s32 lock = rd<s32>(cs + kCsLock) + 1;
    wr<s32>(cs + kCsLock, lock);
    if (lock == 0) {
        wr<u32>(cs + kCsOwner, me);
        wr<s32>(cs + kCsRecursion, 1);
    } else if (rd<u32>(cs + kCsOwner) == me) {
        wr<s32>(cs + kCsRecursion, rd<s32>(cs + kCsRecursion) + 1);
    } else {
        wait_for_objects({cs}, false, std::nullopt);
        wr<u32>(cs + kCsOwner, me);
        wr<s32>(cs + kCsRecursion, 1);
    }
    ps::critical_section_held(+1);
    return 0;
}

KEXPORT(RtlLeaveCriticalSection) {
    const u32 cs = a[0];
    const s32 rec = rd<s32>(cs + kCsRecursion) - 1;
    wr<s32>(cs + kCsRecursion, rec);
    const s32 lock = rd<s32>(cs + kCsLock) - 1;
    wr<s32>(cs + kCsLock, lock);
    if (rec == 0) {
        wr<u32>(cs + kCsOwner, 0);
        if (lock >= 0) set_event(cs);  // hand over to a waiter
    }
    ps::critical_section_held(-1);
    return 0;
}

KEXPORT(RtlTryEnterCriticalSection) {
    const u32 cs = a[0], me = ps::current_kthread();
    if (rd<s32>(cs + kCsLock) == -1) {
        wr<s32>(cs + kCsLock, 0);
        wr<u32>(cs + kCsOwner, me);
        wr<s32>(cs + kCsRecursion, 1);
        ps::critical_section_held(+1);
        return 1;
    }
    if (rd<u32>(cs + kCsOwner) == me) {
        wr<s32>(cs + kCsLock, rd<s32>(cs + kCsLock) + 1);
        wr<s32>(cs + kCsRecursion, rd<s32>(cs + kCsRecursion) + 1);
        ps::critical_section_held(+1);
        return 1;
    }
    return 0;
}

// ---------------------------------------------------------------- strings
// ANSI_STRING / UNICODE_STRING { u16 Length; u16 MaximumLength; ptr Buffer }

KEXPORT(RtlInitAnsiString) {
    const u32 dst = a[0], src = a[1];
    const u32 len = src ? u32(read_cstr(src, 0xFFFE).size()) : 0;
    wr<u16>(dst, u16(len));
    wr<u16>(dst + 2, u16(src ? len + 1 : 0));
    wr<u32>(dst + 4, src);
    return 0;
}

// RtlAnsiStringToUnicodeString(PUNICODE_STRING Dest, PSTRING Src, BOOLEAN Allocate)
KEXPORT(RtlAnsiStringToUnicodeString) {
    const std::string s = read_ansi_string(a[1]);
    const u32 need = u32(s.size() + 1) * 2;
    u32 buf;
    if (a[2] & 0xFF) {
        buf = mm::pool_alloc(need);
        if (!buf) return STATUS_NO_MEMORY;
        wr<u16>(a[0] + 2, u16(need));
        wr<u32>(a[0] + 4, buf);
    } else {
        if (rd<u16>(a[0] + 2) < need) return STATUS_BUFFER_TOO_SMALL;
        buf = rd<u32>(a[0] + 4);
    }
    for (size_t i = 0; i < s.size(); ++i) wr<u16>(buf + u32(2 * i), u8(s[i]));
    wr<u16>(buf + u32(2 * s.size()), 0);
    wr<u16>(a[0], u16(s.size() * 2));
    return STATUS_SUCCESS;
}

// RtlUnicodeStringToAnsiString(PSTRING Dest, PUNICODE_STRING Src, BOOLEAN Allocate)
KEXPORT(RtlUnicodeStringToAnsiString) {
    const u32 n = rd<u16>(a[1]) / 2, src = rd<u32>(a[1] + 4);
    u32 buf;
    if (a[2] & 0xFF) {
        buf = mm::pool_alloc(n + 1);
        if (!buf) return STATUS_NO_MEMORY;
        wr<u16>(a[0] + 2, u16(n + 1));
        wr<u32>(a[0] + 4, buf);
    } else {
        if (rd<u16>(a[0] + 2) < n + 1) return STATUS_BUFFER_TOO_SMALL;
        buf = rd<u32>(a[0] + 4);
    }
    for (u32 i = 0; i < n; ++i) {
        const u16 ch = rd<u16>(src + 2 * i);
        wr<u8>(buf + i, ch < 0x100 ? u8(ch) : u8('?'));
    }
    wr<u8>(buf + n, 0);
    wr<u16>(a[0], u16(n));
    return STATUS_SUCCESS;
}

static int compare_strings(u32 s1, u32 s2, bool ci) {
    const std::string a = read_ansi_string(s1), b = read_ansi_string(s2);
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        int x = u8(a[i]), y = u8(b[i]);
        if (ci) { x = std::toupper(x); y = std::toupper(y); }
        if (x != y) return x - y;
    }
    return int(a.size()) - int(b.size());
}
KEXPORT(RtlCompareString) { return u32(compare_strings(a[0], a[1], a[2] & 0xFF)); }
KEXPORT(RtlEqualString) {
    if (rd<u16>(a[0]) != rd<u16>(a[1])) return 0;
    return compare_strings(a[0], a[1], a[2] & 0xFF) == 0;
}

// ---------------------------------------------------------------- memory
KEXPORT(RtlCompareMemory) {
    u32 i = 0;
    while (i < a[2] && rd<u8>(a[0] + i) == rd<u8>(a[1] + i)) ++i;
    return i;
}
// RtlCompareMemoryUlong(PVOID Source, SIZE_T Length, ULONG Pattern) -> bytes that match
KEXPORT(RtlCompareMemoryUlong) {
    u32 i = 0;
    while (i + 4 <= a[1] && rd<u32>(a[0] + i) == a[2]) i += 4;
    return i;
}
// RtlFillMemoryUlong(PVOID Destination, SIZE_T Length, ULONG Pattern)
KEXPORT(RtlFillMemoryUlong) {
    for (u32 i = 0; i + 4 <= a[1]; i += 4) wr<u32>(a[0] + i, a[2]);
    return 0;
}

// ---------------------------------------------------------------- status / exceptions
KEXPORT(RtlNtStatusToDosError) {
    switch (a[0]) {
        case STATUS_SUCCESS: return 0;
        case STATUS_TIMEOUT: return 258;   // WAIT_TIMEOUT
        case STATUS_PENDING: return 997;   // ERROR_IO_PENDING
        case STATUS_NO_MORE_FILES: return 18;
        case STATUS_UNSUCCESSFUL: return 31;
        case STATUS_NOT_IMPLEMENTED: return 1;
        case STATUS_INVALID_HANDLE: return 6;
        case STATUS_INVALID_PARAMETER: return 87;
        case STATUS_END_OF_FILE: return 38;
        case STATUS_NO_MEMORY: return 8;
        case STATUS_ACCESS_DENIED: return 5;
        case STATUS_BUFFER_TOO_SMALL: return 122;
        case STATUS_OBJECT_NAME_NOT_FOUND: return 2;
        case STATUS_OBJECT_NAME_COLLISION: return 183;
        case STATUS_OBJECT_PATH_NOT_FOUND: return 3;
        case STATUS_FILE_IS_A_DIRECTORY: return 5;
        case STATUS_NOT_A_DIRECTORY: return 267;
        default: return 317;  // ERROR_MR_MID_NOT_FOUND
    }
}

KEXPORT(RtlRaiseException) {
    const u32 rec = a[0];
    std::fprintf(stderr, "[rtl] RtlRaiseException code=%08x flags=%08x address=%08x (from %08x)\n", rd<u32>(rec),
                 rd<u32>(rec + 4), rd<u32>(rec + 12), rd<u32>(c.esp));
    std::abort();  // TODO: SEH dispatch through the fs:[0] chain
}
KEXPORT(RtlUnwind) {
    std::fprintf(stderr, "[rtl] RtlUnwind(frame=%08x ip=%08x) not supported (from %08x)\n", a[0], a[1], rd<u32>(c.esp));
    std::abort();
}

// ---------------------------------------------------------------- time fields
// TIME_FIELDS { CSHORT Year, Month, Day, Hour, Minute, Second, Milliseconds, Weekday }
static s64 days_from_civil(s64 y, unsigned m, unsigned d) {
    y -= m <= 2;
    const s64 era = (y >= 0 ? y : y - 399) / 400;
    const unsigned yoe = unsigned(y - era * 400);
    const unsigned doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
    const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097 + s64(doe) - 719468;  // days since 1970-01-01
}
static void civil_from_days(s64 z, s64& y, unsigned& m, unsigned& d) {
    z += 719468;
    const s64 era = (z >= 0 ? z : z - 146096) / 146097;
    const unsigned doe = unsigned(z - era * 146097);
    const unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    const unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    const unsigned mp = (5 * doy + 2) / 153;
    d = doy - (153 * mp + 2) / 5 + 1;
    m = mp < 10 ? mp + 3 : mp - 9;
    y = s64(yoe) + era * 400 + (m <= 2);
}
constexpr s64 kDaysFrom1601To1970 = 134774;

KEXPORT(RtlTimeToTimeFields) {
    const s64 t = rd<s64>(a[0]);
    const s64 ms_total = t / 10000;
    const s64 days = ms_total / 86400000 - kDaysFrom1601To1970;
    s64 rem = ms_total % 86400000;
    s64 y; unsigned m, d;
    civil_from_days(days, y, m, d);
    const u32 f = a[1];
    wr<s16>(f + 0, s16(y)); wr<s16>(f + 2, s16(m)); wr<s16>(f + 4, s16(d));
    wr<s16>(f + 6, s16(rem / 3600000)); rem %= 3600000;
    wr<s16>(f + 8, s16(rem / 60000)); rem %= 60000;
    wr<s16>(f + 10, s16(rem / 1000));
    wr<s16>(f + 12, s16(rem % 1000));
    wr<s16>(f + 14, s16(((days % 7) + 7 + 4) % 7));  // 1970-01-01 was a Thursday
    return 0;
}

KEXPORT(RtlTimeFieldsToTime) {
    const u32 f = a[0];
    const s16 y = rd<s16>(f), m = rd<s16>(f + 2), d = rd<s16>(f + 4);
    if (m < 1 || m > 12 || d < 1 || d > 31) return 0;
    const s64 days = days_from_civil(y, unsigned(m), unsigned(d)) + kDaysFrom1601To1970;
    const s64 ms = ((days * 24 + rd<s16>(f + 6)) * 60 + rd<s16>(f + 8)) * 60000 + rd<s16>(f + 10) * 1000 + rd<s16>(f + 12);
    wr<s64>(a[1], ms * 10000);
    return 1;
}

}  // namespace vc3::kernel
