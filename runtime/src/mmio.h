// Memory-mapped I/O.
//
// Guest MMIO ranges are left unmapped in the 4 GiB guest reservation. When
// lifted code touches one, the SIGSEGV handler decodes the faulting host
// instruction (Zydis), performs the access through the registered device
// callbacks, patches the host register state and resumes after the
// instruction. Supported host forms: mov/movzx/movsx loads and stores, and
// cmp/test/add/sub/and/or/xor with a memory operand.
#pragma once
#include <cstdint>

namespace vc3::mmio {

using ReadFn = uint32_t (*)(void* ctx, uint32_t addr, unsigned size);
using WriteFn = void (*)(void* ctx, uint32_t addr, uint32_t value, unsigned size);

void register_range(uint32_t base, uint32_t size, const char* name, ReadFn rd, WriteFn wr, void* ctx = nullptr);
void install();

// x86 I/O port space (in/out). Unregistered ports read as all-ones and are logged once.
void register_ports(uint32_t base, uint32_t count, const char* name, ReadFn rd, WriteFn wr, void* ctx = nullptr);

}  // namespace vc3::mmio
