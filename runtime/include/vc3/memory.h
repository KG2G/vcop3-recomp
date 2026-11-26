// Guest physical/virtual memory layout (Chihiro = Xbox with 128 MiB RAM).
//
//   0x00000000 - 0x07FFFFFF  RAM (128 MiB). XBE is loaded at 0x00010000.
//   0x80000000 - 0x87FFFFFF  kernel mirror of RAM (same backing pages)
//   0xF0000000 - 0xF7FFFFFF  contiguous/"physical" mirror used by D3D/MmAllocateContiguousMemory
//   0xFD000000 - 0xFDFFFFFF  NV2A MMIO (PGRAPH, PFIFO, PRAMIN, ...)
//   0xFE800000 - 0xFE87FFFF  MCPX APU MMIO
//   0xFFFF0000 + ordinal     sentinel addresses for kernel HLE thunks (never mapped)
//
// The whole 4 GiB guest range is reserved up front (PROT_NONE) and RAM is
// committed via a memfd mapped at every mirror, so writes through one alias
// are visible through the others. MMIO ranges stay unmapped: touching them
// faults, which is where an MMIO trap handler (or explicit accessors) will go.
#pragma once
#include <cstdint>
#include <string>

namespace vc3::mem {

constexpr uint32_t kRamSize = 128u << 20;
constexpr uint32_t kRamMirrors[] = {0x00000000, 0x80000000, 0xF0000000};
constexpr uint32_t kNv2aBase = 0xFD000000, kNv2aSize = 0x01000000;
constexpr uint32_t kApuBase = 0xFE800000, kApuSize = 0x00080000;

bool init(std::string& err);
// Map RAM [ram_offset, ram_offset+size) a second time at guest address `guest` (e.g. NV2A PRAMIN).
bool map_ram_alias(uint32_t guest, uint32_t ram_offset, uint32_t size);
void shutdown();

}  // namespace vc3::mem
