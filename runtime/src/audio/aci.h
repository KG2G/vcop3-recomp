// MCPX ACI: the AC'97 controller at 0xFEC00000 (IRQ 6).
//
// Codec mixer registers (NAM) at 0x000-0x0FF and an Intel-ICH-style bus master
// (NABM) at 0x100: PCM in (0x00), PCM out (0x10), mic (0x20) and the nForce
// S/PDIF out box (0x70), then GLOB_CNT / GLOB_STA / CAS at 0x2C-0x34.
//
// DSOUND programs the PCM-out and S/PDIF buffer descriptor lists and expects
// them to progress at 48 kHz with IOC / last-valid-buffer interrupts. The sound
// itself is produced from the voice processor's mixbins (apu.cpp), so the DMA
// engines only model timing and status; the buffer contents aren't read.
#pragma once
#include <cstdint>

namespace vc3::audio::aci {

void init();
// Advance the running DMA engines by `frames` stereo sample frames (48 kHz).
void tick(uint32_t frames);

}  // namespace vc3::audio::aci
