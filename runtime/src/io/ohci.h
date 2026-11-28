// OHCI USB host controllers (registers + transfer engine).
#pragma once
#include "io/usb.h"

namespace vc3::io::ohci {

void init();                                          // registers both controllers, starts the frame thread
void attach(int controller, int port, usb::Device* dev);  // plug a device into a root-hub port
// Whether guest RAM [lo, hi) overlaps the USB descriptors (EDs, TDs, HCCA) seen so far -- a guard
// for code that writes big blocks into guest RAM (picture write-backs), which must never land there.
bool overlaps_descriptors(uint32_t lo, uint32_t hi);
// The descriptor range seen so far ([lo, hi); lo >= hi: none yet), for the log.
void descriptor_range(uint32_t& lo, uint32_t& hi);

}  // namespace vc3::io::ohci
