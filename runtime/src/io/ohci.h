// OHCI USB host controllers (registers + transfer engine).
#pragma once
#include "io/usb.h"

namespace vc3::io::ohci {

void init();                                          // registers both controllers, starts the frame thread
void attach(int controller, int port, usb::Device* dev);  // plug a device into a root-hub port

}  // namespace vc3::io::ohci
