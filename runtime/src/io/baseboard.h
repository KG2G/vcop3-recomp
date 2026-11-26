// Chihiro Type-1 base board, as seen from the Xbox side: two USB functions
// on USB0 (Cypress AN2131 microcontrollers running Sega firmware).
//   port 0: AN2131QC  (class 0x60/0) - JVS bus bridge + EEPROMs
//   port 1: AN2131SC  (class 0x60/1) - serial/MIDI ports
// Behind the JVS bridge sits a Sega 837-13551 JVS I/O board.
// Protocol details follow MAME's chihiro driver (src/mame/sega/chihiro.cpp).
#pragma once

namespace vc3::io::baseboard {

void init();  // loads EEPROM dumps, attaches both functions to USB0

}  // namespace vc3::io::baseboard
