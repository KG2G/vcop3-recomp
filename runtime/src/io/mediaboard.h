// Chihiro media board (DIMM board) I/O ports 0x4000-0x40FF.
//
// Identification/status registers per MAME's chihiro driver: board ID "BXAM"
// at 0x20-0x25, "board present" at 0xF0, DIMM size code at 0xF4 (2 = 512 MB).
// The board's sector-command channel (mbcom) is not modelled yet.
#pragma once

namespace vc3::io::mediaboard {

void init();

}  // namespace vc3::io::mediaboard
