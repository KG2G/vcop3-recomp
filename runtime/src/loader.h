#pragma once
#include <string>

#include "xbe.h"

namespace vc3 {

// Copy XBE headers + sections into guest memory and patch the kernel thunk
// table so each import points at an HLE sentinel or a kernel data export.
bool load_xbe(const Xbe& xbe, std::string& err);

}  // namespace vc3
