#pragma once
#include <filesystem>

#include "analysis.h"
#include "lifter.h"

namespace vc3 {

struct EmitOptions {
    std::filesystem::path out_dir;
    size_t funcs_per_file = 250;
};

// Writes generated C++ for every discovered function (decls.h, funcs_NNN.cpp),
// the address -> host function table (func_table.cpp) and functions.txt.
LiftStats emit(const Xbe& xbe, const Analyzer& an, const EmitOptions& opts);

}  // namespace vc3
