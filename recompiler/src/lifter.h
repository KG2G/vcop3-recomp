#pragma once
#include <set>
#include <map>
#include <string>

#include "analysis.h"

namespace vc3 {

// Function entries whose generated code first calls vc3::guest_hook(c, entry) (--hooks-file).
extern std::set<uint32_t> g_hook_entries;

struct LiftStats {
    size_t total = 0, lifted = 0;
    std::map<std::string, size_t> unlifted;  // mnemonic -> count
};

std::string fn_name(uint32_t va);

// Lift one function to a C++ function definition.
std::string lift_function(const Analyzer& an, const Function& f, LiftStats& stats);

}  // namespace vc3
