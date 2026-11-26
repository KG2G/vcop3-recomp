// User settings: one table of options, saved to a file in the per-user config directory
// (SDL_GetPrefPath), edited in the launcher. Each option is also an environment
// variable (VC3_*): a variable set before starting wins over the file, and apply()
// exports the saved values as variables for the rest of the program to read.
#pragma once
#include <string>
#include <vector>

namespace vc3::settings {

struct Choice {
    const char* value;  // stored / exported value
    const char* label;  // shown in the launcher
};

struct Option {
    const char* key;      // name in the settings file
    const char* env;      // environment variable it maps to (nullptr: launcher-only)
    const char* section;  // launcher grouping
    const char* label;
    const char* help;
    std::vector<Choice> choices;
    const char* def;      // default value
};

const std::vector<Option>& options();
void load();  // from the settings file (missing file: defaults)
bool save();  // writes the settings file
std::string get(const char* key);
void set(const char* key, const std::string& value);
// The environment already sets this option (it can't be changed from the launcher).
bool overridden(const char* key);
// Exports every option that isn't overridden as its environment variable.
void apply();
std::string file_path();

}  // namespace vc3::settings
