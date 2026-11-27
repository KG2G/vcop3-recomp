// User settings: one table of options, saved to settings.ini in the data folder (the per-user
// folder, or the player's own install folder: see settings.cpp), edited in the launcher. Each option is also an environment
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
// Make `dir` the data folder ("" = the default per-user folder): settings.ini moves there and is
// found there from now on. False if the folder can't be written (nothing changes then).
bool relocate(const std::string& dir);
// Remove the per-user folder if nothing is left in it (everything lives in the player's own folder).
void tidy_default_folder();
std::string get(const char* key);
void set(const char* key, const std::string& value);
// The environment already sets this option (it can't be changed from the launcher).
bool overridden(const char* key);
// Exports every option that isn't overridden as its environment variable.
void apply();
std::string file_path();

}  // namespace vc3::settings
