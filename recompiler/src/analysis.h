#pragma once
#include <Zydis/Zydis.h>

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "xbe.h"

namespace vc3 {

struct Instruction {
    uint32_t addr;
    ZydisDecodedInstruction insn;
    ZydisDecodedOperand ops[ZYDIS_MAX_OPERAND_COUNT];
};

struct Function {
    uint32_t entry = 0;
    enum class Source { Entry, CallTarget, PointerScan, ImmediateRef, Manual } source = Source::CallTarget;
    std::set<uint32_t> block_starts;
    std::map<uint32_t, Instruction> insns;  // all instructions reachable from entry
    std::set<uint32_t> tail_calls;           // jmp targets that are other functions
    std::vector<uint32_t> jump_tables;       // addresses of recognised `jmp [reg*4+table]`
    bool decode_error = false;
};

struct AnalysisOptions {
    std::set<std::string> code_sections;
    bool pointer_scan = true;
    std::vector<uint32_t> extra_entries;
};

class Analyzer {
public:
    Analyzer(const Xbe& xbe, AnalysisOptions opts);
    void run();

    const std::map<uint32_t, Function>& functions() const { return funcs_; }
    size_t pointer_scan_hits() const { return ptr_hits_; }

    bool is_code(uint32_t va) const;
    bool is_function(uint32_t va) const { return funcs_.count(va) != 0; }
    // Targets of an MSVC-style `jmp [reg*4 + table]`, empty if not a jump table.
    std::vector<uint32_t> read_jump_table(const Instruction& in) const;

private:
    bool decode(uint32_t va, Instruction& out) const;
    void discover_calls();
    void scan_pointers();
    size_t scan_code_section_pointers();
    bool looks_like_function_start(uint32_t va) const;
    void trace(Function& f);

    const Xbe& xbe_;
    AnalysisOptions opts_;
    ZydisDecoder decoder_;
    std::map<uint32_t, Function> funcs_;
    size_t ptr_hits_ = 0;
    std::set<std::pair<uint32_t, uint32_t>> table_ranges_;  // [start, end) of recognised jump tables
};

}  // namespace vc3
