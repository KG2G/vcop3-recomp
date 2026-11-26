// vc3recomp — static recompiler front end for Virtua Cop 3 (Chihiro XBE).
//
//   vc3recomp <input.xbe> <output_dir> [--no-ptr-scan] [--entry 0xADDR]...
//                                      [--code-sections .text,D3D,...] [--stats] [--forms]
//   vc3recomp <blob.bin> <output_dir> --raw 0xBASE --entries-file FILE   (lifter tests)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <map>
#include <sstream>
#include <vector>

#include "analysis.h"
#include "emitter.h"
#include "lifter.h"
#include "xbe.h"

using namespace vc3;

// Sections that hold x86 code. Nearly every XBE section has the "executable"
// flag set, so this is by name. DOLBY is DSP microcode for the APU, not x86.
static const char* kDefaultCodeSections =
    ".text,D3D,D3DX,XGRPH,DSOUND,XPP,PSGSFD00,PSGSFD_I,PSGSFD_B,PSGSFD_P";

static void usage() {
    std::fprintf(stderr,
                 "usage: vc3recomp <input.xbe> <output_dir> [--no-ptr-scan] [--entry 0xADDR]... "
                 "[--code-sections a,b,c]\n");
    std::exit(2);
}

int main(int argc, char** argv) {
    if (argc < 3) usage();
    std::string xbe_path = argv[1];
    EmitOptions eo{.out_dir = argv[2]};
    AnalysisOptions ao;
    std::string code_sections = kDefaultCodeSections;
    bool stats = false, forms = false, raw = false;
    uint32_t raw_base = 0;

    for (int i = 3; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--no-ptr-scan")) ao.pointer_scan = false;
        else if (!std::strcmp(argv[i], "--stats")) stats = true;
        else if (!std::strcmp(argv[i], "--forms")) forms = true;
        else if (!std::strcmp(argv[i], "--raw") && i + 1 < argc) raw_base = std::strtoul(argv[++i], nullptr, 0), raw = true;
        else if (!std::strcmp(argv[i], "--entries-file") && i + 1 < argc) {
            std::FILE* ef = std::fopen(argv[++i], "r");
            if (!ef) { std::perror(argv[i]); return 1; }
            char line[256];
            while (std::fgets(line, sizeof line, ef)) {  // hex addresses, '#' comments
                unsigned long v;
                if (line[0] != '#' && std::sscanf(line, "%lx", &v) == 1) ao.extra_entries.push_back(uint32_t(v));
            }
            std::fclose(ef);
        }
        else if (!std::strcmp(argv[i], "--hooks-file") && i + 1 < argc) {
            std::FILE* hf = std::fopen(argv[++i], "r");
            if (!hf) { std::perror(argv[i]); return 1; }
            char line[256];
            while (std::fgets(line, sizeof line, hf)) {
                unsigned long v;
                if (line[0] != '#' && std::sscanf(line, "%lx", &v) == 1) g_hook_entries.insert(uint32_t(v));
            }
            std::fclose(hf);
        }
        else if (!std::strcmp(argv[i], "--entry") && i + 1 < argc) ao.extra_entries.push_back(std::strtoul(argv[++i], nullptr, 0));
        else if (!std::strcmp(argv[i], "--code-sections") && i + 1 < argc) code_sections = argv[++i];
        else usage();
    }
    std::stringstream ss(code_sections);
    for (std::string s; std::getline(ss, s, ',');) ao.code_sections.insert(s);

    std::string err;
    auto xbe = raw ? Xbe::load_raw(xbe_path, raw_base, err) : Xbe::load(xbe_path, err);
    if (!xbe) {
        std::fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }
    std::printf("XBE: platform=%s base=%08x entry=%08x thunk=%08x sections=%zu kernel imports=%zu\n",
                xbe->platform.c_str(), xbe->base, xbe->entry, xbe->kernel_thunk, xbe->sections.size(),
                xbe->kernel_ordinals.size());

    Analyzer an(*xbe, ao);
    an.run();
    size_t insns = 0, errs = 0;
    for (const auto& [va, f] : an.functions()) {
        insns += f.insns.size();
        errs += f.decode_error;
    }
    std::printf("analysis: %zu functions (%zu from pointer scan), %zu instructions, %zu with decode errors\n",
                an.functions().size(), an.pointer_scan_hits(), insns, errs);

    if (stats) {
        std::map<std::string, size_t> hist;
        for (const auto& [va, f] : an.functions())
            for (const auto& [ia, in] : f.insns) {
                std::string m = ZydisMnemonicGetString(in.insn.mnemonic);
                if (in.insn.attributes & (ZYDIS_ATTRIB_HAS_REP | ZYDIS_ATTRIB_HAS_REPE)) m = "rep " + m;
                if (in.insn.attributes & ZYDIS_ATTRIB_HAS_REPNE) m = "repne " + m;
                ++hist[m];
            }
        std::vector<std::pair<size_t, std::string>> v;
        for (auto& [m, n] : hist) v.push_back({n, m});
        std::sort(v.rbegin(), v.rend());
        for (auto& [n, m] : v) std::printf("%8zu %s\n", n, m.c_str());
    }

    if (forms) {  // operand shapes per mnemonic, for writing/validating the lifter
        std::map<std::string, size_t> hist;
        for (const auto& [va, f] : an.functions())
            for (const auto& [ia, in] : f.insns) {
                std::string k = ZydisMnemonicGetString(in.insn.mnemonic);
                k += " ";
                for (int i = 0; i < in.insn.operand_count_visible; ++i) {
                    const auto& op = in.ops[i];
                    if (i) k += ",";
                    if (op.type == ZYDIS_OPERAND_TYPE_REGISTER) k += ZydisRegisterGetString(op.reg.value);
                    else if (op.type == ZYDIS_OPERAND_TYPE_MEMORY) k += "m" + std::to_string(op.size);
                    else if (op.type == ZYDIS_OPERAND_TYPE_IMMEDIATE) k += "imm" + std::to_string(op.size);
                    else k += "?";
                }
                ++hist[k];
            }
        for (auto& [k, n] : hist) std::printf("%8zu %s\n", n, k.c_str());
    }

    const LiftStats ls = emit(*xbe, an, eo);
    std::printf("lifted: %zu / %zu instructions (%.2f%%)\n", ls.lifted, ls.total,
                ls.total ? 100.0 * double(ls.lifted) / double(ls.total) : 0.0);
    if (!ls.unlifted.empty()) {
        std::vector<std::pair<size_t, std::string>> v;
        for (const auto& [m, n] : ls.unlifted) v.push_back({n, m});
        std::sort(v.rbegin(), v.rend());
        std::printf("unlifted:");
        for (size_t i = 0; i < v.size() && i < 40; ++i) std::printf(" %s(%zu)", v[i].second.c_str(), v[i].first);
        std::printf("\n");
    }
    std::printf("wrote %s\n", eo.out_dir.c_str());
    return 0;
}
