// Guest file-system namespace hooks for device models.
#pragma once
#include <cstdint>
#include <functional>
#include <string>

namespace vc3::kernel::io {

// Byte-addressed raw access; return bytes moved (or -1 on error).
using RawFn = std::function<long(uint64_t offset, uint8_t* buf, uint32_t len)>;

// Register a raw-only device (e.g. "\Device\MbCom") and optional DOS link ("\??\mbcom:").
void register_raw_device(const std::string& device, uint64_t size, RawFn read, RawFn write, const std::string& dos_link = {});
// Add a symbolic link "\??\x:" -> device path.
void add_symlink(const std::string& link, const std::string& target);

}  // namespace vc3::kernel::io
