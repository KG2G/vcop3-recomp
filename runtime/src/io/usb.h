// USB device-side framework for emulated devices on the OHCI root hub.
#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>

namespace vc3::io::usb {

enum class Pid { Setup, Out, In };

constexpr int kStall = -1;  // transfer() results
constexpr int kNak = -2;

struct SetupPacket {
    uint8_t bmRequestType, bRequest;
    uint16_t wValue, wIndex, wLength;
};

class Device {
public:
    virtual ~Device() = default;
    virtual void reset();
    // Move up to `len` bytes for endpoint `ep` (0-15). Returns bytes moved, kStall or kNak.
    virtual int transfer(int ep, Pid pid, uint8_t* buf, int len) = 0;
    uint8_t address = 0;
};

// A device with standard control-endpoint handling and descriptor tables.
// Subclasses implement vendor_request() (fills `ctl_` for IN data stages)
// and bulk().
class StandardDevice : public Device {
public:
    void reset() override;
    int transfer(int ep, Pid pid, uint8_t* buf, int len) override;

protected:
    // Descriptors (raw bytes). config_ must include interface + endpoint descriptors.
    std::vector<uint8_t> device_desc_, config_desc_;
    std::vector<std::vector<uint8_t>> strings_;

    // Non-standard request. Prepare the data stage in ctl_ (IN) and return true,
    // or return false to stall.
    virtual bool vendor_request(const SetupPacket& s) = 0;
    virtual int bulk(int ep, Pid pid, uint8_t* buf, int len) = 0;

    std::vector<uint8_t> ctl_;  // control data stage buffer
    size_t ctl_pos_ = 0;

private:
    SetupPacket setup_{};
    int pending_address_ = -1;
    uint8_t configuration_ = 0;
};

}  // namespace vc3::io::usb
