#include "usb.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace vc3::io::usb {

void Device::reset() { address = 0; }

void StandardDevice::reset() {
    Device::reset();
    pending_address_ = -1;
    configuration_ = 0;
    ctl_.clear();
    ctl_pos_ = 0;
}

int StandardDevice::transfer(int ep, Pid pid, uint8_t* buf, int len) {
    if (ep != 0) return bulk(ep, pid, buf, len);

    if (pid == Pid::Setup) {
        if (len < 8) return kStall;
        setup_ = SetupPacket{buf[0], buf[1], uint16_t(buf[2] | buf[3] << 8), uint16_t(buf[4] | buf[5] << 8),
                             uint16_t(buf[6] | buf[7] << 8)};
        ctl_.clear();
        ctl_pos_ = 0;
        static const bool tr = std::getenv("VC3_TRACE_USB") != nullptr;
        if (tr)
            std::fprintf(stderr, "[usb] dev@%u SETUP %02x %02x value=%04x index=%04x len=%u\n", address,
                         setup_.bmRequestType, setup_.bRequest, setup_.wValue, setup_.wIndex, setup_.wLength);
        const int type = (setup_.bmRequestType >> 5) & 3;
        if (type == 0) {  // standard
            switch (setup_.bRequest) {
                case 0x00: ctl_ = {0, 0}; break;                          // GET_STATUS
                case 0x01: case 0x03: break;                              // CLEAR/SET_FEATURE
                case 0x05: pending_address_ = setup_.wValue & 0x7F; break;  // SET_ADDRESS (after status stage)
                case 0x06: {                                              // GET_DESCRIPTOR
                    const int dtype = setup_.wValue >> 8, index = setup_.wValue & 0xFF;
                    if (dtype == 1) ctl_ = device_desc_;
                    else if (dtype == 2) ctl_ = config_desc_;
                    else if (dtype == 3 && index < int(strings_.size())) ctl_ = strings_[size_t(index)];
                    else return kStall;
                    break;
                }
                case 0x08: ctl_ = {configuration_}; break;                // GET_CONFIGURATION
                case 0x09: configuration_ = uint8_t(setup_.wValue); break;  // SET_CONFIGURATION
                case 0x0A: ctl_ = {0}; break;                             // GET_INTERFACE
                case 0x0B: break;                                         // SET_INTERFACE
                default: return kStall;
            }
        } else if (!vendor_request(setup_)) {
            return kStall;
        }
        if (ctl_.size() > setup_.wLength) ctl_.resize(setup_.wLength);
        return 8;
    }

    const bool data_in = setup_.bmRequestType & 0x80;
    if (pid == Pid::In && data_in) {  // IN data stage
        const int n = std::min<int>(len, int(ctl_.size() - ctl_pos_));
        std::memcpy(buf, ctl_.data() + ctl_pos_, size_t(n));
        ctl_pos_ += size_t(n);
        return n;
    }
    if (pid == Pid::Out && !data_in && len > 0) {  // OUT data stage
        ctl_.insert(ctl_.end(), buf, buf + len);
        return len;
    }
    // Status stage (zero-length, opposite direction).
    if (pending_address_ >= 0) {
        address = uint8_t(pending_address_);
        pending_address_ = -1;
    }
    return 0;
}

}  // namespace vc3::io::usb
