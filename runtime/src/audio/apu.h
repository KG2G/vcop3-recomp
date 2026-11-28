// MCPX APU (audio processing unit) at 0xFE800000, IRQ 5.
//
// Modelled: the front end (FE) method interface DSOUND drives through PIO writes
// at +0x20000, voice lists and the in-memory voice structures, notifiers,
// interrupts, and the voice processor (VP): PCM 8/16/24/32-bit and Xbox ADPCM,
// buffer (SGE) and stream (SSL) voices, looping, pitch, amplitude/filter
// envelopes, per-mixbin volumes, multipass submixes.
//
// Not modelled: the GP/EP DSPs (effects, AC-3 encoding). Their DSP code is accepted
// and ignored; the output is a stereo downmix of the VP mixbins, played through SDL
// (and optionally written to a WAV file). The one effect VC3 uses, the I3DL2 room
// reverb (mixbin 10), is replaced by a generic stereo reverb fed from that send.
#pragma once

namespace vc3::audio::apu {

// `play` opens an SDL audio device (otherwise audio is only produced for VC3_WAV).
void init(bool play);
// Stops the mixing thread and closes the audio device (before SDL_Quit).
void shutdown();
// The pause menu: no sound is made while paused (what was queued is dropped).
void set_paused(bool paused);
// Output volume (1 = 100%), live.
void set_volume(float v);

}  // namespace vc3::audio::apu
