# FFT Freeze (LV2)

A mono LV2 audio effect that "freezes" the spectrum of its input and sustains
it indefinitely, using a phase-vocoder resynthesis. Freeze is triggered by a
MIDI note or a MIDI CC, so it can be played like an instrument (e.g. hold a
note to sustain the sound at the moment it was struck).

- **URI:** `https://example.org/plugins/fft-freeze`
- **Ports:** `in_l` (audio in), `out_l` (audio out), `midi_in` (MIDI)
- **Parameters:** `freeze_on_note`, `fft_size`, `spectral_mode`, `freeze_cc_number`

## Building

Dependencies: [FFTW3](https://www.fftw.org/) and the LV2 core headers.

```sh
make
make install   # copies the bundle to $PREFIX/fft-freeze.lv2 (PREFIX defaults to ~/.lv2)
```

On macOS with Homebrew, if `fftw3.h` isn't found, point the compiler at the
Homebrew prefix:

```sh
CPATH="$(brew --prefix fftw)/include" LIBRARY_PATH="$(brew --prefix fftw)/lib" make
```

## Usage

| Port               | Type         | Description                                                                 |
|--------------------|--------------|-------------------------------------------------------------------------------|
| `in_l` / `out_l`    | Audio        | Mono audio in/out                                                            |
| `midi_in`           | MIDI         | Note On/Off and/or CC messages trigger freeze/unfreeze                      |
| `freeze_on_note`    | Control      | Enables MIDI-triggered freeze (toggle)                                      |
| `fft_size`          | Control      | FFT window size in samples, 256–8192 (default 1024)                         |
| `spectral_mode`     | Control      | Randomizes bin phases at capture for a diffuse, texture-like freeze (toggle)|
| `freeze_cc_number`  | Control      | MIDI CC number that also triggers freeze/unfreeze (default 20)              |

Freeze engages on a Note On (or CC ≥ 64) while not already frozen, and
disengages on the matching Note Off (or CC < 64).

## Technical details

### Algorithm

This is a classic phase-vocoder freeze, not a looper: the input is only ever
read at the instant freeze is triggered. `capture_freeze()` takes one
magnitude/phase snapshot of the spectrum; from then on, `generate_ola_frame()`
resynthesizes it forever by advancing each frequency bin's phase at its own
estimated rate and overlap-adding the resulting frames — the live input is
ignored until the next freeze.

### Analysis and overlap-add

The plugin analyzes with a Hann window at 4x overlap (hop size = `fft_size`/4,
75% overlap) using FFTW3's real-to-complex/complex-to-real transforms. Frames
are combined with a classic single (analysis-only) overlap-add: no separate
synthesis window is applied. An earlier version applied the Hann window a
second time at synthesis, which measurably lost energy — independent
per-bin phase evolution decorrelates a tone's spectral leakage across
neighbouring bins, and squaring the window makes that loss worse. The
overlap-add is normalized by the closed-form Hann COLA constant (`M/2` for
`M`-fold overlap, exact for the fixed 4x overlap used here) rather than an
accumulated running sum, which avoids amplifying noise during the brief
warm-up period before all overlapping frames have contributed.

### Frozen magnitude averaging

Rather than freezing a single analysis frame, `capture_freeze()` averages the
magnitude spectrum over `CAPTURE_AVG_FRAMES` (3) consecutive hops around the
trigger point. This avoids locking onto a single unlucky transient or noise
burst. The synthesis phase, and the phase used for the true-phase-increment
estimate below, still come from the two most recent hops, so pitch tracking
is unaffected by the averaging.

### True phase-increment estimation

For a bin exactly centered on a partial, advancing its phase by the nominal
`2π·k·hop/N` per hop reconstructs it perfectly. For off-bin-centre content —
almost everything in practice — the nominal increment is wrong and causes
audible amplitude pulsation. Instead, each bin's *actual* instantaneous
frequency is estimated by comparing its phase across one hop of history
(unwrapped to the principal value nearest the nominal increment), and that
measured increment is used for resynthesis. Near-silent bins fall back to the
nominal increment, since their phase is unreliable.

### Phase jitter

A slow, small, mean-reverting random walk is added to each bin's phase
increment every hop. Without it, a long freeze is a perfectly periodic,
static signal, which reads as robotic/lifeless. The jitter amplitude is small
enough to avoid reintroducing the pulsation that the true-phase-increment
estimate fixes.

### Freeze/unfreeze crossfade

Switching straight from live passthrough to the frozen overlap-add output
(and back) can produce an audible click, since the two signals are generally
at different instantaneous values. A short (5 ms) crossfade blends between
live and frozen output on both transitions.

### Spectral mode

When enabled, bin phases are randomized at capture instead of taken from the
input, and every bin advances at its nominal (bin-centre) rate. This turns
the frozen spectrum into a diffuse, texture-like drone that keeps the
input's magnitude envelope but discards its phase relationships.
