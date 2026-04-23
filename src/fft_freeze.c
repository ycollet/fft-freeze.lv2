// SPDX-License-Identifier: ISC
// Plugin LV2 "FFT Freeze" — C + FFTW3
// URI du plugin : https://example.org/plugins/fft-freeze

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <fftw3.h>

#include "lv2/core/lv2.h"
#include "lv2/urid/urid.h"
#include "lv2/atom/atom.h"
#include "lv2/atom/util.h"
#include "lv2/midi/midi.h"

#define FFT_FREEZE_URI "https://example.org/plugins/fft-freeze"

typedef enum {
  PORT_IN_L = 0,
  PORT_OUT_L = 1,
  PORT_MIDI_IN = 2,
  PORT_FREEZE_ON_NOTE = 3,
  PORT_FFT_SIZE = 4,
  PORT_SPECTRAL_MODE = 5,
  PORT_FREEZE_CC_NUMBER = 6
} Ports;

typedef struct {
  LV2_URID midi_MidiEvent;
  LV2_URID atom_Sequence;
} URIs;

typedef struct {
  // LV2 ports
  const float*             in_l;
  float*                   out_l;
  const LV2_Atom_Sequence* midi_in;
  const float*             param_freeze_on_note;
  const float*             param_fft_size;
  const float*             param_spectral_mode;
  const float*             param_freeze_cc_number;

  LV2_URID_Map* map;
  URIs          uris;

  uint32_t srate;
  uint32_t fft_size;
  uint32_t hop_size;   // = fft_size / 2  (50% overlap)

  // Continuous input ring buffer — always filled with live audio
  double*  input_ring;
  uint32_t ring_pos;   // next write position

  // FFT work buffers
  double*       time_buf; // FFT input  (fft_size)
  double*       fft_out;  // IFFT output (fft_size)
  fftw_complex* spec;
  fftw_plan     plan_fwd;
  fftw_plan     plan_inv;

  // Hann window
  double* window;

  // Frozen spectrum
  double* mags;     // magnitudes       (fft_size/2 + 1)
  double* phases;   // synthesis phases, advanced each hop
  double* true_inc; // true phase increment per hop, per bin (estimated from signal)

  // Overlap-add output
  double*  ola_buf;      // circular OLA accumulation (fft_size)
  uint32_t ola_read_pos; // next read position in ola_buf
  uint32_t hop_counter;  // samples elapsed since last OLA frame

  bool frozen;

} FFTFreeze;

/* ------------------------------------------------------------------ */

static void map_uris(FFTFreeze* self)
{
  self->uris.midi_MidiEvent = self->map->map(self->map->handle, LV2_MIDI__MidiEvent);
  self->uris.atom_Sequence  = self->map->map(self->map->handle, LV2_ATOM__Sequence);
}

static void make_window(FFTFreeze* self)
{
  for (uint32_t n = 0; n < self->fft_size; ++n)
    self->window[n] = 0.5 - 0.5 * cos((2.0 * M_PI * n) / (double)self->fft_size);
}

static void realloc_fft(FFTFreeze* self, uint32_t new_size)
{
  if (self->plan_fwd) fftw_destroy_plan(self->plan_fwd);
  if (self->plan_inv) fftw_destroy_plan(self->plan_inv);
  free(self->time_buf);
  free(self->fft_out);
  free(self->window);
  free(self->mags);
  free(self->phases);
  free(self->true_inc);
  free(self->input_ring);
  free(self->ola_buf);
  fftw_free(self->spec);

  self->fft_size = new_size;
  self->hop_size = new_size / 2;

  self->time_buf   = (double*)calloc(self->fft_size,         sizeof(double));
  self->fft_out    = (double*)calloc(self->fft_size,         sizeof(double));
  self->window     = (double*)calloc(self->fft_size,         sizeof(double));
  self->mags       = (double*)calloc(self->fft_size/2 + 1,   sizeof(double));
  self->phases     = (double*)calloc(self->fft_size/2 + 1,   sizeof(double));
  self->true_inc   = (double*)calloc(self->fft_size/2 + 1,   sizeof(double));
  // Ring buffer holds 2×fft_size samples so capture_freeze can access the previous hop
  self->input_ring = (double*)calloc(self->fft_size * 2,     sizeof(double));
  self->ola_buf    = (double*)calloc(self->fft_size,         sizeof(double));
  self->spec       = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * (self->fft_size/2 + 1));

  self->plan_fwd = fftw_plan_dft_r2c_1d((int)self->fft_size,
                                         self->time_buf, self->spec, FFTW_MEASURE);
  self->plan_inv = fftw_plan_dft_c2r_1d((int)self->fft_size,
                                         self->spec, self->fft_out, FFTW_MEASURE);

  make_window(self);
  self->ring_pos     = 0;
  self->ola_read_pos = 0;
  self->hop_counter  = 0;
  self->frozen       = false;
}

/* ------------------------------------------------------------------ */

static LV2_Handle instantiate(const LV2_Descriptor*     descriptor,
                              double                    rate,
                              const char*               bundle_path,
                              const LV2_Feature* const* features)
{
  FFTFreeze* self = (FFTFreeze*)calloc(1, sizeof(FFTFreeze));
  if (!self) return NULL;

  self->srate = (uint32_t)lrint(rate);

  for (const LV2_Feature* const* f = features; f && *f; ++f) {
    if (!strcmp((*f)->URI, LV2_URID__map))
      self->map = (LV2_URID_Map*)(*f)->data;
  }
  if (!self->map) { free(self); return NULL; }

  map_uris(self);
  realloc_fft(self, 1024);
  return (LV2_Handle)self;
}

static void connect_port(LV2_Handle instance, uint32_t port, void* data)
{
  FFTFreeze* self = (FFTFreeze*)instance;
  switch ((Ports)port) {
    case PORT_IN_L:             self->in_l                  = (const float*)data;             break;
    case PORT_OUT_L:            self->out_l                 = (float*)data;                   break;
    case PORT_MIDI_IN:          self->midi_in               = (const LV2_Atom_Sequence*)data; break;
    case PORT_FREEZE_ON_NOTE:   self->param_freeze_on_note  = (const float*)data;             break;
    case PORT_FFT_SIZE:         self->param_fft_size        = (const float*)data;             break;
    case PORT_SPECTRAL_MODE:    self->param_spectral_mode   = (const float*)data;             break;
    case PORT_FREEZE_CC_NUMBER: self->param_freeze_cc_number= (const float*)data;             break;
  }
}

/* ------------------------------------------------------------------ */

static inline bool is_note_on(const uint8_t* msg)
{
  return ((msg[0] & 0xF0u) == LV2_MIDI_MSG_NOTE_ON) && msg[2] > 0;
}

static inline bool is_note_off(const uint8_t* msg)
{
  const uint8_t st = msg[0] & 0xF0u;
  return (st == LV2_MIDI_MSG_NOTE_OFF) || (st == LV2_MIDI_MSG_NOTE_ON && msg[2] == 0);
}

static inline bool is_cc_freeze(const uint8_t* msg, uint8_t cc_num)
{
  return ((msg[0] & 0xF0u) == 0xB0u) && (msg[1] == cc_num) && (msg[2] >= 64);
}

/* Capture the ring buffer into time_buf and compute the frozen spectrum. */
static void capture_freeze(FFTFreeze* self)
{
  const bool spectral = self->param_spectral_mode && (*self->param_spectral_mode >= 0.5f);
  const uint32_t K         = self->fft_size / 2 + 1;
  const uint32_t ring_size = self->fft_size * 2;

  // --- Current frame: most recent fft_size samples ---
  for (uint32_t i = 0; i < self->fft_size; ++i) {
    uint32_t idx = (self->ring_pos + i) % ring_size;
    self->time_buf[i] = self->input_ring[idx] * self->window[i];
  }
  fftw_execute(self->plan_fwd);

  // Store magnitudes and initial synthesis phases
  for (uint32_t k = 0; k < K; ++k) {
    double re = self->spec[k][0];
    double im = self->spec[k][1];
    self->mags[k]   = hypot(re, im);
    self->phases[k] = spectral ? ((double)rand() / RAND_MAX) * 2.0 * M_PI
                               : atan2(im, re);
  }

  // --- Estimate true phase increment per bin ---
  // For spectral mode the phase texture is intentionally random; nominal is fine.
  // For normal mode, we compare current phases against the frame one hop earlier
  // to compute the actual instantaneous frequency at each bin.  This eliminates
  // pulsation caused by off-bin-centre components advancing at the wrong rate.
  const double nominal_per_k = 2.0 * M_PI * (double)self->hop_size / (double)self->fft_size;

  if (spectral) {
    for (uint32_t k = 0; k < K; ++k)
      self->true_inc[k] = (double)k * nominal_per_k;
  } else {
    // Previous frame: same window applied to samples shifted back by one hop
    for (uint32_t i = 0; i < self->fft_size; ++i) {
      uint32_t idx = (self->ring_pos + i + ring_size - self->hop_size) % ring_size;
      self->time_buf[i] = self->input_ring[idx] * self->window[i];
    }
    fftw_execute(self->plan_fwd); // spec now holds the previous frame

    for (uint32_t k = 0; k < K; ++k) {
      double phi_old = atan2(self->spec[k][1], self->spec[k][0]);
      double nominal = (double)k * nominal_per_k;
      // Unwrap the phase difference to the principal determination near the nominal
      double dev = self->phases[k] - phi_old - nominal;
      dev -= 2.0 * M_PI * round(dev / (2.0 * M_PI));
      // Fall back to nominal for near-silence bins where phase is unreliable
      self->true_inc[k] = (self->mags[k] > 1e-10) ? nominal + dev : nominal;
    }
  }

  // Reset OLA state
  memset(self->ola_buf, 0, self->fft_size * sizeof(double));
  self->ola_read_pos = 0;
  self->hop_counter  = 0;
  self->frozen       = true;
}

/* Generate one OLA frame: advance phases, IFFT, window, overlap-add into ola_buf. */
static void generate_ola_frame(FFTFreeze* self)
{
  const uint32_t K    = self->fft_size / 2 + 1;
  const double   invN = 1.0 / (double)self->fft_size;

  for (uint32_t k = 0; k < K; ++k) {
    self->phases[k] += self->true_inc[k];  // true instantaneous frequency increment
    self->spec[k][0] = self->mags[k] * cos(self->phases[k]);
    self->spec[k][1] = self->mags[k] * sin(self->phases[k]);
  }

  fftw_execute(self->plan_inv);

  // Apply synthesis window and overlap-add
  for (uint32_t i = 0; i < self->fft_size; ++i) {
    double sample = self->fft_out[i] * invN;
    uint32_t pos  = (self->ola_read_pos + i) % self->fft_size;
    self->ola_buf[pos] += sample;
  }
}

/* ------------------------------------------------------------------ */

static void handle_midi(FFTFreeze* self)
{
  if (!self->midi_in) return;
  if (!self->param_freeze_on_note || *self->param_freeze_on_note < 0.5f) return;

  uint8_t freeze_cc = self->param_freeze_cc_number
                      ? (uint8_t)(*self->param_freeze_cc_number) : 20;

  LV2_ATOM_SEQUENCE_FOREACH(self->midi_in, ev) {
    if (ev->body.type != self->uris.midi_MidiEvent) continue;
    const uint8_t* const msg = (const uint8_t*)(ev + 1);

    if (!self->frozen && (is_note_on(msg) || is_cc_freeze(msg, freeze_cc))) {
      capture_freeze(self);
    } else if (self->frozen && (is_note_off(msg) || (is_cc_freeze(msg, freeze_cc) == false
               && (msg[0] & 0xF0u) == 0xB0u && msg[1] == freeze_cc))) {
      self->frozen = false;
    }
  }
}

static void run(LV2_Handle instance, uint32_t n_samples)
{
  FFTFreeze* self = (FFTFreeze*)instance;

  // Dynamically resize FFT if parameter changed
  if (self->param_fft_size) {
    uint32_t req = (uint32_t)lrintf(*self->param_fft_size);
    if (req != self->fft_size && req >= 256 && req <= 8192)
      realloc_fft(self, req);
  }

  // Always fill the ring buffer with live audio (needed for capture_freeze)
  if (self->in_l) {
    for (uint32_t i = 0; i < n_samples; ++i) {
      self->input_ring[self->ring_pos] = (double)self->in_l[i];
      self->ring_pos = (self->ring_pos + 1) % (self->fft_size * 2);
    }
  }

  handle_midi(self);

  if (!self->in_l || !self->out_l) return;

  if (!self->frozen) {
    // Passthrough
    for (uint32_t i = 0; i < n_samples; ++i)
      self->out_l[i] = self->in_l[i];
  } else {
    // Overlap-add synthesis of the frozen spectrum
    // With Hann window at 50% overlap, the sum of two adjacent windows = 1.0,
    // so no extra gain compensation is needed.
    for (uint32_t i = 0; i < n_samples; ++i) {
      if (self->hop_counter == 0)
        generate_ola_frame(self);

      self->out_l[i] = (float)self->ola_buf[self->ola_read_pos];
      self->ola_buf[self->ola_read_pos] = 0.0; // clear after reading
      self->ola_read_pos = (self->ola_read_pos + 1) % self->fft_size;
      self->hop_counter  = (self->hop_counter  + 1) % self->hop_size;
    }
  }
}

static void cleanup(LV2_Handle instance)
{
  FFTFreeze* self = (FFTFreeze*)instance;
  if (!self) return;

  if (self->plan_fwd) fftw_destroy_plan(self->plan_fwd);
  if (self->plan_inv) fftw_destroy_plan(self->plan_inv);
  fftw_free(self->spec);
  free(self->time_buf);
  free(self->fft_out);
  free(self->window);
  free(self->mags);
  free(self->phases);
  free(self->true_inc);
  free(self->input_ring);
  free(self->ola_buf);
  free(self);
}

static const void* extension_data(const char* uri) { (void)uri; return NULL; }

static const LV2_Descriptor descriptor = {
  FFT_FREEZE_URI,
  instantiate,
  connect_port,
  NULL,
  run,
  NULL,
  cleanup,
  extension_data
};

LV2_SYMBOL_EXPORT
const LV2_Descriptor* lv2_descriptor(uint32_t index)
{
  return (index == 0) ? &descriptor : NULL;
}
