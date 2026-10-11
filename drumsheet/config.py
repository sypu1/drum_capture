"""Central place for everything you are likely to want to tweak.

* DRUMS            - the kit pieces we transcribe, with their MIDI note and
                     where/how they are drawn on the staff.
* TranscribeConfig - thresholds of the onset detector and the drum classifier.
* GRIDS            - the quantisation grids the CLI offers.
"""
from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Kit definition
# ---------------------------------------------------------------------------
# staff_step: vertical position counted in diatonic steps from the BOTTOM staff
#             line (0 = bottom line, 1 = first space, ... 8 = top line).
# head:       "x" for cymbals, "normal" for drums.
# xml:        (display-step, display-octave) used in MusicXML.
DRUMS: dict[str, dict] = {
    "crash":        dict(label="Crash",          short="CR", midi=49, staff_step=10, head="x",      xml=("A", 5)),
    "ride":         dict(label="Ride",           short="RD", midi=51, staff_step=8,  head="x",      xml=("F", 5)),
    "hihat_open":   dict(label="Open hi-hat",    short="OH", midi=46, staff_step=9,  head="x",      xml=("G", 5)),
    "hihat_closed": dict(label="Closed hi-hat",  short="HH", midi=42, staff_step=9,  head="x",      xml=("G", 5)),
    "tom_high":     dict(label="High tom",       short="T1", midi=50, staff_step=7,  head="normal", xml=("E", 5)),
    "tom_mid":      dict(label="Mid tom",        short="T2", midi=47, staff_step=6,  head="normal", xml=("D", 5)),
    "tom_floor":    dict(label="Floor tom",      short="FT", midi=43, staff_step=3,  head="normal", xml=("A", 4)),
    "snare":        dict(label="Snare",          short="SD", midi=38, staff_step=5,  head="normal", xml=("C", 5)),
    "kick":         dict(label="Kick",           short="BD", midi=36, staff_step=1,  head="normal", xml=("F", 4)),
}
# Note: only a model trained on your own kit tells ride from crash and hi-hat.
# The built-in rules and ADTOF write every long-ringing cymbal as "crash".
# Top-to-bottom order used by the grid chart and the legend.
DRUM_ORDER = list(DRUMS.keys())

# Extra General-MIDI notes that we fold onto our classes when READING a MIDI
# file (used by tools/evaluate.py).
MIDI_ALIASES = {35: "kick", 36: "kick", 37: "snare", 38: "snare", 40: "snare",
                42: "hihat_closed", 44: "hihat_closed", 46: "hihat_open",
                49: "crash", 57: "crash", 55: "crash", 52: "crash",
                51: "ride", 53: "ride", 59: "ride",
                41: "tom_floor", 43: "tom_floor", 45: "tom_mid", 47: "tom_mid",
                48: "tom_high", 50: "tom_high"}

# ---------------------------------------------------------------------------
# Quantisation grids: name -> slots per quarter-note beat
# ---------------------------------------------------------------------------
GRIDS = {"8": 2, "16": 4, "8t": 3, "16t": 6}


# ---------------------------------------------------------------------------
# Detector / classifier thresholds
# ---------------------------------------------------------------------------
@dataclass
class TranscribeConfig:
    sr: int = 44100              # analysis sample rate
    n_fft: int = 1024            # short window -> good time resolution
    hop: int = 256               # ~5.8 ms per frame

    # Frequency bands (Hz) whose energy rise marks an onset.
    band_low: tuple = (30, 130)        # kick / floor-tom body
    band_mid: tuple = (150, 1200)      # snare + tom body
    band_high: tuple = (7500, 16000)   # cymbals

    # Onset picking.  Lower `delta` = more (and more false) onsets.
    # The CLI's --sensitivity divides these values.
    delta_low: float = 0.20
    delta_mid: float = 0.20
    delta_high: float = 0.16
    min_gap_s: float = 0.045           # two hits of one band can't be closer
    merge_window_s: float = 0.045      # bands firing this close = same stroke
    compress: float = 30.0             # log compression before differencing

    # A band only counts if its energy is at least this fraction of the
    # typical (median) hit in that band - suppresses bleed and noise.
    min_rel_level: float = 0.12

    weak_rise: float = 0.35            # a kick may ride on a low-band rise this fraction of delta_low
    low_lookback_s: float = 0.06       # low band: compare with the max of this long a past

    # --- classification -----------------------------------------------------
    # "tone" below = power at 120-400 Hz (drum-head ring) relative to something.
    band_wash: tuple = (3000, 7500)    # where cymbal sustain is measured
    # Kick: same pitch as the lowest drum in the recording, and little tone.
    kick_abs_max_hz: float = 180.0     # nothing pitched above this is a kick (phone mics
                                       # lose the fundamental, so a kick can read ~150 Hz)
    kick_min_ref_hz: float = 55.0
    kick_pitch_tol: float = 1.25       # up to this ratio above the lowest drum's pitch
    kick_tone_tol: float = 3.5         # up to this multiple of the typical kick's tone
    snare_over_kick_tone: float = 6.0  # kick+snare: tone at least this multiple
    # A drum (snare/tom) is present if tone >= this fraction of 2-8 kHz power;
    # cymbals alone have almost no tone.
    drum_min_tone: float = 0.45
    # Snare vs tom: (2-8 kHz)/(100-1000 Hz) power 30-90 ms after the attack.
    # Snare wires keep sizzling; a tom is only its pitched body by then.
    snare_sizzle_ratio: float = 0.07
    # Toms: pitch groups further apart than this ratio are different toms.
    tom_gap_ratio: float = 1.12
    tom_floor_max_hz: float = 110.0    # only used when a single tom pitch is heard
    tom_mid_max_hz: float = 160.0
    # Cymbals: share of the stroke still ringing ~0.25 s later (3-7.5 kHz).
    open_min_ring: float = 0.15        # above = open hi-hat, below = closed
    crash_min_ring: float = 0.50       # above = crash / ride
    open_min_ring_short: float = 0.45  # same test at ~90 ms, for fast passages
    retrigger_s: float = 0.060         # same drum again sooner than this = double trigger
