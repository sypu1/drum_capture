"""Write a reference drum MIDI (and optionally render it through a SoundFont)
so the transcriber can be scored against known ground truth.

    python tools/make_test_midi.py out_dir --sf2 path/to/soundfont.sf2 [--kit 0] [--bpm 104] [--backing]

Rendering needs `pip install pyfluidsynth` plus the fluidsynth library
(macOS: `brew install fluid-synth`).
"""
import argparse
from pathlib import Path

import numpy as np
import pretty_midi

K, S, HC, HO, CR, RD, T1, T2, FT = 36, 38, 42, 46, 49, 51, 50, 47, 43


def build(bpm=104.0, beats_per_bar=4, backing=False, seed=0, two_toms=False):
    rng = np.random.default_rng(seed)
    t1, t2, ft = (45, 45, 41) if two_toms else (T1, T2, FT)   # AVL 4-piece kits: rack + floor
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    drums = pretty_midi.Instrument(0, is_drum=True)
    s16 = 60 / bpm / 4
    bar_len = beats_per_bar * 4

    def add(bar, slot, pitch, vel=100):
        t = (bar * bar_len + slot) * s16 + rng.normal(0, 0.004)      # slight human timing
        v = int(np.clip(vel + rng.normal(0, 6), 30, 127))
        drums.notes.append(pretty_midi.Note(v, pitch, max(t, 0), max(t, 0) + 0.1))

    if beats_per_bar == 3:                       # simple waltz: K S S
        for bar in range(12):
            add(bar, 0, K); add(bar, 4, S); add(bar, 8, S)
            for s in range(0, 12, 2):
                add(bar, s, HC, 80)
    else:
        for bar in range(16):
            ride = 8 <= bar < 12                 # bars 9-12 on the ride
            fill = bar in (3, 7, 15)
            if bar % 4 == 0:
                add(bar, 0, CR, 115)
            for s in range(0, 16, 2):            # eighth-note cymbal pattern
                if fill and s >= 8:
                    break
                if bar % 4 == 0 and s == 0:
                    continue
                if ride:
                    add(bar, s, RD, 90)
                elif s == 14 and bar % 2 == 1 and not fill:
                    add(bar, s, HO, 95)          # open hat on the "and" of 4
                else:
                    add(bar, s, HC, 85 if s % 4 == 0 else 70)
            for s in ((0, 6, 8) if bar % 2 == 0 else (0, 8, 10)):
                if not (fill and s >= 8):
                    add(bar, s, K, 110)
            for s in (4, 12):
                if not (fill and s >= 8):
                    add(bar, s, S, 108)
            if bar % 4 == 2:
                add(bar, 3, K, 100)              # kick with no cymbal on top
            if bar >= 12 and not fill:
                add(bar, 7, S, 45)               # ghost note
            if fill:                             # 16th-note tom fill on beats 3-4
                for i, p in enumerate((S, S, t1, t1, t2, t2, ft, ft)):
                    add(bar, 8 + i, p, 100)
    pm.instruments.append(drums)

    if backing:                                  # bass + electric piano = "full song"
        bass, keys = pretty_midi.Instrument(33), pretty_midi.Instrument(4)
        beat = 60 / bpm
        for bar, root in enumerate([40, 45, 43, 47] * 4):
            t0 = bar * 4 * beat
            for e in range(8):
                bass.notes.append(pretty_midi.Note(95, root - 12, t0 + e * beat / 2, t0 + (e + .9) * beat / 2))
            for p in (root + 12, root + 16, root + 19):
                keys.notes.append(pretty_midi.Note(80, p, t0, t0 + 4 * beat * .95))
        pm.instruments += [bass, keys]
    return pm


def render(pm, drum_sf2, kit=0, backing_sf2=None, sr=44100):
    """Render with pyfluidsynth.  `kit` is the drum preset number (FluidR3: 0
    Standard, 8 Room, 16 Power, 32 Jazz).  Drum-only SoundFonts that keep their
    kit in bank 0 (e.g. AVL Drumkits) are handled automatically."""
    import fluidsynth

    fs = fluidsynth.Synth(samplerate=sr, gain=0.5)
    sfid = fs.sfload(str(drum_sf2))
    bank = 128 if fs.sfpreset_name(sfid, 128, kit) else 0
    fs.program_select(9, sfid, bank, kit if bank == 128 else 0)
    events = []
    melodic = [i for i in pm.instruments if not i.is_drum]
    if melodic:
        sfid2 = fs.sfload(str(backing_sf2 or drum_sf2))
    for ch, inst in enumerate(melodic):
        fs.program_select(ch, sfid2, 0, inst.program)
    for inst in pm.instruments:
        ch = 9 if inst.is_drum else melodic.index(inst)
        for n in inst.notes:
            events.append((n.start, 1, ch, n.pitch, n.velocity))
            if not inst.is_drum:
                events.append((n.end, 0, ch, n.pitch, 0))
    events.sort()
    out, now = [], 0.0
    for t, on, ch, pitch, vel in events + [(pm.get_end_time() + 2.0, None, 0, 0, 0)]:
        n = int((t - now) * sr)
        if n > 0:
            out.append(fs.get_samples(n).reshape(-1, 2).mean(axis=1))
            now += n / sr
        if on == 1:
            fs.noteon(ch, pitch, vel)
        elif on == 0:
            fs.noteoff(ch, pitch)
    fs.delete()
    audio = np.concatenate(out).astype(np.float64)
    return audio / (np.abs(audio).max() + 1e-9) * 0.9


def calibration(drum_pitches, n_hits=12, gap=0.9, seed=1):
    """{drum: PrettyMIDI of that drum alone, n_hits strokes soft to hard}."""
    rng = np.random.default_rng(seed)
    out = {}
    for drum, pitch in drum_pitches.items():
        pm = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(0, is_drum=True)
        for i in range(n_hits):
            v = int(np.clip(60 + 60 * i / (n_hits - 1) + rng.normal(0, 4), 40, 127))
            inst.notes.append(pretty_midi.Note(v, pitch, 0.3 + i * gap, 0.4 + i * gap))
        pm.instruments.append(inst)
        out[drum] = pm
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("--sf2", help="SoundFont to render the drums with")
    ap.add_argument("--kit", type=int, default=0, help="drum kit preset number in the SoundFont")
    ap.add_argument("--backing-sf2", help="GM SoundFont for bass/keys (default: same as --sf2)")
    ap.add_argument("--bpm", type=float, default=104)
    ap.add_argument("--beats", type=int, default=4, help="beats per bar (3 or 4)")
    ap.add_argument("--backing", action="store_true", help="add bass + keys (a 'full song')")
    ap.add_argument("--lead-in", type=float, default=0.5, help="silence before the first note (s)")
    ap.add_argument("--two-toms", action="store_true",
                    help="rack tom + floor tom only, on notes 45/41 (for AVL Drumkits SoundFonts)")
    ap.add_argument("--calibration", action="store_true",
                    help="instead of the groove, write one file per drum (for `drumsheet train`)")
    ap.add_argument("--seed", type=int, default=0, help="a different seed = a different performance")
    ap.add_argument("--name", default="test")
    a = ap.parse_args()
    if a.calibration:
        import soundfile as sf
        t1, t2, ft = (45, 45, 41) if a.two_toms else (T1, T2, FT)
        drums = {"kick": K, "snare": S, "hihat_closed": HC, "hihat_open": HO, "crash": CR,
                 "ride": RD, "tom_high": t1, "tom_floor": ft}
        if not a.two_toms:
            drums["tom_mid"] = t2
        d = Path(a.out_dir) / f"{a.name}_calib"
        d.mkdir(parents=True, exist_ok=True)
        for drum, pm in calibration(drums).items():
            sf.write(str(d / f"{drum}.wav"), render(pm, a.sf2, a.kit), 44100)
        print("wrote", d)
        raise SystemExit
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pm = build(a.bpm, a.beats, a.backing, seed=a.seed, two_toms=a.two_toms)
    for inst in pm.instruments:                   # shift everything by the lead-in
        for n in inst.notes:
            n.start += a.lead_in
            n.end += a.lead_in
    ref = pretty_midi.PrettyMIDI(initial_tempo=a.bpm)      # reference = drums only
    ref.instruments.append(pm.instruments[0])
    ref.write(str(out / f"{a.name}_ref.mid"))
    print("wrote", out / f"{a.name}_ref.mid")
    if a.sf2:
        import soundfile as sf
        sf.write(str(out / f"{a.name}.wav"), render(pm, a.sf2, a.kit, a.backing_sf2), 44100)
        print("wrote", out / f"{a.name}.wav")
