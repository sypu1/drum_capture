"""Score a transcription against a reference MIDI (precision / recall / F1 per drum).

Ride notes in the reference are scored as closed hi-hat: a ride played as a
time-keeper is, to this detector, indistinguishable from a closed hat (see
README, "Known limitations").  Pass --ride-as crash to score them as crash, or --ride-as ride for a model
that knows the ride (a trained kit model).

    python tools/evaluate.py reference.mid audio.wav [--separate no] [--tol 0.05]

Scores the raw detected hits (before quantisation), so it measures the
detector + classifier.  Handy when tuning thresholds in drumsheet/config.py.
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pretty_midi  # noqa: E402

from drumsheet.audio_io import load_audio  # noqa: E402
from drumsheet.config import MIDI_ALIASES  # noqa: E402
from drumsheet.separation import get_drum_signal  # noqa: E402
from drumsheet.backends import get_backend  # noqa: E402


def match(ref, est, tol):
    """Greedy one-to-one matching of two sorted time lists."""
    used, tp = set(), 0
    for t in ref:
        best = min((j for j in range(len(est)) if j not in used and abs(est[j] - t) <= tol),
                   key=lambda j: abs(est[j] - t), default=None)
        if best is not None:
            used.add(best); tp += 1
    return tp


def evaluate(ref_midi, hits, tol=0.05, merge_toms=False, ride_as="hihat_closed"):
    ref, est = defaultdict(list), defaultdict(list)
    fold = (lambda d: "tom" if d.startswith("tom") else d) if merge_toms else (lambda d: d)
    for inst in pretty_midi.PrettyMIDI(str(ref_midi)).instruments:
        if inst.is_drum:
            for n in inst.notes:
                if n.pitch in MIDI_ALIASES:
                    drum = ride_as if n.pitch in (51, 53, 59) else MIDI_ALIASES[n.pitch]
                    ref[fold(drum)].append(n.start)
    for h in hits:
        est[fold(h.drum)].append(h.time)
    rows, TP, NR, NE = [], 0, 0, 0
    for d in sorted(set(ref) | set(est)):
        r, e = sorted(ref[d]), sorted(est[d])
        tp = match(r, e, tol)
        p, rc = tp / len(e) if e else 0, tp / len(r) if r else 0
        rows.append((d, len(r), len(e), p, rc, 2 * p * rc / (p + rc) if p + rc else 0))
        TP += tp; NR += len(r); NE += len(e)
    p, rc = TP / max(NE, 1), TP / max(NR, 1)
    rows.append(("ALL", NR, NE, p, rc, 2 * p * rc / (p + rc) if p + rc else 0))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ref"); ap.add_argument("audio")
    ap.add_argument("--separate", default="no", choices=["auto", "yes", "no"])
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--merge-toms", action="store_true", help="score all toms as one class")
    ap.add_argument("--ride-as", default="hihat_closed", choices=["hihat_closed", "crash", "ride"])
    ap.add_argument("--model", default="builtin", help="which model to score (see `python -m drumsheet models`)")
    a = ap.parse_args()
    y = load_audio(a.audio)
    y, how = get_drum_signal(a.audio, y, 44100, a.separate)
    hits = get_backend(a.model).transcribe(y, 44100)
    print(f"model: {a.model}   separation: {how}")
    print(f"{'drum':14s} {'ref':>4s} {'est':>4s} {'prec':>6s} {'rec':>6s} {'F1':>6s}")
    for d, nr, ne, p, r, f in evaluate(a.ref, hits, a.tol, a.merge_toms, a.ride_as):
        print(f"{d:14s} {nr:4d} {ne:4d} {p:6.2f} {r:6.2f} {f:6.2f}")
