"""A drum model trained on YOUR kit, room and microphone.

Idea
----
Every kit sounds different, and so does every microphone.  Instead of general
rules, this learns what each of your drums looks like from short recordings:

  1. Calibration: you record each drum on its own, about ten hits, a second
     apart, soft and hard.  One file per drum.
  2. Those single hits are cut out and mixed with each other at random levels
     to make thousands of examples of "kick + hi-hat", "snare + crash", ... -
     because in real playing drums are mostly hit together.
  3. A small classifier per drum (extremely randomised trees, scikit-learn) learns to answer "is this drum in this stroke?" from the
     stroke's spectrum over its first 200 ms.
  4. Optionally you add real takes with a corrected MIDI file (fix the app's
     MIDI in a DAW or MuseScore, feed it back): those strokes are learned
     exactly as they sound in context.  This is the "retrain as you go" loop.
  5. Or you skip the correcting: play a piece you have the sheet of, and give
     the sheet with your recording.  sheet.py finds every written note in the
     take; the strokes are then learned like those of step 4.  This works on
     its own too, without step 1.

Training takes seconds on a laptop CPU.  The result is saved as kits/<name>.joblib
and appears as model "kit:<name>" everywhere.

Because you name the drums yourself, this is the only model here that can
tell ride from hi-hat from crash.
"""
from __future__ import annotations

import datetime
import os
from dataclasses import replace
from pathlib import Path

import numpy as np

from .audio_io import AudioError, load_audio
from .backends import Backend
from .config import DRUMS, MIDI_ALIASES, TranscribeConfig
from .transcribe import Hit, detect_strokes, drop_retriggers

KITS_DIR = Path(os.environ.get("DRUMSHEET_KITS", Path(__file__).resolve().parent.parent / "kits"))

SR = 44100
N_FFT, HOP, N_MELS = 1024, 256, 48
FPS = SR / HOP
# The stroke is described in three time slices (frames after the onset):
# attack (~0-35 ms), body (~35-90 ms), tail (~90-200 ms).
POOLS = ((-1, 6), (6, 15), (15, 34))
PRE_S = 0.015                      # snippets start this long before the onset
FLOOR_DB = -60.0
THRESHOLD = 0.6                    # a drum is reported if its forest is at least this sure
UNSURE = 2 / 3                     # a stroke no forest is that sure of still gets its likeliest drum, down
                                   # to this fraction of the threshold.  When the playing sounds a little
                                   # unlike the training recordings every forest is less sure, and without
                                   # this half the strokes went missing while the best guess was mostly right.

_MEL = None


def _mel_frames(y: np.ndarray) -> np.ndarray:
    """Mel power spectrogram [N_MELS, frames] of the whole signal."""
    import librosa
    global _MEL
    if _MEL is None:
        _MEL = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=30, fmax=16000)
    S = np.abs(librosa.stft(y.astype(np.float32), n_fft=N_FFT, hop_length=HOP)) ** 2
    return _MEL @ S


def _reference_db(M: np.ndarray) -> float:
    """How loud this recording's strong strokes are (dB)."""
    return float(10 * np.log10(np.percentile(M.sum(axis=0), 99) + 1e-12))


def _features(M: np.ndarray, t: float, ref_db: float) -> np.ndarray:
    """Describe the stroke at time t: its spectrum's shape in three time slices
    (level-independent), plus one number for how loud it is in this recording."""
    i = int(round(t * FPS))
    slices = []
    for a, b in POOLS:
        a, b = max(0, i + a), min(M.shape[1], max(i + b, 1))
        seg = M[:, a:b].mean(axis=1) if b > a else np.zeros(M.shape[0])
        slices.append(10 * np.log10(seg + 1e-12))
    spec = np.concatenate(slices)
    peak = float(spec.max())
    shape = np.maximum(spec - peak, FLOOR_DB)
    # What is NEW in this stroke: how far each band jumped above the ~35 ms
    # before it.  A drum that is merely still ringing from the previous stroke
    # shows up in `shape` but not here.
    a, b = max(0, i - 7), max(1, i - 1)
    before = 10 * np.log10(M[:, a:b].mean(axis=1) + 1e-12)
    rise = np.clip(slices[0] - np.maximum(before, peak + FLOOR_DB), 0, 40)
    return np.concatenate([shape, rise, [np.clip(peak - ref_db, -80, 30)]]).astype(np.float32)


# ---------------------------------------------------------------------------
# Kit files
# ---------------------------------------------------------------------------
def list_kits() -> list[str]:
    return sorted(p.stem for p in KITS_DIR.glob("*.joblib")) if KITS_DIR.exists() else []


def load_kit(name: str) -> dict:
    import joblib
    path = KITS_DIR / f"{name}.joblib"
    if not path.exists():
        raise AudioError(f"No trained kit called '{name}' in {KITS_DIR}.")
    return joblib.load(path)       # only ever load kit files you trained yourself


def delete_kit(name: str) -> None:
    (KITS_DIR / f"{name}.joblib").unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Finding the single hits in a calibration recording
# ---------------------------------------------------------------------------
def find_hits(y: np.ndarray, min_gap_s: float = 0.25) -> list[float]:
    """Onset times of the separate strokes in a one-drum recording."""
    strokes, _ = detect_strokes(y, TranscribeConfig())
    if not strokes:
        return []
    # loudness of each stroke = peak of |y| in the 60 ms after it
    def loud(t):
        a = int(t * SR)
        return float(np.abs(y[a:a + int(0.06 * SR)]).max(initial=0.0))
    times, levels = [], []
    for s in strokes:
        if times and s["time"] - times[-1] < min_gap_s:
            continue
        times.append(s["time"])
        levels.append(loud(s["time"]))
    gate = 0.15 * float(np.median(levels))
    return [t for t, lv in zip(times, levels) if lv >= gate]


def _cut(y: np.ndarray, t: float, t_next: float | None, length_s: float = 0.6) -> np.ndarray:
    """One hit as a snippet starting PRE_S before the onset, faded out at the end."""
    a = max(0, int((t - PRE_S) * SR))
    b = a + int(length_s * SR)
    if t_next is not None:
        b = min(b, int((t_next - 0.03) * SR))
    snip = y[a:b].astype(np.float32).copy()
    fade = min(len(snip), int(0.02 * SR))
    if fade:
        snip[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
    return snip


# ---------------------------------------------------------------------------
# Building training examples
# ---------------------------------------------------------------------------
def _example(parts: list[tuple[np.ndarray, float, float]], rng, length_s: float = 0.45) -> np.ndarray:
    """Mix snippets [(audio, gain, delay_s)] and return the features of the stroke
    at the start.  A random level offset stands in for "there are louder strokes
    elsewhere in the recording"."""
    n = int(length_s * SR)
    lead = int(0.5 * SR)                          # room for an earlier stroke that still rings
    mix = np.zeros(lead + n, dtype=np.float32)
    for audio, gain, delay in parts:
        a = lead + int(delay * SR)                # delay < 0: it was hit before our stroke
        seg = audio[:max(0, len(mix) - a)]
        mix[a:a + len(seg)] += gain * seg
    M = _mel_frames(mix)
    loudest = 10 * np.log10(M.sum(axis=0).max() + 1e-12)
    jitter = rng.uniform(-0.006, 0.006)
    return _features(M, lead / SR + PRE_S + jitter, loudest + rng.uniform(0, 20))


def _synthesise(snips: dict[str, list[np.ndarray]], rng, n_mix: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Singles, random combinations and "nothing new was hit" examples."""
    classes = sorted(snips)
    X, Y = [], []

    def gain():
        return 10 ** (rng.uniform(-18, 0) / 20)

    def label(names):
        return [1 if c in names else 0 for c in classes]

    def pick(c):
        return snips[c][rng.integers(len(snips[c]))]

    def combo(k):
        names = []
        while len(names) < k:
            c = classes[rng.integers(len(classes))]
            # the two hi-hat states cannot sound together
            if c in names or (c.startswith("hihat") and any(n.startswith("hihat") for n in names)):
                if len(classes) <= k:
                    break
                continue
            names.append(c)
        return names

    for c in classes:                               # every single hit, three levels each
        for s in snips[c]:
            for _ in range(3):
                X.append(_example([(s, gain(), 0.0)], rng)); Y.append(label([c]))
    for i in range(n_mix):
        names = combo(2 if i % 3 else 3) if len(classes) > 1 else [classes[0]]
        parts = [(pick(c), gain(), abs(rng.normal(0, 0.004))) for c in names]
        if i % 4 == 0:                              # the NEXT stroke arrives early (fast playing)
            parts.append((pick(classes[rng.integers(len(classes))]), gain(), rng.uniform(0.09, 0.25)))
        if i % 2 == 0:                              # an EARLIER stroke is still ringing underneath
            parts.append((pick(classes[rng.integers(len(classes))]), gain(), -rng.uniform(0.11, 0.45)))
        X.append(_example(parts, rng)); Y.append(label(names))
    for c in classes:                               # a drum still ringing is not a new hit
        for s in snips[c]:
            X.append(_example([(s, gain(), -rng.uniform(0.11, 0.3))], rng)); Y.append(label([]))
    return np.array(X), np.array(Y), classes


def _from_pair(audio_path, midi_path, classes: list[str], tol: float = 0.04):
    """Real strokes with their true labels, from a take and its (corrected) MIDI."""
    import pretty_midi

    y = load_audio(audio_path, SR)
    ref = []
    for inst in pretty_midi.PrettyMIDI(str(midi_path)).instruments:
        if inst.is_drum:
            ref += [(n.start, MIDI_ALIASES[n.pitch]) for n in inst.notes if n.pitch in MIDI_ALIASES]
    if not ref:
        raise AudioError(f"{Path(midi_path).name} has no drum notes (channel 10).")
    strokes, _ = detect_strokes(y, TranscribeConfig())
    M = _mel_frames(y)
    ref_db = _reference_db(M)
    X, Y = [], []
    for s in strokes:
        names = {d for t, d in ref if abs(t - s["time"]) <= tol}
        X.append(_features(M, s["time"], ref_db))
        Y.append([1 if c in names else 0 for c in classes])
    return np.array(X), np.array(Y)


def _from_sheet(audio_path, notes: list, classes: list[str]):
    """Real strokes with the labels the sheet gives them, from a recording of
    someone playing that sheet (notes: sheet.read_sheet()).  Unlike a corrected
    MIDI the sheet is not in time with the take; sheet.align() finds its notes.
    Returns (X, Y, what the alignment found)."""
    from .sheet import align

    y = load_audio(audio_path, SR)
    strokes, _ = detect_strokes(y, TranscribeConfig())
    M = _mel_frames(y)
    ref_db = _reference_db(M)
    sound = np.array([_features(M, s["time"], ref_db) for s in strokes]).reshape(-1, 4 * N_MELS + 1)
    found = align(notes, strokes, len(y) / SR, sound)
    use = [j for j, names in enumerate(found.labels) if names is not None]     # (None: the sheet does not say)
    Y = np.array([[1 if c in found.labels[j] else 0 for c in classes] for j in use]).reshape(-1, len(classes))
    return sound[use], Y, found


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_kit(name: str, drum_files: dict[str, list], pairs: list[tuple] | None = None,
              n_mix: int = 3000, seed: int = 0, log=print, sheets: list[tuple] | None = None) -> dict:
    """Train and save a kit model.

    drum_files: {"kick": [path, ...], "snare": [...], ...}  one-drum recordings
    pairs:      [(audio_path, midi_path), ...]              takes with corrected MIDI
    sheets:     [(audio_path, sheet), ...]                  recordings of you playing a sheet, at any
                                                            tempo.  sheet: a MIDI or MusicXML file, a
                                                            PDF, or its pages as photos [path, ...]
    Returns the saved kit's info dict.
    """
    from sklearn.ensemble import ExtraTreesClassifier
    import joblib
    from .sheet import PAGE_TYPES, read_sheet, sheet_name

    if not name or not all(ch.isalnum() or ch in "-_" for ch in name):
        raise AudioError("Kit name: letters, digits, '-' and '_' only.")
    pairs = pairs or []
    sheets = [(audio, sheet, read_sheet(sheet, log)) for audio, sheet in sheets or []]
    unknown = set(drum_files) - set(DRUMS)
    if unknown:
        raise AudioError(f"Unknown drum(s): {', '.join(sorted(unknown))}. Use: {', '.join(DRUMS)}")
    rng = np.random.default_rng(seed)

    snips: dict[str, list[np.ndarray]] = {}
    for drum, files in drum_files.items():
        for f in files:
            y = load_audio(f, SR)
            times = find_hits(y)
            for i, t in enumerate(times):
                snips.setdefault(drum, []).append(_cut(y, t, times[i + 1] if i + 1 < len(times) else None))
            log(f"      {DRUMS[drum]['label']:14s} {Path(f).name}: {len(times)} hits")
    few = [DRUMS[d]["label"] for d, s in snips.items() if len(s) < 4]
    if few:
        raise AudioError(f"Too few hits found for: {', '.join(few)}. Record at least 6, a second apart.")
    if not snips and not pairs and not sheets:
        raise AudioError("Nothing to train on: give one-drum recordings, a sheet with your recording "
                         "of it, and/or audio+MIDI pairs.")

    classes = sorted(set(snips))
    if pairs and not classes:                     # pairs only: classes = what the MIDI contains
        import pretty_midi
        classes = sorted({MIDI_ALIASES[n.pitch] for a, m in pairs
                          for inst in pretty_midi.PrettyMIDI(str(m)).instruments if inst.is_drum
                          for n in inst.notes if n.pitch in MIDI_ALIASES})
    classes = sorted(set(classes) | {d for _, _, notes in sheets for _, d in notes})   # ... and every drum of the sheets

    def fit(Xa, Ya):
        # one forest per drum: "is this drum in the stroke?"  (Separate forests
        # scored clearly better in tests than one forest answering for all.)
        forests = []
        for j in range(Ya.shape[1]):
            if Ya[:, j].min() == Ya[:, j].max():          # never / always present: nothing to learn
                forests.append(float(Ya[0, j]))
                continue
            forests.append(ExtraTreesClassifier(n_estimators=100, min_samples_leaf=2, max_features=0.25,
                                                n_jobs=-1, random_state=seed).fit(Xa, Ya[:, j]))
        return forests

    def score(P, Y, names):
        """F1 per drum; a drum that is not in the test strokes cannot be scored."""
        out = {}
        for j, c in enumerate(names):
            tp = int(((P[:, j] >= THRESHOLD) & (Y[:, j] == 1)).sum())
            fp = int(((P[:, j] >= THRESHOLD) & (Y[:, j] == 0)).sum())
            fn = int(((P[:, j] < THRESHOLD) & (Y[:, j] == 1)).sum())
            if tp + fn:
                out[c] = round(2 * tp / max(2 * tp + fp + fn, 1), 2)
        return out

    # ---- honest self-test: hold out a quarter of the recorded hits -----------
    report = {}
    if snips and min(len(s) for s in snips.values()) >= 8:
        tr = {c: s[: len(s) - len(s) // 4] for c, s in snips.items()}
        te = {c: s[len(s) - len(s) // 4:] for c, s in snips.items()}
        Xtr, Ytr, _ = _synthesise(tr, rng, n_mix // 2)
        Xte, Yte, recorded = _synthesise(te, rng, 600)
        report = score(_predict(fit(Xtr, Ytr), Xte), Yte, recorded)
        log("      self-test on held-out hits (F1, 1.0 = perfect): " +
            ", ".join(f"{DRUMS[c]['short']} {v:.2f}" for c, v in report.items()))

    # ---- final model on everything ---------------------------------------------
    parts_X, parts_Y = [], []
    if snips:
        Xs, Ys, recorded = _synthesise(snips, rng, n_mix)
        wide = np.zeros((len(Ys), len(classes)), dtype=Ys.dtype)       # a drum only the sheets have
        wide[:, [classes.index(c) for c in recorded]] = Ys             # is in none of the mixes
        parts_X.append(Xs); parts_Y.append(wide)
    n_real = 0
    real = [(Path(audio).name, *_from_pair(audio, midi, classes)) for audio, midi in pairs]
    for audio, sheet, notes in sheets:
        Xp, Yp, found = _from_sheet(audio, notes, classes)
        log(f"      sheet {sheet_name(sheet)}: {found.summary()}")
        if not isinstance(sheet, (str, Path)) or Path(sheet).suffix.lower() in PAGE_TYPES:
            log(f"      (the tempo mark is not read from a picture, 120 is assumed: you played at about "
                f"{120 * found.tempo:.0f} quarter notes a minute)")
        real.append((Path(audio).name, Xp, Yp))
    for take, Xp, Yp in real:
        n_real += len(Xp)
        log(f"      take {take}: {len(Xp)} labelled strokes")
        # real strokes are worth more than synthetic ones
        reps = max(1, min(8, (n_mix // 4) // max(len(Xp), 1)))
        parts_X += [Xp] * reps; parts_Y += [Yp] * reps
    X, Y = np.concatenate(parts_X), np.concatenate(parts_Y)
    if len(X) < 40:
        raise AudioError("Too few strokes to learn from: the takes need to be longer, or to follow their sheets.")
    if not snips and n_real >= 80:
        # nothing but takes, so the self-test above had no single hits to hold out:
        # learn from the first three quarters of each take, test on the last quarter
        cut = [len(Xp) * 3 // 4 for _, Xp, _ in real]
        model = fit(np.concatenate([Xp[:n] for (_, Xp, _), n in zip(real, cut)]),
                    np.concatenate([Yp[:n] for (_, _, Yp), n in zip(real, cut)]))
        report = score(_predict(model, np.concatenate([Xp[n:] for (_, Xp, _), n in zip(real, cut)])),
                       np.concatenate([Yp[n:] for (_, _, Yp), n in zip(real, cut)]), classes)
        log("      self-test on the last quarter of each take (F1, 1.0 = perfect): " +
            ", ".join(f"{DRUMS[c]['short']} {v:.2f}" for c, v in report.items()))
    weak = [DRUMS[c]["label"] for c, v in report.items() if v < 0.75]
    if weak:
        log("      NOT RELIABLE YET for " + ", ".join(weak) + ": the model mixes these up even in its own "
            "recordings. Real playing teaches it more than more single hits: a take with its sheet.")
    clf = fit(X, Y)

    info = dict(name=name, classes=classes, created=datetime.datetime.now().isoformat(timespec="seconds"),
                hits={c: len(s) for c, s in snips.items()}, real_strokes=n_real,
                examples=int(len(X)), self_test=report)
    KITS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(clf=clf, info=info), KITS_DIR / f"{name}.joblib", compress=3)
    log(f"      saved {KITS_DIR / (name + '.joblib')}  ({len(X)} training examples)")
    return info


def _predict(forests: list, X: np.ndarray) -> np.ndarray:
    """Probability per class, shape [strokes, classes]."""
    cols = []
    for f in forests:
        if isinstance(f, float):
            cols.append(np.full(len(X), f))
        else:
            f.n_jobs = 1          # threads cost more than they save on a handful of strokes (live view)
            cols.append(f.predict_proba(X)[:, list(f.classes_).index(1)])
    return np.stack(cols, axis=1)


# ---------------------------------------------------------------------------
# Using a trained kit
# ---------------------------------------------------------------------------
class KitBackend(Backend):
    """Transcribe with a model trained by train_kit().  key = "kit:<name>"."""
    key = ""                                      # (set per instance; not auto-discovered)
    _cache: dict[str, tuple[float, dict]] = {}

    def __init__(self, name: str):
        self.name = name
        self.key = f"kit:{name}"
        self.label = f"My kit: {name}"
        self.description = ("Trained on your own recordings: knows your drums, room and mic, "
                            "including ride vs hi-hat. Only as good as the calibration.")

    def _kit(self) -> dict:
        path = KITS_DIR / f"{self.name}.joblib"
        stamp = path.stat().st_mtime if path.exists() else 0.0
        if self.name not in self._cache or self._cache[self.name][0] != stamp:
            self._cache[self.name] = (stamp, load_kit(self.name))
        return self._cache[self.name][1]

    def transcribe(self, y, sr, sensitivity=1.0):
        kit = self._kit()
        clf, classes = kit["clf"], kit["info"]["classes"]
        cfg = TranscribeConfig()
        if sensitivity != 1.0:
            cfg = replace(cfg, delta_low=cfg.delta_low / sensitivity,
                          delta_mid=cfg.delta_mid / sensitivity, delta_high=cfg.delta_high / sensitivity)
        pad = int(0.25 * SR)                       # so a hit at t=0 has a "before"
        y = np.concatenate([np.zeros(pad, dtype=np.float32), y.astype(np.float32)])
        strokes, _ = detect_strokes(y, cfg)
        if not strokes:
            return []
        M = _mel_frames(y)
        ref_db = _reference_db(M)
        X = np.array([_features(M, s["time"], ref_db) for s in strokes])
        P = _predict(clf, X)
        threshold = float(np.clip(THRESHOLD / sensitivity, 0.2, 0.85))
        hits = []
        for s, x, p in zip(strokes, X, P):
            names = [c for c, v in zip(classes, p) if v >= threshold]
            if not names and p.max() >= UNSURE * threshold:
                names = [classes[int(np.argmax(p))]]   # a stroke, but none is sure: the likeliest drum
            hats = [c for c in names if c.startswith("hihat")]
            if len(hats) == 2:                     # open and closed at once: keep the likelier
                worse = min(hats, key=lambda c: p[classes.index(c)])
                names.remove(worse)
            velocity = int(np.clip(110 + 2.2 * float(x[-1]), 25, 127))   # x[-1]: dB re loud strokes
            hits += [Hit(max(0.0, s["time"] - 0.25), c, velocity) for c in names]
        return drop_retriggers(hits, cfg)
