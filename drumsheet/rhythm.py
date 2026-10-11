"""Tempo, beats, bar lines and quantisation.

beats      - tracked with librosa, so the grid follows gentle tempo drift.
bar lines  - librosa does not find downbeats, so we use what drummers do:
             kick/crash tend to sit on beat 1, snare on the backbeats.
time sig.  - 4/4 unless the kit pattern clearly repeats every 3 beats.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .transcribe import Hit


@dataclass
class Score:
    """The quantised result.  bars[i][slot] = {drum: velocity}."""
    bpm: float
    beats_per_bar: int
    sub: int                                   # grid slots per beat
    bars: list[dict[int, dict[str, int]]] = field(default_factory=list)
    bar_times: list[float] = field(default_factory=list)   # start time of each bar (s)
    title: str = "Drum transcription"
    notes: list[str] = field(default_factory=list)          # processing notes for the footer

    @property
    def slots_per_bar(self) -> int:
        return self.beats_per_bar * self.sub

    def used_drums(self) -> set[str]:
        return {d for bar in self.bars for slot in bar.values() for d in slot}

    def n_notes(self) -> int:
        return sum(len(slot) for bar in self.bars for slot in bar.values())


# ---------------------------------------------------------------------------
# Beat tracking
# ---------------------------------------------------------------------------
def _common_gap(times: np.ndarray, lo: float, hi: float) -> float | None:
    """The most common spacing between consecutive times (within lo..hi seconds)."""
    gaps = np.diff(np.sort(times))
    gaps = gaps[(gaps >= lo) & (gaps <= hi)]
    if len(gaps) < 6:
        return None
    support = [(np.abs(np.log(gaps / g)) < 0.10).sum() for g in gaps]
    centre = gaps[int(np.argmax(support))]
    return float(np.median(gaps[np.abs(np.log(gaps / centre)) < 0.10]))


def refine_beat_period(period: float, hits: list[Hit], onset_env: np.ndarray, fps: float) -> float:
    """Second opinion on the beat tracker's tempo, using what the drums play.

    Beat trackers happily lock onto 3 hi-hat strokes (a dotted rhythm) or onto
    half/double tempo.  Two musical facts fix most of that:
      * the beat is 2, 4 or 8 strokes of the fastest steady pulse (usually the
        hi-hat) - unless the music really is in triplets;
      * the snare backbeat comes every two beats.
    Returns the (possibly corrected) beat period in seconds.
    """
    if len(hits) < 16:
        return period
    stroke_times = np.unique(np.round([h.time for h in hits], 2))
    pulse = _common_gap(stroke_times, 0.07, 0.7)
    if pulse is None:
        return period

    def close(x, values):
        return any(abs(np.log(x / v)) < 0.08 for v in values)

    ratio = period / pulse
    straight, triple = close(ratio, (1, 2, 4, 8)), close(ratio, (1.5, 3, 6))
    if not (straight or triple):
        return period                              # no clear relation: leave it alone
    if triple:
        # Is the music really in threes?  Compare how strongly the rhythm
        # repeats after 3 and 6 pulses with 2 and 4 pulses.
        env = onset_env - onset_env.mean()
        ac = np.correlate(env, env, mode="full")[len(env) - 1:]

        def strength(n_pulses):
            i0, i1 = int(n_pulses * pulse * 0.94 * fps), int(n_pulses * pulse * 1.06 * fps) + 1
            return float(ac[i0:i1].max()) if i1 < len(ac) else 0.0
        if strength(3) + strength(6) >= strength(2) + strength(4):
            return period                          # genuine triplet / shuffle feel

    candidates = [pulse * m for m in (1, 2, 4, 8) if 55 <= 60 / (pulse * m) <= 185]
    if not candidates:
        return period
    snares = np.array([h.time for h in hits if h.drum == "snare"])
    backbeat = _common_gap(snares, 0.5, 4.0) if len(snares) >= 6 else None
    if backbeat:
        for c in candidates:
            if abs(np.log(c / (backbeat / 2))) < 0.08:
                return c
    if straight:
        return period
    return min(candidates, key=lambda c: abs(np.log(60 / c / 100)))


def track_beats(onset_env: np.ndarray, fps: float, duration: float,
                bpm: float | None = None, hits: list[Hit] | None = None) -> tuple[float, np.ndarray]:
    """Return (bpm, beat_times) with beats covering the whole recording."""
    import librosa

    hop = 512
    sr = fps * hop                              # only sr/hop (= frames per second) matters
    kw = dict(onset_envelope=onset_env, sr=sr, hop_length=hop, units="frames")

    def run(forced_bpm=None):
        if forced_bpm:
            _, frames = librosa.beat.beat_track(bpm=forced_bpm, tightness=100, **kw)
        else:
            _, frames = librosa.beat.beat_track(start_bpm=110, tightness=100, **kw)
        return np.asarray(frames, dtype=float) / fps

    beats = run(bpm)
    if len(beats) < 4:                          # too little rhythm to track: fixed grid
        period = 60.0 / (bpm or 120.0)
        return bpm or 120.0, np.arange(0, duration + period, period)

    period = float(np.median(np.diff(beats)))
    if not bpm:
        # Tempo trackers often land on half/double tempo; fold into 70-170 BPM.
        while 60 / period < 70:
            beats = np.sort(np.concatenate([beats, (beats[:-1] + beats[1:]) / 2]))
            period /= 2
        while 60 / period > 170:
            beats = beats[::2]
            period *= 2
        better = refine_beat_period(period, hits or [], onset_env, fps)
        if abs(np.log(better / period)) > 0.04:
            print(f"      tempo corrected from {60 / period:.0f} to {60 / better:.0f} BPM "
                  "(hi-hat pulse / snare backbeat)")
            retry = run(60 / better)
            if len(retry) >= 4:
                beats, period = retry, float(np.median(np.diff(retry)))
    # Extend the grid to the start and end of the audio.
    first = beats[0] - period * np.arange(int(np.ceil(beats[0] / period)) + 1, 0, -1)
    last = beats[-1] + period * np.arange(1, int(np.ceil((duration - beats[-1]) / period)) + 2)
    return 60.0 / period, np.concatenate([first, beats, last])


def _beat_position(beats: np.ndarray, t: float) -> float:
    """Continuous position of time t on the beat grid (3.5 = half-way between beats 3 and 4)."""
    i = int(np.clip(np.searchsorted(beats, t, side="right") - 1, 0, len(beats) - 2))
    return i + (t - beats[i]) / (beats[i + 1] - beats[i])


# ---------------------------------------------------------------------------
# Bar lines / time signature
# ---------------------------------------------------------------------------
def _beat_features(hits: list[Hit], beats: np.ndarray) -> np.ndarray:
    """Per beat: [kick, snare, crash] strength of the hits that land ON the beat."""
    feat = np.zeros((len(beats), 3))
    col = {"kick": 0, "snare": 1, "crash": 2}
    for h in hits:
        if h.drum in col:
            pos = _beat_position(beats, h.time)
            if abs(pos - round(pos)) < 0.2 and 0 <= round(pos) < len(beats):
                feat[int(round(pos)), col[h.drum]] += h.velocity / 100
    return feat


def fix_offbeat_grid(hits: list[Hit], beats: np.ndarray) -> np.ndarray:
    """Beat trackers sometimes lock onto the "and"s.  If kick+snare mostly fall
    half-way between beats, shift the grid by half a beat."""
    on = off = 0.0
    for h in hits:
        if h.drum in ("kick", "snare"):
            frac = _beat_position(beats, h.time) % 1.0
            if frac < 0.2 or frac > 0.8:
                on += h.velocity
            elif 0.3 < frac < 0.7:
                off += h.velocity
    if off > 1.5 * on:
        print("[rhythm] Beat grid sat on the off-beats - shifted by half a beat.")
        return (beats[:-1] + beats[1:]) / 2
    return beats


def guess_beats_per_bar(feat: np.ndarray) -> int:
    """3 if the kick/snare pattern clearly repeats every 3 beats, otherwise 4."""
    def self_similarity(lag):
        a, b = feat[:-lag], feat[lag:]
        if len(a) < 2 * lag:
            return 0.0
        return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    s3, s4, s2 = self_similarity(3), self_similarity(4), self_similarity(2)
    return 3 if s3 > max(s4, s2) + 0.15 else 4


def find_downbeat(feat: np.ndarray, n: int) -> int:
    """Which beat (0..n-1) is most plausibly "1"."""
    best, best_score = 0, -1.0
    for phase in range(n):
        idx = np.arange(len(feat))
        pos = (idx - phase) % n
        kick, snare, crash = feat[:, 0], feat[:, 1], feat[:, 2]
        score = kick[pos == 0].sum() + 1.5 * crash[pos == 0].sum() - 0.5 * snare[pos == 0].sum()
        if n == 4:
            score += 0.5 * kick[pos == 2].sum() + snare[pos == 1].sum() + snare[pos == 3].sum()
        if score > best_score:
            best, best_score = phase, score
    return best


# ---------------------------------------------------------------------------
# Quantisation
# ---------------------------------------------------------------------------
def build_score(hits: list[Hit], onset_env: np.ndarray, fps: float, duration: float,
                sub: int = 4, bpm: float | None = None,
                beats_per_bar: int | None = None) -> Score:
    """Snap hits to the grid and arrange them into bars."""
    tempo, beats = track_beats(onset_env, fps, duration, bpm, hits)
    beats = fix_offbeat_grid(hits, beats)
    feat = _beat_features(hits, beats)
    n = beats_per_bar or guess_beats_per_bar(feat)
    phase = find_downbeat(feat, n)

    score = Score(bpm=round(tempo, 1), beats_per_bar=n, sub=sub)
    if not hits:
        return score

    # Slot index of each hit, counted from beats[0].
    slots = [int(round(_beat_position(beats, h.time) * sub)) for h in hits]
    # Bar 1 starts on the last downbeat at or before the first hit.
    first_beat = min(slots) // sub
    start_beat = first_beat - ((first_beat - phase) % n)
    spb = n * sub
    for h, s in zip(hits, slots):
        rel = s - start_beat * sub
        bar, slot = divmod(rel, spb)
        while len(score.bars) <= bar:
            score.bars.append({})
        cell = score.bars[bar].setdefault(slot, {})
        cell[h.drum] = max(cell.get(h.drum, 0), h.velocity)

    # A hi-hat can't be open and closed at once.
    for bar in score.bars:
        for cell in bar.values():
            if "hihat_open" in cell:
                cell.pop("hihat_closed", None)

    def beat_time(k: int) -> float:               # works a little past either end
        k = int(np.clip(k, 0, len(beats) - 1))
        return float(beats[k])
    score.bar_times = [beat_time(start_beat + i * n) for i in range(len(score.bars))]
    return score


def slot_time(score: Score, bar: int, slot: int) -> float:
    """Time in seconds of a grid slot on a perfectly steady grid at score.bpm.
    Used for the MIDI export, which should play back in strict time."""
    return (bar * score.slots_per_bar + slot) * 60.0 / score.bpm / score.sub


# ---------------------------------------------------------------------------
# Notation helper shared by the PDF renderer and the MusicXML writer
# ---------------------------------------------------------------------------
DIV = 12                                          # duration units per quarter note
# Note values we allow, in DIV units -> (type, dots, is_triplet)
STRAIGHT = {12: ("quarter", 0, False), 9: ("eighth", 1, False),
            6: ("eighth", 0, False), 3: ("16th", 0, False)}
TRIPLET = {12: ("quarter", 0, False), 8: ("quarter", 0, True), 6: ("eighth", 0, False),
           4: ("eighth", 0, True), 2: ("16th", 0, True)}


def beat_events(score: Score, bar: int, beat: int) -> list[tuple[int, dict | None, int]]:
    """Split one beat into notes and rests that exactly fill it.

    Returns [(offset, drums_or_None, duration)], offsets/durations in DIV units.
    A note lasts until the next note in the beat (or the end of the beat).
    """
    table = TRIPLET if score.sub in (3, 6) else STRAIGHT
    unit = DIV // score.sub
    cells = score.bars[bar]
    onsets = [(s * unit, cells[beat * score.sub + s]) for s in range(score.sub)
              if cells.get(beat * score.sub + s)]
    events, pos = [], 0

    def fill(start, length, drums):
        # largest allowed value first; anything left over becomes rests
        while length > 0:
            dur = max(d for d in table if d <= length)
            events.append((start, drums, dur))
            start, length, drums = start + dur, length - dur, None

    for i, (off, drums) in enumerate(onsets):
        if off > pos:
            fill(pos, off - pos, None)
        end = onsets[i + 1][0] if i + 1 < len(onsets) else DIV
        fill(off, end - off, drums)
        pos = end
    if pos < DIV:
        fill(pos, DIV - pos, None)
    return events
