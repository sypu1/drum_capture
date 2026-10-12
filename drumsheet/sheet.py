"""A sheet and a recording of someone playing it: read the sheet, then find
each of its notes in the recording.

(A photo or a PDF of the sheet is read first by omr.py; from there on it is a
sheet like any other.)

Why
---
The best training material for a kit model is real playing with the right
labels (see kit_model.py).  A corrected MIDI gives that, but somebody has to
correct it.  A sheet you played from is already there.  It only says what is
played in which order, though, not when: the take starts somewhere, at its own
tempo, and the tempo wanders.  align() works the timing out.

How align() works
-----------------
1. The sheet is cut into pieces of a few seconds.  For every tempo between
   half and double the sheet's, each piece is slid along the recording to see
   where its notes fall on strokes.  Strokes the piece does not account for
   count against a position.
2. The best chain of positions, one per piece, each near where the piece
   before it predicts, is the alignment.  Pieces may drift, so a wandering
   tempo is followed.  Jumping elsewhere is possible but costs: a repeat that
   was skipped or played once more is followed, the rest stays in order.
3. Each piece is nudged into place exactly, then every couple of beats within
   it (people rush fills), and every note goes to a stroke near its predicted
   time.
4. In a steady groove there is a stroke on every eighth, so the sheet fits
   just as well half a beat off - unless one knows which stroke is which drum.
   To begin with only the detector's bands say that (kick: low, snare and toms:
   mid, cymbals: high), and on a dull recording they say nothing.  So the
   strokes are sorted into groups that sound alike, the first result shows
   which drums go with which group, and the sheet is lined up again knowing
   that.  A result that is half a beat off somewhere sends a drum to two
   groups, which the next round counts against it; a few rounds settle it.

Parts of the take that do not follow the sheet (improvised bars, a section the
sheet does not have) end up with few matches and are left out rather than
learned wrongly.
"""
from __future__ import annotations

import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from .audio_io import AudioError
from .config import DRUMS, MIDI_ALIASES

# ---------------------------------------------------------------------------
# Reading a sheet: [(seconds at the sheet's own tempo, drum), ...]
# ---------------------------------------------------------------------------
SHEET_TYPES = (".mid", ".midi", ".musicxml", ".xml", ".mxl")     # written out note by note
PAGE_TYPES = (".pdf", ".png", ".jpg", ".jpeg", ".heic", ".heif", ".tif", ".tiff", ".bmp", ".webp")   # pictures


def sheet_name(sheet) -> str:
    """What to call a sheet: its file, or its first page and how many follow."""
    pages = [sheet] if isinstance(sheet, (str, Path)) else list(sheet)
    return Path(pages[0]).name + (f" (+{len(pages) - 1} more)" if len(pages) > 1 else "")


def read_sheet(path, log=None) -> list[tuple[float, str]]:
    """path: one file, or the pages of a sheet in order (photos, scans, PDFs)."""
    pages = [Path(path)] if isinstance(path, (str, Path)) else [Path(p) for p in path]
    if len(pages) > 1 or pages[0].suffix.lower() in PAGE_TYPES:
        if any(p.suffix.lower() not in PAGE_TYPES for p in pages):
            raise AudioError("A sheet in several files has to be its pages as photos or PDFs: " + sheet_name(pages))
        from .omr import read_pages                # a picture of a sheet: read by omr.py
        return sorted(read_pages(pages, log))
    path = pages[0]
    ext = path.suffix.lower()
    if ext in (".mid", ".midi"):
        import pretty_midi
        notes = [(float(n.start), MIDI_ALIASES[n.pitch])
                 for inst in pretty_midi.PrettyMIDI(str(path)).instruments if inst.is_drum
                 for n in inst.notes if n.pitch in MIDI_ALIASES]
    elif ext in (".musicxml", ".xml", ".mxl"):
        notes = _read_musicxml(path)
    else:
        raise AudioError(f"Cannot read the sheet {path.name}: it has to be a photo or a PDF of the sheet, "
                         "or the sheet as MIDI (.mid) or MusicXML (.musicxml, .mxl).")
    if not notes:
        raise AudioError(f"{path.name} contains no drum notes.")
    return sorted(notes)


# staff position -> drum, for MusicXML files that do not name their instruments
_BY_POSITION = {(spec["xml"], spec["head"] == "x"): d for d, spec in DRUMS.items()}


def _xml_drum(note, pitch_of: dict) -> str | None:
    inst = note.find("instrument")
    pitch = pitch_of.get(inst.get("id")) if inst is not None else None
    if pitch in MIDI_ALIASES:
        drum = MIDI_ALIASES[pitch]
    else:
        u, head = note.find("unpitched"), note.findtext("notehead") or "normal"
        drum = _BY_POSITION.get(((u.findtext("display-step"), int(u.findtext("display-octave") or 0)),
                                 "x" in head))
        if drum == "hihat_closed" and head == "circle-x":
            drum = "hihat_open"
    tech = note.find("notations/technical")
    if drum == "hihat_closed" and tech is not None and (tech.find("open-string") is not None
                                                        or tech.find("open") is not None):
        drum = "hihat_open"                      # a closed hi-hat with the "o" sign above it
    return drum


def _xml_measure(measure, state: dict, pitch_of: dict):
    """One measure: its drum notes and tempo marks [(quarter notes from the bar
    line, ...)] and its length in quarter notes."""
    pos = last = length = 0.0
    notes, tempos = [], []
    for el in measure:
        if el.tag == "attributes":
            if el.findtext("divisions"):
                state["div"] = float(el.findtext("divisions"))
            simile = el.find("measure-style/measure-repeat")
            if simile is not None:               # "%": play the bar(s) before again
                state["simile"] = int(simile.text or 1) if simile.get("type") == "start" else 0
        elif el.tag in ("backup", "forward"):
            step = float(el.findtext("duration") or 0) / state["div"]
            pos += -step if el.tag == "backup" else step
        elif el.tag in ("direction", "sound"):
            sound = el if el.tag == "sound" else el.find("sound")
            if sound is not None and sound.get("tempo"):
                tempos.append((pos, float(sound.get("tempo"))))
        elif el.tag == "note" and el.find("grace") is None and el.find("cue") is None:
            chord = el.find("chord") is not None
            at = last if chord else pos
            if not chord:
                last = pos
                pos += float(el.findtext("duration") or 0) / state["div"]
            tied = any(t.get("type") == "stop" for t in el.findall("tie"))
            if el.find("rest") is None and el.find("unpitched") is not None and not tied:
                drum = _xml_drum(el, pitch_of)
                if drum:
                    notes.append((at, drum))
        length = max(length, pos)
    return notes, tempos, length


def _playing_order(measures) -> list[int]:
    """The measures in the order they are played: repeat signs and numbered
    endings are followed.  (D.C., D.S. and coda jumps are not.)
    measures: per measure, that measure in every part (any of them may carry the signs)."""
    forward, back, ending, span = set(), {}, {}, None
    for i, column in enumerate(measures):
        stop = False
        for bar in (b for m in column for b in m.findall("barline")):
            rep, end = bar.find("repeat"), bar.find("ending")
            if rep is not None and rep.get("direction") == "forward":
                forward.add(i)
            elif rep is not None:
                back[i] = int(rep.get("times") or 2)
            if end is not None and end.get("type") == "start":
                span = {int(x) for x in (end.get("number") or "").replace(",", " ").split() if x.isdigit()}
            elif end is not None:
                stop = True
        if span is not None:
            ending[i] = span
        if stop:
            span = None
    order, i, start, turn, done = [], 0, 0, 1, {}
    while i < len(measures) and len(order) < 50 * len(measures):
        if i in forward and i != start:
            start, turn = i, 1
        if i in ending and turn not in ending[i]:
            i += 1
            continue
        order.append(i)
        if i in back and done.get(i, 0) < back[i] - 1:
            done[i] = done.get(i, 0) + 1
            turn += 1
            i = start
            continue
        if i in back or (i in ending and ending.get(i + 1) != ending[i]):
            start, turn = i + 1, 1               # this repeat is over
        i += 1
    return order


def _read_musicxml(path: Path) -> list[tuple[float, str]]:
    if path.suffix.lower() == ".mxl":
        with zipfile.ZipFile(path) as z:
            inside = [n for n in z.namelist()
                      if not n.startswith("META-INF") and n.lower().endswith((".xml", ".musicxml"))]
            if not inside:
                raise AudioError(f"{path.name} has no score in it.")
            root = ElementTree.fromstring(z.read(inside[0]))
    else:
        root = ElementTree.parse(path).getroot()
    if root.tag != "score-partwise":
        raise AudioError(f"{path.name} is not a MusicXML score this program can read (score-partwise).")
    # midi-unpitched counts from 1 (MusicXML: 36 is the kick, MIDI note 35) - except in Audiveris's files
    one = 0 if any((s.text or "").startswith("Audiveris") for s in root.iter("software")) else 1
    pitch_of = {m.get("id"): int(m.findtext("midi-unpitched")) - one
                for m in root.iter("midi-instrument") if m.findtext("midi-unpitched")}
    parts, columns = [], []                      # per part: [(notes, tempos, length) per measure]
    for part in root.findall("part"):
        state, rows = {"div": 1.0, "simile": 0}, []
        for i, m in enumerate(part.findall("measure")):
            columns.append([m]) if i == len(columns) else columns[i].append(m)
            notes, tempos, length = _xml_measure(m, state, pitch_of)
            if state["simile"] and not notes and len(rows) >= state["simile"]:
                notes = rows[-state["simile"]][0]
            rows.append((notes, tempos, length))
        parts.append(rows)
    if not parts:
        return []
    order = _playing_order(columns)
    # walk through the measures as played: quarter-note positions, then seconds
    at, events, marks = 0.0, [], []
    for i in order:
        rows = [p[i] for p in parts if i < len(p)]
        for notes, tempos, _ in rows:
            events += [(at + q, drum) for q, drum in notes]
            marks += [(at + q, bpm) for q, bpm in tempos]
        at += max(r[2] for r in rows)
    marks.sort()
    out, t, q0, bpm, k = [], 0.0, 0.0, 120.0, 0
    for q, drum in sorted(events):
        while k < len(marks) and marks[k][0] <= q:
            t += (marks[k][0] - q0) * 60 / bpm
            q0, bpm, k = marks[k][0], marks[k][1], k + 1
        out.append((t + (q - q0) * 60 / bpm, drum))
    return out


# ---------------------------------------------------------------------------
# Lining the sheet up with the recording
# ---------------------------------------------------------------------------
RES = 50                      # positions tried per second in the search (20 ms apart)
PIECE = 3.0                   # the sheet is searched for in pieces this long (seconds of sheet):
                              # short, because each has one tempo and people change theirs within bars
DRIFT = 0.4                   # a piece may land this far from where the one before predicts (s)
WANDER = 0.10                 # how far the tempo of a piece may be from the take's overall tempo
JUMP = 2.0                    # a piece may also land anywhere at all, for the price of this many
                              # pieces' worth of notes
EXTRA = 0.5                   # what a stroke that the sheet does not have costs, in notes.  Without it
                              # a busy groove fits best at half tempo: every note lands on some stroke.
TOL = 0.06                    # a note is a stroke within this of its predicted time (s)
CLEAR = 0.10                  # a stroke this far from every note is not in the sheet at all
STRETCHES = np.exp(np.arange(np.log(0.5), np.log(2.0) + 1e-9, 0.02))   # recording time / sheet time
_MID = {"snare", "tom_high", "tom_mid", "tom_floor"}


def _band(drum: str) -> int:                                 # 0 low, 1 mid, 2 high
    return 0 if drum == "kick" else 1 if drum in _MID else 2


@dataclass
class Alignment:
    labels: list                  # per stroke: the drums the sheet says it is; an empty set = a stray
                                  # onset that is no drum at all; None = cannot tell, do not learn from it
    notes: int                    # strokes written in the sheet (notes played together count once)
    found: int                    # how many of them were found in the recording
    tempo: float                  # the playing's tempo relative to the sheet's (1.0 = as written)
    start: float                  # where the sheet's first note is in the recording (s)
    skipped: list = field(default_factory=list)   # [(from, to)] seconds of recording that do not follow the sheet

    def summary(self) -> str:
        text = (f"{self.found} of the sheet's {self.notes} strokes found, starting at {self.start:.1f} s, "
                f"played at {100 * self.tempo:.0f} % of the sheet's tempo")
        if self.skipped:
            text += "; not following the sheet: " + ", ".join(f"{a:.0f}-{b:.0f} s" for a, b in self.skipped)
        return text


def _nearest(sorted_times: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Index of the closest of sorted_times to each t."""
    j = np.clip(np.searchsorted(sorted_times, t), 1, len(sorted_times) - 1)
    return np.where(np.abs(t - sorted_times[j - 1]) <= np.abs(t - sorted_times[j]), j - 1, j)


def _sound_groups(sound: np.ndarray) -> np.ndarray:
    """Sort the strokes into groups that sound alike: a group number per stroke."""
    from scipy.cluster.vq import kmeans2

    z = (sound - sound.mean(axis=0)) / (sound.std(axis=0) + 1e-6)
    _, size, axes = np.linalg.svd(z, full_matrices=False)
    z = z @ axes[:6].T                                       # the six directions the strokes differ in most
    k = int(np.clip(len(z) // 10, 3, 12))
    best = None
    for seed in range(4):                                    # k-means depends on where it starts
        centres, group = kmeans2(z, k, minit="++", seed=seed)
        spread = float(((z - centres[group]) ** 2).sum())
        if best is None or spread < best[0]:
            best = (spread, group)
    return best[1]


def align(notes: list[tuple[float, str]], strokes: list[dict], duration: float,
          sound: np.ndarray | None = None) -> Alignment:
    """notes: read_sheet();  strokes: transcribe.detect_strokes() of the recording;
    duration: length of the recording in seconds;  sound: optionally one row of
    numbers per stroke describing how it sounds (step 4 above needs it)."""
    from scipy.fft import irfft, rfft
    from scipy.ndimage import gaussian_filter1d

    # the sheet's strokes: notes written at the same moment are one stroke (in a MIDI
    # file that was played in, "the same moment" is a few hundredths of a second wide)
    u, drums = [], []
    for t, d in notes:
        if u and t - u[-1] < 0.03:
            drums[-1].add(d)
        else:
            u.append(t)
            drums.append({d})
    u = np.array(u) - u[0]
    names = sorted({d for ds in drums for d in ds})
    has = np.array([[d in ds for d in names] for ds in drums], dtype=float)
    share = has / has.sum(axis=1, keepdims=True)             # [notes, drums], each note's drums sum to 1
    s_time = np.array([s["time"] for s in strokes])
    if len(u) < 8 or len(s_time) < 8:
        raise AudioError("Too little to line up: the sheet and the recording need at least a few bars.")
    fired = np.array([["low" in s["bands"], "mid" in s["bands"], "high" in s["bands"] or "air" in s["bands"]]
                      for s in strokes], dtype=float)
    by_band = 0.5 * fired[:, [_band(d) for d in names]]      # [strokes, drums]: the drum's band fired

    n_audio = int(duration * RES) + 1
    piece = (u // PIECE).astype(int)
    K = int(piece.max()) + 1
    # each piece reaches from its first position to its end, the last one to the sheet's last note
    length = np.minimum(PIECE, u[-1] + 0.1 - np.arange(K) * PIECE)
    reach = int(PIECE * STRETCHES[-1] * RES) + 2            # the longest a piece gets, in positions
    N = 1 << int(np.ceil(np.log2(n_audio + 2 * reach)))
    sigma = 0.05 * RES
    at = np.clip((s_time * RES).round().astype(int), 0, n_audio - 1)
    starts = np.arange(-reach, n_audio)                      # where a piece can begin
    count = np.concatenate([[0], np.cumsum(np.bincount(at, minlength=n_audio))])     # strokes before a position
    logs = np.log(STRETCHES)
    near = [np.flatnonzero(np.abs(logs - x) <= WANDER) for x in logs]

    def pulses(weight):                                      # a bump at each stroke, fading within ~50 ms
        x = np.zeros(N, dtype=np.float32)
        np.add.at(x, at, weight)
        return gaussian_filter1d(x, sigma) * sigma * np.sqrt(2 * np.pi)

    F_any = rfft(np.minimum(pulses(1.0), 1.0))

    def search(overall, evidence, kinds):
        """Line the sheet up, trying the tempos `overall` (indices of STRETCHES).
        evidence [strokes, drums]: how much each stroke looks like each drum (-1 ... 1).
        kinds: the same evidence for the search, as [(per stroke, per note)] pairs."""
        worth = 1 + share @ evidence.T                       # [notes, strokes]: a note on a stroke counts this much
        F_kind = [(rfft(pulses(per_stroke)), per_note) for per_stroke, per_note in kinds]

        def fit(stretch):
            """[pieces, starts]: what the notes of the piece that land on a stroke are
            worth when it begins there, less what the strokes it leaves unexplained cost."""
            q = (stretch * (u - piece * PIECE) * RES).round().astype(int)
            X = np.zeros((K, N), dtype=np.float32)
            np.add.at(X, (piece, q), 1.0)
            on = irfft(F_any * np.conj(rfft(X, axis=1, workers=-1)), N, axis=1, workers=-1)[:, starts % N]
            acc = 0
            for F, per_note in F_kind:                       # all kinds of evidence in one transform back
                X[:] = 0
                np.add.at(X, (piece, q), per_note)
                acc = acc + F * np.conj(rfft(X, axis=1, workers=-1))
            E = on + irfft(acc, N, axis=1, workers=-1)[:, starts % N]
            ends = starts[None, :] + (stretch * length * RES).round().astype(int)[:, None]
            inside = count[np.clip(ends, 0, n_audio)] - count[np.clip(starts, 0, n_audio)][None, :]
            return E - EXTRA * np.maximum(inside - on, 0)

        def chain(E, stretch):
            """The best chain of piece positions: where each piece begins (s)."""
            before = int(stretch * u[-1] * RES) + 1         # the sheet may begin before the recording
            offs = np.arange(-before, n_audio)              # where the sheet's zero would land
            step = (stretch * PIECE * RES * np.arange(K)).round().astype(int)
            pos = offs[None, :] + step[:, None]
            D = np.where((pos >= -reach) & (pos < n_audio),
                         E[np.arange(K)[:, None], np.clip(pos + reach, 0, len(starts) - 1)], 0.0)
            J, price = int(DRIFT * RES), JUMP * len(u) / K
            back = np.zeros((K, len(offs)), dtype=np.int8)   # step taken to get here; 127 = a jump
            came_from = np.zeros(K, dtype=int)               # ... which came from this position
            best = D[0].copy()
            for k in range(1, K):
                prev = np.pad(best, J, constant_values=-np.inf)
                cand = np.stack([prev[J + d:J + d + len(offs)] - 0.02 * abs(d) for d in range(-J, J + 1)])
                close = cand.max(axis=0)
                came_from[k] = int(best.argmax())
                jumped = close < best.max() - price
                back[k] = np.where(jumped, 127, cand.argmax(axis=0) - J)
                best = D[k] + np.where(jumped, best.max() - price, close)
            j = int(best.argmax())
            path = [j]
            for k in range(K - 1, 0, -1):
                j = int(came_from[k]) if back[k, j] == 127 else j + int(back[k, j])
                path.append(j)
            return (offs[np.array(path[::-1])] + step) / RES

        # A take has one overall tempo, but each piece is played a little faster or
        # slower.  So for an overall tempo, a piece's fit at a position is the best
        # over the tempos near it.  With several overall tempos to choose from: first
        # a quick look at each (how well do its pieces fit anywhere?), then the most
        # promising ones properly.
        if len(overall) > 1:
            peak = np.array([fit(x).max(axis=1) for x in STRETCHES])
            rough = np.array([peak[near[i]].max(axis=0).sum() for i in overall])
            ranked, overall = [overall[i] for i in np.argsort(rough)[::-1]], []
            for i in ranked:
                if len(overall) < 4 and all(abs(logs[i] - logs[j]) >= 0.03 for j in overall):
                    overall.append(i)
        best = None
        for i in overall:
            E = fit(STRETCHES[near[i][0]])
            for j in near[i][1:]:
                np.maximum(E, fit(STRETCHES[j]), out=E)
            result = _settle(float(STRETCHES[i]), chain(E, float(STRETCHES[i])), u, piece, K, length,
                             s_time, worth)
            if best is None or result[0] > best[0]:
                best = result + (i,)
        return best

    def label(p, good):
        """3. Every stroke gets the drums of the note nearest to it, if one is near.
        (So the click of a pedal just before its drum, which the detector takes
        for a stroke of its own, is that drum too - not a second, unnamed thing.)"""
        labels: list = [None] * len(s_time)
        mine = np.flatnonzero(good[piece])
        if not len(mine):
            return labels, 0
        order = mine[np.argsort(p[mine])]
        i = order[_nearest(p[order], s_time)] if len(order) > 1 else np.full(len(s_time), order[0])
        close = np.abs(p[i] - s_time) <= TOL
        for j in np.flatnonzero(close):
            labels[j] = set(drums[i[j]])
        return labels, len(set(i[close]))

    bands = [(0.5 * fired[:, b], share[:, [_band(d) == b for d in names]].sum(axis=1)) for b in range(3)]
    score, p, good, rate, spans, tempo = search(list(range(len(STRETCHES))), by_band, bands)
    labels, found = label(p, good)

    # ---- 4. learn what each drum sounds like here, and line up again ----------------
    if sound is not None and len(sound) == len(s_time):
        group = _sound_groups(np.asarray(sound, dtype=float))
        for _ in range(4):
            known = np.array([l is not None for l in labels])
            if known.sum() < 20:
                break
            is_drum = np.array([[d in (l or ()) for d in names] for l in labels], dtype=float)
            usual = is_drum[known].mean(axis=0)              # how often each drum is played at all
            learned = np.zeros_like(by_band)
            for g in np.unique(group):
                mine = known & (group == g)                  # (2 = a little doubt for small groups)
                p_drum = (is_drum[mine].sum(axis=0) + 2 * usual) / (mine.sum() + 2)
                learned[group == g] = np.clip(np.log((p_drum + 1e-9) / (usual + 1e-9)) / np.log(4), -1, 1)
            evidence = by_band + learned
            result = search([tempo], evidence, bands + [(learned[:, d], share[:, d]) for d in range(len(names))])
            new_labels, new_found = label(result[1], result[2])
            if not result[2].any():
                break
            unchanged = new_labels == labels
            (score, p, good, rate, spans, tempo), labels, found = result, new_labels, new_found
            if unchanged:
                break

    if found < len(u) / 3:
        raise AudioError(f"Could not find the sheet in the recording: only {found} of its {len(u)} strokes "
                         "fit anywhere. Is it the same piece, with the drum part in the file?")
    # A stroke far from every note, inside a stretch that follows the sheet: nothing was hit there.
    ks = np.flatnonzero(good)
    written = np.sort(p[good[piece]])
    for j, t in enumerate(s_time):
        if (labels[j] is None and abs(written[_nearest(written, np.array([t]))[0]] - t) > CLEAR
                and any(spans[k, 0] <= t < spans[k, 1] for k in ks)):
            labels[j] = set()
    # what the recording has between the first and the last piece that does not follow the sheet
    covered, gaps = sorted(map(tuple, spans[ks])), []
    for (_, b), (a, _) in zip(covered, covered[1:]):
        if a - b > 1.0:
            gaps.append((float(b), float(a)))
    return Alignment(labels, len(u), found, float(1 / np.median(rate[ks])),
                     float(spans[ks[0], 0] - rate[ks[0]] * ks[0] * PIECE), gaps)


def _settle(stretch, begin, u, piece, K, length, s_time, worth):
    """Nudge each piece into place exactly, each at its own tempo, then every
    second of sheet within it.  Returns (score, each note's time in the recording,
    which pieces follow the sheet, each piece's recording seconds per sheet second,
    where each piece lies in the recording [from, to])."""
    def value(notes, t):
        """What notes at times t [..., notes] are worth together: each counts if a stroke is there."""
        j = _nearest(s_time, t)
        return (np.exp(-0.5 * ((t - s_time[j]) / 0.02) ** 2) * worth[notes, j]).sum(axis=-1)

    jumps = int((np.abs(np.diff(begin) - stretch * PIECE) > DRIFT + 1.5 / RES).sum())
    begin, rate = begin.astype(float).copy(), np.full(K, stretch)
    shifts = np.arange(-0.06, 0.0601, 0.004)
    rates = stretch * np.exp(np.arange(-WANDER - 0.015, WANDER + 0.0151, 0.005))
    for k in range(K):
        mine = np.flatnonzero(piece == k)
        if len(mine) < 4:
            continue
        v = value(mine, begin[k] + shifts[None, :, None] + rates[:, None, None] * (u[mine] - k * PIECE))
        r, d = np.unravel_index(v.argmax(), v.shape)
        begin[k], rate[k] = begin[k] + shifts[d], rates[r]
    p = begin[piece] + rate[piece] * (u - piece * PIECE)
    # ... then the three seconds around every second of the sheet: a piece has one tempo, people do not
    shifts, rates = np.arange(-0.12, 0.1201, 0.004), np.exp(np.arange(-0.10, 0.1001, 0.01))
    cost = 0.5 * np.abs(shifts)[None, :] + np.abs(np.log(rates))[:, None]      # (when in doubt, leave it)
    fine = p.copy()
    for c in np.arange(0.5, u[-1] + 1, 1.0):
        k = min(int(c // PIECE), K - 1)
        mine = np.flatnonzero((piece == k) & (np.abs(u - c) <= 1.5))
        if len(mine) < 3:
            continue
        here = begin[k] + rate[k] * (c - k * PIECE)
        v = value(mine, here + shifts[None, :, None] + rate[k] * rates[:, None, None] * (u[mine] - c)) - cost
        r, d = np.unravel_index(v.argmax(), v.shape)
        centre = mine[np.abs(u[mine] - c) <= 0.5]
        fine[centre] = here + shifts[d] + rate[k] * rates[r] * (u[centre] - c)
    p = fine
    spans = np.stack([begin, begin + rate * length], axis=1)
    j = _nearest(s_time, p)
    on = np.abs(s_time[j] - p) <= 0.035                      # notes that sit on a stroke
    order = np.sort(p)
    told = np.abs(order[_nearest(order, s_time)] - s_time) <= 0.035            # strokes that have a note
    good, score = np.zeros(K, dtype=bool), 0.0
    for k in range(K):
        sel = piece == k
        mine = (s_time >= spans[k, 0]) & (s_time < spans[k, 1])
        if sel.sum() < 4 or not mine.any():
            continue
        # a piece follows the sheet if most of its notes are played and most of what is played is in it
        good[k] = on[sel].mean() >= 0.6 and told[mine].mean() >= 0.5
        score += (on[sel] * worth[np.flatnonzero(sel), j[sel]]).sum() - EXTRA * (~told[mine]).sum()
    # (a chain that keeps jumping fits a little anywhere: the jumps cost here as they did in the search)
    return score - jumps * JUMP * len(u) / K, p, good, rate, spans
