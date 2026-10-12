"""Reading a drum sheet from a photo, a scan or a PDF.

The symbols on the page are found by Audiveris (https://github.com/Audiveris/audiveris,
a separate program, AGPL).  This module does what comes before and after:

Before: a phone photo is tilted, seen at an angle and unevenly lit, and
Audiveris wants a flat scan.  straighten() finds the direction of the staff
lines and of the stems all over the picture, works out the two vanishing
points from them, and redraws the page as seen from straight above, evenly
lit, at the size Audiveris likes.

After: Audiveris is asked for what it SAW (note heads with their place on the
staff, stems, beams, flags, dots, rests, bar lines), not for the score it
makes of it.  Its score needs the clef and every rest to be right, and a drum
part gives it trouble there.  What the page says is worked out here instead,
with what holds for drum notation:
  * the place on the staff and the shape of the head (x or oval) say which
    drum it is - no clef needed;
  * stems up are one voice, stems down the other;
  * notes that sound together are printed exactly above each other.  So when
    a rest was missed in one voice the other voice still says where its notes
    belong, and a bar is always as long as the time signature says.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from .audio_io import AudioError
from .config import DRUMS

CACHE = Path(os.environ.get("DRUMSHEET_CACHE", Path.home() / ".cache" / "drumsheet")) / "omr"
VERSION = "1"                 # of what is kept in the cache (the pictures as given to Audiveris)

WORK = 1600                   # straighten() studies a copy with its long side this many pixels
INTERLINE = 22                # distance between staff lines the result is scaled to (pixels)
TICKS = 48                    # per quarter note
BEAM = 80                     # timings of a bar followed at once (see _place)
STEPS = (6, 8, 12, 16, 24, 36, 48)    # ... each tried this far after the note before, in case no voice
                                      # says where the next one belongs (32nd ... quarter, triplets too)


# ---------------------------------------------------------------------------
# A photo -> a flat page
# ---------------------------------------------------------------------------
def _even(gray: np.ndarray) -> np.ndarray:
    """Paper 1, ink 0, whatever the lighting: each spot relative to the paper around it."""
    from scipy import ndimage
    h, w = gray.shape
    k = max(8, max(h, w) // 200)
    paper = ndimage.gaussian_filter(ndimage.maximum_filter(gray[::k, ::k], 7), 4)
    paper = ndimage.zoom(paper, (h / paper.shape[0], w / paper.shape[1]), order=1)
    paper = np.pad(paper, ((0, max(0, h - paper.shape[0])), (0, max(0, w - paper.shape[1]))), mode="edge")[:h, :w]
    return np.clip(gray / np.maximum(paper, 1e-3), 0, 1)


def _lean(mask: np.ndarray, size: int) -> np.ndarray:
    """The slope of the thin lines in each window of the picture: [(x, y, slope, how clear), ...].
    The slope is the one along which the ink piles up most when looked along it."""
    slopes = np.tan(np.radians(np.arange(-15, 15.01, 0.5)))

    def piling(u, v, tries):
        out = []
        for s in tries:
            p = v - s * u
            hist = np.bincount(np.round(p - p.min()).astype(np.int64)).astype(np.float64)
            out.append(float((hist ** 2).sum()))
        return np.array(out)

    h, w = mask.shape
    out = []
    for y0 in range(0, max(h - size, 0) + 1, size // 2):
        for x0 in range(0, max(w - size, 0) + 1, size // 2):
            ys, xs = np.nonzero(mask[y0:y0 + size, x0:x0 + size])
            if len(xs) < 150:
                continue
            u, v = xs - size / 2.0, ys - size / 2.0
            coarse = piling(u, v, slopes)
            near = slopes[int(np.argmax(coarse))] + np.linspace(-1, 1, 21) * (slopes[1] - slopes[0])
            fine = piling(u, v, near)
            j = int(np.argmax(fine))
            out.append((x0 + size / 2.0, y0 + size / 2.0, near[j], fine[j] / np.median(coarse)))
    return np.array(out).reshape(-1, 4)


def _vanishing(x, y, slope, weight) -> tuple[float, float, int]:
    """Lines that are parallel on the page meet in one point in a photo: (1, b, c) in
    homogeneous coordinates, slope = (b - c*y) / (1 - c*x).  Least squares for b and c,
    leaving out the windows that disagree.  Returns b, c and how many windows agree."""
    keep = np.ones(len(slope), bool)
    b = c = 0.0
    for _ in range(6):
        A = np.stack([np.ones(keep.sum()), (x * slope - y)[keep]], 1) * weight[keep, None]
        (b, c), *_ = np.linalg.lstsq(A, slope[keep] * weight[keep], rcond=None)
        off = np.abs(slope - (b - c * y) / (1 - c * x))
        agree = off < max(3 * np.median(off[keep]), 0.004)
        if agree.sum() < 4 or (agree == keep).all():
            break
        keep = agree
    return float(b), float(c), int(keep.sum())


def _spacing(ink: np.ndarray) -> float:
    """Distance between the staff lines of a straight page (pixels): the period of its rows of ink."""
    from scipy import ndimage
    rows = ink.sum(1).astype(np.float64)
    rows -= ndimage.uniform_filter1d(rows, 61)
    ac = np.correlate(rows, rows, "full")[len(rows) - 1:]
    hi = min(80, len(ac) - 2)
    if hi <= 6 or ac[0] <= 0:
        return 0.0
    most = float(ac[5:hi].max())
    if most < 0.15 * ac[0]:
        return 0.0                                 # no staff lines to speak of
    # the first peak that is about as high as the highest (twice the spacing fits nearly as well)
    k = next(k for k in range(5, hi) if ac[k] >= 0.7 * most and ac[k] >= ac[k - 1] and ac[k] >= ac[k + 1])
    a, b, c = ac[k - 1], ac[k], ac[k + 1]          # the top of the peak, between the pixels
    return k + 0.5 * (a - c) / (a - 2 * b + c) if a - 2 * b + c else float(k)


def straighten(path: str | Path, out: str | Path) -> str:
    """Photo or scan of a sheet -> a flat, upright, evenly lit picture of it at `out`.
    Returns a few words on what was done."""
    from PIL import Image, ImageOps
    from scipy import ndimage
    path = Path(path)
    try:
        if path.suffix.lower() in (".heic", ".heif"):
            try:
                import pillow_heif
                pillow_heif.register_heif_opener()
            except ImportError:
                raise AudioError(f"{path.name} is a HEIC photo, which needs one more package: pip install "
                                 "pillow-heif. Or send the photo as JPEG.") from None
        im = ImageOps.exif_transpose(Image.open(path)).convert("L")      # phones store "turn me" in EXIF
    except AudioError:
        raise
    except Exception as exc:
        raise AudioError(f"Cannot open the picture {path.name}: {exc}") from None
    W, H = im.size
    f = min(1.0, WORK / max(W, H))
    small = im.resize((max(1, round(W * f)), max(1, round(H * f))), Image.BILINEAR)
    ink = _even(np.asarray(small, np.float32) / 255) < 0.72
    h, w = ink.shape
    # thin strokes only: staff lines say where "level" is, stems and bar lines where "upright" is
    level = ink & ~ndimage.binary_opening(ink, np.ones((4, 1), bool))
    upright = ink & ~ndimage.binary_opening(ink, np.ones((1, 4), bool))
    size = max(96, min(h, w) // 5)
    cx, cy, n = w / 2.0, h / 2.0, max(w, h) / 2.0
    b = c = d = e = 0.0
    said = "as it is"
    fl = _lean(level, size)
    fl = fl[fl[:, 3] > 1.5] if len(fl) else fl
    if len(fl) >= 4:
        b, c, agree = _vanishing((fl[:, 0] - cx) / n, (fl[:, 1] - cy) / n, fl[:, 2], np.sqrt(fl[:, 3]))
        d = -b                                     # (if the stems say nothing: upright is square to level)
        fu = _lean(upright.T, size)                # turned: there x is y and y is x
        fu = fu[fu[:, 3] > 1.3] if len(fu) else fu
        if len(fu) >= 6:
            d, e, _ = _vanishing((fu[:, 0] - cy) / n, (fu[:, 1] - cx) / n, fu[:, 2], np.sqrt(fu[:, 3]))
        if agree < 6:
            c = e = 0.0                            # too little to tell a slant from a turn
        said = f"turned {np.degrees(np.arctan(-b)):+.1f} deg" + (", slant taken out" if abs(c) + abs(e) > 0.02 else "")
    # the two vanishing points go to infinity, then "level" and "upright" onto the axes
    v1, v2 = np.array([1.0, b, c]), np.array([d, 1.0, e])
    line = np.cross(v1, v2)
    P = np.array([[1, 0, 0], [0, 1, 0], [line[0] / line[2], line[1] / line[2], 1.0]])
    U = np.array([[1.0, d], [b, 1.0]])
    M = np.eye(3)
    M[:2, :2] = np.linalg.inv(U / np.linalg.norm(U, axis=0))
    to_unit = np.array([[f / n, 0, -cx / n], [0, f / n, -cy / n], [0, 0, 1.0]])    # full-size pixels -> around 0
    T = M @ P @ to_unit
    q = T @ np.array([[0, 0, 1], [W, 0, 1], [W, H, 1], [0, H, 1.0]]).T
    q = q[:2] / q[2]
    if not np.all(np.isfinite(q)) or (q[0].max() - q[0].min()) * (q[1].max() - q[1].min()) > 12:
        T, said = to_unit, "as it is"              # a wild guess: better not to bend the page
        q = T @ np.array([[0, 0, 1], [W, 0, 1], [W, H, 1], [0, H, 1.0]]).T
        q = q[:2] / q[2]
    x0, y0, span = q[0].min(), q[1].min(), max(q[0].max() - q[0].min(), q[1].max() - q[1].min())

    def redraw(scale, box=None):                   # pixels per unit of the straightened page; box: a part of it
        left, top, right, bottom = box or (0, 0, (q[0].max() - x0) * scale, (q[1].max() - y0) * scale)
        S = np.array([[scale, 0, -x0 * scale - left], [0, scale, -y0 * scale - top], [0, 0, 1.0]])
        back = np.linalg.inv(S @ T)
        return im.transform((int(np.ceil(right - left)), int(np.ceil(bottom - top))), Image.PERSPECTIVE,
                            tuple((back / back[2, 2]).reshape(9)[:8]), Image.BICUBIC, fillcolor=255)

    rough = WORK / span
    seen = np.asarray(redraw(rough), np.float32) / 255
    # The desk the sheet lies on: where even the brightest spot nearby is far darker than the paper would
    # be there (the paper's brightness may fall off across the page; a plane through it allows for that).
    # Left in, its grain becomes thousands of specks of "ink" once the lighting is evened out.
    cell = 4
    top = ndimage.maximum_filter(seen[::cell, ::cell], 9)
    yy, xx = np.mgrid[0:top.shape[0], 0:top.shape[1]]
    A = np.stack([np.ones(top.size), xx.ravel() / top.shape[1], yy.ravel() / top.shape[0]], 1)
    paper = top.ravel() > np.percentile(top, 50)
    for _ in range(3):
        plane = A @ np.linalg.lstsq(A[paper], top.ravel()[paper], rcond=None)[0]
        paper = np.abs(top.ravel() / np.maximum(plane, 1e-3) - 1) < 0.2
        if paper.sum() < 100:
            break
    desk = ndimage.binary_dilation((top.ravel() < 0.62 * plane).reshape(top.shape), iterations=6)
    desk = np.kron(desk, np.ones((cell, cell), bool))
    desk = np.pad(desk, ((0, max(0, seen.shape[0] - desk.shape[0])), (0, max(0, seen.shape[1] - desk.shape[1]))),
                  mode="edge")[:seen.shape[0], :seen.shape[1]]
    ink = (_even(seen) < 0.72) & ~desk
    gap = _spacing(ink)
    # Where the staves are: thin lines running level for a long way, with company one and two lines
    # further up or down.  Only that part of the picture is kept, with room for what stands above and
    # below the staves.
    thin = ink & ~ndimage.binary_opening(ink, np.ones((max(3, int(0.6 * gap)), 1), bool))
    joined = ndimage.binary_closing(ndimage.binary_dilation(thin, np.ones((3, 1), bool)),
                                    np.ones((1, max(5, int(2.5 * gap))), bool))       # (across the stems)
    long = ndimage.binary_opening(joined, np.ones((1, max(20, int(12 * gap))), bool)) if gap else thin[:0]
    if long.any():
        near = ndimage.binary_dilation(long, np.ones((3, 1), bool))

        def moved(by):
            out = np.zeros_like(near)
            if by > 0:
                out[by:] = near[:-by]
            else:
                out[:by] = near[-by:]
            return out
        one, two = int(round(gap)), int(round(2 * gap))
        long &= (moved(one) | moved(-one)) & (moved(two) | moved(-two))
    if not long.any():
        raise AudioError(f"No staff lines found in {path.name}. The whole sheet has to be in the picture, "
                         "sharp, and the right way up.")
    ys, xs = np.nonzero(long)
    scale = min(rough * INTERLINE / gap, 7000 / span)
    box = [int(max(0, xs.min() - 3 * gap)), int(max(0, ys.min() - 7 * gap)),
           int(min(ink.shape[1], xs.max() + 3 * gap)), int(min(ink.shape[0], ys.max() + 7 * gap))]
    a = np.asarray(redraw(scale, [v * scale / rough for v in box]), np.float32) / 255
    g = np.clip((_even(a) - 0.3) / 0.55, 0, 1)
    part = desk[box[1]:box[3], box[0]:box[2]].astype(np.uint8)
    part = ndimage.zoom(part, (a.shape[0] / part.shape[0], a.shape[1] / part.shape[1]), order=0)
    part = np.pad(part, ((0, max(0, a.shape[0] - part.shape[0])), (0, max(0, a.shape[1] - part.shape[1]))),
                  mode="edge")[:a.shape[0], :a.shape[1]]
    g[part > 0] = 1.0
    Image.fromarray((g * 255).astype(np.uint8)).save(out)
    return said


# ---------------------------------------------------------------------------
# Audiveris
# ---------------------------------------------------------------------------
def find_audiveris() -> str:
    names = [os.environ.get("AUDIVERIS"), shutil.which("Audiveris"), shutil.which("audiveris"),
             str(Path.home() / ".local/opt/audiveris/bin/Audiveris"), "/opt/audiveris/bin/Audiveris",
             "/Applications/Audiveris.app/Contents/MacOS/Audiveris"]
    for name in names:
        if name and Path(name).is_file():
            return name
    raise AudioError("Reading a sheet from a photo or a PDF needs the program Audiveris, which is not installed. "
                     "Get it from https://github.com/Audiveris/audiveris/releases (or set AUDIVERIS to where it "
                     "is). Sheets as MIDI or MusicXML are read without it.")


def _audiveris(page: Path, out_dir: Path) -> Path:
    """One picture or PDF -> Audiveris's file of what it saw on it (.omr)."""
    cmd = [find_audiveris(), "-batch", "-transcribe", "-save", "-output", str(out_dir),
           "-constant", "org.audiveris.omr.sheet.ProcessingSwitches.drumNotation=true", "--", str(page)]
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=1800, check=False)
    except subprocess.TimeoutExpired:
        raise AudioError(f"Reading {page.name} took too long.") from None
    found = sorted(out_dir.rglob("*.omr"))
    if not found:
        raise AudioError(f"No staves were found in {page.name}.")
    return found[0]


# ---------------------------------------------------------------------------
# What Audiveris saw -> which drum when
# ---------------------------------------------------------------------------
@dataclass
class _Event:
    """Something with a length at one place in a bar: the notes on one stem, or a rest."""
    x: float
    voice: str                               # "u" stem up, "d" stem down, "w" no stem (whole notes), "" a rest
    ticks: int
    drums: list = field(default_factory=list)
    pitch: float = 0.0                       # (rests: how high it stands, to tell whose it is)
    triplet: bool = False                    # a "3" was seen over it


_REST = {"HALF_REST": 96, "QUARTER_REST": 48, "EIGHTH_REST": 24, "ONE_16TH_REST": 12,
         "ONE_32ND_REST": 6, "ONE_64TH_REST": 3}


def _drum(shape: str, pitch: int, open_sign: bool) -> str | None:
    """Head shape and place on the staff (0 the middle line, 1 per line or space, down is up...
    as Audiveris counts: negative is higher) -> drum.  The usual drum key (PAS / Weinberg)."""
    if "SMALL" in shape or "TRIANGLE" in shape:
        return None                              # grace and cue notes; cowbell and the like
    if "CROSS" in shape or "CIRCLE_X" in shape or "DIAMOND" in shape:
        if pitch <= -6:
            return "crash"
        if pitch in (-5, -3):
            return "hihat_open" if open_sign or "CIRCLE_X" in shape else "hihat_closed"
        if pitch in (-4, -2):
            return "ride"
        if pitch in (-1, 0):
            return "snare"                       # cross stick
        return None                              # below the staff: hi-hat with the foot, no stroke to hear
    if pitch <= -3:
        return "tom_high"
    if pitch == -2:
        return "tom_mid"
    if pitch == -1:
        return "snare"
    if pitch == 0:
        return "tom_mid"
    if pitch in (1, 2):
        return "tom_floor"
    return "kick"


def _box(el) -> tuple[float, float, float, float]:
    b = el.find("bounds")
    return tuple(float(b.get(k)) for k in ("x", "y", "w", "h"))


def _place(slots: list[list[_Event]], length: int, at_start: bool = True) -> list[int]:
    """When each slot of a bar sounds (ticks from the bar line).  slots: the events
    of the bar grouped by where they stand, left to right; a slot sounds at one time.

    Each voice says how far apart its events are (their lengths), counted from the
    bar line and, as a bar is full, from the end.  Where a rest was missed or a note
    misread the voices disagree; the timing most of them agree with wins, and of
    equally good ones the one that fits the spacing on the page best.
    at_start: the first slot stands right behind the bar line, so it is the first beat
    (unless everything else says otherwise) - not a note after a rest that was missed."""
    first = 2.5 if at_start else 0.5
    back = defaultdict(set)                       # counted from the end of the bar, per voice
    for v in {e.voice for s in slots for e in s}:
        t = length
        for k in range(len(slots) - 1, -1, -1):
            for e in slots[k]:
                if e.voice == v:
                    t -= e.ticks
                    back[k].add(t)
    paths = [(0.0, (), {})]                       # points, times so far, {voice: when its last event ends}
    for k, slot in enumerate(slots):
        grown = []
        for points, times, ends in paths:
            before = times[-1] if times else -1
            tries = set(ends.values()) | back[k] | ({0} if k == 0 else {before + step for step in STEPS})
            for t in {t for t in tries if before < t < length} or {before + 3}:    # (no room left: squeezed in)
                gain = sum(1.0 for e in slot if ends.get(e.voice) == t) + (first if k == 0 and t == 0 else 0.0)
                grown.append((points + gain, times + (t,), {**ends, **{e.voice: t + e.ticks for e in slot}}))
        grown.sort(key=lambda p: -p[0])
        paths = grown[:BEAM]
    xs = np.array([np.mean([e.x for e in s]) for s in slots])

    def odd_spacing(times):                       # a longer note gets more room, never less
        dt, dx = np.diff(np.array(times + (length,), float)), np.diff(xs)
        if len(dx) < 2:
            return 0.0
        dt, bad = dt[:-1], 0.0                    # (the room after the last note says nothing)
        for i in range(len(dx)):
            for j in range(i + 1, len(dx)):
                r = dx[i] / max(dx[j], 1e-6)
                if (dt[i] > dt[j] and r < 0.8) or (dt[i] < dt[j] and r > 1.25):
                    bad += 1
                elif dt[i] == dt[j] and not 0.6 < r < 1.67:
                    bad += 0.5
        return bad / len(dx)

    best = max(paths, key=lambda p: p[0] + sum(1.0 for t in p[2].values() if t == length) - 0.8 * odd_spacing(p[1]))
    return list(best[1])


def _read_page(root) -> tuple[list[dict], list[str]]:
    """One page of Audiveris's file -> its bars, in order:
    {"events": [...], "length": ticks or None, "signs": {...}}, and remarks."""
    bars, remarks = [], []
    scale = root.find("scale/interline")
    il = float(scale.get("main")) if scale is not None else 20.0
    page = root.find("page")
    for system in (page.findall("system") if page is not None else []):
        sig = system.find("sig")
        if sig is None:
            continue
        inters = {i.get("id"): i for i in sig.find("inters")}
        rel = defaultdict(list)                   # kind -> [(source id, target id)]
        for r in (sig.find("relations") if sig.find("relations") is not None else []):
            if len(r):
                rel[r[0].tag].append((r.get("source"), r.get("target")))
        # which staves are the drums: all of them, unless this is a score with other instruments too
        staves = [s.get("id") for s in system.iter("staff")]
        if len(staves) > 1:
            clefs = {i.get("staff") for i in inters.values() if i.tag == "clef" and i.get("kind") == "PERCUSSION"}
            if not clefs:
                crosses = Counter(i.get("staff") for i in inters.values()
                                  if i.tag == "head" and "CROSS" in (i.get("shape") or ""))
                clefs = {crosses.most_common(1)[0][0]} if crosses else set(staves)
            staves = [s for s in staves if s in clefs]
        stem_of = dict(rel["head-stem"])
        beams, flags, dots, opened = Counter(), Counter(), Counter(), set()
        for beam, stem in rel["beam-stem"]:
            beams[stem] += 1
        for flag, stem in rel["flag-stem"]:
            digits = [ch for ch in (inters[flag].get("shape") or "") if ch.isdigit()] if flag in inters else []
            flags[stem] += int(digits[0]) if digits else 1
        for dot, target in rel["augmentation"]:
            dots[target] += 1
        for kind in ("chord-articulation", "head-playing", "chord-playing"):
            for a, b in rel[kind]:
                for sign, other in ((a, b), (b, a)):
                    if sign in inters and "OPEN" in (inters[sign].get("shape") or ""):
                        opened.add(other)
        chord_of = {head: chord for chord, head in rel["containment"]
                    if chord in inters and inters[chord].tag == "head-chord"}
        triplet = {}                              # chord or rest-chord id -> how its length shrinks
        for a, b in rel["chord-tuplet"]:
            for chord, sign in ((a, b), (b, a)):
                if sign in inters and inters[sign].tag == "tuplet":
                    triplet[chord] = (4, 6) if "SIX" in (inters[sign].get("shape") or "") else (2, 3)
        rest_chord = {rest: chord for chord, rest in rel["containment"]
                      if chord in inters and inters[chord].tag == "rest-chord"}
        events, groups = [], defaultdict(list)
        for i in inters.values():
            if i.tag == "head" and i.get("staff") in staves:
                groups[stem_of.get(i.get("id")) or ("alone", i.get("staff"), round(_box(i)[0] / il))].append(i)
        for key, heads in groups.items():
            boxes = [_box(h) for h in heads]
            x = float(np.median([b[0] + b[2] / 2 for b in boxes]))
            y = float(np.mean([b[1] + b[3] / 2 for b in boxes]))
            shapes = [h.get("shape") or "" for h in heads]
            base = 192 if any(s.startswith("WHOLE") or s == "BREVE" for s in shapes) else \
                96 if any("VOID" in s for s in shapes) else 48
            voice = "w"
            if key in inters:                     # a stem: which way, and what hangs on it
                sx, sy, sw, sh = _box(inters[key])
                voice = "u" if sy + sh / 2 < y else "d"
                base >>= max(beams[key], flags[key])
            n = max((dots[h.get("id")] for h in heads), default=0)
            ticks = base * 3 // 2 if n == 1 else base * 7 // 4 if n > 1 else base
            chords = {chord_of.get(h.get("id")) for h in heads}
            marked = sorted(chords & set(triplet))
            for chord in marked[:1]:
                ticks = ticks * triplet[chord][0] // triplet[chord][1]
            is_open = bool(chords & opened) or any(h.get("id") in opened for h in heads)
            drums = [d for h in heads
                     if (d := _drum(h.get("shape") or "", round(float(h.get("pitch") or 0)), is_open))]
            if drums:
                events.append(_Event(x, voice, max(ticks, 1), sorted(set(drums)), triplet=bool(marked)))
        for i in inters.values():
            if i.tag == "rest" and i.get("staff") in staves and i.get("shape") in _REST:
                bx = _box(i)
                ticks = _REST[i.get("shape")] * (3 if dots[i.get("id")] else 2) // 2
                how = triplet.get(rest_chord.get(i.get("id")))
                events.append(_Event(bx[0] + bx[2] / 2, "", ticks * how[0] // how[1] if how else ticks,
                                     pitch=float(i.get("pitch") or 0), triplet=bool(how)))
        # time signatures and the signs that change the order of play, by where they stand
        marks = []
        for i in inters.values():
            x = _box(i)[0] if i.find("bounds") is not None else 0.0
            if i.tag.startswith("time") and i.get("time-rational"):
                num, _, den = i.get("time-rational").partition("/")
                if num.isdigit() and den.isdigit() and int(den):
                    marks.append((x, "length", int(num) * 4 * TICKS // int(den)))
            elif i.tag == "ending":               # the bracket over a first or second ending
                marks.append((x, "ending", (i.get("id"), x + _box(i)[2])))
            elif "REPEAT_ONE_BAR" in (i.get("shape") or "") or i.tag == "measure-repeat":
                marks.append((x + _box(i)[2] / 2, "again", None))
        stacks = [(float(s.get("left")), float(s.get("right")), s.get("repeat") or "") for s in system.findall("stack")]
        for left, right, repeat in stacks:
            mine = sorted((e for e in events if left <= e.x < right), key=lambda e: e.x)
            signs = {"start": repeat in ("LEFT", "BOTH"), "end": repeat in ("RIGHT", "BOTH"),
                     "again": False, "ending": None, "length": None}
            for x, kind, what in marks:
                if kind == "length" and (left - 6 * il if left == stacks[0][0] else left) <= x < right:
                    signs["length"] = what
                elif kind == "again" and left <= x < right:
                    signs["again"] = True
                elif kind == "ending" and min(right, what[1]) - max(left, x) > 0.5 * (right - left):
                    signs["ending"] = what[0]     # (which bracket; its number follows from the order)
            bars.append(dict(events=mine, signs=signs, left=left, right=right, il=il))
    return bars, remarks


def _slots(bar: dict) -> list[list[_Event]]:
    """The events of a bar grouped by where they stand: what is printed above each other
    sounds together.  Each rest is given to the voice that has no note there."""
    events, il = bar["events"], bar["il"]
    slots: list[list[_Event]] = []
    for e in events:
        if slots and e.x - slots[-1][0].x <= 0.7 * il and all(o.voice != e.voice or not e.voice for o in slots[-1]):
            slots[-1].append(e)
        else:
            slots.append([e])
    voices = [v for v, _ in Counter(e.voice for e in events if e.voice in ("u", "d")).most_common()]
    for slot in slots:
        for e in slot:
            if not e.voice:
                free = [v for v in voices if all(o.voice != v for o in slot)]
                side = "d" if e.pitch > 0 else "u"
                e.voice = side if side in free or not free else free[0]
        seen = set()                              # (two rests of one voice in a slot: one too many)
        slot[:] = [e for e in slot if e.voice not in seen and not seen.add(e.voice)]
    return slots


def _fill(slots: list[list[_Event]]) -> int:
    """How much of a bar its fullest voice fills (ticks)."""
    total = Counter()
    for slot in slots:
        for e in slot:
            total[e.voice] += e.ticks
    return max(total.values(), default=0)


def _time_bar(slots: list[list[_Event]], length: int, bar_left: float = 0.0, il: float = 20.0) \
        -> tuple[list[tuple[int, str]], int]:
    """The notes of one bar, [(ticks from the bar line, drum), ...], and how long the bar is."""
    if not any(e.drums for slot in slots for e in slot):
        return [], length
    # Triplets are often marked "3" only the first few times, and the mark is easily missed.  A voice
    # that overfills its bar, and fills it exactly with its unmarked short notes as triplets, is that.
    for v in {e.voice for slot in slots for e in slot}:
        mine = [e for slot in slots for e in slot if e.voice == v]
        plain = [e for e in mine if not e.triplet and e.ticks <= 24 and e.ticks % 3 == 0]
        if plain and sum(e.ticks for e in mine) - sum(e.ticks for e in plain) // 3 == length:
            for e in plain:
                e.ticks, e.triplet = e.ticks * 2 // 3, True
    # a bar line that was not seen makes two bars in one (a stray rest must not: hence nearly twice as full)
    span = length * max(1, int(_fill(slots) / length + 0.25))
    times = _place(slots, span, slots[0][0].x - bar_left < 9 * il)
    return sorted({(t, d) for t, slot in zip(times, slots) for e in slot for d in e.drums}), span


def _order(bars: list[dict]) -> list[int]:
    """The bars in the order they are played: repeat signs and first / second endings followed."""
    ending, number, last = {}, 0, None            # bar -> which time through it is played
    for k, bar in enumerate(bars):
        bracket = bar["signs"]["ending"]
        if bracket:
            if bracket != last:
                number = number + 1 if k - 1 in ending else 1
            ending[k] = number
        last = bracket
    order, i, start, turn, done = [], 0, 0, 1, Counter()
    while i < len(bars) and len(order) < 50 * len(bars):
        signs = bars[i]["signs"]
        if signs["start"] and i != start:
            start, turn = i, 1
        if i in ending and ending[i] != turn and not (ending[i] < turn and ending.get(i + 1) is None
                                                      and not signs["end"]):
            i += 1                                # (the last ending is also played every later time)
            continue
        order.append(i)
        if signs["end"] and done[i] < 1:
            done[i] += 1
            turn += 1
            i = start
            continue
        if signs["end"] or (i in ending and ending.get(i + 1) != ending[i]):
            start, turn = i + 1, 1
        i += 1
    return order


def read_omr(path: str | Path) -> tuple[list[tuple[float, str]], dict]:
    """Audiveris's file of one sheet -> [(seconds at 120 to the minute, drum), ...] and
    what there is to say about it: {"bars": n, "pages": n, "remarks": [...]}."""
    bars, remarks = [], []
    with zipfile.ZipFile(path) as z:
        names = sorted((n for n in z.namelist() if n.endswith(".xml") and n.startswith("sheet#")),
                       key=lambda n: int("".join(ch for ch in n.split("/")[0] if ch.isdigit()) or 0))
        for n in names:
            got, said = _read_page(ElementTree.fromstring(z.read(n)))
            bars += got
            remarks += said
    # how long a bar is: what the sheet says, else what most bars add up to
    for bar in bars:
        bar["slots"] = _slots(bar)
    lengths = Counter(_fill(bar["slots"]) for bar in bars)
    lengths.pop(0, None)
    usual = lengths.most_common(1)[0][0] if lengths else 4 * TICKS
    length = None
    for bar in bars:
        length = bar["signs"]["length"] or length
        bar["length"] = length or usual
    # A stem taken for a bar line cuts a bar in two: two short bars side by side that make one full bar.
    k = 0
    while k + 1 < len(bars):
        one, two = bars[k], bars[k + 1]
        a, b = _fill(one["slots"]), _fill(two["slots"])
        if a and b and a + b == one["length"] == two["length"] and one["right"] == two["left"] \
                and not (one["signs"]["end"] or two["signs"]["start"]):
            one["events"] += two["events"]
            one["right"] = two["right"]
            one["signs"] = {**two["signs"], "start": one["signs"]["start"],
                            "ending": one["signs"]["ending"] or two["signs"]["ending"]}
            one["slots"] = _slots(one)
            del bars[k + 1]
        else:
            k += 1
    timed = [_time_bar(bar["slots"], bar["length"], bar["left"], bar["il"]) for bar in bars]
    for k, bar in enumerate(bars):                # "play the bar before again"
        if bar["signs"]["again"] and not timed[k][0] and k:
            timed[k] = timed[k - 1]
    notes, at = [], 0
    for k in _order(bars):
        notes += [((at + t) / TICKS * 0.5, d) for t, d in timed[k][0]]
        at += timed[k][1]
    return sorted(notes), dict(bars=len(bars), pages=len(names), remarks=remarks, seconds=at / TICKS * 0.5)


# ---------------------------------------------------------------------------
def read_pages(pages: list[str | Path], log=None) -> list[tuple[float, str]]:
    """The pages of one sheet, in order (photos, scans, or a PDF) -> [(seconds, drum), ...]."""
    log = log or (lambda *a: None)
    notes, at = [], 0.0
    CACHE.mkdir(parents=True, exist_ok=True)
    for page in map(Path, pages):
        key = hashlib.sha1(VERSION.encode() + page.read_bytes()).hexdigest()[:20]
        kept = CACHE / f"{key}.omr"
        if not kept.exists():
            find_audiveris()                      # (say so before the minute of work, not after)
            with tempfile.TemporaryDirectory(prefix="drumsheet_omr_") as tmp:
                tmp = Path(tmp)
                given, how = page, ""
                if page.suffix.lower() != ".pdf":
                    given = tmp / "page.png"
                    how = " (" + straighten(page, given) + ")"
                log(f"      reading {page.name}{how} ...")
                shutil.copy(_audiveris(given, tmp / "out"), kept)
        got, info = read_omr(kept)
        if not got:
            raise AudioError(f"No drum notes could be read from {page.name}. It has to be drum notation on "
                             "five-line staves, sharp, and with the whole sheet in the picture.")
        count = Counter(d for _, d in got)
        log(f"      {page.name}: {info['bars']} bars, {len(got)} notes read ("
            + ", ".join(f"{DRUMS[d]['label'].lower()} {n}" for d, n in count.most_common()) + ")")
        notes += [(at + t, d) for t, d in got]
        at += info["seconds"]
    return notes
