"""PDF output.  Two looks, both drawn directly with ReportLab (no external
engraving program needed):

* "notation" - standard drum-staff notation: x noteheads for cymbals, round
               ones for drums, stems up, beamed per beat, with rests.
* "grid"     - a drum-machine style chart: one row per drum, one column per
               grid step, with the count written above.

It is a first-draft chart, not publication engraving.  For that, open the
MusicXML export in MuseScore.
"""
from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas

from .config import DRUM_ORDER, DRUMS
from .rhythm import STRAIGHT, TRIPLET, Score, beat_events

SP = 7.0                 # distance between two staff lines (points)
MARGIN = 42.0
GREY = colors.Color(0.45, 0.45, 0.45)
PAGES = {"letter": letter, "a4": A4}


# ---------------------------------------------------------------------------
# Fonts: the built-in Helvetica only covers Latin text.  For titles in other
# scripts (e.g. Chinese/Japanese/Korean song names) switch to a Unicode font.
# ---------------------------------------------------------------------------
def _title_font(text: str) -> str:
    try:
        text.encode("cp1252")
        return "Helvetica-Bold"
    except UnicodeEncodeError:
        pass
    from reportlab.pdfbase.ttfonts import TTFont
    for path in ("/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
                 "/Library/Fonts/Arial Unicode.ttf",
                 "/usr/share/fonts/truetype/arphic/uming.ttc",
                 "C:/Windows/Fonts/msjh.ttc"):
        if Path(path).exists():
            try:
                pdfmetrics.registerFont(TTFont("TitleUnicode", path))
                return "TitleUnicode"
            except Exception:
                continue
    # Last resort: Adobe's built-in CJK fonts (no font file needed).
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    name = ("HeiseiMin-W3" if any("\u3040" <= ch <= "\u30ff" for ch in text) else
            "HYSMyeongJo-Medium" if any("\uac00" <= ch <= "\ud7a3" for ch in text) else
            "MSung-Light")
    pdfmetrics.registerFont(UnicodeCIDFont(name))
    return name


# ---------------------------------------------------------------------------
# Small glyphs
# ---------------------------------------------------------------------------
def _notehead(c: canvas.Canvas, x: float, y: float, drum: str) -> None:
    if DRUMS[drum]["head"] == "x":
        r = SP * 0.46
        c.setLineWidth(1.25)
        c.line(x - r, y - r, x + r, y + r)
        c.line(x - r, y + r, x + r, y - r)
        if drum == "hihat_open":                 # open hi-hat = circled x
            c.setLineWidth(0.8)
            c.circle(x, y, r * 1.55, stroke=1, fill=0)
    else:
        c.saveState()
        c.translate(x, y)
        c.rotate(22)
        c.ellipse(-SP * 0.56, -SP * 0.40, SP * 0.56, SP * 0.40, stroke=0, fill=1)
        c.restoreState()


def _rest(c: canvas.Canvas, x: float, y: float, dur: int, table: dict) -> None:
    """Draw a rest centred on (x, y = middle staff line)."""
    kind, dots, _ = table[dur]
    c.setLineWidth(1.3)
    c.setLineCap(1)
    if kind == "quarter":
        p = c.beginPath()
        p.moveTo(x - 1.8, y + 7.5)
        for dx, dy in ((2.2, 3.0), (-1.6, 0.2), (2.2, -3.6)):
            p.lineTo(x + dx, y + dy)
        p.curveTo(x - 2.5, y - 2.2, x - 3.2, y - 5.5, x + 0.6, y - 7.2)
        c.drawPath(p, stroke=1, fill=0)
    else:
        hooks = 1 if kind == "eighth" else 2
        c.line(x + 2.6, y + 4.2, x - 0.4 - 1.3 * (hooks - 1), y - 6.0 - 3.0 * (hooks - 1))
        for i in range(hooks):
            yy = y + 2.6 - i * 4.4
            xx = x - 2.4 - i * 1.3
            c.circle(xx, yy, 1.45, stroke=0, fill=1)
            c.line(xx, yy - 0.9, x + 2.2 - i * 1.3, yy + 1.2)
    for d in range(dots):
        c.circle(x + 5.5 + 3 * d, y + SP * 0.5, 1.0, stroke=0, fill=1)
    c.setLineCap(0)


# ---------------------------------------------------------------------------
# Shared page furniture
# ---------------------------------------------------------------------------
def _grid_name(sub: int) -> str:
    return {2: "8th notes", 4: "16th notes", 3: "8th-note triplets", 6: "16th-note triplets"}[sub]


def _header(c: canvas.Canvas, score: Score, W: float, H: float, used: list[str]) -> float:
    """Title block + legend.  Returns the y coordinate where music may start."""
    c.setFillColor(colors.black)
    c.setFont(_title_font(score.title), 19)
    c.drawString(MARGIN, H - MARGIN - 10, score.title[:80])
    c.setFont("Helvetica", 10.5)
    c.drawString(MARGIN, H - MARGIN - 28,
                 f"Tempo ~ {score.bpm:.0f} BPM   \u00b7   {score.beats_per_bar}/4   \u00b7   "
                 f"grid: {_grid_name(score.sub)}   \u00b7   {len(score.bars)} bars")
    c.setFont("Helvetica-Oblique", 8)
    c.setFillColor(GREY)
    c.drawRightString(W - MARGIN, H - MARGIN - 6, "Automatic transcription - a first draft, check it by ear")
    for i, note in enumerate(score.notes[:3]):
        c.drawRightString(W - MARGIN, H - MARGIN - 17 - 10 * i, note[:95])
    c.setFillColor(colors.black)

    # legend: a short staff showing where each drum sits
    y0 = H - MARGIN - 82
    x = MARGIN + 58
    c.setFont("Helvetica-Bold", 8)
    c.drawString(MARGIN, y0 + 2 * SP - 3, "Key")
    c.setLineWidth(0.5)
    width = 62 * len(used) + 8
    for i in range(5):
        c.line(x - 32, y0 + i * SP, x - 32 + width, y0 + i * SP)
    c.setFont("Helvetica", 7.5)
    for drum in used:
        y = y0 + DRUMS[drum]["staff_step"] * SP / 2
        if DRUMS[drum]["staff_step"] >= 10:
            c.setLineWidth(0.5)
            c.line(x - 6, y, x + 6, y)
        _notehead(c, x, y, drum)
        c.drawCentredString(x, y0 - 13, DRUMS[drum]["label"])
        x += 62
    return y0 - 30


def _footer(c: canvas.Canvas, W: float, page: int) -> None:
    c.setFont("Helvetica", 7.5)
    c.setFillColor(GREY)
    c.drawCentredString(W / 2, MARGIN - 18, f"- {page} -")
    c.drawRightString(W - MARGIN, MARGIN - 18, "drumsheet")
    c.setFillColor(colors.black)


# ---------------------------------------------------------------------------
# Notation
# ---------------------------------------------------------------------------
STAFF_SYSTEM_H = 13.5 * SP        # vertical room one staff system takes


def _draw_notation_system(c: canvas.Canvas, score: Score, first_bar: int, n_bars: int,
                          bars_per_line: int, x0: float, y0: float, width: float,
                          show_time_sig: bool) -> None:
    """One line of music.  y0 = bottom staff line."""
    table = TRIPLET if score.sub in (3, 6) else STRAIGHT
    lead = 20 + (16 if show_time_sig else 0)          # room for clef (+ time signature)
    bar_w = (width - lead) / bars_per_line
    x_end = x0 + lead + n_bars * bar_w
    y_beam = y0 + 7.6 * SP

    c.setLineWidth(0.55)
    for i in range(5):
        c.line(x0, y0 + i * SP, x_end, y0 + i * SP)
    c.setLineWidth(0.9)
    c.line(x0, y0, x0, y0 + 4 * SP)
    c.rect(x0 + 5, y0 + SP, 2.2, 2 * SP, stroke=0, fill=1)        # percussion clef
    c.rect(x0 + 9.5, y0 + SP, 2.2, 2 * SP, stroke=0, fill=1)
    if show_time_sig:
        c.setFont("Helvetica-Bold", 15)
        c.drawCentredString(x0 + 26, y0 + 2 * SP + 1, str(score.beats_per_bar))
        c.drawCentredString(x0 + 26, y0 + 1, "4")

    for b in range(n_bars):
        bar = first_bar + b
        bx = x0 + lead + b * bar_w
        last = bar == len(score.bars) - 1
        c.setLineWidth(0.9)
        c.line(bx + bar_w, y0, bx + bar_w, y0 + 4 * SP)
        if last:                                                  # final double bar
            c.line(bx + bar_w - 3.5, y0, bx + bar_w - 3.5, y0 + 4 * SP)
            c.rect(bx + bar_w - 1.6, y0, 1.6, 4 * SP, stroke=0, fill=1)
        c.setFont("Helvetica-Oblique", 7.5)
        c.setFillColor(GREY)
        c.drawString(bx + 2, y0 + 4 * SP + 30, str(bar + 1))
        c.setFillColor(colors.black)

        if not score.bars[bar]:                                   # empty bar: whole rest
            c.rect(bx + bar_w / 2 - 4, y0 + 3 * SP - 3.2, 8, 3.2, stroke=0, fill=1)
            continue

        pad = 9.0
        step = (bar_w - pad - 3) / (score.beats_per_bar * 12)     # x per duration unit
        for beat in range(score.beats_per_bar):
            events = beat_events(score, bar, beat)
            beat_x = bx + pad + beat * 12 * step
            notes = [(beat_x + off * step, drums, dur) for off, drums, dur in events if drums]
            for off, drums, dur in events:
                if not drums:
                    _rest(c, beat_x + off * step + 1, y0 + 2 * SP, dur, table)

            stems = []
            for x, drums, dur in notes:
                steps = [DRUMS[d]["staff_step"] for d in drums]
                for d in drums:
                    y = y0 + DRUMS[d]["staff_step"] * SP / 2
                    if DRUMS[d]["staff_step"] >= 10:              # ledger line for the crash
                        c.setLineWidth(0.55)
                        c.line(x - SP * 0.95, y, x + SP * 0.95, y)
                    _notehead(c, x, y, d)
                sx = x + SP * 0.52
                c.setLineWidth(0.85)
                c.line(sx, y0 + min(steps) * SP / 2 + 1, sx, y_beam)
                stems.append(sx)
                if table[dur][1]:                                 # dotted note
                    c.circle(x + SP * 1.15, y0 + min(steps) * SP / 2 + SP * 0.3, 1.0,
                             stroke=0, fill=1)

            beamable = len(notes) >= 2 and all(dur not in (12, 8) for _, _, dur in notes)
            if beamable:
                c.rect(stems[0], y_beam - 2.6, stems[-1] - stems[0], 2.6, stroke=0, fill=1)
                for i, (_, _, dur) in enumerate(notes):
                    if dur in (3, 2):                             # 16th: second beam
                        nxt = i + 1 < len(notes) and notes[i + 1][2] in (3, 2)
                        prv = i > 0 and notes[i - 1][2] in (3, 2)
                        if nxt:
                            c.rect(stems[i], y_beam - 6.6, stems[i + 1] - stems[i], 2.4, stroke=0, fill=1)
                        elif not prv:                             # lone 16th: short stub
                            stub = 5.0 if i == 0 else -5.0
                            c.rect(min(stems[i], stems[i] + stub), y_beam - 6.6, 5.0, 2.4,
                                   stroke=0, fill=1)
            else:
                for sx, (_, _, dur) in zip(stems, notes):
                    flags = 2 if dur in (3, 2) else 1 if dur in (6, 9, 4) else 0
                    c.setLineWidth(1.3)
                    for f in range(flags):
                        c.line(sx, y_beam - f * 4.2, sx + 5.5, y_beam - 6.5 - f * 4.2)
            if any(table[dur][2] for _, _, dur in events):        # triplet figure
                c.setFont("Helvetica-Oblique", 7.5)
                c.drawCentredString(beat_x + 6 * step, y_beam + 3.5, "3")


def _notation_pages(c: canvas.Canvas, score: Score, W: float, H: float,
                    bars_per_line: int, page: int, used: list[str]) -> int:
    y = _header(c, score, W, H, used)
    first = True
    for start in range(0, len(score.bars), bars_per_line):
        if y - STAFF_SYSTEM_H < MARGIN:
            _footer(c, W, page)
            c.showPage()
            page += 1
            y = H - MARGIN - 4
        n = min(bars_per_line, len(score.bars) - start)
        _draw_notation_system(c, score, start, n, bars_per_line, MARGIN, y - 9.5 * SP,
                              W - 2 * MARGIN, first)
        first = False
        y -= STAFF_SYSTEM_H
    _footer(c, W, page)
    return page


# ---------------------------------------------------------------------------
# Grid chart
# ---------------------------------------------------------------------------
def _count_labels(sub: int) -> list[str]:
    return {2: ["", "&"], 4: ["", "e", "&", "a"], 3: ["", "&", "a"],
            6: ["", "", "&", "", "a", ""]}[sub]


def _grid_pages(c: canvas.Canvas, score: Score, W: float, H: float,
                bars_per_line: int, page: int, used: list[str], with_header: bool) -> int:
    # one row per drum; open + closed hi-hat share the hi-hat row
    rows = [d for d in used if d != "hihat_open"]
    if "hihat_open" in used and "hihat_closed" not in rows:
        rows.insert(1 if "crash" in rows else 0, "hihat_closed")
    row_h, label_w = 12.0, 24.0
    sys_h = (len(rows) + 1) * row_h + 26
    y = _header(c, score, W, H, used) + 8 if with_header else H - MARGIN
    spb = score.slots_per_bar
    cell = (W - 2 * MARGIN - label_w) / (bars_per_line * spb)
    counts = _count_labels(score.sub)

    for start in range(0, len(score.bars), bars_per_line):
        if y - sys_h < MARGIN:
            _footer(c, W, page)
            c.showPage()
            page += 1
            y = H - MARGIN
        top = y - 22                                              # top edge of first drum row
        n = min(bars_per_line, len(score.bars) - start)
        x0 = MARGIN + label_w
        for r, drum in enumerate(rows):                           # row labels + shading
            ry = top - (r + 1) * row_h
            if r % 2 == 0:
                c.setFillColor(colors.Color(0.95, 0.95, 0.95))
                c.rect(x0, ry, n * spb * cell, row_h, stroke=0, fill=1)
            c.setFillColor(colors.black)
            c.setFont("Helvetica-Bold", 7.5)
            c.drawString(MARGIN, ry + 3.2, "HH" if drum == "hihat_closed" else DRUMS[drum]["short"])
        bottom = top - len(rows) * row_h
        for s in range(n * spb + 1):                              # vertical lines
            x = x0 + s * cell
            if s % spb == 0:
                c.setStrokeColor(colors.black); c.setLineWidth(1.1)
            elif s % score.sub == 0:
                c.setStrokeColor(GREY); c.setLineWidth(0.5)
            else:
                c.setStrokeColor(colors.Color(0.8, 0.8, 0.8)); c.setLineWidth(0.3)
            c.line(x, bottom, x, top)
        c.setStrokeColor(colors.black)
        c.setLineWidth(0.6)
        c.line(x0, top, x0 + n * spb * cell, top)
        c.line(x0, bottom, x0 + n * spb * cell, bottom)

        for b in range(n):
            bar = start + b
            bx = x0 + b * spb * cell
            c.setFont("Helvetica-Oblique", 7.5)
            c.setFillColor(GREY)
            c.drawString(bx + 1, top + 13, f"bar {bar + 1}")
            c.setFillColor(colors.black)
            for s in range(spb):                                  # the count: 1 e & a ...
                cx = bx + (s + 0.5) * cell
                beat, part = divmod(s, score.sub)
                if part == 0:
                    c.setFont("Helvetica-Bold", 7.5)
                    c.drawCentredString(cx, top + 3, str(beat + 1))
                elif counts[part] and cell > 5:
                    c.setFont("Helvetica", 6)
                    c.drawCentredString(cx, top + 3, counts[part])
                for drum, vel in score.bars[bar].get(s, {}).items():
                    row = "hihat_closed" if drum == "hihat_open" else drum
                    cy = top - (rows.index(row) + 0.5) * row_h
                    r = min(3.4, cell * 0.36)
                    if DRUMS[drum]["head"] == "x":
                        c.setLineWidth(1.6 if drum == "crash" else 1.1)
                        c.line(cx - r, cy - r, cx + r, cy + r)
                        c.line(cx - r, cy + r, cx + r, cy - r)
                        if drum == "hihat_open":
                            c.setLineWidth(0.7)
                            c.circle(cx, cy, r * 1.45, stroke=1, fill=0)
                    else:
                        c.circle(cx, cy, r, stroke=0, fill=1)
        y -= sys_h
    _footer(c, W, page)
    return page


# ---------------------------------------------------------------------------
def render_pdf(score: Score, path: str | Path, style: str = "notation",
               bars_per_line: int = 4, page_size: str = "letter") -> Path:
    """Write the chart.  style: "notation", "grid" or "both"."""
    path = Path(path)
    W, H = PAGES[page_size]
    c = canvas.Canvas(str(path), pagesize=(W, H))
    c.setTitle(score.title)
    c.setAuthor("drumsheet")
    used = [d for d in DRUM_ORDER if d in score.used_drums()] or ["snare"]
    page = 1
    if style in ("notation", "both"):
        page = _notation_pages(c, score, W, H, bars_per_line, page, used)
    if style in ("grid", "both"):
        if style == "both":
            c.showPage()
            page += 1
        page = _grid_pages(c, score, W, H, bars_per_line, page, used, with_header=True)
    c.save()
    return path
