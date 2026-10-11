"""The whole job in one function: audio file in, chart files out."""
from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .audio_io import AudioError, find_music_region, load_audio
from .config import DRUM_ORDER, DRUMS, GRIDS, MIDI_ALIASES, TranscribeConfig
from .export import export_midi, export_musicxml
from .render_pdf import render_pdf
from .rhythm import Score, build_score
from .separation import get_drum_signal
from .backends import all_backends, get_backend
from .transcribe import Hit, onset_envelope


@dataclass
class Options:
    out_dir: str | Path = "output"
    model: str = "builtin"            # which transcription model (see drumsheet/backends)
    hits_from: str | Path | None = None   # take the hits from this MIDI file instead of a model
    title: str | None = None
    separate: str = "auto"            # auto | yes | no
    demucs_model: str = "htdemucs"
    device: str | None = None         # cpu | mps | cuda (None = let Demucs decide)
    bpm: float | None = None          # force a tempo instead of detecting it
    time_signature: str = "auto"      # "auto" or "N/4"
    grid: str = "16"                  # see config.GRIDS
    style: str = "notation"           # notation | grid | both
    bars_per_line: int = 4
    page_size: str = "letter"
    midi: bool = True
    musicxml: bool = True
    sensitivity: float = 1.0
    start: float = 0.0                # only transcribe from here (seconds)
    duration: float | None = None
    trim: bool = True                 # skip a quiet lead-in / tail (talking before a take)
    keep_stem: bool = False           # keep the Demucs output next to the results


@dataclass
class Result:
    score: Score
    files: dict[str, Path] = field(default_factory=dict)
    summary: str = ""
    model: str = ""
    counts: dict[str, int] = field(default_factory=dict)


@dataclass
class Prepared:
    """The audio after loading, trimming and drum separation - ready for any model."""
    path: Path
    drums: "object"                   # mono float array at cfg.sr
    how: str                          # what the separation step did
    window_note: str | None
    sr: int


def _beats_per_bar(text: str) -> int | None:
    if text == "auto":
        return None
    m = re.fullmatch(r"(\d+)/4", text.strip())
    if not m or not 1 <= int(m.group(1)) <= 12:
        raise AudioError(f"Unsupported time signature '{text}'. Use 'auto' or N/4 (e.g. 3/4, 4/4, 5/4). "
                         "For 6/8 or 12/8 use 2/4 or 4/4 with --grid 8t.")
    return int(m.group(1))


def prepare(audio_path: str | Path, opt: Options, log=print, isolate: bool = True) -> Prepared:
    """Steps 1-2: load, pick the part with music, isolate the drums.
    isolate=False: with separation on "auto", skip Demucs (the model copes with a mix)."""
    audio_path = Path(audio_path)
    cfg = TranscribeConfig()
    log(f"[1/5] Loading {audio_path.name}")
    y_full = load_audio(audio_path, cfg.sr)
    total = len(y_full) / cfg.sr
    # Which part to transcribe: what the user asked for, else the part with music.
    start, end = opt.start, (opt.start + opt.duration if opt.duration else total)
    window_note = None
    if not (opt.start or opt.duration) and opt.trim:
        region = find_music_region(y_full, cfg.sr)
        if region:
            start, end = region
            window_note = f"Transcribed {start:.1f}-{end:.1f} s of {total:.1f} s (quiet start/end skipped)"
            log(f"      music found from {start:.1f} s to {end:.1f} s - skipping the rest "
                "(--no-trim keeps everything)")
    a, b = int(start * cfg.sr), int(min(end, total) * cfg.sr)
    y = y_full[a:b]
    if len(y) < cfg.sr // 2:
        raise AudioError("The selected part of the audio is shorter than half a second.")
    log(f"      {len(y) / cfg.sr:.1f} s of audio")

    log("[2/5] Drum separation")
    out_dir = Path(opt.out_dir)
    stem_dir = out_dir / f"{audio_path.stem}_stems" if opt.keep_stem else None
    # The full-mix check listens to the selected part; Demucs gets the whole file.
    mode = opt.separate
    if mode == "auto" and not isolate:
        mode = "no"
        log("      this model was trained on full mixes - no separation needed")
    drums, how = get_drum_signal(audio_path, y, cfg.sr, mode, opt.demucs_model,
                                 opt.device, stem_dir)
    if drums is not y:
        drums = drums[a:b]
    log(f"      {how}")
    return Prepared(audio_path, drums, how, window_note, cfg.sr)


def _hits_from_midi(path: str | Path) -> list[Hit]:
    """Use another program's transcription: every drum note of a MIDI file as a hit."""
    import pretty_midi
    hits = [Hit(float(n.start), MIDI_ALIASES[n.pitch], int(n.velocity))
            for inst in pretty_midi.PrettyMIDI(str(path)).instruments if inst.is_drum
            for n in inst.notes if n.pitch in MIDI_ALIASES]
    if not hits:
        raise AudioError(f"{Path(path).name} contains no drum notes (channel 10).")
    return sorted(hits, key=lambda h: h.time)


def finish(prep: Prepared, opt: Options, model: str, log=print, suffix: str = "") -> Result:
    """Steps 3-5 with one model: name the hits, find tempo and bars, write the files."""
    beats_per_bar = _beats_per_bar(opt.time_signature)
    t0 = time.time()
    if opt.hits_from:
        log(f"[3/5] Taking the hits from {Path(opt.hits_from).name}")
        hits, model_label = _hits_from_midi(opt.hits_from), f"MIDI file {Path(opt.hits_from).name}"
    else:
        backend = get_backend(model)
        log(f"[3/5] Detecting and classifying drum hits - model: {backend.label}")
        hits, model_label = backend.transcribe(prep.drums, prep.sr, opt.sensitivity), backend.label
    if not hits:
        raise AudioError("No drum hits found. Try a higher sensitivity, another model, or "
                         "--separate yes if this is a full song.")
    log(f"      {len(hits)} hits")

    log("[4/5] Tempo, bars and quantisation")
    onset_env, fps = onset_envelope(prep.drums)
    score = build_score(hits, onset_env, fps, len(prep.drums) / prep.sr, GRIDS[opt.grid],
                        opt.bpm, beats_per_bar)
    score.title = opt.title or prep.path.stem.replace("_", " ")
    score.notes = [f"Source: {prep.path.name}", f"Model: {model_label}  |  separation: {prep.how}"]
    if prep.window_note:
        score.notes.insert(1, prep.window_note)
    log(f"      {score.bpm:.0f} BPM, {score.beats_per_bar}/4, {len(score.bars)} bars")

    log("[5/5] Writing files")
    out_dir = Path(opt.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / (prep.path.stem + suffix)
    res = Result(score, model=model)
    res.files["pdf"] = render_pdf(score, Path(str(base) + ".pdf"), opt.style,
                                  opt.bars_per_line, opt.page_size)
    if opt.midi:
        res.files["midi"] = export_midi(score, Path(str(base) + ".mid"))
    if opt.musicxml:
        res.files["musicxml"] = export_musicxml(score, Path(str(base) + ".musicxml"))
    for kind, p in res.files.items():
        log(f"      {kind:9s} {p}")

    counts = Counter(d for bar in score.bars for cell in bar.values() for d in cell)
    res.counts = dict(counts)
    lines = [f"Model: {model_label}",
             f"Tempo: {score.bpm:.0f} BPM   Time signature: {score.beats_per_bar}/4   "
             f"Bars: {len(score.bars)}   Notes: {score.n_notes()}",
             f"Drum separation: {prep.how}",
             "Hits: " + ", ".join(f"{DRUMS[d]['label']} {counts[d]}" for d in DRUM_ORDER if counts[d]),
             f"Done in {time.time() - t0:.1f} s"]
    res.summary = "\n".join(lines)
    return res


def _check(opt: Options) -> None:
    if opt.grid not in GRIDS:
        raise AudioError(f"Unknown grid '{opt.grid}'. Choose from {', '.join(GRIDS)}.")
    _beats_per_bar(opt.time_signature)


def run(audio_path: str | Path, opt: Options | None = None, log=print) -> Result:
    """Audio file in, chart files out, with the model named in opt.model."""
    opt = opt or Options()
    _check(opt)
    isolate = True
    if not opt.hits_from:
        isolate = get_backend(opt.model).wants_isolated_drums   # (also fails early if not installed)
    return finish(prepare(audio_path, opt, log, isolate), opt, opt.model, log)


def run_compare(audio_path: str | Path, opt: Options | None = None, models: list[str] | None = None,
                log=print) -> tuple[dict[str, Result], str]:
    """Run several models on the same prepared audio.  Files get the model name
    in them (take.builtin.pdf, take.adtof.pdf, ...).  Returns the results and a
    side-by-side table of what each model found."""
    opt = opt or Options()
    _check(opt)
    if not models:
        models = [k for k, b in all_backends().items() if b.available()[0]]
    prep = prepare(audio_path, opt, log)
    results: dict[str, Result] = {}
    for m in models:
        log(f"--- {m} ---")
        try:
            results[m] = finish(prep, opt, m, log, suffix="." + m.replace(":", "-"))
        except AudioError as exc:
            log(f"      skipped: {exc}")
    if not results:
        raise AudioError("None of the models produced a result.")
    drums = [d for d in DRUM_ORDER if any(r.counts.get(d) for r in results.values())]
    width = max(len(m) for m in results) + 2
    rows = ["How many of each drum every model found (they should roughly agree):",
            " " * width + " BPM bars " + " ".join(f"{DRUMS[d]['short']:>4s}" for d in drums)]
    for m, r in results.items():
        rows.append(f"{m:<{width}}{r.score.bpm:4.0f} {len(r.score.bars):4d} " +
                    " ".join(f"{r.counts.get(d, 0):4d}" for d in drums))
    return results, "\n".join(rows)
