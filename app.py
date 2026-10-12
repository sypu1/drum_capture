"""Local web UI:  python app.py   ->  http://127.0.0.1:7860

Three tabs, all running on this machine:
  Transcribe a file   upload or record, pick a model, download PDF / MIDI / MusicXML
  Live                watch hits per drum while you play, then chart the take
  Train my kit        teach a model your own drums from short recordings
"""
from __future__ import annotations

import contextlib
import io
import re
import shutil
import tempfile
import warnings
from pathlib import Path

import gradio as gr
from fastapi import FastAPI

from drumsheet.audio_io import AudioError
from drumsheet.backends import all_backends
from drumsheet.config import DRUM_ORDER, DRUMS, GRIDS
from drumsheet.kit_model import train_kit
from drumsheet.live_web import _state as live_state
from drumsheet.live_web import router as live_router
from drumsheet.pipeline import Options, run, run_compare
from drumsheet.sheet import PAGE_TYPES, SHEET_TYPES

warnings.filterwarnings("ignore")

GRID_LABELS = {"16th notes": "16", "8th notes": "8", "8th-note triplets (shuffle)": "8t",
               "16th-note triplets": "16t"}
assert set(GRID_LABELS.values()) == set(GRIDS)


def model_choices() -> list[tuple[str, str]]:
    """(label shown, key) for the model dropdown - re-read every time, so newly
    trained kits and newly installed models appear."""
    return [(b.label if b.available()[0] else f"{b.label} (not installed)", key)
            for key, b in all_backends().items()]


def model_notes() -> str:
    lines = []
    for key, b in all_backends().items():
        ok, hint = b.available()
        lines.append(f"- **{b.label}** - {b.description}" + ("" if ok else f"  \n  _Not installed._ {hint}"))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tab 1: file -> chart
# ---------------------------------------------------------------------------
def transcribe_ui(audio_path, model, compare, title, separate, grid, style, time_sig, bpm,
                  sensitivity, trim=True, start=0, duration=None):
    if not audio_path:
        raise gr.Error("Please upload or record some audio first.")
    out_dir = Path(tempfile.mkdtemp(prefix="drumsheet_ui_"))
    opt = Options(out_dir=out_dir, model=model or "builtin", title=(title or "").strip() or None,
                  separate=separate, grid=GRID_LABELS[grid], style=style, time_signature=time_sig,
                  bpm=float(bpm) if bpm else None, sensitivity=float(sensitivity),
                  trim=bool(trim), start=float(start or 0),
                  duration=float(duration) if duration else None)
    log = io.StringIO()
    try:
        with contextlib.redirect_stdout(log):
            if compare:
                results, table = run_compare(audio_path, opt)
                files = [str(p) for r in results.values() for p in r.files.values()]
                text = table + "\n\n" + "\n\n".join(r.summary for r in results.values())
            else:
                result = run(audio_path, opt)
                files, text = [str(p) for p in result.files.values()], result.summary
    except AudioError as exc:
        raise gr.Error(str(exc))
    return files, text + "\n\n--- log ---\n" + log.getvalue()


# ---------------------------------------------------------------------------
# Tab 3: train a kit model
# ---------------------------------------------------------------------------
def pair_up(uploads, written: tuple, what: str, pages: tuple = ()) -> list[tuple]:
    """[(recording, the file that says what is played in it), ...] from one upload
    box.  The two go together by name: take1.wav + take1.mid, or take1.wav +
    take1_fixed.mid.  written: the extensions of the second kind.  pages: extensions
    of pictures of a sheet, of which a recording may have several (its pages, in the
    order of their names); for those the second of the pair is a list."""
    paths = [Path(f) for f in uploads or []]
    if not pages and any(p.suffix.lower() in PAGE_TYPES for p in paths):
        raise gr.Error(f"This box takes recordings and their {what}s. A sheet as a photo or a PDF goes with "
                       "its recording in the box for sheets.")
    scores = [p for p in paths if p.suffix.lower() in written + pages]
    audios = [p for p in paths if p not in scores]
    if not paths:
        return []
    if not audios or not scores:
        raise gr.Error(f"Every recording needs its {what}, and every {what} its recording: "
                       f"got {len(audios)} recording(s) and {len(scores)} {what} file(s).")
    mine: dict[Path, list[Path]] = {a: [] for a in audios}
    for m in scores:
        match = [a for a in audios if m.stem.startswith(a.stem) or a.stem.startswith(m.stem)]
        match = [a for a in match if len(a.stem) == max(len(b.stem) for b in match)]   # song2.mid: song2.wav, not song.wav
        if len(audios) == 1:
            match = audios
        if len(match) != 1:
            raise gr.Error(f"Could not tell which recording '{m.name}' belongs to. Name them alike, "
                           "e.g. take1.wav and take1.mid" + (", or song.m4a and song_1.jpg, song_2.jpg." if pages else "."))
        mine[match[0]].append(m)
    pairs = []
    for a, files in mine.items():
        if not files:
            raise gr.Error(f"The recording '{a.name}' has no {what}.")
        photos = [f for f in files if f.suffix.lower() in pages]
        if len(files) > 1 and len(photos) < len(files):
            raise gr.Error(f"'{a.name}' has more than one {what}: " + ", ".join(f.name for f in files)
                           + ". Give one, or the pages of one as pictures.")
        # pages in the order of their names, numbers counted as numbers (page2 before page10)
        number = lambda f: [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", f.name)]
        pairs.append((a, sorted(photos, key=number) if photos else files[0]))
    return pairs


def train_ui(name, sheets, takes, *drum_audio):
    name = (name or "").strip()
    if not name or not all(ch.isalnum() or ch in "-_" for ch in name):
        raise gr.Error("Give the kit a name: letters, digits, '-' and '_' only.")
    files = {d: [a] for d, a in zip(DRUM_ORDER, drum_audio) if a}
    pairs = pair_up(takes, (".mid", ".midi"), "MIDI file")
    # a sheet but no recording: the take just made on the Live tab is the recording of it
    sheets, used = list(sheets or []), ""
    if sheets and all(Path(f).suffix.lower() in SHEET_TYPES + PAGE_TYPES for f in sheets) and live_state["take"]:
        sheets.append(str(live_state["take"]))
        used = "(No recording was uploaded with the sheet: used the take last recorded on the Live tab.)\n"
    covers = pair_up(sheets, SHEET_TYPES, "sheet", PAGE_TYPES)
    log = io.StringIO()
    try:
        # keep the recordings: retraining later (with more takes) needs them
        from drumsheet.kit_model import KITS_DIR
        keep = KITS_DIR / f"{name}_recordings"
        keep.mkdir(parents=True, exist_ok=True)
        for d, (path,) in list(files.items()):
            kept = keep / f"{d}{Path(path).suffix}"
            shutil.copy(path, kept)
            files[d] = [kept]
        info = train_kit(name, files, pairs, log=lambda *a: print(*a, file=log), sheets=covers)
    except AudioError as exc:
        raise gr.Error(str(exc))
    text = (f"Trained 'kit:{info['name']}' on {', '.join(DRUMS[c]['label'] for c in info['classes'])}.\n"
            f"It is now in the model list on the other tabs (reload the Live tab).\n\n{used}{log.getvalue()}")
    return text, gr.update(choices=model_choices(), value=f"kit:{info['name']}"), model_notes()


def sheet_pictures_note() -> str:
    """What the Train tab says about photos and PDFs of sheets: how, or that they cannot be read here."""
    from drumsheet.omr import find_audiveris
    try:
        find_audiveris()
    except AudioError:
        return ("(Photos and PDFs cannot be read on this computer yet: that needs the program Audiveris, "
                "see the README.)")
    return ("Photos: one per page, the whole page in the picture, sharp, taken from above; reading takes "
            "about half a minute per page. It has to be printed drum notation on five-line staves, in the "
            "usual drum key (hi-hat x above the top line, snare in the third space, kick in the bottom space).")


SHEET_PICTURES = sheet_pictures_note()

# ---------------------------------------------------------------------------
with gr.Blocks(title="drumsheet", analytics_enabled=False) as demo:   # no telemetry
    gr.Markdown("# drumsheet\nDrum audio in, drum chart out. Runs entirely on this computer.")
    with gr.Tab("Transcribe a file"):
        with gr.Row():
            with gr.Column():
                audio = gr.Audio(sources=["upload", "microphone"], type="filepath",
                                 label="Audio (MP3, WAV, M4A, FLAC) - or record")
                model = gr.Dropdown(model_choices(), value="builtin", label="Model",
                                    info="which transcriber names the drums - try several on the same file")
                compare = gr.Checkbox(False, label="Compare: run every installed model",
                                      info="one set of files per model, plus a table of what each found")
                title = gr.Textbox(label="Title on the chart (optional)")
                separate = gr.Radio(["auto", "yes", "no"], value="auto",
                                    label="Isolate drums with Demucs first?",
                                    info="auto = only if it sounds like a full song")
                with gr.Row():
                    grid = gr.Dropdown(list(GRID_LABELS), value="16th notes", label="Grid")
                    style = gr.Dropdown(["notation", "grid", "both"], value="notation", label="PDF style")
                with gr.Row():
                    time_sig = gr.Dropdown(["auto", "4/4", "3/4", "2/4", "5/4", "6/4", "7/4"],
                                           value="auto", label="Time signature")
                    bpm = gr.Number(label="Tempo (BPM) - leave empty to detect", value=None)
                sensitivity = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="Sensitivity",
                                        info="higher = picks up quieter hits (and more false ones)")
                trim = gr.Checkbox(value=True, label="Skip quiet start and end",
                                   info="talking or counting-in before and after the playing")
                with gr.Row():
                    start = gr.Number(label="Start at (seconds)", value=0)
                    duration = gr.Number(label="Length (seconds) - empty = to the end", value=None)
                button = gr.Button("Make drum sheet", variant="primary")
            with gr.Column():
                files = gr.File(label="Downloads", file_count="multiple")
                summary = gr.Textbox(label="Result", lines=16)
                notes = gr.Markdown(model_notes())
        button.click(transcribe_ui, [audio, model, compare, title, separate, grid, style, time_sig, bpm,
                                     sensitivity, trim, start, duration], [files, summary])

    with gr.Tab("Live"):
        gr.HTML('<iframe src="/live" style="width:100%;height:860px;border:0" '
                'allow="microphone" title="Live capture"></iframe>')

    with gr.Tab("Train my kit"):
        gr.Markdown(
            "Teach a model **your** drums, room and microphone. Any one of these is enough; "
            "together they are better.\n\n"
            "- **A sheet and your cover of it.** Play a piece you have the sheet of, record it on your "
            "setup, and upload both: the recording, and the sheet as a **photo, a PDF, MIDI or "
            "MusicXML**. Play it as written, repeats included; tempo and where the recording starts do "
            "not matter. Best is to record it on the **Live** tab (the model then learns the sound that "
            "tab hears) and to upload only the sheet here: the take you recorded last is used. "
            + SHEET_PICTURES + " Several pieces: name each recording and its sheet alike, "
            "e.g. `song.m4a` + `song.pdf`, or `song.m4a` + `song_1.jpg`, `song_2.jpg`.\n"
            "- **Each drum alone.** For each drum you have, record it **alone, about 10 hits, a second "
            "apart, soft to hard** (or upload such a recording). Skip drums you do not have. Use the "
            "microphone position you will use for playing.\n"
            "- **Takes with a corrected MIDI.** Transcribe a take, fix the app's MIDI in MuseScore or "
            "a DAW, and add both. Name them alike, e.g. `take1.wav` + `take1.mid`.\n\n"
            "Then train - a minute or so - and pick the new model on the other tabs.")
        kit_name = gr.Textbox(label="Name for this kit", placeholder="home")
        # (no file_types here: phones grey out .musicxml and .mxl, which they do not know)
        sheets = gr.File(label="A sheet (photo, PDF, MIDI or MusicXML) and your recording of it",
                         file_count="multiple")
        drum_inputs = []
        for row_start in range(0, len(DRUM_ORDER), 3):
            with gr.Row():
                for d in DRUM_ORDER[row_start:row_start + 3]:
                    drum_inputs.append(gr.Audio(sources=["microphone", "upload"], type="filepath",
                                                label=DRUMS[d]["label"]))
        takes = gr.File(label="Takes with corrected MIDI (optional)", file_count="multiple",
                        file_types=["audio", ".mid", ".midi"])
        train_button = gr.Button("Train", variant="primary")
        train_out = gr.Textbox(label="Training result", lines=14)
        train_button.click(train_ui, [kit_name, sheets, takes, *drum_inputs], [train_out, model, notes])

# Gradio's recorders ask the browser for a microphone the way a phone call would: echo cancelling,
# noise suppression, automatic gain.  That boosts every stroke to full level and reshapes its sound,
# so a kit recorded on the Train tab did not sound like the same kit on the Live tab (which asks for
# the plain microphone) and the trained model was unsure of most strokes.  Ask for the plain one here too.
PLAIN_MICROPHONE = """<script>
(() => {
  const md = navigator.mediaDevices;
  if (!md || !md.getUserMedia) return;
  const ask = md.getUserMedia.bind(md);
  md.getUserMedia = (want) => {
    if (want && want.audio)
      want = { ...want, audio: { ...(want.audio === true ? {} : want.audio), echoCancellation: false,
                                 noiseSuppression: false, autoGainControl: false, channelCount: 1 } };
    return ask(want);
  };
})();
</script>"""

# One server for everything: the live view's routes, then Gradio at "/".
app = FastAPI()
app.include_router(live_router)
app = gr.mount_gradio_app(app, demo, path="/", head=PLAIN_MICROPHONE)


def device_note() -> str:
    """Where the neural models (ADTOF, Demucs) will run.  Said once at startup,
    because without a usable GPU they fall back to the CPU without a word."""
    try:
        import torch
    except ImportError:
        return "PyTorch is not installed: ADTOF and Demucs are unavailable"
    if torch.cuda.is_available():
        return f"ADTOF and Demucs run on the GPU ({torch.cuda.get_device_name(0)})"
    return "ADTOF and Demucs run on the CPU (PyTorch sees no CUDA GPU)"


if __name__ == "__main__":
    import uvicorn
    print(device_note())
    print("drumsheet is running at  http://127.0.0.1:7860   (Ctrl-C to stop)")
    uvicorn.run(app, host="127.0.0.1", port=7860, log_level="warning")
