# drumsheet

Turn drum audio into a drum chart (PDF, MIDI, MusicXML), watch hits live while
you play, and train a model on your own kit. Everything runs locally.

```
audio / microphone ─> (Demucs: isolate drums) ─> MODEL: which drum was hit when ─> tempo, bars, grid ─> PDF / MIDI / MusicXML
                                                   │
                              built-in rules  |  ADTOF neural net  |  a model trained on your kit
```

It produces a **first draft** that you correct by ear. Which model is best
depends on the recording, which is why the model is a dropdown.

## Install (macOS; Linux and Windows work the same way)

Needs Python 3.10 or newer.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
brew install ffmpeg          # for M4A/AAC (WAV, FLAC and MP3 work without it)
```

Optional pieces:

| You want | Install |
|---|---|
| Core only (files to charts, training) | `pip install numpy scipy librosa soundfile reportlab pretty_midi` |
| Web UI and live view | `pip install gradio sounddevice` |
| Full songs with the built-in or kit models | `pip install demucs` (pulls in PyTorch; downloads its model on first use) |
| The ADTOF model | `pip install torch` then `pip install git+https://github.com/xavriley/ADTOF-pytorch.git` |

On macOS the first microphone use asks for permission for your terminal
(System Settings > Privacy & Security > Microphone).

## Quick start

```bash
python app.py                                   # web UI at http://127.0.0.1:7860
python -m drumsheet examples/demo_groove.flac   # or the command line; result in output/
```

The very first run takes about half a minute while librosa compiles; after
that a 3-minute recording takes a few seconds.

## Models

```bash
python -m drumsheet models                      # what is installed
python -m drumsheet take.wav --model adtof
python -m drumsheet take.wav --compare          # every installed model, side by side
```

`--compare` (a checkbox in the web UI) writes `take.builtin.pdf`,
`take.adtof.pdf`, ... and prints how many of each drum every model found.

| Model | Good at | Weak at |
|---|---|---|
| `builtin` - rules, no install | Any recording quality, incl. phone recordings; adapts to the kit | Full songs without Demucs; ride is written as hi-hat |
| `adtof` - neural network trained on thousands of songs | Full songs (no Demucs needed) and clean kit recordings; soft notes | Dull or unusual-sounding recordings; 5 classes only (all cymbals but the hi-hat are "crash"); non-commercial licence |
| `kit:NAME` - trained on your kit | Your drums, room and mic; the only one that separates ride, crash and hi-hat | Anything that does not sound like the calibration recordings |

Measured on a 16-bar test groove rendered through nine sampled drum kits
(F1 over all notes, 1.0 = perfect; hits within 50 ms count):

| | builtin | adtof | kit model |
|---|---|---|---|
| Drums only, nine kits | 0.87 - 0.98 | 0.96 - 0.99 on eight, **0.49 on a dark jazz kit** | 0.85 - 0.98 |
| Same, after adding one corrected take (three kits tried) | - | - | 0.95 - 0.98 |
| Kit plus bass and keys, no Demucs (two mixes) | 0.67, 0.79 | 0.99, 0.99 | not tested |

How the ride was scored differs, because the models differ: as closed hi-hat
for `builtin`, as crash for `adtof`, as its own class for kit models (ride F1
0.87 - 1.00).

**Read these as upper bounds.** The kits are sampled, so every hit of a drum
sounds almost the same - much easier than a real drummer, above all for the
kit model, whose calibration hits come from the same samples. The built-in
thresholds were tuned on these kits. On the one real recording tried (a phone
recording of a kit) `builtin` and `adtof` disagreed strongly, and there was no
reference to say which was right.

Demucs: with separation on `auto`, `adtof` gets the mix as it is (it was
trained on mixes); the other models get a Demucs drum stem when the file sounds
like a full song. Demucs itself could not be run where this was built. Its
hand-off was checked against Demucs' command line, and one run on a Mac showed
the separation step executing; how accurate the models are on Demucs stems has
not been measured.

### Using another program's transcription

```bash
python -m drumsheet take.wav --hits-from other_tool.mid
```

takes the hits from any drum MIDI and only does tempo, bars and the chart.

### Adding a model

Create `drumsheet/backends/yourmodel.py`:

```python
from . import Backend
from ..transcribe import Hit

class MyBackend(Backend):
    key = "mine"
    label = "My model"
    description = "What it is good and bad at."

    def transcribe(self, y, sr, sensitivity=1.0):      # y: mono float array, 44.1 kHz
        return [Hit(time_in_seconds, "snare", velocity_1_to_127), ...]
```

It is picked up automatically by `--model`, the dropdowns and the live view.
Drum names are the keys of `DRUMS` in `drumsheet/config.py`. A model that only
knows "tom" or "hihat" can pass its hits through `transcribe.refine_details()`
to get floor/mid/high toms and open/closed hi-hats (see `backends/adtof.py`).

## Live view

```bash
python app.py                       # then the "Live" tab, or http://127.0.0.1:7860/live
python -m drumsheet live            # the same in the terminal; Ctrl-C to stop
python -m drumsheet live --simulate examples/demo_groove.flac    # try it without a microphone
```

Pick a model and an input, press Start and play: each stroke appears on its
drum's lane about a quarter of a second later (the model needs to hear a
little of what follows a hit). Stop, and the take is saved and turned into a
chart by the normal, more accurate whole-take pass.

The live view was tested by streaming files through it in real time, in a
real browser. **Capturing from an actual microphone could not be tested**
where this was built. If it fails, `python -m drumsheet live --list-devices`
shows what Python can see.

## Train a model on your kit

1. Record each drum **alone: about ten hits, a second apart, soft to hard**,
   with the microphone where it will be when you play. One file per drum,
   named after the drum; skip drums you do not have.
2. Train:

```bash
python -m drumsheet train home --dir calib/     # calib/kick.wav, snare.wav, hihat_closed.wav, hihat_open.wav,
                                                # tom_high.wav, tom_mid.wav, tom_floor.wav, crash.wav, ride.wav
python -m drumsheet train home --record         # or: guided recording, drum by drum (untested: needs a mic)
python -m drumsheet take.wav --model kit:home
```

   or use the **Train my kit** tab, which records or uploads each drum.
3. Improve it with real playing - the most effective step in tests:
   transcribe a take, fix the MIDI in MuseScore or a DAW, and train again
   with both:

```bash
python -m drumsheet train home --dir calib/ --pair take1.wav take1_fixed.mid --pair take2.wav take2_fixed.mid
```

Training takes under a minute on a laptop and prints a self-test score per
drum. How it works: the single hits are cut out and mixed with each other
thousands of times at random levels and offsets (in real playing drums are
mostly hit together), and a small classifier per drum learns "is this drum in
this stroke?" from the stroke's spectrum. Corrected takes add real strokes in
context. Models are saved in `kits/`; training again under the same name
replaces the model, so keep your calibration recordings (the web UI stores
them in `kits/NAME_recordings/`).

This is a kit-specific model, not a fine-tune of the ADTOF network. Fine-tuning
ADTOF would need hours of labelled recordings and is not implemented.

## Options for files

| Option | Meaning |
|---|---|
| `-o DIR` | output folder (default `output/`) |
| `--model NAME`, `--compare [NAMES]`, `--hits-from FILE.mid` | which transcriber (see above) |
| `--title TEXT` | title on the chart (default: file name) |
| `--separate auto\|yes\|no` | run Demucs: `auto` = only if it sounds like a full mix and the model needs it |
| `--demucs-model NAME`, `--device cpu\|mps\|cuda`, `--keep-stem` | Demucs settings |
| `--bpm N` | set the tempo instead of detecting it |
| `--time-signature auto\|N/4` | `auto` = 4/4 unless the pattern clearly repeats every 3 beats |
| `--grid 8\|16\|8t\|16t` | quantisation grid; `8t` = eighth-note triplets (shuffle) |
| `--sensitivity X` | above 1 picks up quieter hits (and more false ones) |
| `--start S`, `--duration S` | transcribe only part of the file |
| `--no-trim` | keep a quiet lead-in and tail (by default talking before/after a take is skipped) |
| `--style notation\|grid\|both` | staff notation, drum-grid chart, or both in one PDF |
| `--bars-per-line N`, `--page-size letter\|a4` | layout |
| `--no-midi`, `--no-musicxml`, `--record SECONDS` | skip exports; record from the microphone |

### Phone recordings

- **Talking before and after** is skipped; the console and the PDF say which
  seconds were transcribed.
- **Dull sound**: the built-in rules also look for hi-hats at 3-7.5 kHz and
  accept a kick that reads as high as 150 Hz. `adtof` has no such adaptation.
- **Do not force Demucs** on a recording of just a kit.
- Tempo is cross-checked against the hi-hat pulse and the snare backbeat. If
  it is still wrong, give `--bpm`.

## Reading the chart

| Drum | Notation | MIDI note |
|---|---|---|
| Crash | x on a ledger line above the staff | 49 |
| Ride (kit models only) | x on the top line | 51 |
| Closed / open hi-hat | x / circled x above the top line | 42 / 46 |
| High / mid / floor tom | round notes: top space / 4th line / 2nd space | 50 / 47 / 43 |
| Snare | round note, 3rd space | 38 |
| Kick | round note, bottom space | 36 |

The PDF is a working chart drawn by this program. For engraving quality or to
edit notes, open the `.musicxml` in MuseScore.

## Code map

| File | Job |
|---|---|
| `drumsheet/config.py` | kit definition and every threshold of the built-in rules |
| `drumsheet/backends/` | the models: `builtin.py`, `adtof.py`, and the plug-in registry |
| `drumsheet/kit_model.py` | training and using a model for your own kit |
| `drumsheet/transcribe.py` | the built-in rules; shared helpers (`refine_details`, `onset_envelope`) |
| `drumsheet/rhythm.py` | beats, tempo check, bar lines, quantisation, the `Score` data model |
| `drumsheet/separation.py`, `audio_io.py` | Demucs, full-mix detection, loading, trimming, recording |
| `drumsheet/render_pdf.py`, `export.py` | PDF, MIDI, MusicXML |
| `drumsheet/pipeline.py` | `run()` and `run_compare()`, used by CLI and web UI |
| `drumsheet/live.py`, `live_web.py`, `live_page.html` | live engine, its web endpoints, the page |
| `drumsheet/cli.py`, `app.py` | command line and web UI |
| `tools/` | scoring against a reference MIDI, test-audio generator |

## Known limitations

- **Full songs** need `adtof`, or Demucs with the other models.
- **Ride** is only recognised by a kit model.
- **Hi-hats under a ringing crash**, **ghost notes** and very fast doubles are
  often missed by the built-in rules; `adtof` and kit models do better on the
  test kits.
- **Kick vs floor tom** can swap in the built-in rules when tuned alike. Tom
  names are relative: with two toms you get "high" and "floor".
- **Flams, drags, buzz rolls, rim clicks, cowbell** are not handled.
- **Time signature**: only N/4; `auto` only decides between 3/4 and 4/4. For
  6/8 or 12/8 use `2/4` or `4/4` with `--grid 8t`. One signature, one tempo
  map and one grid per file; no swing detection.
- **Bar 1** is placed by the kick/snare/crash pattern and can be off by a beat
  or a bar on unusual grooves.
- **Live view** shows hits about 0.25 s late and is a little less accurate
  than the chart made afterwards.

## Measuring

```bash
python tools/evaluate.py examples/demo_groove_reference.mid examples/demo_groove.flac --model builtin
```

prints precision, recall and F1 per drum for any model. With a reference MIDI
for one of your own takes, this tells you which model suits your setup.
`tools/make_test_midi.py` renders the test groove and calibration recordings
through a SoundFont (`pip install pyfluidsynth`, `brew install fluid-synth`).

The demo audio was rendered from the FluidR3 GM SoundFont (MIT licence).
