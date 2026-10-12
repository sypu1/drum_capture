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
| Sheets as photos or PDFs (for training) | [Audiveris](https://github.com/Audiveris/audiveris/releases), a separate program (AGPL): install it, or unpack it and point `AUDIVERIS` at its launcher. `pip install pillow-heif` for iPhone HEIC photos |

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

GPU: with an NVIDIA card and a CUDA build of PyTorch, `adtof` and Demucs run
on the GPU without any option; `python app.py` says at startup which device
they will use. For two minutes of audio the ADTOF network then needs 0.04 s
instead of 3 s (RTX 3090 against its 72-core host). `builtin` and kit models
are not neural networks and always use the CPU. `CUDA_VISIBLE_DEVICES=`
(empty) forces everything onto the CPU.

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

The inputs are the microphones of the computer running drumsheet and "This
browser's microphone". Use the second when the page is open on another device,
a phone next to the kit for example: the page records there and sends the
audio over. Browsers only hand out the microphone on https pages (or on
localhost), so reach drumsheet through something that adds https, such as
`tailscale serve`. Keep the page in front and the screen on: phones stop
recording in the background.

Sounds quieter than -30 dB are never shown as hits ("Ignore quieter than" on
the page, `--gate-db` in the terminal). The mark on the level meter is that
limit, and the bar turns grey for sounds below it: raise the slider until the
room's background stays grey, lower it if soft strokes are missed - it works
while you play. Once you have played, anything more than 24 dB below your
loudest stroke is ignored as well. This matters most for the built-in rules:
they judge loudness only against the rest of the recording, so any tap or
word that gets past the limit is named as some drum. `adtof` was trained to
pick drums out of full songs and mostly ignores other sounds.

The live view was tested by streaming files through it in real time, in a
real browser; the browser's microphone was tested the same way, in Chromium
with a file standing in for its microphone. **Capturing from an actual
microphone, on the computer or on a phone, could not be tested** where this
was built. If it fails, `python -m drumsheet live --list-devices` shows what
Python can see.

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
   Record them the way you will play: the model learns the sound of the
   microphone as much as that of the drums. (The tab's recorder asks the
   browser for the plain microphone, as the Live tab does. It used to get the
   phone-call version, with automatic gain and noise suppression, and a kit
   recorded that way did not know its own drums on the Live tab.)
3. Improve it with real playing - the most effective step in tests:
   transcribe a take, fix the MIDI in MuseScore or a DAW, and train again
   with both:

```bash
python -m drumsheet train home --dir calib/ --pair take1.wav take1_fixed.mid --pair take2.wav take2_fixed.mid
```

Training takes under a minute on a laptop and prints a self-test score per
drum; below 0.75 the result says in so many words that the drum is not
reliable yet. When a trained model plays back less sure than it trained (a
different room, microphone or level), a stroke no drum reaches the threshold
for still gets its likeliest drum rather than disappearing. How it works: the single hits are cut out and mixed with each other
thousands of times at random levels and offsets (in real playing drums are
mostly hit together), and a small classifier per drum learns "is this drum in
this stroke?" from the stroke's spectrum. Corrected takes add real strokes in
context. Models are saved in `kits/`; training again under the same name
replaces the model, so keep your calibration recordings (the web UI stores
them in `kits/NAME_recordings/`).

### From a sheet and your cover of it

The quickest way to get real, labelled playing: play a piece you have the sheet
of, record it on your setup, and give both.

```bash
python -m drumsheet train home --sheet my_cover.wav the_sheet.mid
python -m drumsheet train home --sheet my_cover.m4a the_sheet.pdf
python -m drumsheet train home --sheet my_cover.m4a page1.jpg page2.jpg      # photos of the pages, in order
python -m drumsheet train home --sheet a.wav a.musicxml --sheet b.wav b.mid --dir calib/
```

or the first upload box of the **Train my kit** tab. There, a sheet uploaded
without a recording goes with the take you last recorded on the Live tab,
which is the best recording to use: the model then learns the sound the Live
tab hears. No one-drum recordings are needed, though they still help, above
all for a drum the piece rarely uses.

- The sheet can be **MIDI or MusicXML** (`.mid`, `.musicxml`, `.mxl`), **a
  PDF, or photos of its pages** (see below). Prefer MIDI where there is a
  choice: it has every repeat and jump already written out. From MusicXML and
  from pictures, repeat signs, first and second endings and "%" bars are
  followed; D.C., D.S. and coda jumps are not.
- You do not have to match the sheet's tempo, keep the tempo steady, or start
  the recording at any particular moment: the sheet is lined up with the take
  piece by piece (`drumsheet/sheet.py`), learning on the way what each drum
  sounds like in this recording. A repeat you skipped or played once more is
  followed too. Training prints what it found, for example
  `605 of the sheet's 616 strokes found, starting at 4.0 s, played at 94 % of
  the sheet's tempo`, and which seconds of the take do not follow the sheet.
- The model learns what the sheet says, not what you played. Where the two
  differ - a ghost note that is not written, a fill you simplified - it learns
  something wrong, so pick a piece you play as written.
- Length matters more than anything: a drum needs a few dozen strokes to be
  learned, and a short take has its toms and crashes only a handful of times.

Tested with takes built from the bars of the demo groove in a shuffled order,
the tempo wandering by up to 8 %, with the sheet written at another tempo:
all strokes got the labels of what was really played, also with a bar played
twice, with four bars skipped (one stroke of 387 wrong) and with the timing
20 ms off the sheet (96 % at 35 ms). An unrelated sheet was refused. A kit
trained on one such take alone scored F1 0.98 on another one (`builtin` 0.94,
`adtof` 0.99) - but both takes are made of the same eight bars of the same
sampled kit, so this shows that the route works, not how good the model will
be. Lining up a 3.5-minute take takes about ten seconds.

On real playing it has been tried once, on two phone recordings of one player
(14 s and 8 s: a beat and a tom fill, dull sound with nothing above 4 kHz, the
tempo changing by 9 % within two bars). There was no sheet, so one was written
for each afterwards from the pitches of the strokes - a reading of the takes,
not the player's own sheet. Every written stroke was found in both. A kit
trained on the 14 s take named 14 of the 19 strokes of the other correctly
(`builtin`: 12, with 24 false hits; `adtof` heard nothing); the five misses
were toms and the kick that ends the fill, and the training take has each tom
only twice. **The player's own sheet has not been tried yet.**

#### A photo or a PDF of the sheet

Needs Audiveris (see Install). Takes about half a minute per page; what was
read is kept in `~/.cache/drumsheet/`, so training again is quick.

- **What it has to be:** printed drum notation on five-line staves, in the
  usual drum key: x above the top line hi-hat, x on a ledger line above crash,
  x on the top line ride, third space snare, top space and fourth line the
  rack toms, second space floor tom, bottom space kick. A sheet in another key
  is read as if it were this one, so its drums come out renamed. Handwriting
  and tablature are not read.
- **Photos:** one per page, the whole page in the picture, sharp, taken from
  roughly above. Tilt, slant and uneven light are taken out first
  (`straighten()` in `drumsheet/omr.py`), a curled page is not flattened.
  Let the page fill the picture: a phone then has the staff lines some 25
  pixels apart, and the photo that failed in the tests below had them 13
  apart and blurred. A scan made with the phone's document scanner (a PDF)
  should read best; that has not been tried.
- **What is read:** which drum, when, repeats, endings, "%" bars, triplets
  (also where the "3" is only printed the first few times). **Not read:** the
  "o" of an open hi-hat (it becomes a closed one), the tempo mark (120 is
  assumed; a piece played slower than 60 or faster than 240 is then not
  found), accents and the brackets of ghost notes (those notes count as
  ordinary ones). Grace notes and the hi-hat played with the foot are left
  out.
- Training prints what it read, e.g. `32 bars, 430 notes read (closed hi-hat
  215, kick 86, ...)`. Compare the number of bars with your sheet: if staves
  are missing the photo was too small or too blurred, take it again closer.

How it works: Audiveris finds the symbols; which drum and when is worked out
here from what it saw (place on the staff and shape of the head; stems up and
down as two voices; notes printed above each other sound together), because
its own score falls apart when it misses a clef or a rest.

Tested on three sheets made for the purpose, with known notes, as PDF and as
simulated phone photos (page turned and seen at an angle on a dark desk,
uneven light, blur, noise). No photo taken with a real phone and no sheet from
a publisher has been tried.

| sheet | as PDF | photos |
|---|---|---|
| 13 bars, LilyPond, two voices, fills, ride | 149 of 152 notes | 149, 141 and 148 of 152 (turned 0, 2.5 and 5 degrees); turned 8 degrees with a strong slant 110, and only 60 of those in the right bar |
| 14 bars, LilyPond, sixteenths, repeat with two endings, "%" bars, triplets | 203 of 205 | not tried |
| 32 bars, MuseScore, a full page | 421 of 430 | 421 of 430 turned 1 degree; 178 of 430 turned 2.5 degrees and more blurred (four of the eight staves were not found) |

The notes missing from the PDFs are the open hi-hats and one crash written as
a whole note. Training from the 32-bar PDF and from its photo gave 276 of 285
strokes the same labels as training from the exact sheet. drumsheet's own
chart PDFs read poorly (25 of 100 notes): they are drawn, not set in a music
font. Use the MusicXML or MIDI that comes with them.

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
  A hi-hat played together with a kick leaves no trace on such a recording,
  so it is written without being heard, wherever hi-hat strokes are running
  within three places on both sides of the kick.
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
| `drumsheet/sheet.py` | reading a sheet (MIDI, MusicXML) and finding its notes in a recording of it |
| `drumsheet/omr.py` | a photo or PDF of a sheet: straightening the photo, running Audiveris, working out the notes |
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
