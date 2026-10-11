"""Command line interface.

    python -m drumsheet song.mp3 [options]     transcribe a file (the default)
    python -m drumsheet models                 list the transcription models
    python -m drumsheet train NAME ...         train a model on your own kit
    python -m drumsheet live ...               watch hits live while you play
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

from . import __version__
from .config import GRIDS


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="drumsheet",
        description="Turn a drum recording (or a full song) into a PDF drum chart.",
        epilog="Examples:\n"
               "  python -m drumsheet beat.wav\n"
               "  python -m drumsheet song.mp3 --separate yes --style both -o charts/\n"
               "  python -m drumsheet shuffle.m4a --grid 8t --bpm 96\n"
               "  python -m drumsheet take.wav --model adtof\n"
               "  python -m drumsheet take.wav --compare          (every installed model, side by side)\n"
               "  python -m drumsheet --record 20 --title \"Practice take\"\n\n"
               "Other commands:  python -m drumsheet models | train | live   (each has --help)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("audio", nargs="?", help="audio file (mp3, wav, m4a, flac)")
    p.add_argument("-o", "--out-dir", default="output", help="where to write results (default: output/)")
    p.add_argument("--title", help="title printed on the chart (default: file name)")
    p.add_argument("--record", type=float, metavar="SECONDS",
                   help="record from the microphone instead of reading a file")

    g = p.add_argument_group("model")
    g.add_argument("--model", default="builtin",
                   help="transcription model: builtin (default), adtof, kit:NAME - see `drumsheet models`")
    g.add_argument("--compare", nargs="*", metavar="MODEL",
                   help="run several models on the same audio (no names = all installed); "
                        "files are named take.MODEL.pdf")
    g.add_argument("--hits-from", metavar="FILE.mid",
                   help="skip the model: take the hits from another program's drum MIDI")

    g = p.add_argument_group("separation")
    g.add_argument("--separate", choices=["auto", "yes", "no"], default="auto",
                   help="isolate drums with Demucs first: auto = only if it sounds like a full mix")
    g.add_argument("--demucs-model", default="htdemucs", help="Demucs model name (default: htdemucs)")
    g.add_argument("--device", choices=["cpu", "mps", "cuda"],
                   help="device for Demucs (Apple Silicon: mps)")
    g.add_argument("--keep-stem", action="store_true", help="keep the separated drum stem")

    g = p.add_argument_group("rhythm")
    g.add_argument("--bpm", type=float, help="set the tempo instead of detecting it")
    g.add_argument("--time-signature", default="auto", metavar="N/4",
                   help="auto (4/4 unless clearly 3/4), or e.g. 3/4, 4/4, 6/4")
    g.add_argument("--grid", choices=list(GRIDS), default="16",
                   help="quantisation: 8, 16 (default), 8t = eighth triplets/shuffle, 16t")
    g.add_argument("--sensitivity", type=float, default=1.0,
                   help="above 1 picks up quieter hits (and more false ones), below 1 fewer")
    g.add_argument("--start", type=float, default=0.0, help="start time in seconds")
    g.add_argument("--duration", type=float, help="only transcribe this many seconds")
    g.add_argument("--no-trim", action="store_true",
                   help="do not skip a quiet lead-in / tail (talking before and after a take)")

    g = p.add_argument_group("output")
    g.add_argument("--style", choices=["notation", "grid", "both"], default="notation",
                   help="staff notation, a drum-grid chart, or both in one PDF")
    g.add_argument("--bars-per-line", type=int, default=4)
    g.add_argument("--page-size", choices=["letter", "a4"], default="letter")
    g.add_argument("--no-midi", action="store_true", help="skip the MIDI export")
    g.add_argument("--no-musicxml", action="store_true", help="skip the MusicXML export")
    p.add_argument("--version", action="version", version=f"drumsheet {__version__}")
    return p


def _models(argv: list[str]) -> int:
    from .backends import all_backends
    from .kit_model import KITS_DIR, load_kit
    argparse.ArgumentParser(prog="drumsheet models", description="List the transcription models.").parse_args(argv)
    for key, b in all_backends().items():
        ok, hint = b.available()
        print(f"{key:18s} {'ready' if ok else 'NOT INSTALLED':14s} {b.label}")
        print(f"{'':18s} {b.description}")
        if not ok:
            print(f"{'':18s} {hint}")
        if key.startswith("kit:"):
            info = load_kit(key[4:])["info"]
            print(f"{'':18s} drums: {', '.join(info['classes'])}; trained {info['created']}")
    print(f"\nKit models are stored in {KITS_DIR}")
    return 0


def _train(argv: list[str]) -> int:
    from .audio_io import AudioError, record_microphone
    from .config import DRUMS
    from .kit_model import train_kit

    p = argparse.ArgumentParser(
        prog="drumsheet train",
        description="Train a model on your own kit from short one-drum recordings "
                    "(about 10 hits each, a second apart, soft and hard).",
        epilog="Examples:\n"
               "  python -m drumsheet train home --dir calib/      (calib/kick.wav, calib/snare.wav, ...)\n"
               "  python -m drumsheet train home --kick k.wav --snare s.wav --hihat_closed hh.wav\n"
               "  python -m drumsheet train home --record          (guided recording, drum by drum)\n"
               "  python -m drumsheet train home --dir calib/ --pair take1.wav take1_fixed.mid\n"
               "Then:  python -m drumsheet take.wav --model kit:home",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("name", help="name for the kit model (letters, digits, - and _)")
    p.add_argument("--dir", help="folder with one recording per drum, named after the drum: "
                                 + ", ".join(f"{d}.wav" for d in DRUMS))
    for d, spec in DRUMS.items():
        p.add_argument(f"--{d}", nargs="+", metavar="FILE", help=f"recording(s) of the {spec['label'].lower()} alone")
    p.add_argument("--pair", nargs=2, action="append", metavar=("AUDIO", "MIDI"), default=[],
                   help="a real take plus its corrected drum MIDI (repeatable)")
    p.add_argument("--record", action="store_true", help="record each drum now, guided, from the microphone")
    p.add_argument("--seconds", type=float, default=12.0, help="length of each guided recording (default 12)")
    args = p.parse_args(argv)
    warnings.filterwarnings("ignore")
    try:
        files: dict[str, list] = {}
        if args.dir:
            folder = Path(args.dir)
            if not folder.is_dir():
                raise AudioError(f"Not a folder: {folder}")
            for f in sorted(folder.iterdir()):
                if f.stem in DRUMS and f.suffix.lower() in (".wav", ".flac", ".mp3", ".m4a", ".aif", ".aiff"):
                    files.setdefault(f.stem, []).append(f)
        for d in DRUMS:
            if getattr(args, d):
                files.setdefault(d, []).extend(getattr(args, d))
        if args.record:
            folder = Path("kits") / f"{args.name}_recordings"
            folder.mkdir(parents=True, exist_ok=True)
            print("For each drum: press Enter, then hit it about ten times, a second apart,\n"
                  "from soft to hard. Type s + Enter to skip a drum you do not have.")
            for d, spec in DRUMS.items():
                if input(f"\n{spec['label']} - Enter to record, s to skip: ").strip().lower() == "s":
                    continue
                files.setdefault(d, []).append(record_microphone(args.seconds, folder / f"{d}.wav"))
        print(f"Training kit model '{args.name}'")
        info = train_kit(args.name, files, [tuple(x) for x in args.pair])
    except AudioError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1
    print(f"\nDone. Use it with:  --model kit:{info['name']}   (drums: {', '.join(info['classes'])})")
    return 0


def _live(argv: list[str]) -> int:
    from .audio_io import AudioError
    from .backends import get_backend
    from .live import FileSource, LiveSession, MicSource, list_input_devices, run_terminal
    from .pipeline import Options, run

    p = argparse.ArgumentParser(
        prog="drumsheet live",
        description="Show drum hits in the terminal while you play (Ctrl-C to stop), then make "
                    "a chart from the take. For the graphical live view run `python app.py`.")
    p.add_argument("--model", default="builtin", help="builtin, adtof or kit:NAME")
    p.add_argument("--device", help="input device number or name (see --list-devices)")
    p.add_argument("--list-devices", action="store_true")
    p.add_argument("--simulate", metavar="FILE", help="play a file through the live view instead of the mic")
    p.add_argument("--speed", type=float, default=1.0, help="with --simulate: playback speed")
    p.add_argument("--sensitivity", type=float, default=1.0)
    p.add_argument("--gate-db", type=float, default=-45.0, help="ignore input quieter than this (dBFS)")
    p.add_argument("-o", "--out-dir", default="output")
    p.add_argument("--no-chart", action="store_true", help="do not make a PDF from the take afterwards")
    args = p.parse_args(argv)
    warnings.filterwarnings("ignore")
    if args.list_devices:
        for d in list_input_devices() or [dict(index="-", name="(no input devices found)", default=False)]:
            print(f"{d['index']:>3} {'*' if d['default'] else ' '} {d['name']}")
        return 0
    try:
        backend = get_backend(args.model)
        device = int(args.device) if args.device and args.device.isdigit() else args.device
        source = FileSource(args.simulate, args.speed) if args.simulate else MicSource(device)
        session = LiveSession(source, backend, sensitivity=args.sensitivity, gate_db=args.gate_db)
        print(f"Live - model: {backend.label}. " + ("" if args.simulate else "Play! Ctrl-C to stop."))
        run_terminal(session)
        take = session.save(Path(args.out_dir) / "live_take.wav")
        print(f"\n{len(session.hits)} hits shown live. Take saved to {take}")
        if not args.no_chart:
            result = run(take, Options(out_dir=args.out_dir, model=args.model, separate="no",
                                       sensitivity=args.sensitivity, title="Live take"))
            print("\n" + result.summary)
    except AudioError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    commands = {"models": _models, "train": _train, "live": _live}
    if argv and argv[0] in commands:
        return commands[argv[0]](argv[1:])
    args = build_parser().parse_args(argv)
    warnings.filterwarnings("ignore")                 # keep librosa's chatter off the console
    from .audio_io import AudioError, record_microphone
    from .pipeline import Options, run, run_compare

    try:
        if args.record:
            Path(args.out_dir).mkdir(parents=True, exist_ok=True)
            args.audio = str(record_microphone(args.record, Path(args.out_dir) / "recording.wav"))
        if not args.audio:
            print("error: give an audio file, or use --record SECONDS  (see --help)", file=sys.stderr)
            return 2
        if not 1 <= args.bars_per_line <= 8:
            raise AudioError("--bars-per-line must be between 1 and 8.")
        opt = Options(out_dir=args.out_dir, model=args.model, hits_from=args.hits_from,
                      title=args.title, separate=args.separate,
                      demucs_model=args.demucs_model, device=args.device, bpm=args.bpm,
                      time_signature=args.time_signature, grid=args.grid, style=args.style,
                      bars_per_line=args.bars_per_line, page_size=args.page_size,
                      midi=not args.no_midi, musicxml=not args.no_musicxml,
                      sensitivity=args.sensitivity, start=args.start, duration=args.duration,
                      trim=not args.no_trim,
                      keep_stem=args.keep_stem)
        if args.compare is not None:
            _, table = run_compare(args.audio, opt, args.compare or None)
            print("\n" + table)
            return 0
        result = run(args.audio, opt)
    except AudioError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    print("\n" + result.summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
