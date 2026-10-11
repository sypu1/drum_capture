"""Getting a drums-only signal out of a full mix with Demucs.

Demucs (htdemucs) is a neural source separator that runs locally.  It is an
optional dependency: without it the program still runs, but a full song is
then transcribed as-is, which gives many wrong notes (bass and guitars get
mistaken for drums).
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from .audio_io import AudioError, load_audio


def demucs_available() -> bool:
    return importlib.util.find_spec("demucs") is not None


def looks_like_full_mix(y: np.ndarray, sr: int, threshold_db: float = -12.0) -> tuple[bool, float]:
    """Guess whether `y` contains sustained instruments besides drums.

    Between two hits a drum kit goes (nearly) quiet in the 100-1000 Hz range,
    while bass, guitars, keys and vocals keep that range filled.  So we compare
    the quiet moments (20th percentile of the frame energy) with the loud ones
    (90th percentile).  Returns (is_full_mix, floor_in_dB).

    Measured on the test material: isolated kits -20 dB and below, kit plus
    bass and keys around -5 dB.
    """
    import librosa

    n = int(30 * sr)
    seg = y[max(0, (len(y) - n) // 2):][:n]                # 30 s from the middle
    S = np.abs(librosa.stft(seg, n_fft=2048, hop_length=512)) ** 2
    f = librosa.fft_frequencies(sr=sr, n_fft=2048)
    E = S[(f >= 100) & (f < 1000)].sum(axis=0)
    floor_db = float(10 * np.log10((np.percentile(E, 20) + 1e-12) / (np.percentile(E, 90) + 1e-12)))
    return floor_db > threshold_db, floor_db


def separate_with_demucs(path: str | Path, sr: int, model: str = "htdemucs",
                         device: str | None = None, keep_dir: str | Path | None = None) -> np.ndarray:
    """Run `python -m demucs --two-stems drums` and return the drum stem.

    The Demucs command line is used (rather than its Python API) because it is
    the most stable interface across Demucs versions.
    """
    work = Path(keep_dir) if keep_dir else Path(tempfile.mkdtemp(prefix="drumsheet_"))
    work.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "demucs", "--two-stems", "drums", "-n", model,
           "-o", str(work), str(path)]
    if device:
        cmd += ["-d", device]
    print(f"[demucs] Separating drums with '{model}'. The first run downloads the model; "
          "a full song can take a few minutes on a laptop CPU ...")
    # Demucs' own progress bar and error messages go straight to the console.
    res = subprocess.run(cmd, stdout=subprocess.DEVNULL)
    stems = sorted(work.rglob("drums.wav"), key=lambda p: p.stat().st_mtime)
    if res.returncode != 0 or not stems:
        raise AudioError(f"Demucs failed (exit code {res.returncode}) - see its messages above.")
    print(f"[demucs] Drum stem: {stems[-1]}")
    return load_audio(stems[-1], sr=sr)


def get_drum_signal(path: str | Path, y: np.ndarray, sr: int, mode: str = "auto",
                    model: str = "htdemucs", device: str | None = None,
                    keep_dir: str | Path | None = None) -> tuple[np.ndarray, str]:
    """Return (drum_signal, description_of_what_was_done).

    mode: "auto" (separate only if it sounds like a full mix), "yes", "no".
    """
    if mode == "no":
        return y, "none (input used as-is)"
    if mode == "auto":
        full, floor_db = looks_like_full_mix(y, sr)
        print(f"      between-hit level {floor_db:.0f} dB -> "
              f"{'sounds like a full mix' if full else 'sounds like drums only'}")
        if not full:
            return y, "none (input sounds like isolated drums)"
    if not demucs_available():
        print("[warn] This looks like a full song, but Demucs is not installed, so the drums\n"
              "       cannot be isolated.  Transcribing the mix as-is - expect many wrong notes.\n"
              "       Install it with:  pip install demucs")
        return y, "NOT DONE - Demucs missing, full mix transcribed as-is"
    try:
        return separate_with_demucs(path, sr, model, device, keep_dir), f"Demucs ({model})"
    except AudioError as exc:
        print(f"[warn] {exc}\n[warn] Transcribing the unseparated audio instead.")
        return y, "FAILED - Demucs error, full mix transcribed as-is"
