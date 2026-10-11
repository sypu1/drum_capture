"""Loading audio files and (optionally) recording from the microphone."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

SUPPORTED = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aif", ".aiff", ".aac"}


class AudioError(RuntimeError):
    """Raised for any problem the user can fix (bad file, no ffmpeg, ...)."""


def load_audio(path: str | Path, sr: int = 44100, start: float = 0.0,
               duration: float | None = None) -> np.ndarray:
    """Return a mono float32 signal at sample rate `sr`.

    WAV/FLAC/MP3 are read directly; anything libsndfile can't open (typically
    M4A/AAC) is converted through ffmpeg first.
    """
    import librosa

    path = Path(path)
    if not path.exists():
        raise AudioError(f"File not found: {path}")
    if path.suffix.lower() not in SUPPORTED:
        print(f"[warn] Unusual extension '{path.suffix}', trying anyway.")

    try:
        import soundfile as sf
        sf.info(str(path))                       # can libsndfile open it?
        y, _ = librosa.load(str(path), sr=sr, mono=True, offset=start, duration=duration)
    except Exception:
        y = _load_via_ffmpeg(path, sr, start, duration)

    if y.size < sr // 2:
        raise AudioError("Audio is shorter than half a second - nothing to transcribe.")
    peak = float(np.max(np.abs(y)))
    if peak < 1e-4:
        raise AudioError("Audio appears to be silent.")
    return (y / peak).astype(np.float32)         # peak-normalise


def _load_via_ffmpeg(path: Path, sr: int, start: float, duration: float | None) -> np.ndarray:
    import soundfile as sf

    if shutil.which("ffmpeg") is None:
        raise AudioError(
            f"Could not decode {path.name}. Install ffmpeg (macOS: `brew install ffmpeg`) "
            "to read MP3/M4A files.")
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "decoded.wav"
        cmd = ["ffmpeg", "-v", "error", "-y", "-ss", str(start), "-i", str(path)]
        if duration:
            cmd += ["-t", str(duration)]
        cmd += ["-ac", "1", "-ar", str(sr), str(wav)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0 or not wav.exists():
            reason = (res.stderr.strip().splitlines() or ["unknown error"])[-1]
            raise AudioError(f"Could not decode {path.name} - is it a valid audio file? ({reason})")
        y, _ = sf.read(str(wav), dtype="float32")
    return y


def find_music_region(y: np.ndarray, sr: int, drop_db: float = 12.0,
                      pad_s: float = 0.5) -> tuple[float, float] | None:
    """Where the playing starts and stops, in seconds - or None if the whole
    file is music.  Meant for phone recordings that begin and end with talking,
    counting-in or shuffling about: only a quiet lead-in and tail are cut,
    never anything in the middle.

    "Quiet" = the level averaged over 2 s is more than drop_db below the loud
    parts of the recording.
    """
    from scipy.ndimage import uniform_filter1d

    hop = int(0.05 * sr)
    n = len(y) // hop
    if n < 80:                                    # under 4 s: nothing to trim
        return None
    rms = np.sqrt(np.mean(y[:n * hop].reshape(n, hop) ** 2, axis=1))
    smooth = uniform_filter1d(rms, size=40, mode="nearest")
    level = 20 * np.log10(smooth / (np.percentile(smooth, 90) + 1e-12) + 1e-12)
    loud = np.flatnonzero(level > -drop_db)
    if len(loud) == 0:
        return None
    start = max(0.0, loud[0] * 0.05 - pad_s)
    end = min(len(y) / sr, (loud[-1] + 1) * 0.05 + pad_s)
    if start < 1.0 and end > len(y) / sr - 1.0:
        return None
    return (0.0 if start < 1.0 else start), end


def record_microphone(seconds: float, out_path: str | Path, sr: int = 44100) -> Path:
    """Record `seconds` of mono audio from the default input device to a WAV."""
    try:
        import sounddevice as sd
        import soundfile as sf
    except Exception as exc:  # ImportError, or PortAudio missing
        raise AudioError(
            "Microphone recording needs the optional 'sounddevice' package "
            "(`pip install sounddevice`).") from exc
    print(f"[rec] Recording {seconds:.0f} s from the default microphone ... play now!")
    try:
        data = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32")
        sd.wait()
    except Exception as exc:
        raise AudioError(f"Recording failed: {exc}. On macOS, allow microphone access "
                         "for your terminal in System Settings > Privacy & Security.") from exc
    out_path = Path(out_path)
    sf.write(str(out_path), data, sr)
    print(f"[rec] Saved recording to {out_path}")
    return out_path
