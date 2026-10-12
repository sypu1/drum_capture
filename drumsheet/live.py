"""Live capture: show drum hits while you play, keep the take for a chart.

How it works
------------
Audio arrives in small blocks (from the microphone of this computer or of the
browser showing the live view, or from a file played at real-time speed for
testing).  About ten times a second the last few seconds
are handed to the selected model - the same one used for files - and every hit
that is new and old enough to be "settled" (default 0.25 s) is sent to the
display.  So what you see lags your playing by roughly a quarter of a second:
fine for watching, not for triggering sounds.

The whole take is recorded.  When you stop, it goes through the normal
pipeline (which sees the take as a whole and is more accurate than the live
guess) to make the PDF.

Pieces
    MicSource / BrowserSource / FileSource   where the audio comes from
    LiveEngine               rolling window -> newly settled hits
    LiveSession              thread that ties source, engine and listeners together
"""
from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

import numpy as np

from .audio_io import AudioError, load_audio
from .transcribe import Hit

SR = 44100


# ---------------------------------------------------------------------------
# Audio sources.  read() returns whatever audio has arrived since the last call.
# ---------------------------------------------------------------------------
class MicSource:
    """The computer's microphone / audio interface, via the sounddevice package."""

    def __init__(self, device: int | str | None = None):
        try:
            import sounddevice as sd
        except Exception as exc:
            raise AudioError("Live capture needs the 'sounddevice' package: pip install sounddevice") from exc
        self._q: queue.Queue = queue.Queue()
        self.finished = False
        self._rate = SR
        try:
            self._stream = sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=1024,
                                          device=device, callback=self._callback)
        except Exception:
            # device refuses 44.1 kHz: open at its own rate and convert block by block
            try:
                self._rate = int(sd.query_devices(device, "input")["default_samplerate"])
                self._stream = sd.InputStream(samplerate=self._rate, channels=1, dtype="float32",
                                              blocksize=1024, device=device, callback=self._callback)
            except Exception as exc:
                raise AudioError(f"Could not open the microphone: {exc}. On macOS allow microphone "
                                 "access for your terminal in System Settings > Privacy & Security.") from exc
        self._stream.start()

    def _callback(self, indata, frames, when, status):      # runs on the audio thread
        self._q.put(indata[:, 0].copy())

    def read(self) -> np.ndarray:
        blocks = []
        while True:
            try:
                blocks.append(self._q.get_nowait())
            except queue.Empty:
                break
        if not blocks:
            return np.zeros(0, dtype=np.float32)
        y = np.concatenate(blocks)
        if self._rate != SR:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(SR, self._rate)
            y = resample_poly(y, SR // g, self._rate // g).astype(np.float32)
        return y

    def close(self):
        self.finished = True
        self._stream.stop()
        self._stream.close()


class BrowserSource:
    """Audio recorded by the browser that shows the live view and sent here in
    pieces - for when the page is open on another machine (a phone) than the one
    running drumsheet, whose own microphone MicSource would use."""

    def __init__(self, rate: int):
        self._q: queue.Queue = queue.Queue()
        self.finished = False
        self._resample = None
        if rate != SR:
            # soxr comes with librosa.  It keeps its state from block to block, so the
            # joins are seamless (resample_poly, block by block, leaves a click at each).
            import soxr
            self._resample = soxr.ResampleStream(rate, SR, 1, dtype="float32").resample_chunk

    def push(self, block: np.ndarray) -> None:               # runs on the web server's thread
        self._q.put(block)

    def read(self) -> np.ndarray:
        blocks = []
        while True:
            try:
                blocks.append(self._q.get_nowait())
            except queue.Empty:
                break
        if not blocks:
            return np.zeros(0, dtype=np.float32)
        y = np.concatenate(blocks).astype(np.float32)
        return self._resample(y) if self._resample else y

    def close(self):
        self.finished = True


class FileSource:
    """A file fed in at real-time speed - for trying the live view without a
    microphone, and for testing.  speed > 1 plays faster than real time."""

    def __init__(self, path: str | Path, speed: float = 1.0):
        self._y = load_audio(path, SR)
        self._speed = speed
        self._t0 = None                        # the clock starts at the first read
        self._pos = 0
        self.finished = False

    def read(self) -> np.ndarray:
        if self._t0 is None:
            self._t0 = time.time()
        due = min(len(self._y), int((time.time() - self._t0) * self._speed * SR))
        out = self._y[self._pos:due]
        self._pos = due
        self.finished = due >= len(self._y)
        return out

    def close(self):
        self.finished = True


def list_input_devices() -> list[dict]:
    """[{index, name, default}] for every device that can record."""
    try:
        import sounddevice as sd
        default = sd.default.device[0]
        return [dict(index=i, name=d["name"], default=(i == default))
                for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]
    except Exception:
        return []


# ---------------------------------------------------------------------------
class LiveEngine:
    """Rolling window in, newly settled hits out."""

    def __init__(self, backend, window_s: float = 4.0, settle_s: float = 0.25,
                 gate_db: float = -30.0, sensitivity: float = 1.0):
        self.backend = backend
        self.window = int(window_s * SR)
        self.settle_s = settle_s
        self.set_gate(gate_db)
        self.sensitivity = sensitivity
        self._buf = np.zeros(0, dtype=np.float32)
        self._total = 0                        # samples received so far
        self._done_until = 0.0                 # everything before this has been reported
        self._last: dict[str, float] = {}      # last reported time per drum (against repeats)
        self._loudest = 0.0                    # loudest stroke heard this session

    @property
    def now(self) -> float:
        return self._total / SR

    def set_gate(self, gate_db: float) -> None:
        """Sounds peaking below this level (dBFS) are never hits.  The built-in rules
        judge loudness only relative to what else is in the window, so without a
        gate every tap and word in a quiet room would be named as some drum."""
        self.gate = 10 ** (gate_db / 20)

    @property
    def floor(self) -> float:
        """What a stroke has to reach right now: the gate, or 6 % of the loudest stroke."""
        return max(self.gate, 0.06 * self._loudest)

    def feed(self, block: np.ndarray) -> None:
        self._total += len(block)
        self._buf = np.concatenate([self._buf, block.astype(np.float32)])[-self.window:]

    def process(self) -> list[Hit]:
        """Run the model on the window; return hits not reported before, in session time."""
        t0 = self.now - len(self._buf) / SR
        until = self.now - self.settle_s
        if until <= self._done_until or len(self._buf) < SR // 2:
            return []
        peak = float(np.abs(self._buf).max())
        self._loudest = max(self._loudest * 0.999, peak)
        # Ignore anything far quieter than real playing: room noise, talking.
        floor = self.floor
        out: list[Hit] = []
        if peak >= floor:
            try:
                hits = self.backend.transcribe(self._buf, SR, self.sensitivity)
            except Exception:
                hits = []                      # a hiccup in one window must not stop the session
            for h in hits:
                t = h.time + t0
                # the 30 ms overlap catches hits whose time moved slightly between windows
                if not (self._done_until - 0.03 < t <= until):
                    continue
                if t - self._last.get(h.drum, -1.0) < 0.06:
                    continue
                a = int(h.time * SR)
                if float(np.abs(self._buf[max(0, a - 220):a + 1800]).max(initial=0.0)) < floor:
                    continue
                self._last[h.drum] = t
                out.append(Hit(round(t, 3), h.drum, h.velocity))
        self._done_until = until
        return sorted(out, key=lambda h: h.time)


# ---------------------------------------------------------------------------
class LiveSession:
    """Runs source -> engine in a background thread and tells listeners."""

    def __init__(self, source, backend, step_s: float = 0.1, **engine_kw):
        self.source = source
        self.engine = LiveEngine(backend, **engine_kw)
        self.step_s = step_s
        self.hits: list[Hit] = []
        self.level_db = -120.0
        self.error: str | None = None
        self._audio: list[np.ndarray] = []
        self._listeners: list[queue.Queue] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.started = time.time()

    def start(self) -> "LiveSession":
        self._thread.start()
        return self

    def listen(self) -> queue.Queue:
        """A queue that receives dicts: {"type": "hit"|"level"|"end", ...}."""
        q: queue.Queue = queue.Queue()
        self._listeners.append(q)
        return q

    def _tell(self, event: dict) -> None:
        for q in self._listeners:
            q.put(event)

    @property
    def running(self) -> bool:
        return self._thread.is_alive()

    def _run(self) -> None:
        last = 0.0
        try:
            while not self._stop.is_set():
                block = self.source.read()
                if not len(block):
                    if self.source.finished:
                        break
                    time.sleep(0.005)
                    continue
                self._audio.append(block)
                # If a lot arrived at once (the model was busy), work through it in
                # steps so nothing scrolls out of the window unseen.
                n = int(self.step_s * SR)
                for i in range(0, len(block), n):
                    piece = block[i:i + n]
                    self.engine.feed(piece)
                    if self.engine.now - last < self.step_s and i + n < len(block):
                        continue
                    last = self.engine.now
                    self.level_db = float(20 * np.log10(np.sqrt(np.mean(piece ** 2)) + 1e-9))
                    for h in self.engine.process():
                        self.hits.append(h)
                        self._tell(dict(type="hit", t=h.time, drum=h.drum, vel=h.velocity))
                    # peak and gate let the display show which sounds are too quiet to count
                    peak_db, gate_db = 20 * np.log10(np.array([np.abs(piece).max(), self.engine.floor]) + 1e-9)
                    self._tell(dict(type="level", t=round(self.engine.now, 2), db=round(self.level_db, 1),
                                    peak=round(float(peak_db), 1), gate=round(float(gate_db), 1)))
        except Exception as exc:                 # report instead of dying silently
            self.error = str(exc)
        finally:
            self.source.close()
            self._tell(dict(type="end", t=round(self.engine.now, 2), error=self.error))

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def wait(self) -> None:
        self._thread.join()

    def save(self, path: str | Path) -> Path:
        """Write the recorded take as a WAV file."""
        import soundfile as sf
        if not self._audio:
            raise AudioError("Nothing was recorded.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), np.concatenate(self._audio), SR)
        return path


# ---------------------------------------------------------------------------
# Terminal view
# ---------------------------------------------------------------------------
def run_terminal(session: LiveSession, drums: list[str] | None = None, out=print) -> None:
    """Print one line per stroke with a column per drum, until the session ends
    (Ctrl-C stops a microphone session)."""
    from .config import DRUM_ORDER, DRUMS

    drums = drums or DRUM_ORDER
    q = session.listen()
    out("  time   " + " ".join(f"{DRUMS[d]['short']:>3s}" for d in drums))
    pending: list[dict] = []

    def flush():
        if pending:
            names = {e["drum"] for e in pending}
            out(f"{pending[0]['t']:6.2f}   " + " ".join("  #" if d in names else "  ." for d in drums))
            pending.clear()

    session.start()
    try:
        while True:
            e = q.get()
            if e["type"] == "hit":
                if pending and e["t"] - pending[0]["t"] > 0.04:
                    flush()
                pending.append(e)
            elif e["type"] == "level":
                flush()
            elif e["type"] == "end":
                flush()
                if e.get("error"):
                    out(f"error: {e['error']}")
                break
    except KeyboardInterrupt:
        session.stop()
