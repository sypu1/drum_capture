"""Pluggable transcription models ("backends").

A backend turns a mono drum signal into a list of Hit(time, drum, velocity).
Everything else - separation, tempo, bars, PDF, the live view - is shared.

Shipped backends
    builtin      rule-based, no extra install, works on any audio
    adtof        ADTOF neural network (needs: PyTorch + adtof-pytorch)
    kit:<name>   a model you trained on your own kit ("Train my kit")

Adding your own
    Create drumsheet/backends/<something>.py containing a subclass of Backend
    with a unique `key`.  It is discovered automatically and shows up in
    `python -m drumsheet models`, in --model and in the web UI's dropdown.
    See builtin.py for the smallest possible example.
"""
from __future__ import annotations

import importlib
import pkgutil

import numpy as np

from ..transcribe import Hit


class Backend:
    """Base class.  Subclasses set the three attributes and implement transcribe()."""
    key: str = ""                 # short id used on the command line
    label: str = ""               # shown in the dropdown
    description: str = ""         # one line: what it is good / bad at
    # False for models trained on full songs: with separation on "auto" they are
    # given the mix as it is instead of a Demucs drum stem.
    wants_isolated_drums: bool = True

    def available(self) -> tuple[bool, str]:
        """(usable right now?, how to install it if not)."""
        return True, ""

    def transcribe(self, y: np.ndarray, sr: int, sensitivity: float = 1.0) -> list[Hit]:
        """y: mono float signal at `sr` (44100) Hz.  Return hits sorted by time,
        with drum names from config.DRUMS."""
        raise NotImplementedError


def _discover() -> dict[str, Backend]:
    found: dict[str, Backend] = {}
    for mod in pkgutil.iter_modules(__path__):
        module = importlib.import_module(f"{__name__}.{mod.name}")
        for obj in vars(module).values():
            if isinstance(obj, type) and issubclass(obj, Backend) and obj is not Backend and obj.key:
                found[obj.key] = obj()
    return found


def all_backends() -> dict[str, Backend]:
    """Every backend, including one per trained kit model.  Ordered for display."""
    found = _discover()
    from ..kit_model import KitBackend, list_kits
    ordered = {k: found[k] for k in ("builtin", "adtof") if k in found}
    ordered.update({k: v for k, v in sorted(found.items()) if k not in ordered})
    for name in list_kits():
        ordered[f"kit:{name}"] = KitBackend(name)
    return ordered


def get_backend(key: str) -> Backend:
    from ..audio_io import AudioError
    backends = all_backends()
    if key not in backends:
        raise AudioError(f"Unknown model '{key}'. Available: {', '.join(backends)}")
    ok, hint = backends[key].available()
    if not ok:
        raise AudioError(f"Model '{key}' is not installed. {hint}")
    return backends[key]
