"""The built-in rule-based transcriber (see drumsheet/transcribe.py)."""
from __future__ import annotations

from . import Backend
from ..transcribe import transcribe


class BuiltinBackend(Backend):
    key = "builtin"
    label = "Built-in rules"
    description = ("No install, adapts to the recording (works on phone recordings). "
                   "Cannot tell ride from hi-hat; weak on full mixes without Demucs.")

    def transcribe(self, y, sr, sensitivity=1.0):
        hits, _, _ = transcribe(y, sensitivity=sensitivity)
        return hits
