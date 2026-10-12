"""ADTOF Frame-RNN drum transcription network (Zehren et al.), PyTorch port.

Install (not bundled - the original ADTOF is licensed CC BY-NC-SA 4.0, i.e.
non-commercial use only):

    pip install torch
    pip install git+https://github.com/xavriley/ADTOF-pytorch.git

The network was trained on thousands of full songs, so it copes with bass,
guitars and vocals in the signal.  It knows five classes: kick, snare, tom,
hi-hat, cymbal.  refine_details() then splits toms by pitch and hi-hats into
open/closed with the same measurements the built-in rules use.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io

import numpy as np

from . import Backend
from ..transcribe import Hit, drop_retriggers, refine_details

# ADTOF's output classes (General MIDI notes) -> our names
ADTOF_CLASSES = {35: "kick", 38: "snare", 47: "tom", 42: "hihat", 49: "crash"}


class AdtofBackend(Backend):
    key = "adtof"
    label = "ADTOF neural network"
    description = ("Trained on full songs: the best choice for mixes and studio recordings. "
                   "Five classes only (no ride); non-commercial licence.")
    wants_isolated_drums = False  # trained on full mixes
    _model = None                 # loaded once, shared

    def available(self):
        if importlib.util.find_spec("torch") is None or importlib.util.find_spec("adtof_pytorch") is None:
            return False, ("Install with:  pip install torch  &&  "
                           "pip install git+https://github.com/xavriley/ADTOF-pytorch.git")
        return True, ""

    @classmethod
    def _load(cls):
        if cls._model is None:
            import adtof_pytorch as A
            import torch
            model = A.create_frame_rnn_model(A.calculate_n_bins())
            with contextlib.redirect_stdout(io.StringIO()):       # it prints while loading
                model = A.load_pytorch_weights(model, A.get_default_weights_path(), strict=False)
            model.eval()
            cls._model = model.to("cuda" if torch.cuda.is_available() else "cpu")   # NVIDIA GPU if usable
        return cls._model

    def activations(self, y: np.ndarray) -> np.ndarray:
        """Per-class hit probability, 100 frames per second: array [frames, 5]."""
        import torch
        from adtof_pytorch.audio import create_adtof_processor

        proc = create_adtof_processor()                           # 44.1 kHz, 100 fps log-filterbank
        spec = proc.apply_filterbank(proc.compute_stft(y.astype(np.float32))).T[:, :, None]
        model = self._load()
        x = torch.from_numpy(np.ascontiguousarray(spec)).float().unsqueeze(0)
        with torch.no_grad():
            out = model(x.to(next(model.parameters()).device))
        return out[0].cpu().numpy()

    def transcribe(self, y, sr, sensitivity=1.0):
        from adtof_pytorch import FRAME_RNN_THRESHOLDS, LABELS_5, PeakPicker

        act = self.activations(y)
        thresholds = [t / sensitivity for t in FRAME_RNN_THRESHOLDS]
        peaks = PeakPicker(thresholds=thresholds, fps=100).pick(act[None], labels=LABELS_5)[0]
        hits = []
        for i, pitch in enumerate(LABELS_5):
            for t in peaks[pitch]:
                strength = float(act[min(int(round(t * 100)), len(act) - 1), i])
                hits.append(Hit(float(t), ADTOF_CLASSES[pitch], int(np.clip(45 + 80 * strength, 20, 127))))
        hits.sort(key=lambda h: h.time)
        return drop_retriggers(refine_details(y, hits))
