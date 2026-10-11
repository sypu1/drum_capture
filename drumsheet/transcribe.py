"""Drum transcription: find the strokes and decide which drum(s) each one is.

The approach is a transparent signal-processing heuristic, not a neural net:

1. ONSETS - A drum kit splits fairly well by frequency: kick lives at the
   bottom, snare/toms in the middle, cymbals at the top.  We track the energy
   in those three bands and mark a stroke wherever a band's energy jumps.
   Bands that fire within ~30 ms of each other are one stroke (e.g. kick+hat).

2. CLASSIFY - For each stroke we look at its spectrum and at how long the
   cymbals keep ringing:
     low band fired  -> kick, unless it is pitched clearly above the lowest
                        drum of the recording or has too much "drum tone"
                        above 120 Hz - then it is a tom
     mid band fired  -> snare  if the 2-8 kHz "sizzle" of the snare wires is
                        still there 30-90 ms after the stroke, else a tom
                        (floor / mid / high by pitch, relative to the other
                        toms heard in the recording).  A cymbal that leaks
                        into the mid band is recognised by its lack of tone.
     high band fired -> closed hi-hat (dead before the next stroke), open
                        hi-hat (still ringing a little), crash/ride (still
                        ringing a lot).

Every threshold lives in config.TranscribeConfig.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import TranscribeConfig


@dataclass
class Hit:
    time: float        # seconds
    drum: str          # key of config.DRUMS
    velocity: int      # 1..127


# ---------------------------------------------------------------------------
# 1. Onset detection
# ---------------------------------------------------------------------------
def _band(S: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return S[(freqs >= lo) & (freqs < hi)].sum(axis=0)


def _flux(E: np.ndarray, compress: float, lag: int = 2, vs_recent_max: int = 0) -> np.ndarray:
    """Positive change of log-compressed band energy ("how hard did it jump").

    Normally the change over `lag` frames.  With vs_recent_max = N the level is
    compared with the highest level of the previous N frames instead: used for
    the low band, where a ringing kick's energy ripples (a 23 ms window sees
    barely one cycle of a 50 Hz tone) and each ripple would look like a new hit.
    """
    ref = np.percentile(E, 99) + 1e-9
    L = np.log1p(compress * E / ref) / np.log1p(compress)
    if vs_recent_max:
        from scipy.ndimage import maximum_filter1d
        n = vs_recent_max
        past = maximum_filter1d(L, size=n, origin=(n - 1) // 2, mode="nearest")  # max of L[t-n+1..t]
        d = np.zeros_like(L)
        d[1:] = L[1:] - past[:-1]
        return np.maximum(d, 0.0)
    d = np.zeros_like(L)
    d[lag:] = L[lag:] - L[:-lag]
    return np.maximum(d, 0.0)


def _pick(env: np.ndarray, delta: float, min_gap_frames: int, fps: float) -> np.ndarray:
    """Peaks of `env` that stand out from their local surroundings."""
    from scipy.ndimage import median_filter
    from scipy.signal import find_peaks

    local = median_filter(env, size=int(0.3 * fps) | 1, mode="nearest")
    peaks, _ = find_peaks(env - local, height=delta, distance=max(1, min_gap_frames),
                          prominence=delta * 0.5)
    return peaks


def detect_strokes(y: np.ndarray, cfg: TranscribeConfig):
    """Return (strokes, context).

    strokes: list of dicts {time, bands: {band_name: strength}}
    context: spectrogram bits the classifier needs.
    """
    import librosa

    S = np.abs(librosa.stft(y, n_fft=cfg.n_fft, hop_length=cfg.hop))
    freqs = librosa.fft_frequencies(sr=cfg.sr, n_fft=cfg.n_fft)
    fps = cfg.sr / cfg.hop
    # "air" (3-7.5 kHz) is a second cymbal band: on phone recordings and dark
    # cymbals there is too little above 7.5 kHz for the "high" band to react.
    energy = {"low": _band(S, freqs, *cfg.band_low),
              "mid": _band(S, freqs, *cfg.band_mid),
              "air": _band(S, freqs, *cfg.band_wash),
              "high": _band(S, freqs, *cfg.band_high)}
    deltas = {"low": cfg.delta_low, "mid": cfg.delta_mid,
              "air": cfg.delta_high, "high": cfg.delta_high}

    raw = []                                   # (time, band, strength)
    onset_times, envs = {}, {}
    for name, E in energy.items():
        env = _flux(E, cfg.compress,
                    vs_recent_max=int(cfg.low_lookback_s * fps) if name == "low" else 0)
        envs[name] = env
        peaks = _pick(env, deltas[name], int(cfg.min_gap_s * fps), fps)
        onset_times[name] = peaks / fps
        raw += [(p / fps, name, float(env[p])) for p in peaks]
    raw.sort()

    # Merge bands that fire almost together into one stroke.
    strokes: list[dict] = []
    for t, band, strength in raw:
        if strokes and t - strokes[-1]["time"] <= cfg.merge_window_s:
            b = strokes[-1]["bands"]
            b[band] = max(b.get(band, 0.0), strength)
        else:
            strokes.append({"time": t, "bands": {band: strength}})

    # Also note how much the low band rose at each stroke even if it did not
    # reach the onset threshold (needed for a kick played into a ringing kick).
    w = int(cfg.merge_window_s * fps)
    for st in strokes:
        i = int(round(st["time"] * fps))
        st["low_rise"] = float(envs["low"][max(0, i - w):i + w + 1].max(initial=0.0))
    energy["wash"] = energy["air"]                        # cymbal sustain is measured here
    ctx = {"energy": energy, "fps": fps, "onset_times": onset_times,
           "onset_env": _flux(S.sum(axis=0), cfg.compress)}
    return strokes, ctx


# ---------------------------------------------------------------------------
# 2. Features + classification
# ---------------------------------------------------------------------------
def _stroke_features(y: np.ndarray, t: float, cfg: TranscribeConfig) -> dict:
    """Spectral measurements of one stroke."""
    def spectrum(start_s, n):
        a = max(0, int((t + start_s) * cfg.sr))
        seg = y[a:a + n]
        seg = np.pad(seg, (0, n - len(seg)))
        return np.abs(np.fft.rfft(seg * np.hanning(n))) ** 2, np.fft.rfftfreq(n, 1 / cfg.sr)

    P, f = spectrum(-0.008, 4096)                  # the whole stroke (~93 ms)
    L, fl = spectrum(0.030, 2646)                  # its tail, 30-90 ms after the attack

    def power(P_, f_, lo, hi):
        return float(P_[(f_ >= lo) & (f_ < hi)].sum())

    def peak_hz(lo, hi):
        """Strongest frequency between lo and hi (parabolic interpolation)."""
        idx = np.flatnonzero((f >= lo) & (f < hi))
        k = idx[np.argmax(P[idx])]
        a_, b_, c_ = np.log(P[k - 1:k + 2] + 1e-20)
        shift = 0.5 * (a_ - c_) / (a_ - 2 * b_ + c_ - 1e-20)
        return float((k + np.clip(shift, -0.5, 0.5)) * (f[1] - f[0]))

    return {"p_low": power(P, f, *cfg.band_low), "p_body": power(P, f, 150, 1000),
            "p_high": power(P, f, *cfg.band_high),
            "p_head": power(P, f, 120, 400),                          # drum-head tone
            "p_noise": power(P, f, 2000, 8000),                       # cymbals / wires
            "sizzle": power(L, fl, 2000, 8000),                       # late snare-wire noise
            "sizzle_ratio": power(L, fl, 2000, 8000) / (power(L, fl, 100, 1000) + 1e-12),
            "low_peak": peak_hz(35, 400), "tom_peak": peak_hz(60, 500)}


def _ring(E: np.ndarray, fps: float, t: float, lo_s: float, hi_s: float,
          t_next: float | None) -> float | None:
    """How much of the stroke's energy in band-envelope E is still there between
    lo_s and hi_s after the stroke: 0 = fully died away, 1 = as loud as the peak.
    The level just before the stroke is subtracted, so a stroke played over an
    already-ringing cymbal is not mistaken for a ringing one.  The window stops
    short of the next stroke; returns None if that leaves no room to measure."""
    i = int(round(t * fps))
    pre = float(E[max(0, i - 5):max(1, i - 1)].mean())
    peak = float(E[i:i + int(0.035 * fps) + 1].max(initial=0.0))
    if t_next is not None:
        room = (t_next - t) - 0.02
        if room < hi_s:                            # slide the window earlier
            lo_s, hi_s = max(lo_s * 0.6, room - (hi_s - lo_s)), room
            if hi_s - lo_s < 0.02 or lo_s < 0.05:
                return None
    a, b = i + int(lo_s * fps), min(i + int(hi_s * fps), len(E))
    if b - a < 2:
        return None
    return max(0.0, float(E[a:b].mean()) - pre) / (peak - pre + 1e-9)


def _name_toms(pitches: list[float], cfg: TranscribeConfig) -> list[str]:
    """Floor / mid / high for each tom stroke.  Kits are tuned differently, so
    we look for up to three pitch groups among the toms actually played
    (a tiny 1-D k-means on log-pitch) instead of using fixed frequencies."""
    if not pitches:
        return []
    x = np.log(np.asarray(pitches))
    best = (np.zeros(len(x), dtype=int), 1)
    for k in (2, 3):
        if len(x) < 2 * k:
            break
        centres = np.quantile(x, (np.arange(k) + 0.5) / k)
        for _ in range(20):
            assign = np.argmin(np.abs(x[:, None] - centres[None, :]), axis=1)
            centres = np.array([x[assign == j].mean() if np.any(assign == j) else centres[j]
                                for j in range(k)])
        sizes = np.bincount(assign, minlength=k)
        # accept k groups only if they are clearly different drums
        if sizes.min() >= 2 and np.all(np.diff(centres) >= np.log(cfg.tom_gap_ratio)):
            best = (assign, k)
    assign, k = best
    if k == 3:
        names = ["tom_floor", "tom_mid", "tom_high"]
    elif k == 2:
        names = ["tom_floor", "tom_high"]
    else:                                          # a single tom: go by absolute pitch
        hz = float(np.exp(np.median(x)))
        names = ["tom_floor" if hz < cfg.tom_floor_max_hz else
                 "tom_mid" if hz < cfg.tom_mid_max_hz else "tom_high"]
    return [names[j] for j in assign]


def _velocity(strength: float, ref: float) -> int:
    return int(np.clip(40 + 70 * strength / (ref + 1e-9), 20, 127))


def classify_strokes(y: np.ndarray, strokes: list[dict], ctx: dict,
                     cfg: TranscribeConfig) -> list[Hit]:
    fps = ctx["fps"]
    feats = [_stroke_features(y, s["time"], cfg) for s in strokes]
    times = [s["time"] for s in strokes]

    def typical(key, band, pct):
        """Level of a solid hit in this band."""
        vals = [f[key] for f, s in zip(feats, strokes) if band in s["bands"]]
        return float(np.percentile(vals, pct)) if vals else 0.0

    # Low/mid: 75th percentile, because many strokes firing there are only the
    # faint click of another drum.  High: median, since most are real hi-hats.
    level = {"low": typical("p_low", "low", 75), "mid": typical("p_body", "mid", 75),
             "high": typical("p_high", "high", 50)}
    ref_strength = {b: float(np.median([s["bands"][b] for s in strokes if b in s["bands"]] or [1.0]))
                    for b in ("low", "mid", "high")}

    # What does this recording's kick look like?  It is the lowest-pitched
    # drum and the one with the least tone above 120 Hz, so take the low end
    # of both measures over all solid low-band strokes.
    solid = [f for f, s in zip(feats, strokes)
             if "low" in s["bands"] and f["p_low"] >= 0.3 * level["low"]]
    kick_hz = float(np.percentile([f["low_peak"] for f in solid], 25)) if solid else 60.0
    kick_tone = float(np.percentile([f["p_head"] / (f["p_low"] + 1e-12) for f in solid], 40)) \
        if solid else 0.05

    # The mid-band level must not be set by the kick: a kick is far louder
    # than a snare at 150-1000 Hz in some recordings, and soft snare strokes
    # would then be thrown out as "bleed".  So measure it without the kicks.
    def kick_guess(f):
        return (f["p_low"] >= cfg.min_rel_level * level["low"]
                and f["low_peak"] <= min(cfg.kick_abs_max_hz,
                                         cfg.kick_pitch_tol * max(kick_hz, cfg.kick_min_ref_hz))
                and f["p_head"] / (f["p_low"] + 1e-12) <= cfg.kick_tone_tol * kick_tone)
    non_kick = [f["p_body"] for f, s in zip(feats, strokes)
                if "mid" in s["bands"] and not kick_guess(f)]
    if len(non_kick) >= 4:
        level["mid"] = min(level["mid"], float(np.percentile(non_kick, 75)))

    # ---- pass 1: label every stroke ----------------------------------------
    labelled = []
    for k, (s, f) in enumerate(zip(strokes, feats)):
        b, t = s["bands"], s["time"]
        # Next stroke, ignoring anything within 70 ms (a crash swells slowly
        # and can re-trigger a band a moment after the stroke itself).
        t_next = next((u for u in times[k + 1:] if u - t >= 0.07), None)
        # faint energy in a band is bleed from another drum, not a hit
        low_fired = "low" in b or s["low_rise"] >= cfg.weak_rise * cfg.delta_low
        low_ok = low_fired and f["p_low"] >= cfg.min_rel_level * level["low"]
        mid_ok = "mid" in b and (f["p_body"] >= cfg.min_rel_level * level["mid"] or
                                 f["p_low"] >= cfg.min_rel_level * level["low"])   # deep toms
        high_ok = "high" in b and f["p_high"] >= cfg.min_rel_level * level["high"]
        tone = f["p_head"] / (f["p_low"] + 1e-12)

        # --- cymbal ---
        def cymbal_type(short_band="high"):
            wash = _ring(ctx["energy"]["wash"], fps, t, 0.22, 0.30, t_next)
            if wash is not None:
                return ("crash" if wash >= cfg.crash_min_ring else
                        "hihat_open" if wash >= cfg.open_min_ring else "hihat_closed")
            # fast playing: only a short look is possible
            short = _ring(ctx["energy"][short_band], fps, t, 0.07, 0.11, t_next)
            return "hihat_open" if short is not None and short >= cfg.open_min_ring_short \
                else "hihat_closed"

        cym = cymbal_type() if high_ok else None

        # --- kick ---
        kick = False
        if low_ok and f["low_peak"] <= cfg.kick_abs_max_hz:
            same_pitch = f["low_peak"] <= cfg.kick_pitch_tol * max(kick_hz, cfg.kick_min_ref_hz)
            # (a crash adds tone of its own, so skip the tone test under a crash)
            kick_like = tone <= cfg.kick_tone_tol * kick_tone or cym == "crash"
            kick = same_pitch and kick_like
        if "low" in b and low_ok and not kick:
            mid_ok = True                          # a low drum that is not the kick -> tom

        # --- snare / tom ---
        body = None
        if kick:
            # kick + snare together: far more drum tone than the kick has alone
            if mid_ok and tone >= cfg.snare_over_kick_tone * kick_tone and cym != "crash":
                body = "snare"
        elif mid_ok and f["p_head"] >= cfg.drum_min_tone * f["p_noise"]:
            # (the tone test above rejects cymbals that leak into the mid band)
            body = "snare" if f["sizzle_ratio"] >= cfg.snare_sizzle_ratio else "tom"

        # A snare's own ring reaches the cymbal band, so "open" is not
        # trustworthy on top of a snare; keep only closed or crash.
        if body == "snare" and cym == "hihat_open":
            cym = "hihat_closed"
        labelled.append(dict(time=t, kick=kick, body=body, cym=cym, f=f, b=b, weak=False,
                             ctype=cymbal_type, noisy=f["p_head"] < cfg.drum_min_tone * f["p_noise"]))

    # ---- pass 1b: cymbals the top band did not catch ------------------------
    # A stroke that is no drum at all and is mostly 2-8 kHz noise is a cymbal,
    # even if nothing happened above 7.5 kHz (dull recording, dark cymbal).
    orphans = [h for h in labelled if not (h["kick"] or h["body"] or h["cym"]) and h["noisy"]]
    if orphans:
        noise_ref = float(np.median([h["f"]["p_noise"] for h in orphans]))
        for h in orphans:
            if h["f"]["p_noise"] >= cfg.min_rel_level * noise_ref:
                h["cym"] = h["ctype"]("air")
                h["weak"] = True
    # If most cymbals had to be found that way, the recording is dull, and a
    # cymbal under a kick or snare cannot be heard directly either: assume one
    # wherever the 3-7.5 kHz band reacted, and let pass 2 keep it only where
    # the cymbal pattern is running on both sides.
    n_clear = sum(1 for h in labelled if h["cym"] and not h["weak"])
    n_weak = sum(1 for h in labelled if h["weak"])
    if n_weak > n_clear:
        for h in labelled:
            if (h["kick"] or h["body"]) and not h["cym"] and ("air" in h["b"] or "high" in h["b"]):
                h["cym"] = "hihat_closed"
                h["weak"] = True

    # ---- pass 2: sanity-check simultaneous hits -----------------------------
    # snare + cymbal: snare wires reach right up into the cymbal band, so a
    # snare alone looks like snare + hi-hat.  Keep the cymbal only where the
    # cymbal pattern is actually running on both sides of the snare.
    cym_times = np.array([h["time"] for h in labelled if h["cym"] and h["body"] != "snare"
                          and not (h["weak"] and (h["kick"] or h["body"]))])
    if len(cym_times) >= 2:
        window = float(np.clip(1.3 * np.median(np.diff(cym_times)), 0.15, 0.7))
        for h in labelled:
            if h["cym"] and (h["body"] == "snare" or (h["weak"] and (h["kick"] or h["body"]))):
                d = cym_times - h["time"]
                before = np.any((d < -0.03) & (d >= -window))
                after = np.any((d > 0.03) & (d <= window))
                if not (before and after) and h["cym"] != "crash":
                    h["cym"] = None

    # ---- pass 3: name the toms ----------------------------------------------
    toms = [h for h in labelled if h["body"] == "tom"]
    for h, name in zip(toms, _name_toms([h["f"]["tom_peak"] for h in toms], cfg)):
        h["body"] = name

    # ---- emit hits ----------------------------------------------------------
    hits: list[Hit] = []
    for h in labelled:
        b = h["b"]
        if h["kick"]:
            hits.append(Hit(h["time"], "kick", _velocity(b.get("low", 0), ref_strength["low"])))
        if h["body"]:
            strength = b.get("mid", b.get("low", 0))
            hits.append(Hit(h["time"], h["body"], _velocity(strength, ref_strength["mid"])))
        if h["cym"]:
            hits.append(Hit(h["time"], h["cym"], _velocity(b.get("high", 0), ref_strength["high"])))

    # One drum cannot be struck twice within a few ms: drop double triggers.
    hits.sort(key=lambda h: h.time)
    last: dict[str, float] = {}
    kept = []
    for h in hits:
        family = "tom" if h.drum.startswith("tom") else "hat" if h.drum.startswith("hihat") else h.drum
        if h.time - last.get(family, -1.0) >= cfg.retrigger_s:
            kept.append(h)
            last[family] = h.time
    return kept


def onset_envelope(y: np.ndarray, cfg: TranscribeConfig | None = None) -> tuple[np.ndarray, float]:
    """Overall "something was hit" curve used for beat tracking, and its frame rate.
    Independent of which model names the drums."""
    import librosa

    cfg = cfg or TranscribeConfig()
    S = np.abs(librosa.stft(y, n_fft=cfg.n_fft, hop_length=cfg.hop))
    return _flux(S.sum(axis=0), cfg.compress), cfg.sr / cfg.hop


def drop_retriggers(hits: list[Hit], cfg: TranscribeConfig | None = None) -> list[Hit]:
    """One drum cannot be struck twice within a few ms: keep the first."""
    cfg = cfg or TranscribeConfig()
    last: dict[str, float] = {}
    kept = []
    for h in sorted(hits, key=lambda h: h.time):
        family = "tom" if h.drum.startswith("tom") else "hat" if h.drum.startswith("hihat") else h.drum
        if h.time - last.get(family, -1.0) >= cfg.retrigger_s:
            kept.append(h)
            last[family] = h.time
    return kept


def refine_details(y: np.ndarray, hits: list[Hit], cfg: TranscribeConfig | None = None) -> list[Hit]:
    """Fill in what a coarse model leaves open.

    Models such as ADTOF only know "tom" and "hihat".  This turns
      "tom"   into tom_floor / tom_mid / tom_high (by pitch, relative to the
              other toms in the recording), and
      "hihat" into hihat_closed / hihat_open (by how long it rings),
    using the same measurements as the built-in rules.  Other hits pass through.
    """
    import librosa

    cfg = cfg or TranscribeConfig()
    if not any(h.drum in ("tom", "hihat") for h in hits):
        return hits
    S = np.abs(librosa.stft(y, n_fft=cfg.n_fft, hop_length=cfg.hop))
    freqs = librosa.fft_frequencies(sr=cfg.sr, n_fft=cfg.n_fft)
    fps = cfg.sr / cfg.hop
    wash, high = _band(S, freqs, *cfg.band_wash), _band(S, freqs, *cfg.band_high)
    times = sorted({round(h.time, 3) for h in hits})

    toms = [h for h in hits if h.drum == "tom"]
    pitches = [_stroke_features(y, h.time, cfg)["tom_peak"] for h in toms]
    for h, name in zip(toms, _name_toms(pitches, cfg)):
        h.drum = name
    for h in hits:
        if h.drum != "hihat":
            continue
        t_next = next((u for u in times if u - h.time >= 0.07), None)
        ring = _ring(wash, fps, h.time, 0.22, 0.30, t_next)
        if ring is not None:
            is_open = ring >= cfg.open_min_ring
        else:
            short = _ring(high, fps, h.time, 0.07, 0.11, t_next)
            is_open = short is not None and short >= cfg.open_min_ring_short
        # with a snare on top the ring is the snare's, not the hi-hat's
        with_snare = any(o.drum == "snare" and abs(o.time - h.time) < 0.03 for o in hits)
        h.drum = "hihat_open" if is_open and not with_snare else "hihat_closed"
    return hits


def transcribe(y: np.ndarray, cfg: TranscribeConfig | None = None,
               sensitivity: float = 1.0) -> tuple[list[Hit], np.ndarray, float]:
    """Full transcription.  Returns (hits, onset_envelope, envelope_frames_per_second).

    sensitivity > 1 finds more (quieter) strokes, < 1 fewer.
    """
    from dataclasses import replace

    cfg = cfg or TranscribeConfig()
    if sensitivity != 1.0:
        cfg = replace(cfg, delta_low=cfg.delta_low / sensitivity,
                      delta_mid=cfg.delta_mid / sensitivity,
                      delta_high=cfg.delta_high / sensitivity)
    # Pad the start so a hit at t=0 still has a "before" to rise from.
    pad_s = 0.25
    y = np.concatenate([np.zeros(int(pad_s * cfg.sr), dtype=y.dtype), y])
    strokes, ctx = detect_strokes(y, cfg)
    hits = classify_strokes(y, strokes, ctx, cfg) if strokes else []
    for h in hits:
        h.time = max(0.0, h.time - pad_s)
    env = ctx["onset_env"][int(round(pad_s * ctx["fps"])):]
    return hits, env, ctx["fps"]
