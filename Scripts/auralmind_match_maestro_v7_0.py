#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AuralMind Match — Maestro v7.0 (EXPERT)
Reference-based mastering with psychoacoustic-safe Match EQ, stem-aware processing (Demucs),
Key-Aware Harmonic Glow, true transient preservation, and a streaming-perfect True-Peak limiter.

Core goals:
- Zero “muffled” outcomes from darker references (air-preserve guardrails + freq-dependent strength curve).
- Transients survive loudness targets (snare transient detect + micro-attack restoration AFTER limiting).
- Vocals de-essed without killing hats/synth air (stem-aware mastering via Demucs, optional).
- Guaranteed <= target true-peak (oversampled true-peak chasing limiter + hard guarantee).

Dependencies:
  pip install numpy scipy soundfile librosa pyloudnorm matplotlib
Optional (stem-aware):
  pip install torch demucs

PowerShell example (your exact flow):
python auralmind_match_maestro_v7_0.py
  --reference "C:/Users/goku/Downloads/Brent Faiyaz - Pistachios [Official Video].mp3
  "
  --target   "C:/Users/goku/Downloads/Vegas - top teir (20).wav"
  --out      "C:/Users/goku/Desktop/Vegas_Top_Teir_MASTER_v7_0.wav"
  --report   "C:/Users/goku/Desktop/Vegas_Top_Teir_MASTER_v7_0_Report.md"
  --target_lufs -11.0 --target_peak_dbfs -1.0
  --enable_spatial --enable_movement
  --enable_key_glow --enable_transient_restore

"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import librosa
import pyloudnorm as pyln

from scipy.signal import (
    butter,
    filtfilt,
    firwin2,
    resample_poly,
    fftconvolve,
)
from scipy.ndimage import maximum_filter1d

# Optional plotting
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# Optional Demucs (stem-aware mastering)
try:
    import torch
    from demucs import pretrained
    from demucs.apply import apply_model

    _HAS_DEMUCS = True
except Exception:
    _HAS_DEMUCS = False


# =============================================================================
# Logging
# =============================================================================

LOG = logging.getLogger("auralmind")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")


# =============================================================================
# Utility
# =============================================================================


def db_to_lin(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def lin_to_db(x: float, eps: float = 1e-12) -> float:
    return float(20.0 * math.log10(max(eps, float(x))))


def ensure_finite(x: np.ndarray, name: str) -> np.ndarray:
    if not np.isfinite(x).all():
        LOG.warning("Non-finite values detected in %s; replacing with 0.", name)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def stereoize(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        x = np.stack([x, x], axis=1)
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    return x.astype(np.float32, copy=False)


def to_mono(x: np.ndarray) -> np.ndarray:
    x = stereoize(x)
    return 0.5 * (x[:, 0] + x[:, 1])


def max_abs(x: np.ndarray) -> float:
    return float(np.max(np.abs(x)))


def clamp01(v: float) -> float:
    return float(np.clip(v, 0.0, 1.0))


def safe_headroom(x: np.ndarray, headroom_db: float = -6.0) -> np.ndarray:
    """Ensure internal chain headroom (prevents accidental clipping during DSP stacking)."""
    x = stereoize(x)
    peak = max_abs(x)
    if peak < 1e-12:
        return x
    target = db_to_lin(headroom_db)
    if peak > target:
        x = x * (target / peak)
    return x.astype(np.float32)


# =============================================================================
# Filters
# =============================================================================


def butter_highpass(sr: int, hz: float, order: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    nyq = 0.5 * sr
    w = np.clip(hz / nyq, 1e-6, 0.999999)
    return butter(order, w, btype="highpass")


def butter_lowpass(sr: int, hz: float, order: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    nyq = 0.5 * sr
    w = np.clip(hz / nyq, 1e-6, 0.999999)
    return butter(order, w, btype="lowpass")


def apply_iir(x: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    x = stereoize(x)
    y0 = filtfilt(b, a, x[:, 0]).astype(np.float32)
    y1 = filtfilt(b, a, x[:, 1]).astype(np.float32)
    return np.stack([y0, y1], axis=1)


def peaking_biquad(sr: int, f0: float, q: float, gain_db: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Classic peaking EQ biquad (RBJ Audio EQ Cookbook).
    Returns (b, a).
    """
    f0 = float(max(10.0, min(0.49 * sr, f0)))
    q = float(max(0.1, q))
    A = float(10.0 ** (gain_db / 40.0))  # sqrt of linear gain
    w0 = 2.0 * math.pi * (f0 / sr)
    alpha = math.sin(w0) / (2.0 * q)

    b0 = 1.0 + alpha * A
    b1 = -2.0 * math.cos(w0)
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * math.cos(w0)
    a2 = 1.0 - alpha / A

    b = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float32)
    a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float32)
    return b, a


def apply_peaking_bank(x: np.ndarray, sr: int, bands: List[Tuple[float, float, float]]) -> np.ndarray:
    """
    bands: list of (f0, Q, gain_db).
    Cascades biquads.
    """
    y = stereoize(x)
    for (f0, q, g) in bands:
        b, a = peaking_biquad(sr, f0, q, g)
        y = apply_iir(y, b, a)
    return y.astype(np.float32)


# =============================================================================
# Spectral analysis + Match EQ
# =============================================================================


def average_spectrum_db_librosa(
    x_mono: np.ndarray, sr: int, n_fft: int = 8192, hop: int = 2048
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Efficient average magnitude spectrum using librosa STFT.
    """
    x_mono = ensure_finite(x_mono.astype(np.float32, copy=False), "avg_spec_mono")
    S = np.abs(librosa.stft(x_mono, n_fft=n_fft, hop_length=hop, window="hann", center=True))
    mag = np.mean(S, axis=1).astype(np.float32)
    mag_db = (20.0 * np.log10(mag + 1e-12)).astype(np.float32)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft).astype(np.float32)
    return freqs, mag_db


def smooth_curve_db(freqs: np.ndarray, db_curve: np.ndarray, smooth_hz: float) -> np.ndarray:
    if smooth_hz <= 0:
        return db_curve.astype(np.float32)
    freqs = freqs.astype(np.float32)
    db_curve = db_curve.astype(np.float32)
    df = float(np.median(np.diff(freqs)) + 1e-12)
    win = int(max(5, round(smooth_hz / df)))
    if win % 2 == 0:
        win += 1
    pad = win // 2
    x = np.pad(db_curve, (pad, pad), mode="edge")
    kernel = np.ones(win, dtype=np.float32) / float(win)
    return np.convolve(x, kernel, mode="valid").astype(np.float32)


def _air_preserve_guardrails(freqs: np.ndarray, delta_db: np.ndarray) -> np.ndarray:
    """
    Prevent 'muffling' by limiting negative EQ moves in the air band and ultra-high.
    """
    f = freqs.astype(np.float32)
    d = delta_db.astype(np.float32).copy()

    # Deep subs: avoid over-cutting fundamental translation
    sub = f < 35.0
    if np.any(sub):
        d[sub] = np.maximum(d[sub], -3.0)

    # Low bass: prevent reference-chasing from flattening punch
    low = (f >= 35.0) & (f < 80.0)
    if np.any(low):
        d[low] = np.clip(d[low], -4.0, 4.0)

    # Air band: do not over-reduce above 8k
    air = f >= 8000.0
    if np.any(air):
        d[air] = np.clip(d[air], -2.0, 4.0)

    # “Sparkle” above 12k: ultra-conservative on cuts
    sparkle = f >= 12000.0
    if np.any(sparkle):
        d[sparkle] = np.clip(d[sparkle], -1.5, 3.0)

    return d.astype(np.float32)


def match_strength_curve(
    freqs: np.ndarray,
    base_strength: float,
    lo_hz: float = 80.0,
    hi_hz: float = 10000.0,
    lo_factor: float = 0.55,
    hi_factor: float = 0.60,
) -> np.ndarray:
    """
    Smooth frequency-dependent match EQ strength:
      - weaker below lo_hz (subs)
      - weaker above hi_hz (air)
    """
    f = np.maximum(freqs.astype(np.float32), 1.0)
    base = float(np.clip(base_strength, 0.0, 1.5))

    def smoothstep(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, 0.0, 1.0).astype(np.float32)
        return x * x * (3.0 - 2.0 * x)

    # low taper
    lo_hz = float(max(30.0, lo_hz))
    lo_factor = float(np.clip(lo_factor, 0.0, 1.0))
    low = f <= lo_hz
    s = np.full_like(f, base, dtype=np.float32)
    if np.any(low):
        t = (np.log2(f[low]) - math.log2(20.0)) / (math.log2(lo_hz) - math.log2(20.0) + 1e-6)
        t = smoothstep(t)
        s[low] = base * (lo_factor + (1.0 - lo_factor) * t)

    # high taper
    hi_hz = float(max(2000.0, hi_hz))
    hi_factor = float(np.clip(hi_factor, 0.0, 1.0))
    high = f >= hi_hz
    if np.any(high):
        nyq = float(np.max(f))
        t = (np.log2(f[high]) - math.log2(hi_hz)) / (math.log2(max(hi_hz * 1.5, nyq)) - math.log2(hi_hz) + 1e-6)
        t = smoothstep(t)
        s[high] = base * ((1.0 - t) + hi_factor * t)

    return s.astype(np.float32)


def minimum_phase_fir_from_mag(mag: np.ndarray, nfft: int) -> np.ndarray:
    """
    Minimum-phase impulse via real-cepstrum trick.
    """
    mag = np.maximum(mag.astype(np.float32), 1e-8)
    log_mag = np.log(mag)
    cep = np.fft.irfft(log_mag, nfft).real.astype(np.float32)

    cep_min = np.zeros_like(cep)
    cep_min[0] = cep[0]
    cep_min[1 : nfft // 2] = 2.0 * cep[1 : nfft // 2]
    if nfft % 2 == 0:
        cep_min[nfft // 2] = cep[nfft // 2]

    spec_min = np.fft.rfft(cep_min)
    H_min = np.exp(spec_min).astype(np.complex64)
    h = np.fft.irfft(H_min, nfft).real.astype(np.float32)
    return h


def design_match_fir(
    ref_mono: np.ndarray,
    tgt_mono: np.ndarray,
    sr: int,
    numtaps: int,
    max_gain_db: float,
    smooth_hz: float,
    eq_phase: str,
    minphase_nfft: int,
    match_strength: float,
    match_lo_hz: float,
    match_hi_hz: float,
    match_lo_factor: float,
    match_hi_factor: float,
    enable_guardrails: bool = True,
) -> Tuple[np.ndarray, Dict]:
    """
    Psychoacoustic-safe match EQ FIR with:
    - smoothed delta curve
    - air-preserve guardrails
    - frequency-dependent strength curve
    - minimum-phase option to reduce pre-ringing
    """
    freqs, ref_db = average_spectrum_db_librosa(ref_mono, sr=sr)
    _, tgt_db = average_spectrum_db_librosa(tgt_mono, sr=sr)

    delta_db = (ref_db - tgt_db).astype(np.float32)
    delta_db = np.clip(delta_db, -float(max_gain_db), float(max_gain_db)).astype(np.float32)
    delta_db = smooth_curve_db(freqs, delta_db, smooth_hz=smooth_hz)

    if enable_guardrails:
        delta_db = _air_preserve_guardrails(freqs, delta_db)

    strength = match_strength_curve(
        freqs,
        base_strength=match_strength,
        lo_hz=match_lo_hz,
        hi_hz=match_hi_hz,
        lo_factor=match_lo_factor,
        hi_factor=match_hi_factor,
    )
    delta_db = (delta_db * strength).astype(np.float32)

    desired_mag = (10.0 ** (delta_db / 20.0)).astype(np.float32)

    nyq = 0.5 * sr
    f_norm = np.clip(freqs / nyq, 0.0, 1.0).astype(np.float32)
    f_norm[0] = 0.0
    f_norm[-1] = 1.0

    # FIR design
    numtaps = int(max(513, numtaps))
    if numtaps % 2 == 0:
        numtaps += 1

    h = firwin2(numtaps, f_norm, desired_mag, window="hann").astype(np.float32)

    if eq_phase.lower() == "minimum":
        nfft = int(max(4096, minphase_nfft))
        H = np.abs(np.fft.rfft(h, nfft)).astype(np.float32)
        h_mp = minimum_phase_fir_from_mag(H, nfft=nfft)
        if len(h_mp) > numtaps:
            h_mp = h_mp[:numtaps]
        h = h_mp.astype(np.float32)

    # DC normalization
    dc = float(np.sum(h))
    if abs(dc) > 1e-9:
        h = (h / dc).astype(np.float32)

    # Capture a downsampled EQ curve for plotting/report
    idx = np.linspace(0, len(freqs) - 1, 512).astype(int)
    info = {
        "enabled": True,
        "numtaps": int(numtaps),
        "max_gain_db": float(max_gain_db),
        "smooth_hz": float(smooth_hz),
        "eq_phase": str(eq_phase),
        "minphase_nfft": int(minphase_nfft),
        "match_strength": float(match_strength),
        "match_lo_hz": float(match_lo_hz),
        "match_hi_hz": float(match_hi_hz),
        "match_lo_factor": float(match_lo_factor),
        "match_hi_factor": float(match_hi_factor),
        "guardrails": bool(enable_guardrails),
        "eq_curve": {
            "freqs_hz": freqs[idx].astype(float).tolist(),
            "delta_db": delta_db[idx].astype(float).tolist(),
        },
    }
    return h, info


def apply_fir_fft(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    FFT convolution for speed vs np.convolve (especially on long tracks / larger taps).
    """
    x = stereoize(x)
    h = h.astype(np.float32)
    y0 = fftconvolve(x[:, 0], h, mode="same").astype(np.float32)
    y1 = fftconvolve(x[:, 1], h, mode="same").astype(np.float32)
    return np.stack([y0, y1], axis=1)


# =============================================================================
# Stem separation (Demucs) — optional
# =============================================================================


def demucs_separate_stems(
    audio: np.ndarray,
    sr: int,
    model_name: str = "htdemucs",
    device: str = "cpu",
    split: bool = True,
    overlap: float = 0.25,
) -> Tuple[Dict[str, np.ndarray], Dict]:
    """
    Separate target into stems using Demucs:
      sources typically: ['drums','bass','other','vocals']
    Returns dict of stereo stems at original sr.
    """
    if not _HAS_DEMUCS:
        raise RuntimeError("Demucs/Torch not available. Install: pip install torch demucs")

    x = stereoize(audio)
    model = pretrained.get_model(model_name)
    model.eval()

    # Demucs models have a native sample rate
    model_sr = int(getattr(model, "samplerate", sr))
    x_rs = x
    if model_sr != sr:
        x_rs = resample_poly(x, model_sr, sr, axis=0).astype(np.float32)

    wav = torch.from_numpy(x_rs.T).unsqueeze(0)  # [1, C, T]
    dev = torch.device(device)

    try:
        model.to(dev)
        wav = wav.to(dev)
        with torch.no_grad():
            sources = apply_model(
                model,
                wav,
                device=dev,
                split=split,
                overlap=float(overlap),
                progress=False,
            )
    finally:
        # Keep memory predictable on CUDA
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    # sources: [1, S, C, T]
    src_names = list(getattr(model, "sources", []))
    if not src_names:
        # fallback naming
        src_names = [f"stem_{i}" for i in range(int(sources.shape[1]))]

    stems_rs: Dict[str, np.ndarray] = {}
    for i, name in enumerate(src_names):
        stem = sources[0, i].detach().cpu().numpy().T.astype(np.float32)  # [T, C]
        stems_rs[name] = stereoize(stem)

    # Resample back to original sr
    stems: Dict[str, np.ndarray] = {}
    if model_sr != sr:
        for k, v in stems_rs.items():
            stems[k] = resample_poly(v, sr, model_sr, axis=0).astype(np.float32)
    else:
        stems = stems_rs

    info = {
        "enabled": True,
        "model": str(model_name),
        "device": str(device),
        "model_sr": int(model_sr),
        "split": bool(split),
        "overlap": float(overlap),
        "stems": list(stems.keys()),
    }
    return stems, info


# =============================================================================
# De-esser (stem-aware focus)
# =============================================================================


def dynamic_deesser(
    audio: np.ndarray,
    sr: int,
    band_low: float = 6000.0,
    band_high: float = 10000.0,
    threshold_db: float = -22.0,
    ratio: float = 2.0,
    attack_ms: float = 2.0,
    release_ms: float = 60.0,
) -> Tuple[np.ndarray, Dict]:
    """
    Band-limited dynamic attenuation for sibilance.
    """
    x = stereoize(audio)

    # bandpass: HP -> LP
    b1, a1 = butter_highpass(sr, band_low, order=2)
    b2, a2 = butter_lowpass(sr, band_high, order=2)
    band = apply_iir(apply_iir(x, b1, a1), b2, a2)
    rest = x - band

    env = np.maximum(np.abs(band[:, 0]), np.abs(band[:, 1])).astype(np.float32)
    eps = 1e-12

    att = math.exp(-1.0 / (sr * (attack_ms / 1000.0) + eps))
    rel = math.exp(-1.0 / (sr * (release_ms / 1000.0) + eps))

    e = 0.0
    env_s = np.zeros_like(env)
    for i in range(len(env)):
        v = float(env[i])
        if v > e:
            e = v + att * (e - v)
        else:
            e = v + rel * (e - v)
        env_s[i] = e

    thr = db_to_lin(threshold_db)
    gr = np.ones_like(env_s, dtype=np.float32)
    over = env_s > thr
    if np.any(over):
        gr[over] = (thr / (env_s[over] + eps)) ** (1.0 - 1.0 / float(max(1.01, ratio)))

    band_c = band.copy()
    band_c[:, 0] *= gr
    band_c[:, 1] *= gr

    y = rest + band_c
    info = {
        "enabled": True,
        "band_low": float(band_low),
        "band_high": float(band_high),
        "threshold_db": float(threshold_db),
        "ratio": float(ratio),
        "min_gain_db": float(lin_to_db(float(np.min(gr)))),
    }
    return y.astype(np.float32), info


# =============================================================================
# Key detection + Key-Aware Harmonic Glow
# =============================================================================


_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=np.float32)
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=np.float32)
_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def estimate_key_ks(
    x_mono: np.ndarray,
    sr: int,
    hop: int = 2048,
) -> Dict:
    """
    Krumhansl-Schmuckler key estimation using average chroma (CQT chroma).
    """
    x_mono = ensure_finite(x_mono.astype(np.float32, copy=False), "key_est_mono")
    chroma = librosa.feature.chroma_cqt(y=x_mono, sr=sr, hop_length=hop)
    chroma_mean = np.mean(chroma, axis=1).astype(np.float32)
    chroma_mean = chroma_mean / (np.sum(chroma_mean) + 1e-12)

    # Compare to rotated profiles
    scores = []
    for k in range(12):
        maj = np.roll(_MAJOR_PROFILE, k)
        minr = np.roll(_MINOR_PROFILE, k)
        maj = maj / (np.sum(maj) + 1e-12)
        minr = minr / (np.sum(minr) + 1e-12)
        smaj = float(np.dot(chroma_mean, maj))
        smin = float(np.dot(chroma_mean, minr))
        scores.append((k, smaj, smin))

    best = max(scores, key=lambda t: max(t[1], t[2]))
    k_idx, smaj, smin = best
    mode = "major" if smaj >= smin else "minor"
    tonic = _NOTE_NAMES[k_idx]
    key_name = f"{tonic} {mode}"

    # scale degrees in semitones from tonic
    if mode == "major":
        degrees = [0, 2, 4, 5, 7, 9, 11]
    else:
        degrees = [0, 2, 3, 5, 7, 8, 10]

    scale_pitch_classes = sorted([(k_idx + d) % 12 for d in degrees])
    scale_notes = [_NOTE_NAMES[i] for i in scale_pitch_classes]

    return {
        "key": key_name,
        "tonic": tonic,
        "mode": mode,
        "score_major": float(smaj),
        "score_minor": float(smin),
        "scale_pitch_classes": scale_pitch_classes,
        "scale_notes": scale_notes,
    }


def build_scale_glow_bands(
    sr: int,
    tonic_idx: int,
    mode: str,
    base_gain_db: float = 0.8,
    q: float = 7.0,
    f_min: float = 120.0,
    f_max: float = 8000.0,
) -> List[Tuple[float, float, float]]:
    """
    Construct a small peaking EQ bank that boosts partials aligned to the detected scale.
    The effect is a “harmonic glow”: subtle overtone lift that feels *in-key* and musical.

    Strategy:
      - For each scale degree, create peaks across octaves within [f_min, f_max].
      - Keep boost tiny (0.4..1.2 dB) and let Q keep it elegant (no harshness).
    """
    # tonic pitch class -> midi note mapping anchor (C4=60)
    # We'll generate frequencies for pitch classes across octaves:
    # Use MIDI note numbers and convert to Hz via librosa.midi_to_hz.
    if mode == "major":
        degrees = [0, 2, 4, 5, 7, 9, 11]
    else:
        degrees = [0, 2, 3, 5, 7, 8, 10]

    pitch_classes = [(tonic_idx + d) % 12 for d in degrees]
    bands: List[Tuple[float, float, float]] = []

    # Octave sweep: MIDI 36 (C2) to MIDI 108 (C8)
    for midi in range(36, 109):
        pc = midi % 12
        if pc in pitch_classes:
            f0 = float(librosa.midi_to_hz(midi))
            if f0 < f_min or f0 > f_max:
                continue
            # gentle psychoacoustic tilt: emphasize upper mids slightly
            # (where "glow" feels like presence rather than mud)
            weight = 0.75 + 0.25 * np.clip((f0 - 200.0) / 2500.0, 0.0, 1.0)
            gain = float(np.clip(base_gain_db * weight, 0.35, 1.25))
            bands.append((f0, float(q), gain))

    # Reduce density: keep only the most useful “glow band anchors”
    # by choosing a sparse set across the spectrum.
    if len(bands) > 18:
        # pick every Nth by log spacing
        freqs = np.array([b[0] for b in bands], dtype=np.float32)
        order = np.argsort(freqs)
        bands = [bands[i] for i in order.tolist()]
        keep = []
        step = max(1, len(bands) // 18)
        for i in range(0, len(bands), step):
            keep.append(bands[i])
        bands = keep[:18]

    return bands


def key_aware_harmonic_glow(
    audio: np.ndarray,
    sr: int,
    detected_key: Dict,
    glow_gain_db: float = 0.8,
    glow_q: float = 7.0,
    mix: float = 0.55,
) -> Tuple[np.ndarray, Dict]:
    """
    Apply subtle in-key harmonic glow using a sparse peaking EQ bank.
    """
    x = stereoize(audio)
    tonic = str(detected_key.get("tonic", "C"))
    mode = str(detected_key.get("mode", "major"))
    tonic_idx = _NOTE_NAMES.index(tonic) if tonic in _NOTE_NAMES else 0

    bands = build_scale_glow_bands(
        sr=sr,
        tonic_idx=tonic_idx,
        mode=mode,
        base_gain_db=float(glow_gain_db),
        q=float(glow_q),
        f_min=140.0,
        f_max=7800.0,
    )

    if not bands:
        return x, {"enabled": False, "reason": "no_bands"}

    y = apply_peaking_bank(x, sr=sr, bands=bands)
    y = (1.0 - float(clamp01(mix))) * x + float(clamp01(mix)) * y

    info = {
        "enabled": True,
        "key": str(detected_key.get("key", "")),
        "glow_gain_db": float(glow_gain_db),
        "glow_q": float(glow_q),
        "mix": float(mix),
        "bands": [{"f0": float(f0), "q": float(q), "gain_db": float(g)} for (f0, q, g) in bands],
    }
    return y.astype(np.float32), info


# =============================================================================
# Stereo spatial + movement
# =============================================================================


def mid_side(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = stereoize(x)
    mid = 0.5 * (x[:, 0] + x[:, 1])
    side = 0.5 * (x[:, 0] - x[:, 1])
    return mid.astype(np.float32), side.astype(np.float32)


def from_mid_side(mid: np.ndarray, side: np.ndarray) -> np.ndarray:
    l = (mid + side).astype(np.float32)
    r = (mid - side).astype(np.float32)
    return np.stack([l, r], axis=1)


def stereo_widener_freq(
    audio: np.ndarray,
    sr: int,
    side_hp_hz: float = 180.0,
    width_mid: float = 1.07,
    width_hi: float = 1.25,
    corr_min: float = 0.0,
) -> Tuple[np.ndarray, Dict]:
    x = stereoize(audio)
    mid, side = mid_side(x)

    b, a = butter_highpass(sr, side_hp_hz, order=2)
    side_hp = filtfilt(b, a, side).astype(np.float32)
    side_lp = (side - side_hp).astype(np.float32)

    side_out = side_lp + side_hp * float(width_hi)
    mid_out = mid * float(width_mid)
    y = from_mid_side(mid_out, side_out)

    corr = float(np.corrcoef(y[:, 0], y[:, 1])[0, 1] if len(y) > 10 else 1.0)
    if corr < corr_min:
        alpha = float(np.clip((corr_min - corr) / (abs(corr_min) + 1e-6), 0.0, 1.0))
        y = (1.0 - alpha) * y + alpha * x

    return y.astype(np.float32), {
        "enabled": True,
        "side_hp_hz": float(side_hp_hz),
        "width_mid": float(width_mid),
        "width_hi": float(width_hi),
        "corr": float(corr),
        "corr_min": float(corr_min),
    }


def movement_automation(audio: np.ndarray, sr: int, amount: float = 0.12) -> Tuple[np.ndarray, Dict]:
    """
    Section-aware movement: micro width modulation driven by mid energy envelope.
    Subtle enough for mastering, but adds “alive” shimmer in melodic-trap textures.
    """
    x = stereoize(audio)
    mid, side = mid_side(x)

    env = np.abs(mid)
    win = int(max(16, round(sr * 0.03)))
    kernel = np.ones(win, dtype=np.float32) / float(win)
    env_s = np.convolve(env, kernel, mode="same").astype(np.float32)
    env_n = (env_s - np.min(env_s)) / (float(np.max(env_s) - np.min(env_s) + 1e-12))
    mod = (1.0 + float(amount) * (env_n - 0.5)).astype(np.float32)

    y = from_mid_side(mid, side * mod)
    return y.astype(np.float32), {"enabled": True, "amount": float(amount)}


# =============================================================================
# Soft clip (pre-limiter)
# =============================================================================


def oversampled_soft_clip(
    audio: np.ndarray,
    drive_db: float = 2.2,
    mix: float = 0.20,
    oversample: int = 4,
) -> np.ndarray:
    x = stereoize(audio)
    os = int(max(1, oversample))
    drive = db_to_lin(drive_db)
    eps = 1e-12

    if os > 1:
        xo = resample_poly(x, os, 1, axis=0).astype(np.float32)
    else:
        xo = x

    sat = np.tanh(xo * drive).astype(np.float32) / float(np.tanh(drive) + eps)
    yo = (1.0 - float(clamp01(mix))) * xo + float(clamp01(mix)) * sat

    if os > 1:
        y = resample_poly(yo, 1, os, axis=0).astype(np.float32)
    else:
        y = yo.astype(np.float32)

    return y


# =============================================================================
# True transient preservation: snare detect + micro-attack restore
# =============================================================================


def detect_snare_transients(
    drums: np.ndarray,
    sr: int,
    band_low: float = 1300.0,
    band_high: float = 5200.0,
    hop: int = 256,
    pre_max: int = 6,
    post_max: int = 6,
    delta: float = 0.20,
    wait: int = 12,
) -> Tuple[np.ndarray, Dict]:
    """
    Detect transient positions (snare-like peaks) on drums stem via onset strength in a high-mid band.
    Returns sample indices (int array).
    """
    d = stereoize(drums)
    dm = to_mono(d)

    # band focus
    b1, a1 = butter_highpass(sr, band_low, order=2)
    b2, a2 = butter_lowpass(sr, band_high, order=2)
    band = filtfilt(b1, a1, dm).astype(np.float32)
    band = filtfilt(b2, a2, band).astype(np.float32)

    onset_env = librosa.onset.onset_strength(y=band, sr=sr, hop_length=hop)
    # Peak picking
    peaks = librosa.util.peak_pick(
        onset_env,
        pre_max=pre_max,
        post_max=post_max,
        pre_avg=pre_max,
        post_avg=post_max,
        delta=delta,
        wait=wait,
    )
    times = librosa.frames_to_samples(peaks, hop_length=hop).astype(np.int64)
    info = {
        "enabled": True,
        "count": int(times.size),
        "band_low": float(band_low),
        "band_high": float(band_high),
        "hop": int(hop),
    }
    return times, info


def micro_attack_restore(
    audio: np.ndarray,
    sr: int,
    transient_samples: np.ndarray,
    restore_db: float = 1.2,
    band_low: float = 1500.0,
    band_high: float = 6500.0,
    attack_ms: float = 1.4,
    decay_ms: float = 38.0,
    mix: float = 0.60,
) -> Tuple[np.ndarray, Dict]:
    """
    Post-limiter micro-attack restoration:
      - isolates a presence band (snare crack + consonant snap)
      - injects a short transient-shaped boost on detected hits

    This is a mastering-grade “punch return” layer:
      it restores articulation without raising fullband peaks dangerously.
    """
    x = stereoize(audio)
    if transient_samples.size == 0:
        return x, {"enabled": False, "reason": "no_transients"}

    # isolate high-mid band
    b1, a1 = butter_highpass(sr, band_low, order=2)
    b2, a2 = butter_lowpass(sr, band_high, order=2)
    band = apply_iir(apply_iir(x, b1, a1), b2, a2)

    env = np.zeros((len(x),), dtype=np.float32)
    att_n = int(max(1, round(sr * (attack_ms / 1000.0))))
    dec_n = int(max(att_n + 4, round(sr * (decay_ms / 1000.0))))

    # build transient envelope as exponential-ish pulse
    peak_gain = db_to_lin(restore_db) - 1.0  # extra gain amount
    peak_gain = float(np.clip(peak_gain, 0.0, 0.35))  # safe bound

    for t in transient_samples:
        t = int(t)
        if t < 0 or t >= len(env):
            continue
        # attack ramp
        a0 = t
        a1s = min(len(env), t + att_n)
        if a1s > a0:
            env[a0:a1s] = np.maximum(env[a0:a1s], np.linspace(0.0, peak_gain, a1s - a0, dtype=np.float32))
        # decay
        d0 = a1s
        d1s = min(len(env), t + dec_n)
        if d1s > d0:
            decay = np.linspace(peak_gain, 0.0, d1s - d0, dtype=np.float32)
            env[d0:d1s] = np.maximum(env[d0:d1s], decay)

    # Apply envelope to band (additive boost)
    y = x.copy()
    boost = band.copy()
    boost[:, 0] *= env
    boost[:, 1] *= env

    mix = float(clamp01(mix))
    y = y + mix * boost

    info = {
        "enabled": True,
        "restore_db": float(restore_db),
        "band_low": float(band_low),
        "band_high": float(band_high),
        "attack_ms": float(attack_ms),
        "decay_ms": float(decay_ms),
        "mix": float(mix),
        "transients": int(transient_samples.size),
    }
    return y.astype(np.float32), info


# =============================================================================
# Loudness + True-Peak limiter (streaming-perfect)
# =============================================================================


def measure_lufs(audio: np.ndarray, sr: int) -> float:
    meter = pyln.Meter(sr)
    return float(meter.integrated_loudness(stereoize(audio)))


def true_peak_lin(audio: np.ndarray, oversample: int = 8) -> float:
    x = stereoize(audio)
    os = int(max(1, oversample))
    if os > 1:
        xo = resample_poly(x, os, 1, axis=0)
    else:
        xo = x
    return float(np.max(np.abs(xo)))


def true_peak_chasing_limiter(
    audio: np.ndarray,
    sr: int,
    ceiling_dbfs: float = -1.0,
    tp_oversample: int = 8,
    lookahead_ms: float = 2.7,
    attack_ms: float = 0.25,
    release_ms: float = 70.0,
    ceiling_chase_strength: float = 1.0,
) -> Tuple[np.ndarray, Dict]:
    """
    Oversampled true-peak chasing limiter with a hard guarantee <= ceiling.
    """
    x = stereoize(ensure_finite(audio, "tp_limit_in"))
    os = int(max(1, tp_oversample))
    ceiling = db_to_lin(ceiling_dbfs)
    eps = 1e-12

    if os > 1:
        y_os = resample_poly(x, os, 1, axis=0).astype(np.float32)
    else:
        y_os = x.astype(np.float32, copy=True)

    os_sr = float(sr * os)
    peak = np.max(np.abs(y_os), axis=1).astype(np.float32)

    la = int(max(1, round(os_sr * (lookahead_ms / 1000.0))))
    if la > 1:
        peak_la = maximum_filter1d(peak, size=la, mode="nearest").astype(np.float32)
    else:
        peak_la = peak

    desired = np.minimum(1.0, ceiling / (peak_la + eps)).astype(np.float32)

    # Smooth gain
    a_att = math.exp(-1.0 / (os_sr * (max(0.05, attack_ms) / 1000.0) + eps))
    a_rel = math.exp(-1.0 / (os_sr * (max(attack_ms, release_ms) / 1000.0) + eps))

    g = np.ones_like(desired, dtype=np.float32)
    cur = 1.0
    chase = float(np.clip(ceiling_chase_strength, 0.25, 1.5))

    for i in range(len(desired)):
        d = float(desired[i])
        # slight “ceiling chase” bias: if we're close to ceiling, lean tighter
        d = d ** chase
        if d < cur:
            cur = d + a_att * (cur - d)
        else:
            cur = d + a_rel * (cur - d)
        g[i] = cur

    y_os[:, 0] *= g
    y_os[:, 1] *= g

    if os > 1:
        y = resample_poly(y_os, 1, os, axis=0).astype(np.float32)
    else:
        y = y_os.astype(np.float32)

    # HARD guarantee (rare tiny overshoot due to resampling edge)
    tp = true_peak_lin(y, oversample=os)
    if tp > ceiling:
        y *= (ceiling / (tp + eps))

    info = {
        "enabled": True,
        "ceiling_dbfs": float(ceiling_dbfs),
        "tp_oversample": int(tp_oversample),
        "min_gain_db": float(lin_to_db(float(np.min(g)))),
        "true_peak_dbfs": float(lin_to_db(true_peak_lin(y, oversample=os))),
        "ceiling_chase_strength": float(chase),
    }
    return y.astype(np.float32), info


def finalize_master(
    audio: np.ndarray,
    sr: int,
    target_lufs: float = -11.0,
    target_peak_dbfs: float = -1.0,
    tp_oversample: int = 8,
    iters: int = 3,
    clip_drive_db: float = 2.2,
    clip_mix: float = 0.20,
    clip_oversample: int = 4,
    limit_lookahead_ms: float = 2.7,
    limit_attack_ms: float = 0.25,
    limit_release_ms: float = 70.0,
    ceiling_chase_strength: float = 1.0,
) -> Tuple[np.ndarray, Dict]:
    """
    Iteratively converge to LUFS while enforcing true peak ceiling.
    """
    x = stereoize(audio)
    g_total_db = 0.0
    limiter_info: Dict = {}

    iters = int(max(1, iters))
    for _ in range(iters):
        cur_lufs = measure_lufs(x, sr)
        gain_db = float(target_lufs - cur_lufs)
        x *= db_to_lin(gain_db)
        g_total_db += gain_db

        x = oversampled_soft_clip(x, drive_db=clip_drive_db, mix=clip_mix, oversample=clip_oversample)

        x, limiter_info = true_peak_chasing_limiter(
            x,
            sr=sr,
            ceiling_dbfs=target_peak_dbfs,
            tp_oversample=tp_oversample,
            lookahead_ms=limit_lookahead_ms,
            attack_ms=limit_attack_ms,
            release_ms=limit_release_ms,
            ceiling_chase_strength=ceiling_chase_strength,
        )

    info = {
        "enabled": True,
        "target_lufs": float(target_lufs),
        "final_lufs": float(measure_lufs(x, sr)),
        "gain_db_total": float(g_total_db),
        "target_peak_dbfs": float(target_peak_dbfs),
        "final_true_peak_dbfs": float(lin_to_db(true_peak_lin(x, oversample=tp_oversample))),
        "iters": int(iters),
        "limiter": limiter_info,
    }
    return x.astype(np.float32), info


# =============================================================================
# Report helpers (plots)
# =============================================================================


def plot_eq_delta(report_path: Path, eq_info: Dict) -> Optional[str]:
    if plt is None:
        return None
    curve = (eq_info or {}).get("eq_curve")
    if not curve:
        return None

    freqs = np.asarray(curve.get("freqs_hz", []), dtype=np.float32)
    delta = np.asarray(curve.get("delta_db", []), dtype=np.float32)
    if freqs.size == 0 or delta.size == 0 or freqs.size != delta.size:
        return None

    png = report_path.with_name(f"{report_path.stem}_EQ_Delta.png")
    plt.figure(figsize=(9.5, 4.0))
    plt.semilogx(freqs, delta)
    plt.axhline(0.0, linewidth=1.0)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("EQ Delta (dB)")
    plt.title("Match EQ Delta (guardrails + freq-strength curve)")
    plt.grid(True, which="both", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(png, dpi=170)
    plt.close()
    return png.name


def plot_spectrum_overlay(report_path: Path, sr: int, pre: np.ndarray, post: np.ndarray) -> Optional[str]:
    if plt is None:
        return None
    f1, pre_db = average_spectrum_db_librosa(to_mono(pre), sr=sr)
    f2, post_db = average_spectrum_db_librosa(to_mono(post), sr=sr)

    png = report_path.with_name(f"{report_path.stem}_Spectrum_Overlay.png")
    plt.figure(figsize=(9.5, 4.0))
    plt.semilogx(f1, pre_db, label="Pre")
    plt.semilogx(f2, post_db, label="Post")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("Average Spectrum — Pre vs Post")
    plt.grid(True, which="both", linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png, dpi=170)
    plt.close()
    return png.name


def write_report(path: Path, payload: Dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    md = []
    md.append("# Enhancement Report — AuralMind Maestro v7.0\n")

    plots = payload.get("plots", {}) or {}
    if "eq_delta" in plots:
        md.append("## Match EQ Delta Curve\n")
        md.append(f"![EQ Delta]({plots['eq_delta']})\n")
    if "spectrum_overlay" in plots:
        md.append("## Spectrum Overlay\n")
        md.append(f"![Spectrum]({plots['spectrum_overlay']})\n")

    md.append("## Debug JSON\n")
    md.append("```json")
    md.append(json.dumps(payload, indent=2))
    md.append("```")

    path.write_text("\n".join(md), encoding="utf-8")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    ap = argparse.ArgumentParser(description="AuralMind Maestro v7.0 — expert mastering (stems + key glow + TP limiter + transient restore)")

    ap.add_argument("--reference", required=True, type=str)
    ap.add_argument("--target", required=True, type=str)
    ap.add_argument("--out", required=True, type=str)
    ap.add_argument("--report", required=True, type=str)

    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("--headroom_db", type=float, default=-6.0)

    # Match EQ
    ap.add_argument("--fir_taps", type=int, default=2049)
    ap.add_argument("--max_eq_db", type=float, default=7.0)
    ap.add_argument("--eq_smooth_hz", type=float, default=80.0)
    ap.add_argument("--match_strength", type=float, default=0.85)
    ap.add_argument("--match_strength_lo_hz", type=float, default=80.0)
    ap.add_argument("--match_strength_hi_hz", type=float, default=10000.0)
    ap.add_argument("--match_strength_lo_factor", type=float, default=0.55)
    ap.add_argument("--match_strength_hi_factor", type=float, default=0.60)
    ap.add_argument("--eq_phase", type=str, default="minimum", choices=["linear", "minimum"])
    ap.add_argument("--eq_minphase_nfft", type=int, default=16384)
    ap.add_argument("--disable_eq_guardrails", action="store_true")

    # Stem-aware mastering (Demucs)
    ap.add_argument("--enable_stems", action="store_true")
    ap.add_argument("--demucs_model", type=str, default="htdemucs")
    ap.add_argument("--demucs_device", type=str, default="cpu")
    ap.add_argument("--demucs_overlap", type=float, default=0.25)

    # De-ess (vocals stem recommended)
    ap.add_argument("--deess_low", type=float, default=6000.0)
    ap.add_argument("--deess_high", type=float, default=10000.0)
    ap.add_argument("--deess_threshold_db", type=float, default=-22.0)
    ap.add_argument("--deess_ratio", type=float, default=2.0)

    # Key-aware harmonic glow
    ap.add_argument("--enable_key_glow", action="store_true")
    ap.add_argument("--glow_gain_db", type=float, default=0.85)
    ap.add_argument("--glow_q", type=float, default=7.0)
    ap.add_argument("--glow_mix", type=float, default=0.55)

    # Spatial + movement
    ap.add_argument("--enable_spatial", action="store_true")
    ap.add_argument("--stereo_side_hp_hz", type=float, default=180.0)
    ap.add_argument("--stereo_width_mid", type=float, default=1.07)
    ap.add_argument("--stereo_width_hi", type=float, default=1.25)
    ap.add_argument("--stereo_corr_min", type=float, default=0.00)

    ap.add_argument("--enable_movement", action="store_true")
    ap.add_argument("--rhythm_amount", type=float, default=0.12)

    # Transient restore
    ap.add_argument("--enable_transient_restore", action="store_true")
    ap.add_argument("--attack_restore_db", type=float, default=1.2)
    ap.add_argument("--attack_restore_mix", type=float, default=0.60)

    # Finalize / limiter
    ap.add_argument("--target_lufs", type=float, default=-11.0)
    ap.add_argument("--target_peak_dbfs", type=float, default=-1.0)
    ap.add_argument("--tp_oversample", type=int, default=8)
    ap.add_argument("--finalize_iters", type=int, default=3)

    ap.add_argument("--clip_drive_db", type=float, default=2.2)
    ap.add_argument("--clip_mix", type=float, default=0.20)
    ap.add_argument("--clip_oversample", type=int, default=4)

    ap.add_argument("--limit_lookahead_ms", type=float, default=2.7)
    ap.add_argument("--limit_attack_ms", type=float, default=0.25)
    ap.add_argument("--limit_release_ms", type=float, default=70.0)
    ap.add_argument("--ceiling_chase_strength", type=float, default=1.0)

    args = ap.parse_args()

    ref_path = Path(args.reference)
    tgt_path = Path(args.target)
    out_path = Path(args.out)
    rep_path = Path(args.report)

    # Load audio
    ref, ref_sr = sf.read(ref_path, always_2d=True, dtype="float32")
    tgt, tgt_sr = sf.read(tgt_path, always_2d=True, dtype="float32")
    ref = stereoize(ref)
    tgt = stereoize(tgt)

    sr = int(args.sr)
    if ref_sr != sr:
        ref = resample_poly(ref, sr, int(ref_sr), axis=0).astype(np.float32)
    if tgt_sr != sr:
        tgt = resample_poly(tgt, sr, int(tgt_sr), axis=0).astype(np.float32)

    # Headroom
    ref = safe_headroom(ref, headroom_db=float(args.headroom_db))
    tgt = safe_headroom(tgt, headroom_db=float(args.headroom_db))

    pre = tgt.copy()

    # Key detection (for glow + report)
    key_info = estimate_key_ks(to_mono(tgt), sr=sr)

    # Optional stem separation
    stems_info: Dict = {"enabled": False}
    stems: Optional[Dict[str, np.ndarray]] = None
    snare_times = np.array([], dtype=np.int64)
    snare_info: Dict = {"enabled": False}

    if args.enable_stems:
        if not _HAS_DEMUCS:
            LOG.warning("Stem separation requested but Demucs not available. Continuing without stems.")
        else:
            stems, stems_info = demucs_separate_stems(
                tgt,
                sr=sr,
                model_name=str(args.demucs_model),
                device=str(args.demucs_device),
                split=True,
                overlap=float(args.demucs_overlap),
            )

    # STEM-AWARE PROCESSING
    if stems is not None:
        # Expected names: drums, bass, other, vocals
        drums = stems.get("drums", np.zeros_like(tgt))
        bass = stems.get("bass", np.zeros_like(tgt))
        other = stems.get("other", np.zeros_like(tgt))
        vocals = stems.get("vocals", np.zeros_like(tgt))

        # Vocals: de-ess ONLY on vocals (protect hats/air in drums+other)
        vocals, vocals_deess = dynamic_deesser(
            vocals,
            sr=sr,
            band_low=float(args.deess_low),
            band_high=float(args.deess_high),
            threshold_db=float(args.deess_threshold_db),
            ratio=float(args.deess_ratio),
        )

        # Key glow (gentle) on “other” + a little on vocals
        glow_info = {"enabled": False}
        if args.enable_key_glow:
            other, glow_info_other = key_aware_harmonic_glow(
                other, sr=sr, detected_key=key_info, glow_gain_db=float(args.glow_gain_db), glow_q=float(args.glow_q), mix=float(args.glow_mix)
            )
            vocals, glow_info_voc = key_aware_harmonic_glow(
                vocals, sr=sr, detected_key=key_info, glow_gain_db=float(args.glow_gain_db) * 0.75, glow_q=float(args.glow_q), mix=float(args.glow_mix) * 0.45
            )
            glow_info = {"enabled": True, "other": glow_info_other, "vocals": glow_info_voc}

        # Transient detection on drums BEFORE limiting (for later restore)
        if args.enable_transient_restore:
            snare_times, snare_info = detect_snare_transients(drums, sr=sr)

        # Recombine (stem bus)
        x = (drums + bass + other + vocals).astype(np.float32)

        stem_block_info = {
            "enabled": True,
            "vocals_deess": vocals_deess,
            "key_glow": glow_info,
            "snare_detect": snare_info,
        }
    else:
        # No stems path
        x = tgt.copy()
        stem_block_info = {"enabled": False, "reason": "stems_disabled_or_unavailable"}

        # Key glow on full mix (subtle)
        glow_info = {"enabled": False}
        if args.enable_key_glow:
            x, glow_info = key_aware_harmonic_glow(
                x, sr=sr, detected_key=key_info, glow_gain_db=float(args.glow_gain_db), glow_q=float(args.glow_q), mix=float(args.glow_mix)
            )

        # Transient detection on full mix (less precise)
        if args.enable_transient_restore:
            snare_times, snare_info = detect_snare_transients(x, sr=sr)

        stem_block_info = {
            "enabled": False,
            "key_glow": glow_info,
            "snare_detect": snare_info,
        }

    # MATCH EQ (global, but protected against muffling)
    h, eq_info = design_match_fir(
        ref_mono=to_mono(ref),
        tgt_mono=to_mono(x),
        sr=sr,
        numtaps=int(args.fir_taps),
        max_gain_db=float(args.max_eq_db),
        smooth_hz=float(args.eq_smooth_hz),
        eq_phase=str(args.eq_phase),
        minphase_nfft=int(args.eq_minphase_nfft),
        match_strength=float(args.match_strength),
        match_lo_hz=float(args.match_strength_lo_hz),
        match_hi_hz=float(args.match_strength_hi_hz),
        match_lo_factor=float(args.match_strength_lo_factor),
        match_hi_factor=float(args.match_strength_hi_factor),
        enable_guardrails=(not bool(args.disable_eq_guardrails)),
    )
    x = apply_fir_fft(x, h)

    # Spatial + movement
    spatial_info = {"enabled": False}
    if args.enable_spatial:
        x, spatial_info = stereo_widener_freq(
            x,
            sr=sr,
            side_hp_hz=float(args.stereo_side_hp_hz),
            width_mid=float(args.stereo_width_mid),
            width_hi=float(args.stereo_width_hi),
            corr_min=float(args.stereo_corr_min),
        )

    movement_info = {"enabled": False}
    if args.enable_movement:
        x, movement_info = movement_automation(x, sr=sr, amount=float(args.rhythm_amount))

    # FINALIZE (LUFS + streaming-safe TP limiter)
    x, finalize_info = finalize_master(
        x,
        sr=sr,
        target_lufs=float(args.target_lufs),
        target_peak_dbfs=float(args.target_peak_dbfs),
        tp_oversample=int(args.tp_oversample),
        iters=int(args.finalize_iters),
        clip_drive_db=float(args.clip_drive_db),
        clip_mix=float(args.clip_mix),
        clip_oversample=int(args.clip_oversample),
        limit_lookahead_ms=float(args.limit_lookahead_ms),
        limit_attack_ms=float(args.limit_attack_ms),
        limit_release_ms=float(args.limit_release_ms),
        ceiling_chase_strength=float(args.ceiling_chase_strength),
    )

    # POST-LIMITER TRANSIENT RESTORE (true punch return)
    transient_restore_info = {"enabled": False}
    if args.enable_transient_restore and snare_times.size > 0:
        x, transient_restore_info = micro_attack_restore(
            x,
            sr=sr,
            transient_samples=snare_times,
            restore_db=float(args.attack_restore_db),
            mix=float(args.attack_restore_mix),
        )
        # safety: re-limit true peak to guarantee ceiling after attack restore
        x, post_tp_info = true_peak_chasing_limiter(
            x,
            sr=sr,
            ceiling_dbfs=float(args.target_peak_dbfs),
            tp_oversample=int(args.tp_oversample),
            lookahead_ms=float(args.limit_lookahead_ms),
            attack_ms=float(args.limit_attack_ms),
            release_ms=float(args.limit_release_ms),
            ceiling_chase_strength=float(args.ceiling_chase_strength),
        )
        transient_restore_info["post_tp_limiter"] = post_tp_info

    # Export
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, x, sr, subtype="PCM_24")

    # Metrics
    payload: Dict = {
        "version": "v7.0",
        "paths": {"reference": str(ref_path), "target": str(tgt_path), "out": str(out_path)},
        "pre": {
            "lufs": float(measure_lufs(pre, sr)),
            "true_peak_dbfs": float(lin_to_db(true_peak_lin(pre, oversample=int(args.tp_oversample)))),
            "peak_dbfs": float(lin_to_db(max_abs(pre))),
        },
        "post": {
            "lufs": float(measure_lufs(x, sr)),
            "true_peak_dbfs": float(lin_to_db(true_peak_lin(x, oversample=int(args.tp_oversample)))),
            "peak_dbfs": float(lin_to_db(max_abs(x))),
        },
        "key_info": key_info,
        "stems_info": stems_info,
        "stem_block": stem_block_info,
        "match_eq": eq_info,
        "spatial": spatial_info,
        "movement": movement_info,
        "finalize": finalize_info,
        "transient_restore": transient_restore_info,
        "plots": {},
    }

    # Plots
    eq_png = plot_eq_delta(rep_path, eq_info=eq_info)
    if eq_png:
        payload["plots"]["eq_delta"] = eq_png

    spec_png = plot_spectrum_overlay(rep_path, sr=sr, pre=pre, post=x)
    if spec_png:
        payload["plots"]["spectrum_overlay"] = spec_png

    # Report
    write_report(rep_path, payload)

    LOG.info("Done. Export: %s", out_path)
    LOG.info("Report: %s", rep_path)
    if args.enable_stems and not _HAS_DEMUCS:
        LOG.info("Note: Demucs was not installed — stem-aware mastering was skipped.")


if __name__ == "__main__":
    main()

"""
# =============================================================================
# MUSICAL PRESETS (PowerShell)
# =============================================================================
# 1) CINEMATIC PUNCH (Trap, forward drums, clean air, no muffling)
# python auralmind_match_maestro_v7_0.py --reference "C:/Users/goku/Downloads/Brent Faiyaz - Pistachios [Official Video].mp3" --target   "C:/Users/goku/Downloads/Vegas - top teir (20).wav" --out      "C:/Users/goku/Desktop/Vegas_Top_Teir_PUNCH_v7_0.wav" --report   "C:/Users/goku/Desktop/Vegas_Top_Teir_PUNCH_v7_0_Report.md" --target_lufs -11.0 --target_peak_dbfs -1.0 --fir_taps 2049 --match_strength 0.85 --max_eq_db 7 --eq_smooth_hz 80 --enable_key_glow --glow_gain_db 0.85 --glow_mix 0.55 --enable_spatial --stereo_width_mid 1.07 --stereo_width_hi 1.25 --enable_movement --rhythm_amount 0.12 --enable_transient_restore --attack_restore_db 1.2 --attack_restore_mix 0.60 --enable_stems --demucs_model htdemucs --demucs_device cpu

# 2) AIRY STREAMING (Extra clarity + shimmer, still safe and smooth)
# python auralmind_match_maestro_v7_0.py --reference "C:/Users/goku/Downloads/Brent Faiyaz - Pistachios [Official Video].mp3" --target   "C:/Users/goku/Downloads/Vegas - top teir (20).wav" --out      "C:/Users/goku/Desktop/Vegas_Top_Teir_AIRY_v7_0.wav" --report   "C:/Users/goku/Desktop/Vegas_Top_Teir_AIRY_v7_0_Report.md" --target_lufs -11.0 --target_peak_dbfs -1.0 --match_strength 0.78 --max_eq_db 6 --eq_smooth_hz 95 --match_strength_hi_factor 0.72 --enable_key_glow --glow_gain_db 0.95 --glow_mix 0.60 --enable_spatial --stereo_width_mid 1.06 --stereo_width_hi 1.28 --enable_movement --rhythm_amount 0.10 --enable_stems --demucs_model htdemucs --demucs_device cpu

# 3) LOUD CLUB (Harder density, still no crackles, still <= -1.0 dBTP)
# python auralmind_match_maestro_v7_0.py --reference "C:/Users/goku/Downloads/Brent Faiyaz - Pistachios [Official Video].mp3" --target   "C:/Users/goku/Downloads/Vegas - top teir (20).wav" --out      "C:/Users/goku/Desktop/Vegas_Top_Teir_LOUD_v7_0.wav" --report   "C:/Users/goku/Desktop/Vegas_Top_Teir_LOUD_v7_0_Report.md" --target_lufs -10.2 --target_peak_dbfs -1.0 --clip_drive_db 2.6 --clip_mix 0.25 --tp_oversample 8 --finalize_iters 4 --ceiling_chase_strength 1.15 --enable_transient_restore --attack_restore_db 1.0 --attack_restore_mix 0.55 --enable_stems --demucs_model htdemucs --demucs_device cpu
#
# STEM-AWARE MASTERING (Best vocal de-ess, hats preserved): Add: --enable_stems --demucs_device cpu Example:   ... --enable_stems --demucs_model htdemucs --demucs_device cpu
"""