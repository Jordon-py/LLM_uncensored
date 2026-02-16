#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AuralMind Match — Maestro v7.1 (NO-REGRESSION UPGRADE)
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

--------------------------------------------------
Possibly use havent yet, 

max quality resample and bit change 
./ffmpeg-install/bin/ffmpeg -hide_banner -i "input.wav" \
  -af "aresample=resampler=soxr:precision=32:cheby=1" \
  -ar 44100 -c:a pcm_s24le "out_44k1_24b_max.wav"

high quality
./ffmpeg-install/bin/ffmpeg -hide_banner -i "input.wav" \
  -af "aresample=resampler=soxr:precision=28:cheby=0" \
  -ar 44100 -c:a pcm_s24le "out_44k1_24b.wav"
----------------------------------------------------------

PowerShell example (your exact flow):
python auralmind_match_maestro_v7_1.py --preset innovative_trap --reference "C:/Users/goku/Downloads/Brent Faiyaz - Pistachios [Official Video].mp3" --target   "C:/Users/goku/Downloads/Vegas - top teir (20).wav" --out "C:/Users/goku/Desktop/Vegas_Top_Teir_MASTER_innovative_v7_1.wav" --report   "C:/Users/goku/Desktop/Vegas_Top_Teir_MASTER_v7_1_Report.md" --target_lufs -11.0 --target_peak_dbfs -1.0 --enable_spatial --enable_movement --enable_key_glow --enable_transient_restore --enable_stem_separation --enable_mono_sub

"""

from __future__ import annotations

MAESTRO_VERSION = "v7.1"


import os
import sys
import time
import shutil
import argparse
import json
import math
import logging
import faulthandler
from contextlib import contextmanager

LOG = logging.getLogger("auralmind")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", filemode='w', filename='auralmind.log' )

import tempfile
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
try:
    import librosa
    HAVE_LIBROSA = True
except Exception:
    librosa = None
    HAVE_LIBROSA = False

try:
    import pyloudnorm as pyln
    HAVE_PYLOUDNORM = True
except Exception:
    pyln = None
    HAVE_PYLOUDNORM = False

from scipy.signal import (
    butter,
    filtfilt,
    firwin2,
    resample_poly,
    fftconvolve,
)
from scipy.ndimage import maximum_filter1d

# Optional progress bar (batch mode)
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

# Optional performance acceleration (Numba)
try:
    from numba import njit  # type: ignore
except Exception:  # pragma: no cover
    def njit(*args, **kwargs):
        def _wrap(fn):
            return fn
        return _wrap


# Optional plotting
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None
import torchaudio
from torchaudio import load, save  
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
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", filemode='w', filename='auralmind.log' )



# =============================================================================
# Utility - Export Guard + SoXR SRC (FFmpeg)
# =============================================================================

SRC_PRESETS = {
    # Matches your examples
    "hq":  {"precision": 28, "cheby": 0, "codec": "pcm_s24le"},
    "max": {"precision": 32, "cheby": 1, "codec": "pcm_s24le"},
}

def _which_ffmpeg(user_path: str | None = None) -> str | None:
    """
    Resolve ffmpeg path.
    - Prefer user-provided --ffmpeg
    - Else search PATH
    """
    if user_path:
        p = Path(user_path)
        if p.exists():
            return str(p)
    return shutil.which("ffmpeg")

def _ffmpeg_soxr_resample(
    ffmpeg_path: str,
    in_wav: Path,
    out_wav: Path,
    out_sr: int,
    precision: int,
    cheby: int,
    codec: str,
) -> None:
    """
    Run FFmpeg SoXR resample:
      aresample=resampler=soxr:precision=XX:cheby=Y

    Notes:
    - Writes WAV (codec controls bit depth/PCM format).
    - Uses -y overwrite + quiet-ish logging.
    """
    filt = f"aresample=resampler=soxr:precision={int(precision)}:cheby={int(cheby)}"
    cmd = [
        ffmpeg_path,
        "-hide_banner", "-y",
        "-loglevel", "error",
        "-i", str(in_wav),
        "-af", filt,
        "-ar", str(int(out_sr)),
        "-c:a", str(codec),
        str(out_wav),
    ]
    subprocess.run(cmd, check=True)

def export_guarded_wav(
    audio: np.ndarray,
    input_sr: int,
    out_path: Path,
    export_sr: int,
    src_preset: str,
    ffmpeg_path: str | None,
    subtype_on_direct_write: str = "PCM_24",
) -> dict:
    """
    Export Guard:
      - If input_sr == export_sr: direct soundfile write (no SRC)
      - Else: SoXR SRC via FFmpeg using preset-driven settings

    Returns dict describing exactly what happened (for report + logs).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    export_sr = int(export_sr)
    input_sr = int(input_sr)

    info = {
        "input_sr": input_sr,
        "export_sr": export_sr,
        "src_applied": False,
        "src_backend": None,
        "src_preset": src_preset,
        "soxr": None,
        "codec": None,
        "direct_write_subtype": None,
        "path": str(out_path),
    }

    # Guard: no SRC needed
    if input_sr == export_sr:
        sf.write(out_path, stereoize(audio), input_sr, subtype=subtype_on_direct_write)
        info["src_applied"] = False
        info["src_backend"] = "none"
        info["direct_write_subtype"] = subtype_on_direct_write
        LOG.info("Export Guard: input_sr == export_sr (%s). Skipping SRC. subtype=%s", input_sr, subtype_on_direct_write)
        return info

    # SRC needed -> SoXR (FFmpeg)
    preset = SRC_PRESETS.get(str(src_preset).lower(), SRC_PRESETS["hq"])
    precision = int(preset["precision"])
    cheby = int(preset["cheby"])
    codec = str(preset["codec"])

    ff = _which_ffmpeg(ffmpeg_path)
    if not ff:
        # Hard fail because you explicitly requested SoXR application when SR differs
        raise RuntimeError(
            "Export Guard: export_sr differs from input_sr, but ffmpeg not found. "
            "Install ffmpeg or pass --ffmpeg /path/to/ffmpeg."
        )

    info["src_applied"] = True
    info["src_backend"] = "ffmpeg_soxr"
    info["soxr"] = {"precision": precision, "cheby": cheby}
    info["codec"] = codec

    LOG.info(
        "Export Guard: SRC required (%s -> %s). Using SoXR preset=%s precision=%s cheby=%s codec=%s ffmpeg=%s",
        input_sr, export_sr, src_preset, precision, cheby, codec, ff
    )

    # Write temp WAV at input_sr, then FFmpeg resample to final out_path
    with tempfile.TemporaryDirectory() as td:
        tmp_in = Path(td) / "tmp_in.wav"
        sf.write(tmp_in, stereoize(audio), input_sr, subtype="FLOAT")  # lossless-ish staging
        _ffmpeg_soxr_resample(
            ffmpeg_path=ff,
            in_wav=tmp_in,
            out_wav=out_path,
            out_sr=export_sr,
            precision=precision,
            cheby=cheby,
            codec=codec,
        )

    return info


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
# Utility - Audio Loading
# =============================================================================

def load_audio_any(path: Path, sr: int) -> Tuple[np.ndarray, Dict]:
    """
    Robust audio loader: tries soundfile first (fast), falls back to librosa (slow but supports mp3 etc).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # 1. Try SoundFile (fastest) - supports wav, flac, ogg, etc.
    try:
        y, orig_sr = sf.read(str(path), always_2d=True)  # (samples, channels)
        y = y.astype(np.float32, copy=False)

        if orig_sr != sr:
            if HAVE_LIBROSA:
                # librosa supports n-d resampling via axis in modern versions,
                # but transpose method is most compatible across installs.
                y = librosa.resample(y.T, orig_sr=orig_sr, target_sr=sr).T.astype(np.float32)
            else:
                gcd = math.gcd(orig_sr, sr)
                y = resample_poly(y, sr // gcd, orig_sr // gcd, axis=0).astype(np.float32)

        return y, {"sr": sr, "orig_sr": orig_sr, "source": "soundfile"}

    
    except Exception as e:
        LOG.debug("SoundFile load failed for %s: %s. Trying librosa.", path, e)
    
    # 2. Try librosa (supports mp3, m4a via ffmpeg/audioread)
    if HAVE_LIBROSA:
        try:
            # librosa.load returns (mono, sr) by default unless mono=False
            y, orig_sr = librosa.load(str(path), sr=sr, mono=False)
            if y.ndim == 1:
                # mono -> (samples, 1) to match (N, C) convention?
                # Wait, this script uses (N, 2) mostly. 
                # Librosa returns (C, N).
                y = y.T
            else:
                y = y.T
            return y.astype(np.float32), {"sr": sr, "orig_sr": orig_sr, "source": "librosa"}
        except Exception as e:
            raise RuntimeError(f"Could not load audio {path}: {e}")
    else:
        raise RuntimeError(f"Could not load {path} (SoundFile failed and Librosa not installed).")


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

def estimate_tilt_db_per_oct_from_reference(
    ref_mono: np.ndarray,
    tgt_mono: np.ndarray,
    sr: int,
    pivot_hz: float = 1000.0,
    f_lo: float = 200.0,
    f_hi: float = 8000.0,
) -> float:
    freqs, ref_db = average_spectrum_db_librosa(ref_mono, sr=sr)
    _, tgt_db = average_spectrum_db_librosa(tgt_mono, sr=sr)

    f = freqs.astype(np.float32)
    mask = (f >= float(f_lo)) & (f <= float(f_hi))
    if np.sum(mask) < 16:
        return 0.0

    x = np.log2(f[mask] / float(pivot_hz)).astype(np.float32)
    y = (ref_db[mask] - tgt_db[mask]).astype(np.float32)  # desired delta

    denom = float(np.sum(x * x) + 1e-12)
    slope = float(np.sum(x * y) / denom)  # dB per octave
    return float(np.clip(slope, -0.35, 0.35))


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


def scale_shimmer_exciter(
    audio: np.ndarray,
    sr: int,
    detected_key: Dict,
    drive: float = 1.6,
    mix: float = 0.06,
    band_gain_db: float = 1.0,
    q: float = 9.0,
) -> Tuple[np.ndarray, Dict]:
    """
    Scale-Shimmer Exciter (music-theory grounded):
      - emphasizes partials aligned to the detected scale
      - saturates ONLY that scale-energy slice
      - returns subtle "singing" brilliance without random harshness
    """
    x = stereoize(audio)
    mix = float(np.clip(mix, 0.0, 0.25))
    if mix <= 1e-6:
        return x, {"enabled": False}

    tonic = str(detected_key.get("tonic", "C"))
    mode = str(detected_key.get("mode", "major"))
    tonic_idx = _NOTE_NAMES.index(tonic) if tonic in _NOTE_NAMES else 0

    bands = build_scale_glow_bands(
        sr=sr,
        tonic_idx=tonic_idx,
        mode=mode,
        base_gain_db=float(band_gain_db),
        q=float(q),
        f_min=700.0,
        f_max=12000.0,
    )
    if not bands:
        return x, {"enabled": False, "reason": "no_bands"}

    emphasized = apply_peaking_bank(x, sr=sr, bands=bands)
    isolated = (emphasized - x).astype(np.float32)
    sat = np.tanh(isolated * float(np.clip(drive, 1.0, 3.0))).astype(np.float32)

    y = (x + mix * sat).astype(np.float32)

    info = {
        "enabled": True,
        "key": str(detected_key.get("key", "")),
        "tonic": tonic,
        "mode": mode,
        "drive": float(drive),
        "mix": float(mix),
        "band_gain_db": float(band_gain_db),
        "q": float(q),
        "bands_used": int(len(bands)),
        "bands": [{"f0": float(f0), "q": float(q), "gain_db": float(g)} for (f0, q, g) in bands],
    }
    return y, info


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

# Alias for compatibility with v7.1 functions references
ms_to_lr = from_mid_side



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
# v7.1 Enhancements — Mono Sub Anchor + Tilt EQ + Groove Glue + HookLift
# =============================================================================

def mono_sub_anchor(
    audio: np.ndarray,
    sr: int,
    cutoff_hz: float = 120.0,
    mix: float = 1.0,
    order: int = 4,
) -> Tuple[np.ndarray, Dict]:
    """
    Mono Sub Anchor (≤ cutoff_hz)

    Why it matters (modern trap / club translation):
      - Sub energy must survive phone speakers → car systems → club rigs consistently.
      - Stereo sub can collapse unpredictably when encoded to AAC/Opus, or when played in mono venues.
      - Mono-anchoring the low band keeps the *weight* stable while preserving stereo texture above it.

    Implementation:
      - Convert to Mid/Side
      - Low-pass the Side
      - Attenuate the Side-low band bymix` (mix=1 => fully mono below cutoff)

    Guardrails:
      -mix` is clamped to [0..1]
      - Uses filtfilt (zero-phase) to avoid phase smear in the sub
    """
    x = stereoize(audio)
    cutoff = float(np.clip(cutoff_hz, 40.0, 220.0))
    mix = float(np.clip(mix, 0.0, 1.0))
    mid, side = mid_side(x)

    b, a = butter(int(max(2, order)), cutoff / (0.5 * sr), btype="low")
    side_lo = filtfilt(b, a, side).astype(np.float32)
    side_new = (side - mix * side_lo).astype(np.float32)

    y = ms_to_lr(mid.astype(np.float32), side_new)
    info = {"enabled": True, "cutoff_hz": cutoff, "mix": mix}
    return y.astype(np.float32), info

def design_tilt_fir(
    sr: int,
    numtaps: int = 513,
    tilt_db_per_oct: float = 0.0,
    pivot_hz: float = 1000.0,
    max_tilt_db: float = 3.0,
    guard_lo_hz: float = 35.0,
    guard_hi_hz: float = 14000.0,
    phase: str = "minimum",
) -> Tuple[np.ndarray, Dict]:
    """
    Tilt EQ = a smooth spectral slope around a pivot frequency.

    Music-theory intuition:
      - Perception of “brightness” and “warmth” is heavily linked to spectral slope.
      - Melodic trap often benefits from a *slight upward tilt* (air + presence) while keeping 808 weight controlled.

    This stage is intentionally *guardrailed*:
      - it tapers below ~35 Hz (avoid infra rumble)
      - it tapers above ~14 kHz (avoid hashy ultra-air)
      - it clamps overall tilt magnitude to +/- max_tilt_db

    The result is a *safe, modern* macro-tonal move, not a harsh EQ.
    """
    tilt_db_per_oct = float(tilt_db_per_oct)
    if abs(tilt_db_per_oct) < 1e-6:
        h = np.array([1.0], dtype=np.float32)
        return h, {"enabled": False}

    numtaps = int(max(129, numtaps))
    if numtaps % 2 == 0:
        numtaps += 1

    nyq = 0.5 * sr
    pivot = float(np.clip(pivot_hz, 150.0, 6000.0))
    max_tilt_db = float(np.clip(max_tilt_db, 0.5, 6.0))

    # Log-spaced magnitude design for firwin2
    npts = 256
    freqs = np.geomspace(20.0, nyq, num=npts).astype(np.float32)
    # dB tilt around pivot: +slope per octave above pivot, - below
    tilt = tilt_db_per_oct * (np.log2(freqs / pivot)).astype(np.float32)
    tilt = np.clip(tilt, -max_tilt_db, max_tilt_db).astype(np.float32)

    # Guardrails taper
    lo = float(max(20.0, guard_lo_hz))
    hi = float(min(nyq * 0.98, guard_hi_hz))
    if hi <= lo:
        hi = lo + 1.0

    taper = np.ones_like(tilt, dtype=np.float32)
    taper[freqs < lo] = (freqs[freqs < lo] / lo).astype(np.float32)
    taper[freqs > hi] = (hi / freqs[freqs > hi]).astype(np.float32)
    taper = np.clip(taper, 0.0, 1.0)

    tilt = (tilt * taper).astype(np.float32)

    mag = (10.0 ** (tilt / 20.0)).astype(np.float32)

    f_norm = np.clip(freqs / nyq, 0.0, 1.0).astype(np.float32)
    f_norm[0] = 0.0
    f_norm[-1] = 1.0

    h = firwin2(numtaps, f_norm, mag).astype(np.float32)

    if phase == "minimum":
        nfft = int(2 ** math.ceil(math.log2(max(2048, numtaps * 8))))
        H = np.fft.rfft(h, nfft)
        h = minimum_phase_fir_from_mag(np.abs(H).astype(np.float32), nfft)[:numtaps].astype(np.float32)

    info = {
        "enabled": True,
        "tilt_db_per_oct": tilt_db_per_oct,
        "pivot_hz": pivot,
        "max_tilt_db": max_tilt_db,
        "guard_lo_hz": lo,
        "guard_hi_hz": hi,
        "numtaps": int(numtaps),
    }
    return h, info


@njit(cache=True)
def _tanh_sat(x: np.ndarray, drive: float) -> np.ndarray:
    # Small helper for speed; Numba will inline this per-block.
    return np.tanh(x * drive).astype(np.float32)


def groove_glue_saturator(
    audio: np.ndarray,
    sr: int,
    base_drive: float = 1.15,
    dynamic: float = 0.20,
    mix: float = 0.12,
) -> Tuple[np.ndarray, Dict]:
    """
    Groove Glue Saturation (original v7.1 feature)

    Trend + theory:
      - Modern melodic trap masters use *micro* saturation/clipping to increase perceived density,
        but the “feel” comes from making it *react* to groove/energy rather than being static.
      - This stage modulates saturation drive using an envelope follower on the MID channel.
        It subtly thickens loud moments and stays transparent in quieter parts.

    This is a mastering-safe, low-distortion “motion glue”:
      - no multiband pumping
      - no obvious compression artifacts
      - measurable improvement in crest factor stability + perceived loudness
    """
    mix = float(np.clip(mix, 0.0, 0.45))
    if mix <= 1e-6:
        return audio.astype(np.float32), {"enabled": False}

    x = stereoize(audio)
    mid, side = mid_side(x)

    env = np.abs(mid).astype(np.float32)
    win = int(max(64, round(sr * 0.03)))
    kernel = np.ones(win, dtype=np.float32) / float(win)
    env_s = np.convolve(env, kernel, mode="same").astype(np.float32)
    env_n = (env_s - np.min(env_s)) / float(np.max(env_s) - np.min(env_s) + 1e-12)

    drive = float(np.clip(base_drive, 1.0, 2.5))
    dyn = float(np.clip(dynamic, 0.0, 0.65))
    drive_t = drive * (1.0 + dyn * (env_n - 0.5)).astype(np.float32)

    mid_sat = np.tanh(mid * drive_t).astype(np.float32)

    mid_out = (1.0 - mix) * mid + mix * mid_sat
    y = ms_to_lr(mid_out.astype(np.float32), side.astype(np.float32))
    info = {"enabled": True, "mix": mix, "base_drive": drive, "dynamic": dyn}
    return y.astype(np.float32), info


def build_section_lift_mask(
    audio: np.ndarray,
    sr: int,
    win_s: float = 0.80,
    percentile: float = 75.0,
    attack_s: float = 0.25,
    release_s: float = 0.90,
) -> np.ndarray:
    x = stereoize(audio)
    m = to_mono(x)
    win = int(max(256, round(sr * win_s)))
    kernel = np.ones(win, dtype=np.float32) / float(win)

    rms = np.sqrt(np.convolve(m * m, kernel, mode="same") + 1e-12).astype(np.float32)
    thr = np.percentile(rms, float(np.clip(percentile, 50.0, 95.0)))

    target = (rms >= thr).astype(np.float32)

    # smooth gate (attack/release)
    a = math.exp(-1.0 / (sr * max(0.02, attack_s) + 1e-12))
    r = math.exp(-1.0 / (sr * max(0.05, release_s) + 1e-12))

    env = np.zeros_like(target, dtype=np.float32)
    cur = 0.0
    for i in range(target.size):
        d = float(target[i])
        if d > cur:
            cur = d + a * (cur - d)
        else:
            cur = d + r * (cur - d)
        env[i] = cur

    return np.clip(env, 0.0, 1.0).astype(np.float32)

def hooklift(
    audio: np.ndarray,
    sr: int,
    mix: float = 0.20,
    width_gain: float = 0.18,
    width_hp_hz: float = 1600.0,
    air_hz: float = 8500.0,
    air_gain: float = 0.14,
    shimmer_drive: float = 1.55,
    shimmer_mix: float = 0.35,
) -> Tuple[np.ndarray, Dict]:
    """
    HookLift (v7.1) — “bright chorus lift” without harshness.

    This combines:
      1) High-band side boost (wider hooks, but correlation-guarded by existing spatial stage)
      2) Gentle air shelf
      3) “Shimmer” harmonic excitement on highs only (soft, de-ess friendly)

    It is intentionally *mixable* to avoid over-brightening.
    """
    mix = float(np.clip(mix, 0.0, 0.65))
    if mix <= 1e-6:
        return audio.astype(np.float32), {"enabled": False}

    x = stereoize(audio)
    mid, side = mid_side(x)

    hp = float(np.clip(width_hp_hz, 600.0, 6000.0))
    b, a = butter(2, hp / (0.5 * sr), btype="high")
    side_hi = filtfilt(b, a, side).astype(np.float32)
    side_boosted = (side + width_gain * side_hi).astype(np.float32)

    # Air shelf using a gentle peak-ish approximation: highpass then add back
    air_hz = float(np.clip(air_hz, 4000.0, 16000.0))
    b2, a2 = butter(2, air_hz / (0.5 * sr), btype="high")
    air = filtfilt(b2, a2, mid).astype(np.float32)
    mid_air = (mid + air_gain * air).astype(np.float32)

    # Shimmer saturation on the air band
    air_sat = np.tanh(air * float(np.clip(shimmer_drive, 1.0, 3.0))).astype(np.float32)
    mid_air2 = (mid_air + shimmer_mix * air_sat).astype(np.float32)

    y = ms_to_lr(mid_air2, side_boosted)
    y = (1.0 - mix) * x + mix * y
    info = {"enabled": True, "mix": mix, "width_gain": width_gain, "air_hz": air_hz, "air_gain": air_gain}
    return y.astype(np.float32), info

# =============================================================================
# Loudness + True-Peak limiter (streaming-perfect)
# =============================================================================


def measure_lufs(audio: np.ndarray, sr: int) -> float:
    """Integrated loudness in LUFS (ITU-R BS.1770 via pyloudnorm)."""
    if not HAVE_PYLOUDNORM:
        raise RuntimeError(
            "pyloudnorm is required for LUFS measurement/targeting. "
            "Install it with: pip install pyloudnorm"
        )
    x = stereoize(audio)
    meter = pyln.Meter(sr)  # BS.1770
    loudness = meter.integrated_loudness(x)
    return float(loudness)
def true_peak_lin(audio: np.ndarray, oversample: int = 8) -> float:
    x = stereoize(audio)
    os = int(max(1, oversample))
    if os > 1:
        xo = resample_poly(x, os, 1, axis=0)
    else:
        xo = x
    return float(np.max(np.abs(xo)))



@njit(cache=True)
def _gain_smooth(desired: np.ndarray, a_att: float, a_rel: float, chase: float) -> np.ndarray:
    g = np.empty_like(desired, dtype=np.float32)
    cur = 1.0
    for i in range(desired.size):
        d = float(desired[i])
        d = d ** chase
        if d < cur:
            cur = d + a_att * (cur - d)
        else:
            cur = d + a_rel * (cur - d)
        g[i] = cur
    return g

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

    # Numba-accelerated gain smoothing (huge speedup on long files)
    g = _gain_smooth(desired.astype(np.float32), a_att, a_rel, chase).astype(np.float32)

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
    if plt is None or not HAVE_LIBROSA:
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


def _extract_shimmer_plot_info(shimmer_info: Dict) -> Tuple[List[Dict], str]:
    if not shimmer_info or not shimmer_info.get("enabled"):
        return [], ""
    info = shimmer_info
    if "other" in shimmer_info or "vocals" in shimmer_info:
        for key in ("other", "vocals"):
            candidate = shimmer_info.get(key, {})
            if isinstance(candidate, dict) and candidate.get("bands"):
                info = candidate
                break
    bands = info.get("bands") or []
    return bands, str(info.get("key", ""))


def plot_scale_shimmer_overlay(
    report_path: Path,
    sr: int,
    post: np.ndarray,
    shimmer_info: Dict,
) -> Optional[str]:
    if plt is None or not HAVE_LIBROSA:
        return None
    bands, key = _extract_shimmer_plot_info(shimmer_info)
    if not bands:
        return None

    f, post_db = average_spectrum_db_librosa(to_mono(post), sr=sr)
    band_freqs = np.asarray([b.get("f0", 0.0) for b in bands], dtype=np.float32)
    if band_freqs.size == 0:
        return None

    f_min = float(f[0])
    f_max = float(f[-1])
    mask = (band_freqs >= f_min) & (band_freqs <= f_max)
    band_freqs = band_freqs[mask]
    if band_freqs.size == 0:
        return None

    band_levels = np.interp(band_freqs, f, post_db).astype(np.float32)

    png = report_path.with_name(f"{report_path.stem}_Scale_Shimmer.png")
    plt.figure(figsize=(9.5, 4.0))
    plt.semilogx(f, post_db, color="0.4", label="Post Spectrum")
    plt.scatter(band_freqs, band_levels, color="#f6c945", s=26, label="Scale Shimmer Bands")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    title = "Scale Shimmer Bands"
    if key:
        title = f"{title} - {key}"
    plt.title(title)
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
    md.append("# Enhancement Report - AuralMind Maestro v7.1\n")

    plots = payload.get("plots", {}) or {}
    if "eq_delta" in plots:
        md.append("## Match EQ Delta Curve\n")
        md.append(f"![EQ Delta]({plots['eq_delta']})\n")
    if "spectrum_overlay" in plots:
        md.append("## Spectrum Overlay\n")
        md.append(f"![Spectrum]({plots['spectrum_overlay']})\n")
    if "scale_shimmer" in plots:
        md.append("## Scale Shimmer Bands\n")
        md.append(f"![Scale Shimmer]({plots['scale_shimmer']})\n")

    md.append("## Debug JSON\n")
    md.append("```json")
    md.append(json.dumps(payload, indent=2))
    md.append("```")

    path.write_text("\n".join(md), encoding="utf-8")


# =============================================================================
# Main
# =============================================================================

PRESETS = {
  "airy_streaming": {
    "target_lufs": -13.0,
    "target_peak_dbfs": -1.0,

    # Match EQ: open the air band + reduce “presence-forward / air-suppressed” risk
    "fir_taps": 4097,
    "match_strength": 0.58,
    "match_lo_hz": 120.0,
    "match_hi_hz": 14000.0,
    "max_eq_db": 4.8,
    "eq_smooth_hz": 110.0,
    "match_strength_hi_factor": 0.72,

    # Theory color (subtle): keep harmonic “glow” musical, not thick
    "enable_key_glow": True,
    "glow_gain_db": 0.75,
    "glow_mix": 0.55,

    # Space: slightly more hi width for “air”, but codec-safe
    "enable_spatial": True,
    "stereo_width_mid": 1.06,
    "stereo_width_hi": 1.28,

    # Motion: reduce a touch for streaming translation
    "enable_movement": True,
    "rhythm_amount": 0.15,

    # Transients: keep clean (avoid bright-edge exaggeration)
    "enable_transient_restore": True,
    "attack_restore_db": 1.00,
    "attack_restore_mix": 0.55,

    # Low end: restore dimension while keeping mono compatibility
    "enable_mono_sub": True,
    "mono_sub_hz": 110.0,
    "mono_sub_mix": 0.70,

    # Tilt: keep gentle lift, pivot slightly above vocal fundamentals
    "tilt_db_per_oct": 0.14,
    "tilt_pivot_hz": 1100.0,

    # Hook polish: slightly less than before, more “expensive” than “hyped”
    "enable_hooklift": True,
    "hooklift_mix": 0.20,

    "enable_groove_glue": True,
    "groove_glue_mix": 0.10,
    "groove_glue_dynamic": 0.16,

    # Prevent LUFS overshoot in streaming
    "finalize_iters": 2,
    "tp_oversample": 8,
    "ceiling_chase_strength": 0.95,
  },

  "loud_club": {
    "target_lufs": -10.5,
    "target_peak_dbfs": -1.0,

    # Match EQ: keep moderate so clipping/limiting defines punch, not EQ spikes
    "fir_taps": 2049,
    "match_strength": 0.50,
    "match_lo_hz": 120.0,
    "match_hi_hz": 14000.0,
    "max_eq_db": 5.0,
    "eq_smooth_hz": 85.0,
    "match_strength_hi_factor": 0.62,

    "enable_key_glow": True,
    "glow_gain_db": 0.75,
    "glow_mix": 0.55,

    "finalize_iters": 3,
    "tp_oversample": 8,
    "clip_oversample": 8,
    "clip_drive_db": 2.8,
    "clip_mix": 0.22,
    "ceiling_chase_strength": 1.12,

    # Punch: slightly more restore, but keep mix moderate
    "enable_transient_restore": True,
    "attack_restore_db": 1.25,
    "attack_restore_mix": 0.58,

    # Don’t over-mono the low end (club wants weight + width above sub)
    "enable_mono_sub": True,
    "mono_sub_hz": 120.0,
    "mono_sub_mix": 0.82,

    # Bright but not brittle
    "tilt_db_per_oct": 0.12,
    "tilt_pivot_hz": 950.0,

    "enable_hooklift": True,
    "hooklift_mix": 0.18,

    "enable_groove_glue": True,
    "groove_glue_mix": 0.14,
    "groove_glue_dynamic": 0.26,
  },

  "balanced_v7": {
    "target_lufs": -11.4,
    "target_peak_dbfs": -1.0,

    # Lower match vs original (0.76 was high-risk for “muffled-when-loud”)
    "fir_taps": 4097,
    "match_strength": 0.63,
    "match_lo_hz": 120.0,
    "match_hi_hz": 14000.0,
    "max_eq_db": 5.8,
    "eq_smooth_hz": 95.0,
    "match_strength_hi_factor": 0.72,

    "enable_key_glow": True,
    "glow_gain_db": 0.80,
    "glow_mix": 0.45,

    "enable_spatial": True,
    "stereo_width_mid": 1.05,
    "stereo_width_hi": 1.25,

    "enable_movement": True,
    "rhythm_amount": 0.10,

    "enable_transient_restore": True,
    "attack_restore_db": 1.05,
    "attack_restore_mix": 0.55,

    "enable_mono_sub": True,
    "mono_sub_hz": 115.0,
    "mono_sub_mix": 0.78,

    "tilt_db_per_oct": 0.11,
    "tilt_pivot_hz": 1050.0,

    "enable_hooklift": False,

    "enable_groove_glue": True,
    "groove_glue_mix": 0.10,
    "groove_glue_dynamic": 0.18,

    "finalize_iters": 2,
    "tp_oversample": 8,
    "ceiling_chase_strength": 1.0,
  },

  "innovative_trap": {
    "target_lufs": -10.7,
    "target_peak_dbfs": -1.0,

    "fir_taps": 4097,
    "finalize_iters": 2,
    "tp_oversample": 8,

    # Match EQ: allow true “air polish”
    "match_strength": 0.56,
    "match_lo_hz": 120.0,
    "match_hi_hz": 16000.0,
    "max_eq_db": 5.6,
    "eq_smooth_hz": 90.0,
    "match_strength_hi_factor": 0.70,

    # Glow: keep energy, reduce thickness
    "enable_key_glow": True,
    "glow_gain_db": 1.00,
    "glow_mix": 0.55,

    # Shimmer: de-resonate (Q↓), reduce gain/mix—more “expensive air”, less “ring”
    "enable_scale_shimmer": True,
    "shimmer_drive": 1.55,
    "shimmer_mix": 0.04,
    "shimmer_band_gain_db": 0.7,
    "shimmer_q": 5.5,

    "enable_spatial": True,
    "stereo_width_mid": 1.08,
    "stereo_width_hi": 1.32,

    # Next-gen motion: keep it, but avoid smear
    "enable_movement": True,
    "rhythm_amount": 0.20,

    "enable_transient_restore": True,
    "attack_restore_db": 1.25,
    "attack_restore_mix": 0.58,

    # Low end: tighter mono, more space above it
    "enable_mono_sub": True,
    "mono_sub_hz": 110.0,
    "mono_sub_mix": 0.69,

    "tilt_db_per_oct": 0.14,
    "tilt_pivot_hz": 950.0,

    "enable_hooklift": True,
    "hooklift_mix": 0.23,

    "enable_groove_glue": True,
    "groove_glue_mix": 0.14,
    "groove_glue_dynamic": 0.30,

    "ceiling_chase_strength": 1.05,
  },

  "creative_trap_theory": {
    "target_lufs": -11.1,
    "target_peak_dbfs": -1.0,

    "fir_taps": 3097,
    "match_strength": 0.58,
    "match_lo_hz": 115.0,
    "match_hi_hz": 15000.0,
    "max_eq_db": 6.0,
    "eq_smooth_hz": 95.0,
    "match_strength_hi_factor": 0.75,

    # Theory-forward, but controlled
    "enable_key_glow": True,
    "glow_gain_db": 1.05,
    "glow_mix": 0.55,

    "enable_spatial": True,
    "stereo_width_mid": 1.06,
    "stereo_width_hi": 1.28,

    "enable_movement": True,
    "rhythm_amount": 0.14,

    "enable_transient_restore": True,
    "attack_restore_db": 1.10,
    "attack_restore_mix": 0.56,

    "enable_mono_sub": True,
    "mono_sub_hz": 110.0,
    "mono_sub_mix": 0.78,

    "tilt_db_per_oct": 0.13,
    "tilt_pivot_hz": 1100.0,

    "enable_hooklift": True,
    "hooklift_mix": 0.28,

    "enable_groove_glue": True,
    "groove_glue_mix": 0.12,
    "groove_glue_dynamic": 0.24,

    "finalize_iters": 2,
    "tp_oversample": 8,
    "ceiling_chase_strength": 1.0,
  },
}

    


def _setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Configure console + optional file logging (batch-friendly)."""
    lvl = getattr(logging, str(level).upper(), logging.INFO)
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="w", encoding="utf-8"))
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
    )


def _list_audio_files(folder: Path) -> List[Path]:
    exts = {".wav", ".mp3", ".flac", ".aiff", ".aif", ".ogg", ".m4a"}
    files = []
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return files


def _derive_out_paths(
    target_path: Path,
    out_dir: Path,
    report_dir: Path,
    preset: str,
    suffix: str = "",
) -> Tuple[Path, Path]:
    stem = target_path.stem
    safe_suffix = suffix.strip()
    tag = f"_{preset}_{MAESTRO_VERSION}"
    if safe_suffix:
        tag = f"_{safe_suffix}{tag}"
    out_wav = out_dir / f"{stem}{tag}.wav"
    out_rep = report_dir / f"{stem}{tag}_Report.md"
    return out_wav, out_rep


def _parser_defaults(ap: argparse.ArgumentParser) -> Dict[str, object]:
    d = {}
    for a in ap._actions:
        if not a.dest or a.dest == "help":
            continue
        d[a.dest] = a.default
    return d


def _apply_preset_defaults(args: argparse.Namespace, preset_name: str, defaults: Dict[str, object]) -> None:
    preset = PRESETS.get(preset_name, {})
    for k, v in preset.items():
        if not hasattr(args, k):
            continue

        cur = getattr(args, k)
        default = defaults.get(k, None)

        # Only apply preset if user did NOT override (cur still equals parser default)
        if cur == default:
            setattr(args, k, v)


def process_one(
    args: argparse.Namespace,
    ref_path: Path,
    tgt_path: Path,
    out_path: Path,
    report_path: Path,
) -> Dict:
    """
    Single-target mastering pipeline.

    IMPORTANT: This restores the full v7.0 DSP chain order, then adds optional v7.1 stages:
      1) Load + resample
      2) Demucs stems (default ON) + stem-aware de-ess + key glow + scale shimmer + transient detect (drums)
      3) Match EQ FIR (psychoacoustic-safe)
      4) v7.1 Mono Sub Anchor (optional)
      5) v7.1 Tilt EQ (optional)
      6) Spatial widening (v7.0)
      7) Movement automation (v7.0)
      8) v7.1 Groove Glue (optional)
      9) v7.1 HookLift (optional)
     10) Finalize (LUFS + clipper + TP chasing limiter) (v7.0)
     11) Post-limiter transient restore + post TP guard (v7.0)

    Returns: payload dict (report data).
    """
    sr = int(args.sr)

    LOG.info("Loading reference: %s", ref_path)
    ref, _ = load_audio_any(ref_path, sr=sr)

    LOG.info("Loading target: %s", tgt_path)
    tgt, _ = load_audio_any(tgt_path, sr=sr)

    pre = tgt.copy()

    stems_required = not bool(args.allow_no_stems)
    stems_enabled = not bool(args.disable_stems)  # v7.1 default ON
    if getattr(args, "enable_stems", False):
        stems_enabled = True  # backwards compat

    stems = None
    stems_info: Dict = {"enabled": False}

    if stems_enabled:
        if not _HAS_DEMUCS:
            msg = "Demucs stems requested (default in v7.1) but Demucs is not installed/available."
            if stems_required:
                raise RuntimeError(msg + " Install demucs (pip install demucs torch) or pass --allow_no_stems.")
            LOG.warning(msg + " Falling back to no-stems because --allow_no_stems was set.")
        else:
            LOG.info("Running Demucs separation (model=%s device=%s)...", args.demucs_model, args.demucs_device)
            stems, stems_info = demucs_separate_stems(
                tgt,
                sr=sr,
                model_name=str(args.demucs_model),
                device=str(args.demucs_device),
                split=True,
                overlap=float(args.demucs_overlap),
            )

    # ---------------------------------------------------------------------
    # STEM-AWARE PROCESSING (v7.0 canonical)
    # ---------------------------------------------------------------------
    snare_times = np.array([], dtype=np.int64)
    snare_info = {"enabled": False}
    glow_info = {"enabled": False}
    shimmer_info = {"enabled": False}
    stem_block_info: Dict = {"enabled": False}

    x = tgt.copy()

    # FIX #6: Headroom early
    ref = safe_headroom(ref, headroom_db=float(args.headroom_db))
    # Note: we work on 'x' mostly, but let's safe headroom 'x' which is 'tgt' copy
    x = safe_headroom(x, headroom_db=float(args.headroom_db))

    if stems is not None:
        drums = stems.get("drums", np.zeros_like(tgt))
        bass = stems.get("bass", np.zeros_like(tgt))
        other = stems.get("other", np.zeros_like(tgt))
        vocals = stems.get("vocals", np.zeros_like(tgt))

        # De-ess ONLY vocals (preserve hats/air elsewhere)
        vocals_deess, deess_info = dynamic_deesser(
            vocals,
            sr=sr,
            band_low=float(args.deess_low),
            band_high=float(args.deess_high),
            threshold_db=float(args.deess_threshold_db),
            ratio=float(args.deess_ratio),
            attack_ms=2.0,
            release_ms=60.0,
        )

        vocals = vocals_deess.astype(np.float32)

        # Key glow + scale shimmer on OTHER + VOCALS buses (musical sparkle)

        glow_info_other = {"enabled": False}
        glow_info_voc = {"enabled": False}
        
        # KEY DETECT FROM MONO TARGET (Fix #5)
        tgt_key_est = None
        if args.enable_key_glow or args.enable_scale_shimmer:
            if not HAVE_LIBROSA:
                raise RuntimeError("librosa required for Key Glow / Scale Shimmer (pip install librosa)")
            tgt_key_est = estimate_key_ks(to_mono(x), sr)

        if args.enable_key_glow:
            other, glow_info_other = key_aware_harmonic_glow(
                other,
                sr=sr,
                detected_key=tgt_key_est,
                glow_gain_db=float(args.glow_gain_db),
                glow_q=float(args.glow_q),
                mix=float(args.glow_mix),
            )
            vocals, glow_info_voc = key_aware_harmonic_glow(
                vocals,
                sr=sr,
                detected_key=tgt_key_est,
                glow_gain_db=float(args.glow_gain_db),
                glow_q=float(args.glow_q),
                mix=float(args.glow_mix),
            )
            glow_info = {"enabled": True, "other": glow_info_other, "vocals": glow_info_voc}

        if args.enable_scale_shimmer:
            shimmer_info_other = {"enabled": False}
            shimmer_info_voc = {"enabled": False}
            other, shimmer_info_other = scale_shimmer_exciter(
                other,
                sr=sr,
                detected_key=tgt_key_est,
                drive=float(args.shimmer_drive),
                mix=float(args.shimmer_mix),
                band_gain_db=float(args.shimmer_band_gain_db),
                q=float(args.shimmer_q),
            )
            vocals, shimmer_info_voc = scale_shimmer_exciter(
                vocals,
                sr=sr,
                detected_key=tgt_key_est,
                drive=float(args.shimmer_drive),
                mix=float(args.shimmer_mix),
                band_gain_db=float(args.shimmer_band_gain_db),
                q=float(args.shimmer_q),
            )
            shimmer_info = {"enabled": True, "other": shimmer_info_other, "vocals": shimmer_info_voc}

        # Transient detection on DRUMS before limiting (for later restore)
        if args.enable_transient_restore:
            snare_times, snare_info = detect_snare_transients(drums, sr=sr)

        x = (drums + bass + other + vocals).astype(np.float32)
        stem_block_info = {
            "enabled": True,
            "vocals_deess": deess_info,
            "key_glow": glow_info,
            "scale_shimmer": shimmer_info,
            "snare_detect": snare_info,
        }

    else:
        # Key glow on full mix
        tgt_key_est = None
        if args.enable_key_glow or args.enable_scale_shimmer:
            if not HAVE_LIBROSA:
                raise RuntimeError("librosa required for Key Glow / Scale Shimmer (pip install librosa)")
            tgt_key_est = estimate_key_ks(to_mono(x), sr)

        if args.enable_key_glow:
            x, glow_info = key_aware_harmonic_glow(
                x,
                sr=sr,
                detected_key=tgt_key_est,
                glow_gain_db=float(args.glow_gain_db),
                glow_q=float(args.glow_q),
                mix=float(args.glow_mix),
            )

        if args.enable_scale_shimmer:
            x, shimmer_info = scale_shimmer_exciter(
                x,
                sr=sr,
                detected_key=tgt_key_est,
                drive=float(args.shimmer_drive),
                mix=float(args.shimmer_mix),
                band_gain_db=float(args.shimmer_band_gain_db),
                q=float(args.shimmer_q),
            )

        # Transient detection on full mix (less precise)
        if args.enable_transient_restore:
            snare_times, snare_info = detect_snare_transients(x, sr=sr)

        stem_block_info = {
            "enabled": False,
            "key_glow": glow_info,
            "scale_shimmer": shimmer_info,
            "snare_detect": snare_info,
        }

    # ---------------------------------------------------------------------
    # MATCH EQ (v7.0 canonical)
    # ---------------------------------------------------------------------
    LOG.info("Designing Match EQ FIR (taps=%s strength=%.2f)...", args.fir_taps, args.match_strength)
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

    # ---------------------------------------------------------------------
    # v7.1 Mono Sub Anchor (optional)
    # ---------------------------------------------------------------------
    mono_sub_info = {"enabled": False}
    if (not bool(args.disable_mono_sub)) and bool(args.enable_mono_sub):
        x, mono_sub_info = mono_sub_anchor(
            x,
            sr=sr,
            cutoff_hz=float(args.mono_sub_hz),
            mix=float(args.mono_sub_mix),
        )

    # ---------------------------------------------------------------------
    # v7.1 Tilt EQ (optional)
    # ---------------------------------------------------------------------
    tilt_info = {"enabled": False}
    # Auto-Tilt Automation (Feature 1)
    if bool(args.auto_tilt) and abs(float(args.tilt_db_per_oct)) < 1e-6:
        slope = estimate_tilt_db_per_oct_from_reference(
            to_mono(ref), to_mono(x), sr=sr, pivot_hz=float(args.tilt_pivot_hz)
        )
        slope *= float(np.clip(args.auto_tilt_strength, 0.0, 1.0))
        args.tilt_db_per_oct = float(slope)
        tilt_info["auto"] = True
        tilt_info["auto_strength"] = float(args.auto_tilt_strength)

    if abs(float(args.tilt_db_per_oct)) > 1e-6:
        ht, t_inf = design_tilt_fir(
            sr=sr,
            numtaps=int(args.tilt_taps),
            tilt_db_per_oct=float(args.tilt_db_per_oct),
            pivot_hz=float(args.tilt_pivot_hz),
            max_tilt_db=float(args.tilt_max_db),
            guard_lo_hz=float(args.tilt_guard_lo_hz),
            guard_hi_hz=float(args.tilt_guard_hi_hz),
            phase=str(args.eq_phase),
        )
        x = apply_fir_fft(x, ht)
        tilt_info.update(t_inf)

    # ---------------------------------------------------------------------
    # Spatial + movement (v7.0)
    # ---------------------------------------------------------------------
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

    # ---------------------------------------------------------------------
    # v7.1 Groove Glue (optional)
    # ---------------------------------------------------------------------
    groove_info = {"enabled": False}
    if bool(args.enable_groove_glue):
        x, groove_info = groove_glue_saturator(
            x,
            sr=sr,
            base_drive=float(args.groove_glue_drive),
            dynamic=float(args.groove_glue_dynamic),
            mix=float(args.groove_glue_mix),
        )

    # ---------------------------------------------------------------------
    # v7.1 HookLift (optional)
    # ---------------------------------------------------------------------
    hook_info = {"enabled": False}
    if (not bool(args.disable_hooklift)) and bool(args.enable_hooklift):
        if bool(getattr(args, "enable_hooklift_auto", False)):
            # Auto-HookLift (Feature 2)
            mask = build_section_lift_mask(
                x,
                sr=sr,
                percentile=float(args.hooklift_auto_percentile),
            )
            # Apply lift fully then mix via mask
            lifted, h_inf = hooklift(x, sr=sr, mix=float(args.hooklift_mix))
            # x is broadcast, mask needs shape match [N] -> [N,1] or similar
            # mask is 1D [N], x is [N,2]
            mask_col = mask[:, None]
            x = (1.0 - mask_col) * x + mask_col * lifted
            hook_info = h_inf
            hook_info["auto"] = True
            hook_info["auto_percentile"] = float(args.hooklift_auto_percentile)
        else:
            x, hook_info = hooklift(
                x,
                sr=sr,
                mix=float(args.hooklift_mix),
            )

    # ---------------------------------------------------------------------
    # FINALIZE (LUFS + TP limiter) (v7.0)
    # ---------------------------------------------------------------------
    LOG.info("Finalizing (target_lufs=%.2f, TP<=%.2f dBFS)...", args.target_lufs, args.target_peak_dbfs)
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

    # POST-LIMITER TRANSIENT RESTORE (v7.0)
    transient_restore_info = {"enabled": False}
    if args.enable_transient_restore and snare_times.size > 0:
        x, transient_restore_info = micro_attack_restore(
            x,
            sr=sr,
            transient_samples=snare_times,
            restore_db=float(args.attack_restore_db),
            mix=float(args.attack_restore_mix),
        )
        # second TP limiter pass to guarantee ceiling after transient injection
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
    report_path.parent.mkdir(parents=True, exist_ok=True)

    sf.write(out_path, x, sr, subtype="PCM_24")

    payload: Dict = {
        "version": MAESTRO_VERSION,
        "paths": {"reference": str(ref_path), "target": str(tgt_path), "out": str(out_path)},
        "stems": stems_info,
        "stem_block": stem_block_info,
        "match_eq": eq_info,
        "mono_sub": mono_sub_info,
        "tilt_eq": tilt_info,
        "spatial": spatial_info,
        "movement": movement_info,
        "groove_glue": groove_info,
        "hooklift": hook_info,
        "finalize": finalize_info,
        "transient_restore": transient_restore_info,
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
    }

    plots: Dict[str, str] = {}
    eq_plot = plot_eq_delta(report_path, eq_info)
    if eq_plot:
        plots["eq_delta"] = eq_plot
    spectrum_plot = plot_spectrum_overlay(report_path, sr, pre, x)
    if spectrum_plot:
        plots["spectrum_overlay"] = spectrum_plot
    shimmer_plot = plot_scale_shimmer_overlay(report_path, sr, x, stem_block_info.get("scale_shimmer", {}))
    if shimmer_plot:
        plots["scale_shimmer"] = shimmer_plot
    payload["plots"] = plots

    write_report(report_path, payload)
    LOG.info("Wrote: %s", out_path)
    LOG.info("Report: %s", report_path)
    return payload


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="AuralMind Match Maestro v7.1 — v7.0 chain + batch + presets + mono-sub + tilt + groove glue"
    )

    # Preset / batch I/O
    ap.add_argument("--preset", default="balanced_v7", choices=list(PRESETS.keys()), help="Named preset bundle.")
    ap.add_argument("--reference", required=True, type=str, help="Reference audio path.")
    ap.add_argument("--target", required=False, type=str, help="Target audio path (single-file mode).")
    ap.add_argument("--target_dir", required=False, type=str, help="Folder of target files (batch mode).")
    ap.add_argument("--out", required=False, type=str, help="Output wav path (single-file mode).")
    ap.add_argument("--report", required=False, type=str, help="Markdown report path (single-file mode).")
    ap.add_argument("--out_dir", required=False, type=str, help="Output folder for batch mode.")
    ap.add_argument("--report_dir", required=False, type=str, help="Report folder for batch mode.")
    ap.add_argument("--suffix", default="", type=str, help="Optional suffix inserted before preset/version tag.")
    ap.add_argument("--dry_run", action="store_true", help="List work without rendering outputs.")
    ap.add_argument("--jobs", type=int, default=10, help="Batch concurrency (advanced). Default 1 (safe).")

    ap.add_argument("--log_level", default="INFO", type=str, help="Logging level.")
    ap.add_argument("--log_file", default=None, type=str, help="Optional log file path.")

    # Core audio options (v7.0)
    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("--headroom_db", type=float, default=-6.0)

    # Match EQ (v7.0)
    ap.add_argument("--fir_taps", type=int, default=4097)
    ap.add_argument("--max_eq_db", type=float, default=7.0)
    ap.add_argument("--eq_smooth_hz", type=float, default=80.0)
    ap.add_argument("--match_strength", type=float, default=0.69)
    ap.add_argument("--match_strength_lo_hz", type=float, default=120.0)
    ap.add_argument("--match_strength_hi_hz", type=float, default=9000.0)
    ap.add_argument("--match_strength_lo_factor", type=float, default=0.72)
    ap.add_argument("--match_strength_hi_factor", type=float, default=0.69)
    ap.add_argument("--disable_eq_guardrails", action="store_true")
    ap.add_argument("--eq_phase", type=str, default="minimum", choices=["minimum", "linear"])
    ap.add_argument("--eq_minphase_nfft", type=int, default=16384)

    # De-ess (v7.0, vocals bus)
    ap.add_argument("--deess_low", type=float, default=6000.0)
    ap.add_argument("--deess_high", type=float, default=10000.0)
    ap.add_argument("--deess_threshold_db", type=float, default=-22.0)
    ap.add_argument("--deess_ratio", type=float, default=2.0)

    # Key Glow (v7.0)
    ap.add_argument("--enable_key_glow", action="store_true")
    ap.add_argument("--glow_gain_db", type=float, default=0.85)
    ap.add_argument("--glow_q", type=float, default=1.2)
    ap.add_argument("--glow_mix", type=float, default=0.55)

    # Scale Shimmer Exciter (v7.1)
    ap.add_argument("--enable_scale_shimmer", action="store_true")
    ap.add_argument("--shimmer_drive", type=float, default=1.6)
    ap.add_argument("--shimmer_mix", type=float, default=0.06)
    ap.add_argument("--shimmer_band_gain_db", type=float, default=1.0)
    ap.add_argument("--shimmer_q", type=float, default=9.0)

    # Spatial (v7.0)
    ap.add_argument("--enable_spatial", action="store_true")
    ap.add_argument("--stereo_side_hp_hz", type=float, default=180.0)
    ap.add_argument("--stereo_width_mid", type=float, default=1.07)
    ap.add_argument("--stereo_width_hi", type=float, default=1.25)
    ap.add_argument("--stereo_corr_min", type=float, default=0.00)

    # Movement (v7.0)
    ap.add_argument("--enable_movement", action="store_true")
    ap.add_argument("--rhythm_amount", type=float, default=0.15)

    # Transient restore (v7.0)
    ap.add_argument("--enable_transient_restore", action="store_true")
    ap.add_argument("--attack_restore_db", type=float, default=1.2)
    ap.add_argument("--attack_restore_mix", type=float, default=0.60)

    # Finalize / limiter (v7.0)
    ap.add_argument("--target_lufs", type=float, default=-11.0)
    ap.add_argument("--target_peak_dbfs", type=float, default=-1.0)
    ap.add_argument("--tp_oversample", type=int, default=8)
    ap.add_argument("--finalize_iters", type=int, default=3)
    ap.add_argument("--clip_drive_db", type=float, default=2.2)
    ap.add_argument("--clip_mix", type=float, default=0.20)
    ap.add_argument("--clip_oversample", type=int, default=4)
    ap.add_argument("--limit_lookahead_ms", type=float, default=2.7)
    ap.add_argument("--limit_attack_ms", type=float, default=1.0)
    ap.add_argument("--limit_release_ms", type=float, default=60.0)
    ap.add_argument("--ceiling_chase_strength", type=float, default=1.0)

    # Demucs stems (v7.1 default ON)
    ap.add_argument("--enable_stems", action="store_true", help="Deprecated: stems are ON by default in v7.1.")
    ap.add_argument("--disable_stems", action="store_true", help="Disable stems (not recommended).")
    ap.add_argument("--allow_no_stems", action="store_true", help="Allow fallback when Demucs missing.")
    ap.add_argument("--demucs_model", type=str, default="htdemucs")
    ap.add_argument("--demucs_device", type=str, default="cpu")
    ap.add_argument("--demucs_overlap", type=float, default=0.25)

    # v7.1 Mono Sub Anchor
    ap.add_argument("--enable_mono_sub", action="store_true", help="Enable Mono Sub Anchor (recommended).")
    ap.add_argument("--disable_mono_sub", action="store_true", help="Force-disable Mono Sub Anchor.")
    ap.add_argument("--mono_sub_hz", type=float, default=120.0)
    ap.add_argument("--mono_sub_mix", type=float, default=1.0)

    # v7.1 Tilt EQ
    ap.add_argument("--tilt_db_per_oct", type=float, default=0.13, help="Tilt slope in dB/oct (positive=brighter).")
    ap.add_argument("--tilt_pivot_hz", type=float, default=1000.0)
    ap.add_argument("--tilt_max_db", type=float, default=3.0)
    ap.add_argument("--tilt_guard_lo_hz", type=float, default=35.0)
    ap.add_argument("--tilt_guard_hi_hz", type=float, default=14000.0)
    ap.add_argument("--tilt_taps", type=int, default=513)
    ap.add_argument("--auto_tilt", action="store_true", help="Auto-estimate tilt slope from reference (safe).")
    ap.add_argument("--auto_tilt_strength", type=float, default=0.80)

    # v7.1 Groove Glue
    ap.add_argument("--enable_groove_glue", action="store_true")
    ap.add_argument("--groove_glue_drive", type=float, default=1.15)
    ap.add_argument("--groove_glue_dynamic", type=float, default=0.23)
    ap.add_argument("--groove_glue_mix", type=float, default=0.12)

    # v7.1 HookLift
    ap.add_argument("--enable_hooklift", action="store_true")
    ap.add_argument("--disable_hooklift", action="store_true")
    ap.add_argument("--hooklift_mix", type=float, default=0.23)
    ap.add_argument("--enable_hooklift_auto", action="store_true")
    ap.add_argument("--hooklift_auto_percentile", type=float, default=75.0)

    return ap


def self_test() -> None:
    """
    Minimal self-test:
      - validates preset names
      - validates Match EQ + limiter functions exist
      - validates chain order marker strings
    """
    assert isinstance(PRESETS, dict) and len(PRESETS) > 0
    assert "legacy_v7" in PRESETS
    for fn in [design_match_fir, finalize_master, true_peak_chasing_limiter, mono_sub_anchor, design_tilt_fir]:
        assert callable(fn)
    LOG.info("Self-test OK.")


def main() -> None:
    ap = _build_parser()
    args = ap.parse_args()
    preset_name = str(args.preset)
    defaults = _parser_defaults(ap)
    _apply_preset_defaults(args, preset_name, defaults)


    _setup_logging(args.log_level, args.log_file)
    ref_path = Path(args.reference)

    is_batch = bool(args.target_dir)
    if is_batch:
        target_dir = Path(args.target_dir)
        targets = _list_audio_files(target_dir)
        if not targets:
            raise RuntimeError(f"No audio files found in {target_dir}")
        out_dir = Path(args.out_dir) if args.out_dir else (target_dir / "maestro_out")
        rep_dir = Path(args.report_dir) if args.report_dir else (target_dir / "maestro_reports")
        out_dir.mkdir(parents=True, exist_ok=True)
        rep_dir.mkdir(parents=True, exist_ok=True)

        LOG.info("Batch mode: %d files | preset=%s", len(targets), preset_name)

        iterable = targets
        if tqdm is not None:
            iterable = tqdm(targets, desc="Maestro v7.1", unit="file")

        for tgt_path in iterable:
            out_path, rep_path = _derive_out_paths(tgt_path, out_dir, rep_dir, preset_name, suffix=str(args.suffix))
            if args.dry_run:
                LOG.info("[DRY] %s -> %s", tgt_path.name, out_path.name)
                continue
            try:
                process_one(args, ref_path, tgt_path, out_path, rep_path)
            except Exception as e:
                LOG.exception("FAILED %s: %s", tgt_path, e)

        LOG.info("Batch complete.")
        return

    # Single-file mode validation
    if not args.target or not args.out or not args.report:
        raise RuntimeError("Single-file mode requires --target, --out, and --report (or use --target_dir for batch).")

    tgt_path = Path(args.target)
    out_path = Path(args.out)
    rep_path = Path(args.report)

    if args.dry_run:
        LOG.info("[DRY] %s -> %s", tgt_path.name, out_path.name)
        return

    process_one(args, ref_path, tgt_path, out_path, rep_path)



if __name__ == "__main__":
    main()

"""
# =============================================================================
# v7.1 Quick Commands (PowerShell)
# =============================================================================
# Single file:
# python auralmind_match_maestro_v7_1.py --preset cinematic_punch --reference "C:\\Users\\goku\\Downloads\\Brent Faiyaz - Pistachios [Official Video].mp3" --target "C:\\Users\\goku\\Downloads\\Don't let me down.wav" --out "C:/out/master.wav" --report "master_Report.md"

# Batch folder:
# python auralmind_match_maestro_v7_1.py --preset airy_streaming --reference "C:/path/ref.wav" --target_dir "C:/folder/targets" --out_dir "C:/folder/out" --report_dir "C:/folder/reports"
#
# Legacy v7.0-like:
# python auralmind_match_maestro_v7_1.py --preset legacy_v7 --reference "..." --target "..." --out "..." --report "..."
"""
