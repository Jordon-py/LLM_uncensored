#!/usr/bin/env python3
"""
Reflexion Architect X - LF–W1 Algorithm v3.1
=============================================
Primary Mission: Transform Melodic Trap masters to SURPASS reference quality.

Operational Axes:
1. LOGIC:      Linear-Phase Spectral Matching (Steal the frequency curve).
2. STRATEGY:   Sub-Bass Phase Alignment + Mono-Lock (Perfect translation).
3. CREATIVITY: Prism Air Exciter + Mid-Presence Saturation (Emotional depth).

Usage:
  # Single file
  python auralmind_reflexion_x.py --reference "C:\\Users\\goku\\Downloads\\Brent Faiyaz - Pistachios [Official Video].wav" --target "C:\\Users\\goku\\Downloads\\Vegas - top teir (20).wav" --out "Vegas-Top Tier.wav" --preset trap_hifi

  # Batch (directory or multiple files)
  python auralmind_reflexion_x.py --reference "ref.wav" --target "mixes/" --out_dir "out/" --report_dir "out/reports/" --preset trap_streaming

Notes:
  - `pyloudnorm` is optional; if installed it's used for LUFS, otherwise an internal BS.1770-style fallback runs.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import soundfile as sf
import librosa
try:
    import pyloudnorm as pyln  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    pyln = None
from scipy.signal import firwin2, fftconvolve, savgol_filter, butter, filtfilt, lfilter

# -----------------------------
# REFLEXION LOGGING SYSTEM
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | REFLEXION-X | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
LOG = logging.getLogger("ReflexionX")

# -----------------------------
# PRESETS (Ease of use)
# -----------------------------

PRESETS: Dict[str, Dict[str, float]] = {
    # Safer ceiling + gentler enhancement for streaming platforms.
    "trap_streaming": {
        "sr": 44100.0,
        "fir_taps": 2049.0,
        "max_eq_db": 9.0,
        "eq_smooth_semitones": 3.5,
        "sub_align_cutoff_hz": 120.0,
        "bass_root_boost_db": 0.6,
        "bass_octave_boost_db": 0.3,
        "bass_fifth_boost_db": 0.15,
        "bass_q": 0.9,
        "drive": 0.12,
        "ceiling_db": -1.0,
    },
    # Default "master-quality" trap profile.
    "trap_hifi": {
        "sr": 48000.0,
        "fir_taps": 4097.0,
        "max_eq_db": 10.0,
        "eq_smooth_semitones": 3.0,
        "sub_align_cutoff_hz": 120.0,
        "bass_root_boost_db": 0.8,
        "bass_octave_boost_db": 0.4,
        "bass_fifth_boost_db": 0.2,
        "bass_q": 1.0,
        "drive": 0.15,
        "ceiling_db": -0.6,
    },
    # Loud / punchy, use with caution.
    "trap_aggressive": {
        "sr": 48000.0,
        "fir_taps": 4097.0,
        "max_eq_db": 12.0,
        "eq_smooth_semitones": 2.5,
        "sub_align_cutoff_hz": 120.0,
        "bass_root_boost_db": 1.2,
        "bass_octave_boost_db": 0.6,
        "bass_fifth_boost_db": 0.3,
        "bass_q": 1.1,
        "drive": 0.18,
        "ceiling_db": -0.3,
    },
}

# -----------------------------
# I/O & PRE-PROCESSING (The "Senses")
# -----------------------------

def _require_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found. Install it to decode non-WAV formats.")
    return ffmpeg

def decode_audio(path: Path, sr: int) -> Tuple[np.ndarray, int]:
    """Load audio into the Reflexion Matrix (Float32 Stereo)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input signature not found: {path}")

    # Try fast path
    if path.suffix.lower() in {".wav", ".flac", ".aiff"}:
        try:
            y, file_sr = sf.read(path, always_2d=True)
            y = y.astype(np.float32)
            if file_sr != sr:
                y = librosa.resample(y.T, orig_sr=file_sr, target_sr=sr).T.astype(np.float32)
            return _ensure_stereo(y), sr
        except Exception:
            pass

    # Fallback to FFmpeg
    ffmpeg = _require_ffmpeg()
    with tempfile.TemporaryDirectory() as td:
        tmp_wav = Path(td) / "temp_decode.wav"
        subprocess.run([
            ffmpeg, "-y", "-v", "error", "-i", str(path),
            "-vn", "-ac", "2", "-ar", str(sr), "-f", "wav", "-c:a", "pcm_f32le",
            str(tmp_wav)
        ], check=True)
        y, _ = sf.read(tmp_wav, always_2d=True)
        return _ensure_stereo(y.astype(np.float32)), sr

def _ensure_stereo(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1: return np.column_stack((x, x))
    if x.shape[1] == 1: return np.repeat(x, 2, axis=1)
    if x.shape[1] > 2: return x[:, :2]
    return x

def _ensure_finite(x: np.ndarray) -> np.ndarray:
    if not np.isfinite(x).all():
        LOG.warning("Non-finite samples detected. Repairing signal topology.")
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x

# -----------------------------
# METRICS & ANALYSIS (The "Logic Core")
# -----------------------------

@dataclass
class SonicMetrics:
    lufs_i: float
    crest_db: float
    spectral_centroid: float
    sub_energy: float
    air_energy: float
    key: Optional[str] = None
    key_confidence: float = 0.0
    bass_note: Optional[str] = None
    bass_freq_hz: float = 0.0


def _biquad_highpass(sr: int, f0_hz: float, q: float) -> Tuple[np.ndarray, np.ndarray]:
    # RBJ cookbook high-pass (used for BS.1770 K-weighting fallback).
    w0 = 2.0 * math.pi * (f0_hz / sr)
    cos_w0 = math.cos(w0)
    sin_w0 = math.sin(w0)
    alpha = sin_w0 / (2.0 * q)
    b0 = (1.0 + cos_w0) * 0.5
    b1 = -(1.0 + cos_w0)
    b2 = (1.0 + cos_w0) * 0.5
    a0 = 1.0 + alpha
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha
    b = (np.array([b0, b1, b2], dtype=np.float64) / a0).astype(np.float64)
    a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
    return b, a


def _biquad_high_shelf(sr: int, f0_hz: float, gain_db: float, slope: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    # RBJ cookbook high-shelf (used for BS.1770 K-weighting fallback).
    if slope <= 0:
        raise ValueError("slope must be > 0")
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * (f0_hz / sr)
    cos_w0 = math.cos(w0)
    sin_w0 = math.sin(w0)
    alpha = (sin_w0 / 2.0) * math.sqrt((A + 1.0 / A) * (1.0 / slope - 1.0) + 2.0)
    two_sqrt_A_alpha = 2.0 * math.sqrt(A) * alpha

    b0 = A * ((A + 1.0) + (A - 1.0) * cos_w0 + two_sqrt_A_alpha)
    b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * cos_w0)
    b2 = A * ((A + 1.0) + (A - 1.0) * cos_w0 - two_sqrt_A_alpha)
    a0 = (A + 1.0) - (A - 1.0) * cos_w0 + two_sqrt_A_alpha
    a1 = 2.0 * ((A - 1.0) - (A + 1.0) * cos_w0)
    a2 = (A + 1.0) - (A - 1.0) * cos_w0 - two_sqrt_A_alpha
    b = (np.array([b0, b1, b2], dtype=np.float64) / a0).astype(np.float64)
    a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
    return b, a


def _k_weight(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Approximate ITU-R BS.1770 K-weighting filter (2 biquad cascade).
    Used only when `pyloudnorm` isn't available.
    """
    b_hp, a_hp = _biquad_highpass(sr, 38.13547087602444, 0.5003270373238773)
    b_hs, a_hs = _biquad_high_shelf(sr, 1681.974450955533, gain_db=4.0, slope=1.0)
    y = lfilter(b_hp, a_hp, audio, axis=0)
    if isinstance(y, tuple):
        y = y[0]
    y2 = lfilter(b_hs, a_hs, y, axis=0)
    if isinstance(y2, tuple):
        y2 = y2[0]
    return np.asarray(y2)


def integrated_lufs(audio: np.ndarray, sr: int) -> float:
    """
    Integrated loudness in LUFS.

    Uses `pyloudnorm` when available. If not installed, falls back to an internal
    BS.1770-style implementation (K-weighting + absolute/relative gating).
    """
    audio = _ensure_finite(_ensure_stereo(audio))
    if pyln is not None:
        try:
            meter = pyln.Meter(sr)
            return float(meter.integrated_loudness(audio))
        except Exception as e:
            LOG.warning("pyloudnorm loudness failed (%s). Falling back to internal LUFS.", e)

    y = np.asarray(_k_weight(audio, sr))
    block = int(round(0.400 * sr))
    hop = int(round(0.100 * sr))
    if y.shape[0] < block:
        ms = float(np.mean(np.sum(y * y, axis=1)))
        return float(-0.691 + 10.0 * math.log10(max(ms, 1e-12)))

    starts = np.arange(0, y.shape[0] - block + 1, hop, dtype=np.int64)
    ms_blocks = np.empty(len(starts), dtype=np.float64)
    for i, s in enumerate(starts):
        seg = y[s : s + block, :]
        ms_blocks[i] = float(np.mean(np.sum(seg * seg, axis=1)))

    l_blocks = -0.691 + 10.0 * np.log10(np.maximum(ms_blocks, 1e-12))
    abs_gate = -70.0
    keep_abs = l_blocks > abs_gate
    if not np.any(keep_abs):
        return float(np.max(l_blocks))

    ms_abs = ms_blocks[keep_abs]
    l_ungated = float(-0.691 + 10.0 * math.log10(max(float(np.mean(ms_abs)), 1e-12)))
    rel_gate = l_ungated - 10.0
    keep_rel = l_blocks > rel_gate
    keep = keep_abs & keep_rel
    ms_final = ms_blocks[keep] if np.any(keep) else ms_abs
    ms = float(np.mean(ms_final))
    return float(-0.691 + 10.0 * math.log10(max(ms, 1e-12)))


_NOTE_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_NOTE_TO_PC = {n: i for i, n in enumerate(_NOTE_NAMES_SHARP)}
_NOTE_TO_PC.update({"Db": 1, "Eb": 3, "Gb": 6, "Ab": 8, "Bb": 10})

_KRUMHANSL_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    dtype=np.float64,
)
_KRUMHANSL_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    dtype=np.float64,
)
_KRUMHANSL_MAJOR /= np.linalg.norm(_KRUMHANSL_MAJOR) + 1e-12
_KRUMHANSL_MINOR /= np.linalg.norm(_KRUMHANSL_MINOR) + 1e-12


def estimate_key(mono: np.ndarray, sr: int) -> Tuple[Optional[str], float]:
    """
    Lightweight key estimate (Krumhansl-Schmuckler via chroma).

    Returns (key_name, confidence), where confidence is the separation between the best and
    runner-up template correlations (higher is better).
    """
    mono = _ensure_finite(mono.astype(np.float32, copy=False))
    if mono.size < sr // 2:
        return None, 0.0

    chroma = librosa.feature.chroma_stft(y=mono, sr=sr, n_fft=4096, hop_length=2048)
    chroma_mean = chroma.mean(axis=1)
    norm = float(np.linalg.norm(chroma_mean))
    if not np.isfinite(norm) or norm <= 1e-12:
        return None, 0.0
    chroma_mean = chroma_mean / norm

    best: Tuple[str, float] = ("C major", float("-inf"))
    runner_up: Tuple[str, float] = ("C major", float("-inf"))
    for pc in range(12):
        maj = float(np.dot(chroma_mean, np.roll(_KRUMHANSL_MAJOR, pc)))
        if maj > best[1]:
            runner_up = best
            best = (f"{_NOTE_NAMES_SHARP[pc]} major", maj)
        elif maj > runner_up[1]:
            runner_up = (f"{_NOTE_NAMES_SHARP[pc]} major", maj)

        minor = float(np.dot(chroma_mean, np.roll(_KRUMHANSL_MINOR, pc)))
        if minor > best[1]:
            runner_up = best
            best = (f"{_NOTE_NAMES_SHARP[pc]} minor", minor)
        elif minor > runner_up[1]:
            runner_up = (f"{_NOTE_NAMES_SHARP[pc]} minor", minor)

    confidence = float(max(0.0, best[1] - runner_up[1]))
    return best[0], confidence


def estimate_bass_note(mono: np.ndarray, sr: int, fmin_hz: float = 25.0, fmax_hz: float = 140.0) -> Tuple[Optional[str], float]:
    """
    Estimate the dominant bass note via long-term low-band spectrum peak.
    """
    mono = _ensure_finite(mono.astype(np.float32, copy=False))
    if mono.size < sr // 4:
        return None, 0.0

    n_fft = 8192
    hop = 2048
    S = np.abs(librosa.stft(mono, n_fft=n_fft, hop_length=hop, window="hann", center=True)) + 1e-12
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    band = (freqs >= fmin_hz) & (freqs <= min(fmax_hz, sr / 2 - 1.0))
    if not np.any(band):
        return None, 0.0

    band_mag = np.mean(S[band, :], axis=1)
    peak_i = int(np.argmax(band_mag))
    f0 = float(freqs[band][peak_i])
    if not np.isfinite(f0) or f0 <= 0:
        return None, 0.0
    return str(librosa.hz_to_note(f0, octave=True)), f0


def analyze_audio(audio: np.ndarray, sr: int) -> SonicMetrics:
    """Perform Deep Spectral Forensics."""
    mono = np.mean(audio, axis=1)
    try:
        lufs = integrated_lufs(audio, sr)
    except Exception as e:
        LOG.warning("LUFS analysis failed (%s). Falling back to RMS-based estimate.", e)
        rms = float(np.sqrt(np.mean(mono * mono)))
        lufs = float(-0.691 + 20.0 * math.log10(max(rms, 1e-12)))

    rms = np.sqrt(np.mean(mono**2))
    peak = np.max(np.abs(mono))
    crest = 20 * np.log10(peak / (rms + 1e-9))

    sc = librosa.feature.spectral_centroid(y=mono, sr=sr).mean()

    # Energy bands (Simple FFT)
    S = np.abs(librosa.stft(mono, n_fft=2048))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    sub_idx = np.where(freqs < 60)[0]
    air_idx = np.where(freqs > 10000)[0]

    sub_e = np.mean(S[sub_idx, :])
    air_e = np.mean(S[air_idx, :])

    key, key_conf = estimate_key(mono, sr)
    bass_note, bass_hz = estimate_bass_note(mono, sr)

    return SonicMetrics(
        lufs_i=float(lufs),
        crest_db=float(crest),
        spectral_centroid=float(sc),
        sub_energy=float(sub_e),
        air_energy=float(air_e),
        key=key,
        key_confidence=float(key_conf),
        bass_note=bass_note,
        bass_freq_hz=float(bass_hz),
    )

def db(x): return 20 * np.log10(np.maximum(x, 1e-12))
def undb(x): return 10 ** (x / 20.0)

# -----------------------------
# DSP MODULES (The "Evolutionary Tools")
# -----------------------------

def _smooth_curve_semitones(freqs_hz: np.ndarray, curve_db: np.ndarray, smooth_semitones: float) -> np.ndarray:
    """
    Pitch-aware smoothing for match-EQ curves.

    Smoothing on a semitone axis (log-frequency) better respects musical harmonic spacing
    than smoothing directly over linear FFT bins.
    """
    if smooth_semitones <= 0:
        return curve_db
    freqs_hz = np.asarray(freqs_hz, dtype=np.float64)
    curve_db = np.asarray(curve_db, dtype=np.float64)
    if freqs_hz.shape != curve_db.shape:
        raise ValueError("freqs_hz and curve_db must have the same shape")

    mask = freqs_hz > 0
    if not np.any(mask):
        return curve_db

    f = freqs_hz[mask]
    y = curve_db[mask]

    semitones = np.asarray(12.0 * np.log2(f / 55.0), dtype=np.float64)  # A1 reference; only relative spacing matters
    step = 0.25  # quarter-semitone grid keeps detail but avoids bin-level overfitting
    grid = np.arange(semitones[0], semitones[-1] + step, step, dtype=np.float64)
    y_grid = np.asarray(np.interp(grid, semitones, np.asarray(y, dtype=np.float64)), dtype=np.float64)

    win = int(round(max(3.0, smooth_semitones / step)))
    if win % 2 == 0:
        win += 1
    # Ensure odd window length <= len(y_grid)
    if win >= len(y_grid):
        win = len(y_grid) - 1 if (len(y_grid) % 2 == 0) else len(y_grid)
    if win >= 5:
        poly = 3 if win >= 7 else 2
        y_grid = np.asarray(
            savgol_filter(y_grid, window_length=win, polyorder=min(poly, win - 1)),
            dtype=np.float64,
        )

    y_smooth = np.asarray(np.interp(semitones, grid, y_grid), dtype=np.float64)
    out = curve_db.copy()
    out[mask] = y_smooth
    return out


def design_linear_phase_eq(
    ref: np.ndarray,
    tgt: np.ndarray,
    sr: int,
    taps: int = 2049,
    max_eq_db: float = 12.0,
    smooth_semitones: float = 3.0,
) -> np.ndarray:
    """
    LOGIC AXIS: Match the reference tonal balance using Linear Phase FIR.
    Reasoning: Linear phase preserves the transient punch of trap drums.
    """
    LOG.info("LOGIC: Computing Spectral DNA Match...")

    def get_spectrum(x):
        # Long window for spectral accuracy
        S = np.abs(librosa.stft(np.mean(x, axis=1), n_fft=4096))
        return db(np.mean(S, axis=1))

    ref_spec = get_spectrum(ref)
    tgt_spec = get_spectrum(tgt)

    # Calculate Delta
    delta = ref_spec - tgt_spec

    # Frequency mapping
    freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
    # Pitch-aware smoothing (semitone domain)
    delta_smooth = _smooth_curve_semitones(freqs, delta, smooth_semitones=smooth_semitones)

    # Intelligent Clamping (Don't boost mud or insane hiss)
    delta_smooth = np.clip(delta_smooth, -float(max_eq_db), float(max_eq_db))
    target_mag = undb(delta_smooth)

    # FIR Design
    # Normalize freqs to 0-1 for firwin2
    freqs_norm = freqs / (sr / 2)
    freqs_norm[0], freqs_norm[-1] = 0.0, 1.0

    h = np.asarray(firwin2(taps, freqs_norm, target_mag, window="hann"), dtype=np.float64)
    return h.astype(np.float32)

def apply_fir(audio: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Apply the DNA Match."""
    out = np.zeros_like(audio)
    for i in range(2):
        out[:, i] = fftconvolve(audio[:, i], h, mode='same')
    return out

def align_sub_phase(audio: np.ndarray, sr: int, cutoff=120.0) -> Tuple[np.ndarray, float]:
    """
    STRATEGY AXIS: Align Sub-Bass Phase.
    Reasoning: Trap 808s often drift in stereo. This aligns L/R for max impact.
    """
    # 1. Split Lows
    b, a = butter(4, cutoff / (sr / 2), btype="low", output="ba")
    lows = np.array([filtfilt(b, a, audio[:, 0]), filtfilt(b, a, audio[:, 1])]).T
    highs = audio - lows

    # 2. Correlate to find lag
    # Avoid full `np.correlate` on long tracks (O(N^2)).
    decim = 4
    l_d = np.ascontiguousarray(lows[::decim, 0], dtype=np.float64)
    r_d = np.ascontiguousarray(lows[::decim, 1], dtype=np.float64)
    n = int(min(l_d.shape[0], r_d.shape[0]))
    if n < 16:
        lag = 0
    else:
        l_d = l_d[:n]
        r_d = r_d[:n]

        # Search only within the range we would ever apply anyway.
        max_shift_ms = 4.0
        max_lag = int(round((max_shift_ms / 1000.0) * (sr / decim)))
        max_lag = max(1, min(max_lag, n - 1))

        def _best_lag(a: np.ndarray, b: np.ndarray, max_lag: int) -> int:
            a = np.asarray(a, dtype=np.float64).ravel()
            b = np.asarray(b, dtype=np.float64).ravel()
            m = int(min(a.shape[0], b.shape[0]))
            if m < 2:
                return 0
            a = a[:m] - float(np.mean(a[:m]))
            b = b[:m] - float(np.mean(b[:m]))

            best_lag = 0
            best_corr = -float("inf")
            max_lag = int(max(1, min(max_lag, m - 1)))
            for lag in range(-max_lag, max_lag + 1):
                if lag >= 0:
                    x = a[lag:]
                    y = b[: m - lag]
                    denom = m - lag
                else:
                    k = -lag
                    x = a[: m - k]
                    y = b[k:]
                    denom = m - k
                if denom <= 0:
                    continue
                corr = float(np.dot(x, y) / float(denom))
                if corr > best_corr:
                    best_corr = corr
                    best_lag = lag
            return int(best_lag)

        lag = _best_lag(l_d, r_d, max_lag=max_lag)

    lag_ms = (lag * decim / sr) * 1000.0

    def _shift_zeros(x: np.ndarray, shift: int) -> np.ndarray:
        if shift == 0:
            return x
        shift = int(shift)
        y = np.zeros_like(x)
        if abs(shift) >= x.shape[0]:
            return y
        if shift > 0:
            y[shift:] = x[:-shift]
        else:
            s = -shift
            y[:-s] = x[s:]
        return y

    if abs(lag_ms) > 0.1 and abs(lag_ms) < 4.0:
        LOG.info(f"STRATEGY: Detected 808 Phase Drift: {lag_ms:.2f}ms. Aligning...")
        # Shift Right channel to match Left
        shift_samples = int(lag * decim)
        lows[:, 1] = _shift_zeros(lows[:, 1], shift_samples)

    # 3. Force Mono (The "Sub-Anchor")
    # Average L/R for a rock-solid center
    mono_lows = np.mean(lows, axis=1)
    lows_aligned = np.column_stack((mono_lows, mono_lows))

    return (highs + lows_aligned), lag_ms


def harmonic_bass_anchor(
    audio: np.ndarray,
    sr: int,
    key: Optional[str],
    key_confidence: float,
    bass_note: Optional[str],
    bass_freq_hz: float,
    root_boost_db: float = 0.8,
    octave_boost_db: float = 0.4,
    fifth_boost_db: float = 0.2,
    q: float = 1.0,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    MUSIC THEORY MODULE: Harmonic Bass Anchor.

    Reinforces the root + octave (+ optional fifth) to make the 808 note read clearly on
    small speakers while keeping sub translation solid.
    """
    info: Dict[str, float] = {}
    audio = _ensure_finite(_ensure_stereo(audio))

    if root_boost_db == 0.0 and octave_boost_db == 0.0 and fifth_boost_db == 0.0:
        return audio, info

    # Prefer the estimated key root if confidence is reasonable; otherwise fall back to the dominant bass note.
    anchor_note: Optional[str] = None
    anchor_source: str = "none"
    if key and key_confidence >= 0.08:
        anchor_note = key.split()[0]
        anchor_source = "key"
    elif bass_note:
        anchor_note = "".join([c for c in bass_note if not c.isdigit()])
        anchor_source = "bass"

    if not anchor_note:
        return audio, info

    pc = _NOTE_TO_PC.get(anchor_note)
    if pc is None:
        return audio, info

    root_midi = 24 + pc  # C1 = 24; trap 808 fundamentals commonly live around octave 1
    root_hz = float(librosa.midi_to_hz(root_midi))
    octave_hz = root_hz * 2.0
    fifth_hz = float(librosa.midi_to_hz(root_midi + 7))

    def apply_peak(x: np.ndarray, f0: float, gain_db: float) -> np.ndarray:
        if gain_db == 0.0 or f0 <= 0.0 or f0 >= (sr / 2 - 1.0):
            return x
        # filtfilt applies the magnitude response twice -> halve the requested gain.
        A = 10.0 ** ((gain_db / 2.0) / 40.0)
        w0 = 2.0 * math.pi * (f0 / sr)
        cos_w0 = math.cos(w0)
        sin_w0 = math.sin(w0)
        alpha = sin_w0 / (2.0 * max(q, 1e-6))
        b0 = 1.0 + alpha * A
        b1 = -2.0 * cos_w0
        b2 = 1.0 - alpha * A
        a0 = 1.0 + alpha / A
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha / A
        b = np.array([b0, b1, b2], dtype=np.float64) / a0
        a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
        return filtfilt(b, a, x, axis=0).astype(np.float32)

    out = audio
    out = apply_peak(out, root_hz, root_boost_db)
    out = apply_peak(out, octave_hz, octave_boost_db)
    out = apply_peak(out, fifth_hz, fifth_boost_db)

    info.update(
        {
            "anchor_source": 1.0 if anchor_source == "key" else 0.0,
            "anchor_root_hz": float(root_hz),
            "anchor_octave_hz": float(octave_hz),
            "anchor_fifth_hz": float(fifth_hz),
            "root_boost_db": float(root_boost_db),
            "octave_boost_db": float(octave_boost_db),
            "fifth_boost_db": float(fifth_boost_db),
        }
    )
    if bass_freq_hz > 0:
        info["detected_bass_hz"] = float(bass_freq_hz)

    LOG.info(
        "THEORY: Harmonic Bass Anchor (%s) root=%.1fHz octave=%.1fHz fifth=%.1fHz",
        anchor_source,
        root_hz,
        octave_hz,
        fifth_hz,
    )
    return out, info


def prism_air_exciter(audio: np.ndarray, sr: int, drive: float = 0.15) -> np.ndarray:
    """
    CREATIVITY AXIS: The 'Prism' Effect.
    Reasoning: Reference tracks often have 'expensive' air. EQ isn't enough.
    We generate new harmonics in the side channel to widen and brighten.
    """
    LOG.info(f"CREATIVITY: Engaging Prism Air Exciter (Drive: {drive})...")

    # 1. Isolate Highs (Source)
    b_hp, a_hp = butter(2, 3500 / (sr / 2), btype="high", output="ba")
    highs = filtfilt(b_hp, a_hp, audio, axis=0)

    # 2. Generate Harmonics (Distortion)
    # Tanh creates odd harmonics, Abs creates even. Mix them.
    harmonics = np.tanh(highs * 2.0) + (np.abs(highs) * 0.5)

    # 3. Filter Harmonics (Keep only the sheen)
    b_air, a_air = butter(2, 6000 / (sr / 2), btype="high", output="ba")
    air_sheen = filtfilt(b_air, a_air, harmonics, axis=0)

    # 4. Inject into SIDE channel only
    # Mid = (L+R)/2, Side = (L-R)/2
    mid = np.mean(audio, axis=1)
    side = (audio[:, 0] - audio[:, 1]) * 0.5

    # Add sheen to side (creates width)
    side_new = side + (np.mean(air_sheen, axis=1) * drive)

    # Decode M/S
    l_new = mid + side_new
    r_new = mid - side_new

    return np.column_stack((l_new, r_new))

def dynamic_presence(audio: np.ndarray, sr: int, amount: float = 0.1) -> np.ndarray:
    """
    CREATIVITY AXIS: Mid-Side Vocal Bloomer.
    Reasoning: Brings vocals forward without volume automation.
    """
    mid = np.mean(audio, axis=1)
    side = (audio[:, 0] - audio[:, 1]) * 0.5

    # Bandpass the vocal range on Mid
    b, a = butter(
        2,
        [800 / (sr / 2), 4000 / (sr / 2)],
        btype="band",
        output="ba",
    )
    vocal_band = filtfilt(b, a, mid)

    # Parallel Saturation on vocal band
    sat_vocals = np.tanh(vocal_band * 1.5)

    # Blend back
    mid_new = mid + (sat_vocals * amount)

    # Reconstruct
    l = mid_new + side
    r = mid_new - side
    return np.column_stack((l, r))

def trap_clipper(audio: np.ndarray, ceiling_db: float = -0.5) -> np.ndarray:
    """
    STRATEGY AXIS: Soft-Clip Limiter.
    Reasoning: Trap needs to be loud. Tanh provides that 'saturated' loudness
    typical of the genre, surpassing standard digital limiting.
    """
    ceiling = 10**(ceiling_db/20)

    # 1. Hard drive into tanh (Simulates analog rail clipping)
    # Automatically gain stage based on peak
    current_peak = np.max(np.abs(audio))
    target_drive = 1.0
    if current_peak > ceiling:
        # Push just enough to round off the peaks
        target_drive = current_peak / ceiling

    driven = audio * target_drive
    clipped = np.tanh(driven)

    # Normalize back to ceiling
    return clipped * ceiling

# -----------------------------
# REFLEXION REPORTING (The "Self-Critique")
# -----------------------------

def generate_report(
    path: Path,
    metrics_pre: SonicMetrics,
    metrics_post: SonicMetrics,
    ref_metrics: SonicMetrics,
    lag: float,
    bass_anchor_info: Optional[Dict[str, float]] = None,
    fir_taps: int = 2049,
    sub_align_cutoff_hz: float = 120.0,
    ceiling_db: float = -0.6,
):

    # Meta-Critique Logic
    critiques = []

    d_crest = metrics_pre.crest_db - metrics_post.crest_db
    if d_crest > 4.0:
        critiques.append(f"⚠️ **Dynamic Reduction High ({d_crest:.1f}dB):** The master is significantly denser. Ensure transients still punch.")
    else:
        critiques.append("✅ **Dynamics Preserved:** Transient integrity maintained within competitive range.")

    if metrics_post.air_energy > (ref_metrics.air_energy * 1.2):
         critiques.append("⚠️ **High Frequency Alert:** Air energy exceeds reference by >20%. Check for harshness.")
    else:
         critiques.append("✅ **Spectral Balance:** High-end luminance matches or tastefully exceeds reference.")

    def _fmt_note(note: Optional[str], hz: float) -> str:
        if note and hz > 0:
            return f"{note} ({hz:.1f} Hz)"
        return "—"

    bass_anchor_line = ""
    if bass_anchor_info:
        src = "key" if float(bass_anchor_info.get("anchor_source", 0.0)) >= 0.5 else "bass"
        root_hz = float(bass_anchor_info.get("anchor_root_hz", 0.0))
        oct_hz = float(bass_anchor_info.get("anchor_octave_hz", 0.0))
        fifth_hz = float(bass_anchor_info.get("anchor_fifth_hz", 0.0))
        bass_anchor_line = f"* Harmonic Bass Anchor ({src}): root {root_hz:.1f} Hz, octave {oct_hz:.1f} Hz, fifth {fifth_hz:.1f} Hz."

    report = f"""
# Reflexion Architect X - Mastering Report
**Date:** {time.strftime("%Y-%m-%d %H:%M:%S")}
**Algorithm:** LF-W1 v3.1 (Hybrid)

## 1. Sonic Forensics
| Metric | Reference | Input (Mix) | **Master (Output)** |
| :--- | :--- | :--- | :--- |
| **LUFS (Int)** | {ref_metrics.lufs_i:.1f} | {metrics_pre.lufs_i:.1f} | **{metrics_post.lufs_i:.1f}** |
| **Crest Factor** | {ref_metrics.crest_db:.1f} dB | {metrics_pre.crest_db:.1f} dB | **{metrics_post.crest_db:.1f} dB** |
| **Spectral Centroid** | {int(ref_metrics.spectral_centroid)} Hz | {int(metrics_pre.spectral_centroid)} Hz | **{int(metrics_post.spectral_centroid)} Hz** |
| **Key (Est.)** | {ref_metrics.key or "—"} | {metrics_pre.key or "—"} | **{metrics_post.key or "—"}** |
| **Bass Note (Est.)** | {_fmt_note(ref_metrics.bass_note, ref_metrics.bass_freq_hz)} | {_fmt_note(metrics_pre.bass_note, metrics_pre.bass_freq_hz)} | **{_fmt_note(metrics_post.bass_note, metrics_post.bass_freq_hz)}** |

## 2. Enhancement Log
* **Logic Phase:** Pitch-aware Linear-Phase FIR EQ match ({fir_taps} taps).
* **Strategy Phase:**
    * Sub-Bass Alignment: {lag:.2f}ms drift corrected.
    * Mono-Lock applied < {sub_align_cutoff_hz:.0f}Hz.
    {bass_anchor_line}
* **Creativity Phase:**
    * Prism Air Exciter engaged (Side-Channel Harmonics).
    * Mid-Presence Vocal saturation applied.
    * Trap Soft-Clipper applied (Ceiling: {ceiling_db:.1f}dB).

## 3. Reflexion Meta-Critique (Self-Evolving Analysis)
> "Did I merely match the reference, or did I elevate the art?"

{chr(10).join([f"- {c}" for c in critiques])}

---
*Generated by AuralMind Reflexion X*
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    LOG.info(f"Report compiled: {path}")

# -----------------------------
# MAIN LOOP
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="AuralMind Reflexion X - The Next-Gen Trap Master")
    parser.add_argument("--reference", required=True, type=Path, help="Reference track")
    parser.add_argument(
        "--target",
        required=True,
        type=Path,
        nargs="+",
        help="Mix(es) to master (files and/or directories)",
    )
    parser.add_argument("--out", type=Path, default=None, help="Output file path (single target only)")
    parser.add_argument("--out_dir", type=Path, default=None, help="Output directory (batch mode)")
    parser.add_argument("--report", type=Path, default=None, help="Report path (single target only)")
    parser.add_argument("--report_dir", type=Path, default=None, help="Report output directory (batch mode)")

    parser.add_argument("--preset", choices=sorted(PRESETS.keys()), default=None, help="Preset profile")
    parser.add_argument("--sr", type=int, default=None, help="Sample Rate")
    parser.add_argument("--fir_taps", type=int, default=None, help="FIR taps for match EQ")
    parser.add_argument("--max_eq_db", type=float, default=None, help="Max match-EQ boost/cut (dB)")
    parser.add_argument("--eq_smooth_semitones", type=float, default=None, help="Match-EQ smoothing in semitones")
    parser.add_argument("--sub_align_cutoff_hz", type=float, default=None, help="Sub mono/phase cutoff (Hz)")
    parser.add_argument("--bass_root_boost_db", type=float, default=None, help="Bass anchor root boost (dB)")
    parser.add_argument("--bass_octave_boost_db", type=float, default=None, help="Bass anchor octave boost (dB)")
    parser.add_argument("--bass_fifth_boost_db", type=float, default=None, help="Bass anchor fifth boost (dB)")
    parser.add_argument("--bass_q", type=float, default=None, help="Bass anchor filter Q")
    parser.add_argument("--drive", type=float, default=None, help="Prism Air Drive (0.0 - 0.5)")
    parser.add_argument("--ceiling_db", type=float, default=None, help="Trap clipper ceiling (dBFS)")
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    preset = PRESETS.get(args.preset or "", {})

    def pick(name: str, default):
        v = getattr(args, name)
        if v is not None:
            return v
        if name in preset:
            return preset[name]
        return default

    sr = int(pick("sr", 44100))
    fir_taps = int(pick("fir_taps", 2049))
    max_eq_db = float(pick("max_eq_db", 12.0))
    eq_smooth_semitones = float(pick("eq_smooth_semitones", 3.0))
    sub_align_cutoff_hz = float(pick("sub_align_cutoff_hz", 120.0))
    bass_root_boost_db = float(pick("bass_root_boost_db", 0.8))
    bass_octave_boost_db = float(pick("bass_octave_boost_db", 0.4))
    bass_fifth_boost_db = float(pick("bass_fifth_boost_db", 0.2))
    bass_q = float(pick("bass_q", 1.0))
    drive = float(pick("drive", 0.15))
    ceiling_db = float(pick("ceiling_db", -0.6))

    supported_exts = {".wav", ".flac", ".aiff", ".aif", ".mp3", ".m4a", ".ogg", ".aac"}

    def gather_targets(paths: List[Path]) -> List[Path]:
        out: List[Path] = []
        for p in paths:
            p = Path(p)
            if p.is_dir():
                out.extend(sorted([f for f in p.iterdir() if f.is_file() and f.suffix.lower() in supported_exts]))
            else:
                out.append(p)
        # Deduplicate while preserving order
        seen: set[str] = set()
        uniq: List[Path] = []
        for p in out:
            key = str(p.resolve()) if p.exists() else str(p)
            if key not in seen:
                uniq.append(p)
                seen.add(key)
        return uniq

    targets = gather_targets(args.target)
    if not targets:
        raise SystemExit("No target files found.")
    missing = [str(p) for p in targets if not p.exists()]
    if missing:
        raise SystemExit(f"Target file(s) not found: {', '.join(missing)}")

    batch_mode = len(targets) > 1
    if batch_mode and args.out is not None:
        raise SystemExit("When processing multiple targets, use --out_dir instead of --out.")
    if batch_mode and args.report is not None:
        raise SystemExit("When processing multiple targets, use --report_dir instead of --report.")

    out_dir = args.out_dir
    report_dir = args.report_dir
    if batch_mode:
        out_dir = out_dir or Path("reflexion_out")
        report_dir = report_dir or out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)

    # 1. INITIALIZATION
    LOG.info("Initializing Reflexion Loop...")
    ref, _ = decode_audio(args.reference, sr)

    # 2. ANALYSIS (Phase 1)
    LOG.info("Phase 1: Deep Spectral Forensics...")
    m_ref = analyze_audio(ref, sr)
    LOG.info("Reference LUFS: %.1f | Key: %s", m_ref.lufs_i, m_ref.key or "-")

    for i, target_path in enumerate(targets, start=1):
        LOG.info("------------------------------------------------")
        LOG.info("Target %d/%d: %s", i, len(targets), target_path)

        tgt, _ = decode_audio(target_path, sr)
        m_pre = analyze_audio(tgt, sr)
        LOG.info("Input LUFS: %.1f | Key: %s | Bass: %s", m_pre.lufs_i, m_pre.key or "-", m_pre.bass_note or "-")

        # 3. LOGIC PHASE (Spectral Matching)
        LOG.info("Phase 2: Logic Core (Spectral Matching)...")
        h_match = design_linear_phase_eq(
            ref,
            tgt,
            sr,
            taps=fir_taps,
            max_eq_db=max_eq_db,
            smooth_semitones=eq_smooth_semitones,
        )
        processed = apply_fir(tgt, h_match)

        # 4. STRATEGY PHASE (The Low End)
        LOG.info("Phase 3: Strategy Core (Sub-Bass Architecture)...")
        processed, lag = align_sub_phase(processed, sr, cutoff=sub_align_cutoff_hz)
        processed, bass_anchor_info = harmonic_bass_anchor(
            processed,
            sr,
            key=m_pre.key,
            key_confidence=m_pre.key_confidence,
            bass_note=m_pre.bass_note,
            bass_freq_hz=m_pre.bass_freq_hz,
            root_boost_db=bass_root_boost_db,
            octave_boost_db=bass_octave_boost_db,
            fifth_boost_db=bass_fifth_boost_db,
            q=bass_q,
        )

        # 5. CREATIVITY PHASE (The Vibe)
        LOG.info("Phase 4: Creativity Core (Harmonic Sculpting)...")
        processed = dynamic_presence(processed, sr, amount=0.12)
        processed = prism_air_exciter(processed, sr, drive=drive)

        # 6. FINALIZING (Loudness & Safety)
        LOG.info("Phase 5: Final Polish (Trap Clipping)...")
        try:
            pre_clip_lufs = integrated_lufs(processed, sr)
            gain_needed = m_ref.lufs_i - pre_clip_lufs
        except Exception as e:
            LOG.warning("Pre-clip LUFS failed (%s). Using input LUFS delta.", e)
            pre_clip_lufs = m_pre.lufs_i
            gain_needed = m_ref.lufs_i - m_pre.lufs_i

        gain_apply = gain_needed * 0.85  # leave room for saturation/clipper to do the final loudness shaping
        LOG.info("Pre-clip LUFS: %.1f | Applying %.1f dB gain", pre_clip_lufs, gain_apply)
        processed = processed * (10 ** (gain_apply / 20.0))
        processed = trap_clipper(processed, ceiling_db=ceiling_db)
        m_post = analyze_audio(processed, sr)

        if batch_mode:
            assert out_dir is not None
            assert report_dir is not None
            out_path = out_dir / f"{target_path.stem}_reflexion_master.wav"
            report_path = report_dir / f"{target_path.stem}_reflexion_master.md"
        else:
            out_path = args.out or Path("reflexion_master.wav")
            report_path = args.report or out_path.with_suffix(".md")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_path, processed, sr, subtype="PCM_24")
        generate_report(
            report_path,
            m_pre,
            m_post,
            m_ref,
            lag,
            bass_anchor_info=bass_anchor_info,
            fir_taps=fir_taps,
            sub_align_cutoff_hz=sub_align_cutoff_hz,
            ceiling_db=ceiling_db,
        )

        LOG.info("Wrote master: %s", out_path)
        LOG.info("Wrote report: %s", report_path)

    LOG.info("------------------------------------------------")
    LOG.info("MISSION COMPLETE.")
    LOG.info("------------------------------------------------")

if __name__ == "__main__":
    main()
