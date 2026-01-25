#!/usr/bin/env python3
"""
AuralMind v5 — Reference-Based Mastering Enhancer
=================================================

What this script does (musically):
----------------------------------
This is a *reference-guided mastering chain* designed for modern melodic trap,
where you typically want:

- Controlled, centered sub (808s translate everywhere)
- Clean punch (kick/snare stay forward)
- Smooth, expensive top-end (air without harshness)
- Tonal coherence (melody + vocal sit in the same "color space" as the reference)
- Reliable loudness matching (LUFS) with true-peak-safe output

Pipeline:
---------
1) Decode/normalize inputs to float32 stereo at --sr.
2) Measure loudness + long-term spectrum.
3) Design a linear-phase FIR EQ to nudge target toward reference tone.
   - NEW: frequency-dependent EQ guardrails (safer subs + smoother highs)
4) Apply FIR EQ to target.
5) Sub-bass phase align + low-end mono focus (optional).
6) Sub-Anchor low-end stabilizer (optional).
7) NEW: Adaptive Harmonic Balancer (innovative feature)
   - Infers *key + mode* (major/minor) from the audio
   - Applies musically-aware, *even-harmonic warmth* to the MID channel
   - Adjusts drive based on tonal density (chroma entropy) for coherence
8) Harmonic luminance sculptor (air sheen) (optional).
9) Air & Presence: dynamic de-essing + mid expressiveness (optional).
10) Optional gentle broadband glue compression (only if crest differs from reference).
11) LUFS match to the reference.
12) NEW: true-peak-safe limiter scaling (oversampled), optional.
13) Write output audio + Markdown report.
    python auralmind_match_v5.py
  --reference "C:/Users/goku/Downloads/Lil Wayne_She Will.mp3"
  --target "C:\\Users\\goku\\Downloads\\Its still love baby (1).wav"
  --out "C:/Users/goku/Downloads/Vegas - its all love_hifinew.wav"
  --report "C:/Users/goku/Downloads/Vegas - its all love_hifi_report.md"
   --sr 48000
  --fir_taps 4097
  --max_eq_db 8
  --eq_smooth_hz 90
  --enable_compression
  --sub_align_cutoff_hz 120 --sub_align_max_ms 1.5 --sub_align_mono_strength 0.60
  --sub_anchor_cutoff_hz 125 --sub_anchor_threshold_db -22 --sub_anchor_ratio 2.4 --sub_anchor_attack_ms 8 --sub_anchor_release_ms 170
  --sub_anchor_sat_mix 0.08 --sub_anchor_sat_drive_db 2.4
  --hb_bp_lo 200 --hb_bp_hi 4200 --hb_mix 0.075 --hb_drive_db 2.2 --hb_dyn_depth_db 2.3 --hb_env_ms 85
  --hb_asym_major 0.060 --hb_asym_minor 0.120 --hb_tension_sensitivity 0.33
  --luminance_hp_hz 7200 --luminance_mix 0.055 --luminance_drive_db 1.6 --luminance_dyn_depth_db 2.0
  --deesser_hp_hz 6500 --deesser_lp_hz 12000 --deesser_thresh 0.12 --deesser_max_reduction_db 4 --deesser_env_ms 25
  --presence_bp_lo 850 --presence_bp_hi 3600 --presence_mix 0.060 --presence_drive_db 1.5 --presence_dyn_depth_db 2.0 --presence_env_ms 70
  --limiter_mode true_peak --limiter_oversample 4 --target_peak_dbfs -1.0
  --dither --dither_amount 1.0
  --log_level INFO

Design philosophy:
------------------
- Tone first (spectral match), then low-end stability, then musical harmonic depth,
  then top-end control, then loudness + safety.
- Conservative defaults; melodic trap can be loud, but "hi-fi" is about control,
  not destruction.

Constraints:
------------
This script does NOT perform offline rendering simulation here.
It is intended for real-world use in Python.

Dependencies:
-------------
numpy, scipy, soundfile, librosa, pyloudnorm
ffmpeg recommended for mp3/other formats
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import soundfile as sf
import librosa
import pyloudnorm as pyln
from scipy.signal import firwin, firwin2, fftconvolve, savgol_filter, resample_poly

LOG = logging.getLogger("auralmind")


# -----------------------------
# IO utilities
# -----------------------------

def _require_ffmpeg() -> str:
    """Return the path to ffmpeg if installed, else raise a helpful error."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install ffmpeg or provide WAV inputs./n"
            "Windows: choco install ffmpeg/n"
            "macOS: brew install ffmpeg/n"
            "Linux: sudo apt-get install ffmpeg"
        )
    return ffmpeg


def _run(cmd: List[str]) -> None:
    LOG.debug("Running: %s", " ".join(cmd))
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}/n{p.stderr}")


def decode_with_ffmpeg_to_wav(input_path: Path, out_wav_path: Path, sr: int, channels: int = 2) -> None:
    """Decode any audio to float32 WAV at specified sample rate and channel count."""
    ffmpeg = _require_ffmpeg()
    cmd = [
        ffmpeg, "-y",
        "-i", str(input_path),
        "-vn",
        "-ac", str(channels),
        "-ar", str(sr),
        "-acodec", "pcm_f32le",
        str(out_wav_path),
    ]
    _run(cmd)


def load_audio_any_format(path: Path, sr: int) -> Tuple[np.ndarray, int]:
    """
    Load audio (wav/mp3/...) as float32 stereo at sr.

    Uses:
      - soundfile for WAV/FLAC/AIFF
      - ffmpeg fallback for other formats
    """
    path = Path(path)
    if path.suffix.lower() in {".wav", ".flac", ".aiff", ".aif"}:
        y, file_sr = sf.read(path, always_2d=True)
        y = y.astype(np.float32)
        if file_sr != sr:
            y = librosa.resample(y.T, orig_sr=file_sr, target_sr=sr).T.astype(np.float32)
        if y.shape[1] == 1:
            y = np.repeat(y, 2, axis=1)
        elif y.shape[1] > 2:
            y = y[:, :2]
        return y, sr

    # Non-wav: use ffmpeg to temp WAV
    with tempfile.TemporaryDirectory() as td:
        tmp_wav = Path(td) / "tmp_decode.wav"
        decode_with_ffmpeg_to_wav(path, tmp_wav, sr=sr, channels=2)
        y, file_sr = sf.read(tmp_wav, always_2d=True)
        y = y.astype(np.float32)
        if file_sr != sr:
            y = librosa.resample(y.T, orig_sr=file_sr, target_sr=sr).T.astype(np.float32)
        return y, sr


# -----------------------------
# Numeric utilities
# -----------------------------

def ensure_finite(x: np.ndarray, name: str = "array") -> np.ndarray:
    """Replace NaN/inf with zeros and warn."""
    if not np.isfinite(x).all():
        LOG.warning("%s contained NaN/inf; replacing with zeros.", name)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x.astype(np.float32, copy=False)


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Stereo -> mono (mid) using simple average."""
    if audio.ndim == 1:
        return audio.astype(np.float32, copy=False)
    if audio.ndim == 2 and audio.shape[1] >= 2:
        return (0.5 * (audio[:, 0] + audio[:, 1])).astype(np.float32)
    return audio.reshape(-1).astype(np.float32, copy=False)


def db(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(x), eps))


def undb(dbv: np.ndarray) -> np.ndarray:
    return 10.0 ** (dbv / 20.0)


def peak_dbfs(audio: np.ndarray) -> float:
    p = float(np.max(np.abs(audio)))
    return float(db(np.array([p]))[0])


def rms_dbfs(audio: np.ndarray) -> float:
    r = float(np.sqrt(np.mean(audio * audio) + 1e-12))
    return float(db(np.array([r]))[0])


# -----------------------------
# Analysis
# -----------------------------

@dataclass
class AudioMetrics:
    sr: int
    lufs_i: float
    rms_dbfs: float
    peak_dbfs: float
    crest_db: float
    spectral_centroid_hz: float
    band_db: Dict[str, float]
    duration_s: float


def measure_lufs_integrated(mono: np.ndarray, sr: int) -> float:
    """Integrated LUFS using ITU-R BS.1770 gating via pyloudnorm."""
    meter = pyln.Meter(sr)
    return float(meter.integrated_loudness(mono.astype(np.float64)))


def band_energy_db(mono: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Rough band energies (for reporting/diagnostics only).
    These bands map to musical roles in trap mastering.
    """
    n_fft = 4096
    hop = 512
    S = np.abs(librosa.stft(mono, n_fft=n_fft, hop_length=hop, window="hann")) + 1e-12
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    def band(f_lo, f_hi) -> float:
        m = (freqs >= f_lo) & (freqs < f_hi)
        if not np.any(m):
            return -120.0
        val = float(np.mean(S[m, :]))
        return float(db(np.array([val]))[0])

    return {
        "sub": band(20.0, 60.0),                      # sub weight (808 rumble)
        "low": band(60.0, 250.0),                     # punch + body
        "mid": band(250.0, 2000.0),                   # vocal/melody core
        "high": band(2000.0, 12000.0),                # bite + articulation
        "air": band(12000.0, min(20000.0, sr / 2 - 1.0)),  # sheen + space
    }


def spectral_centroid(mono: np.ndarray, sr: int) -> float:
    """Brightness proxy; higher centroid -> brighter/edgier perception."""
    c = librosa.feature.spectral_centroid(y=mono, sr=sr)
    return float(np.mean(c))


def compute_metrics(audio: np.ndarray, sr: int) -> AudioMetrics:
    x = ensure_finite(audio, "audio_for_metrics")
    mono = to_mono(x)
    lufs = measure_lufs_integrated(mono, sr)
    rmsv = rms_dbfs(mono)
    peakv = peak_dbfs(mono)
    crest = float(peakv - rmsv)
    bands = band_energy_db(mono, sr)
    cent = spectral_centroid(mono, sr)
    dur = float(len(mono) / sr)
    return AudioMetrics(
        sr=sr,
        lufs_i=float(lufs),
        rms_dbfs=float(rmsv),
        peak_dbfs=float(peakv),
        crest_db=float(crest),
        spectral_centroid_hz=float(cent),
        band_db=bands,
        duration_s=dur,
    )


def average_spectrum_db(mono: np.ndarray, sr: int, n_fft: int = 8192) -> Tuple[np.ndarray, np.ndarray]:
    """
    Long-term average magnitude spectrum in dB.
    This is the backbone for reference tone matching.
    """
    S = np.abs(librosa.stft(mono, n_fft=n_fft, hop_length=n_fft // 8, window="hann")) + 1e-12
    mag = np.mean(S, axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    spec_db = db(mag)
    return freqs, spec_db


# -----------------------------
# Spectral match EQ (linear-phase FIR)
# -----------------------------

def _freq_guardrails_db(freqs: np.ndarray, delta_db: np.ndarray,
                        low_sub_hz: float = 35.0,
                        low_hz: float = 90.0,
                        high_hz: float = 8000.0,
                        sub_cap_db: float = 3.0,
                        low_cap_db: float = 6.0,
                        high_cap_db: float = 4.0) -> np.ndarray:
    """
    Frequency-dependent guardrails:
    - Extremely low subs rarely benefit from big corrective boosts/cuts
    - Very high frequencies can become harsh quickly

    This preserves "hi-fi trap" polish: tight subs + smooth air.
    """
    out = delta_db.copy()

    m_sub = freqs < low_sub_hz
    m_low = (freqs >= low_sub_hz) & (freqs < low_hz)
    m_high = freqs >= high_hz

    out[m_sub] = np.clip(out[m_sub], -sub_cap_db, sub_cap_db)
    out[m_low] = np.clip(out[m_low], -low_cap_db, low_cap_db)
    out[m_high] = np.clip(out[m_high], -high_cap_db, high_cap_db)
    return out


def design_match_fir(
    ref_mono: np.ndarray,
    tgt_mono: np.ndarray,
    sr: int,
    numtaps: int = 1025,
    max_gain_db: float = 12.0,
    smooth_hz: float = 80.0,
    bass_extra_db: float = 0.0,
    enable_guardrails: bool = True,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Build a linear-phase FIR EQ that nudges target spectrum toward reference spectrum.

    Musically:
    - This is "macro tone" alignment, not surgical resonance fixing.
    - Smoothing prevents chasing hats/room noise.
    - Guardrails protect trap fundamentals + air from over-correction.
    """
    numtaps = int(numtaps)
    if numtaps % 2 == 0:
        numtaps += 1
    max_gain_db = float(max_gain_db)

    freqs, ref_db = average_spectrum_db(ref_mono, sr)
    _, tgt_db = average_spectrum_db(tgt_mono, sr)

    delta_db = ref_db - tgt_db

    # Smooth delta to avoid narrow notches/boosts (keeps the sound "coherent").
    df = float(freqs[1] - freqs[0])
    win = max(5, int(smooth_hz / max(df, 1e-9)))
    if win % 2 == 0:
        win += 1

    max_win = len(delta_db) // 2 * 2 - 1
    win = int(min(win, max_win))
    win = max(win, 5)
    if win % 2 == 0:
        win += 1

    delta_smooth = savgol_filter(delta_db, window_length=win, polyorder=3, mode="interp")

    # Clip correction to avoid extreme EQ moves.
    delta_smooth = np.clip(delta_smooth, -max_gain_db, max_gain_db)

    # Optional guardrails (recommended for trap translation).
    if enable_guardrails:
        delta_smooth = _freq_guardrails_db(freqs, delta_smooth)

    # Optional extra bass shelf (rarely needed; keep at 0 unless intentional).
    if bass_extra_db != 0.0:
        bass_mask = freqs < 90.0
        delta_smooth[bass_mask] = np.clip(delta_smooth[bass_mask] + bass_extra_db,
                                          -max_gain_db, max_gain_db)

    desired_mag = undb(delta_smooth)

    nyq = sr / 2.0
    f_norm = np.clip(freqs / nyq, 0.0, 1.0)

    # firwin2 expects first=0 and last=1
    f_norm[0] = 0.0
    f_norm[-1] = 1.0

    h = firwin2(numtaps, f_norm, desired_mag, window="hann")

    info = {
        "numtaps": float(numtaps),
        "max_gain_db": float(max_gain_db),
        "smooth_hz": float(smooth_hz),
        "bass_extra_db": float(bass_extra_db),
        "guardrails": bool(enable_guardrails),
        "min_gain_db_applied": float(np.min(delta_smooth)),
        "max_gain_db_applied": float(np.max(delta_smooth)),
    }
    return h.astype(np.float32), info


def apply_fir_stereo(audio: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Apply FIR to stereo audio using FFT convolution."""
    x = ensure_finite(audio, "audio_for_fir")
    if x.ndim == 1:
        y = fftconvolve(x, h, mode="same")
        return y.astype(np.float32)
    if x.ndim == 2:
        yL = fftconvolve(x[:, 0], h, mode="same")
        yR = fftconvolve(x[:, 1], h, mode="same")
        return np.stack([yL, yR], axis=1).astype(np.float32)
    raise ValueError("Audio array must be 1D or 2D (stereo).")


def _design_fir_lowpass(sr: int, cutoff_hz: float, numtaps: int = 513) -> np.ndarray:
    cutoff_hz = float(np.clip(cutoff_hz, 10.0, sr / 2 - 10.0))
    return firwin(numtaps, cutoff=cutoff_hz, fs=sr, pass_zero=True, window="hann").astype(np.float32)


def _design_fir_highpass(sr: int, cutoff_hz: float, numtaps: int = 513) -> np.ndarray:
    cutoff_hz = float(np.clip(cutoff_hz, 10.0, sr / 2 - 10.0))
    return firwin(numtaps, cutoff=cutoff_hz, fs=sr, pass_zero=False, window="hann").astype(np.float32)


# -----------------------------
# Sub-bass phase alignment + mono focus
# -----------------------------

def _fractional_shift(x: np.ndarray, shift_samples: float) -> np.ndarray:
    """
    Fractional delay via linear interpolation (small shifts only).
    Positive shift delays the signal (moves it later).

    Musically:
    - Aligning sub phase makes the 808 feel "locked" and solid in mono playback.
    """
    if abs(shift_samples) < 1e-6:
        return x.astype(np.float32, copy=False)

    n = len(x)
    idx = np.arange(n, dtype=np.float32) - float(shift_samples)
    idx0 = np.floor(idx).astype(np.int64)
    idx1 = idx0 + 1
    w = idx - idx0

    idx0 = np.clip(idx0, 0, n - 1)
    idx1 = np.clip(idx1, 0, n - 1)

    y = (1.0 - w) * x[idx0] + w * x[idx1]
    return y.astype(np.float32)


def _estimate_delay_samples(a: np.ndarray, b: np.ndarray, sr: int,
                            max_shift_ms: float = 2.0, decim: int = 8) -> int:
    """Estimate integer-sample delay between a and b using local cross-correlation."""
    max_shift_samp = int(sr * max_shift_ms / 1000.0)
    if max_shift_samp < 1:
        return 0

    ad = a[::decim]
    bd = b[::decim]
    maxd = max_shift_samp // decim

    if len(ad) < 16 or len(bd) < 16:
        return 0

    lags = np.arange(-maxd, maxd + 1, dtype=np.int32)
    best_lag = 0
    best_val = -1e18

    for lag in lags:
        if lag < 0:
            x1 = ad[-lag:]
            x2 = bd[: len(x1)]
        else:
            x1 = ad[: len(ad) - lag]
            x2 = bd[lag: lag + len(x1)]
        if len(x1) < 16:
            continue
        val = float(np.sum(x1 * x2))
        if val > best_val:
            best_val = val
            best_lag = int(lag)

    return int(best_lag * decim)


def align_subbass_phase(
    audio: np.ndarray,
    sr: int,
    cutoff_hz: float = 120.0,
    max_shift_ms: float = 2.0,
    mono_strength: float = 0.6,
    numtaps: int = 513,
) -> Tuple[np.ndarray, float]:
    """
    Align low-band phase between L/R and optionally reduce sub stereo width.

    Why it matters:
    - Trap subs must collapse gracefully to mono (phones, clubs, cars).
    - Phase alignment tightens the perceived punch of the 808.
    """
    if audio.ndim != 2 or audio.shape[1] != 2:
        return audio.astype(np.float32, copy=False), 0.0

    h_lp = _design_fir_lowpass(sr, cutoff_hz=cutoff_hz, numtaps=numtaps)
    low = apply_fir_stereo(audio, h_lp)
    high = (audio - low).astype(np.float32)

    left = low[:, 0]
    right = low[:, 1]
    lag_samples = _estimate_delay_samples(left, right, sr=sr, max_shift_ms=max_shift_ms, decim=8)

    if lag_samples != 0:
        right_aligned = _fractional_shift(right, shift_samples=float(lag_samples))
        low_aligned = np.stack([left, right_aligned], axis=1).astype(np.float32)
    else:
        low_aligned = low.astype(np.float32, copy=False)

    mono_strength = float(np.clip(mono_strength, 0.0, 1.0))
    if mono_strength > 0.0:
        mid = 0.5 * (low_aligned[:, 0] + low_aligned[:, 1])
        side = 0.5 * (low_aligned[:, 0] - low_aligned[:, 1])
        side *= (1.0 - mono_strength)
        low_aligned = np.stack([mid + side, mid - side], axis=1)

    lag_ms = float(lag_samples / sr * 1000.0)
    return (low_aligned + high).astype(np.float32), lag_ms


# -----------------------------
# Harmonic luminance (high-band saturation)
# -----------------------------

def apply_harmonic_luminance(
    audio: np.ndarray,
    sr: int,
    highpass_hz: float = 6500.0,
    mix: float = 0.08,
    drive_db: float = 2.0,
    dyn_depth_db: float = 3.0,
    numtaps: int = 513,
    env_ms: float = 50.0,
) -> np.ndarray:
    """
    Add smooth high-band harmonic sheen with level-adaptive drive.

    Musical rationale:
    - "Air" should feel like polish, not harsh EQ.
    - Dynamic drive adds shimmer when the track is quiet,
      and relaxes when the hats/vocals get intense (prevents ice-pick highs).
    """
    mix = float(np.clip(mix, 0.0, 1.0))
    if mix <= 0.0:
        return audio.astype(np.float32, copy=False)

    h_hp = _design_fir_highpass(sr, cutoff_hz=highpass_hz, numtaps=numtaps)
    high = apply_fir_stereo(audio, h_hp)

    mono_high = to_mono(high)

    win = int(max(8, sr * env_ms / 1000.0))
    kernel = np.ones(win, dtype=np.float32) / float(win)
    env = np.sqrt(np.convolve(mono_high * mono_high, kernel, mode="same") + 1e-12)
    env_norm = env / (np.max(env) + 1e-12)

    drive_curve_db = float(drive_db) + float(dyn_depth_db) * (1.0 - env_norm)
    drive_curve_db = np.clip(drive_curve_db, 0.0, 12.0)

    drive_lin = (10.0 ** (drive_curve_db / 20.0)).astype(np.float32)[:, None]

    sat = np.tanh(high * drive_lin)
    norm = np.tanh(drive_lin) + 1e-6
    sat = sat / norm

    return (audio + mix * sat).astype(np.float32)


# -----------------------------
# Trap enhancements (Sub-Anchor)
# -----------------------------

def _ms_encode(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Mid/Side encode: returns (mid, side) as mono arrays."""
    mid = 0.5 * (x[:, 0] + x[:, 1])
    side = 0.5 * (x[:, 0] - x[:, 1])
    return mid.astype(np.float32), side.astype(np.float32)


def _ms_decode(mid: np.ndarray, side: np.ndarray) -> np.ndarray:
    """Mid/Side decode: returns stereo (L,R)."""
    l = mid + side
    r = mid - side
    return np.stack([l, r], axis=1).astype(np.float32)


def compress_broadband(
    audio: np.ndarray,
    sr: int,
    threshold_db: float = -18.0,
    ratio: float = 2.0,
    attack_ms: float = 10.0,
    release_ms: float = 120.0,
    makeup_db: float = 0.0,
) -> np.ndarray:
    """
    Simple broadband compressor (RMS envelope + smoothed gain).

    Musical rationale:
    - Gentle glue can make the master feel "finished".
    - We keep it conservative to preserve trap transients.
    """
    x = ensure_finite(audio, "audio_for_compression")
    mono = to_mono(x)

    frame = int(0.02 * sr)  # 20ms
    hop = int(0.01 * sr)    # 10ms
    if frame < 16 or hop < 8:
        return x.astype(np.float32)

    rms_vals = []
    positions = []
    for i in range(0, len(mono) - frame, hop):
        seg = mono[i:i + frame]
        rms_vals.append(float(np.sqrt(np.mean(seg * seg) + 1e-12)))
        positions.append(i)

    if not rms_vals:
        LOG.debug("Compression skipped (audio shorter than analysis frame).")
        return x.astype(np.float32)

    rms_vals = np.array(rms_vals, dtype=np.float32)
    level_db = db(rms_vals)

    over_db = np.maximum(0.0, level_db - threshold_db)
    gr_db = over_db - (over_db / max(1e-6, ratio))
    desired_gain_db = -gr_db

    tau_a = max(1e-4, attack_ms / 1000.0)
    tau_r = max(1e-4, release_ms / 1000.0)
    a_a = math.exp(-hop / (sr * tau_a))
    a_r = math.exp(-hop / (sr * tau_r))

    sm = np.zeros_like(desired_gain_db, dtype=np.float32)
    sm[0] = desired_gain_db[0]
    for i in range(1, len(sm)):
        if desired_gain_db[i] < sm[i - 1]:
            sm[i] = a_a * sm[i - 1] + (1.0 - a_a) * desired_gain_db[i]
        else:
            sm[i] = a_r * sm[i - 1] + (1.0 - a_r) * desired_gain_db[i]

    # Interpolate smoothed gain to sample domain (hop resolution).
    gain_db = np.zeros(len(mono), dtype=np.float32)
    for idx, pos in enumerate(positions):
        gain_db[pos:pos + hop] = sm[idx]
    if positions[-1] + hop < len(gain_db):
        gain_db[positions[-1] + hop:] = sm[-1]

    gain_lin = undb(gain_db + makeup_db).astype(np.float32)

    if x.ndim == 1:
        return (x * gain_lin).astype(np.float32)

    yL = x[:, 0] * gain_lin
    yR = x[:, 1] * gain_lin
    return np.stack([yL, yR], axis=1).astype(np.float32)


def sub_anchor(
    audio: np.ndarray,
    sr: int,
    cutoff_hz: float = 120.0,
    comp_threshold_db: float = -24.0,
    ratio: float = 2.5,
    attack_ms: float = 8.0,
    release_ms: float = 160.0,
    sat_mix: float = 0.10,
    sat_drive_db: float = 3.0,
    numtaps: int = 513,
) -> np.ndarray:
    """
    Trap-centric low-end stabilizer (sub-band compression + tiny saturation).

    Musical rationale:
    - 808 note-to-note swings can feel unstable.
    - Gentle control preserves vibe while improving translation.
    """
    if audio.ndim != 2 or audio.shape[1] != 2:
        return audio.astype(np.float32, copy=False)

    cutoff_hz = float(max(20.0, min(cutoff_hz, 300.0)))
    sat_mix = float(np.clip(sat_mix, 0.0, 1.0))

    h_lp = _design_fir_lowpass(sr, cutoff_hz=cutoff_hz, numtaps=numtaps)
    sub = apply_fir_stereo(audio, h_lp)
    high = (audio - sub).astype(np.float32)

    sub_c = compress_broadband(
        sub,
        sr=sr,
        threshold_db=float(comp_threshold_db),
        ratio=float(max(1.0, ratio)),
        attack_ms=float(max(0.5, attack_ms)),
        release_ms=float(max(5.0, release_ms)),
        makeup_db=0.0,
    )

    if sat_mix <= 0.0 or sat_drive_db <= 0.0:
        sub_out = sub_c
    else:
        drive = float(10.0 ** (float(sat_drive_db) / 20.0))
        sat = np.tanh(sub_c * drive)
        sat /= (np.tanh(drive) + 1e-6)
        sub_out = (1.0 - sat_mix) * sub_c + sat_mix * sat

    return (sub_out + high).astype(np.float32)


# -----------------------------
# NEW: Adaptive Harmonic Balancer (innovative feature)
# -----------------------------

_KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=np.float32)
_KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=np.float32)
_KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _rotate(v: np.ndarray, k: int) -> np.ndarray:
    return np.roll(v, k)


def infer_key_mode(mono: np.ndarray, sr: int) -> Tuple[str, str, float, float]:
    """
    Infer musical key + mode using a Krumhansl-Schmuckler style template match.

    Returns:
      (key_name, mode, confidence, chroma_entropy)

    Notes:
    - This is a best-effort heuristic, not a perfect transcription.
    - In melodic trap, the tonal center is usually stable enough for this to be useful.

    Why it matters for mastering:
    - Warmth and harmonic density feel more "coherent" when enhancement follows
      the track's tonal gravity (major vs minor emotional intent).
    """
    mono = ensure_finite(mono, "mono_for_key")
    if len(mono) < sr:
        return "C", "minor", 0.0, 1.0

    # Chroma extraction (fast, robust enough)
    chroma = librosa.feature.chroma_stft(y=mono, sr=sr, n_fft=4096, hop_length=1024)
    chroma_mean = np.mean(chroma, axis=1).astype(np.float32)
    chroma_mean = np.maximum(chroma_mean, 1e-9)
    chroma_mean /= float(np.sum(chroma_mean))

    # Tonal density proxy: entropy in [0,1]
    ent = -float(np.sum(chroma_mean * np.log(chroma_mean))) / math.log(12.0)

    # Normalize templates
    maj = _KS_MAJOR / float(np.sum(_KS_MAJOR))
    minr = _KS_MINOR / float(np.sum(_KS_MINOR))

    best = ("C", "major", -1e9)
    scores = []

    for k in range(12):
        s_maj = float(np.dot(chroma_mean, _rotate(maj, k)))
        s_min = float(np.dot(chroma_mean, _rotate(minr, k)))
        scores.append(s_maj)
        scores.append(s_min)

        if s_maj > best[2]:
            best = (_KEY_NAMES[k], "major", s_maj)
        if s_min > best[2]:
            best = (_KEY_NAMES[k], "minor", s_min)

    # Confidence = separation from median score
    med = float(np.median(scores))
    conf = float(np.clip((best[2] - med) / max(1e-9, abs(best[2])), 0.0, 1.0))

    return best[0], best[1], conf, ent


def _even_harmonic_waveshaper(x: np.ndarray, drive_lin: np.ndarray, asym: float) -> np.ndarray:
    """
    Waveshaper that introduces *even-order* harmonics via gentle asymmetry.

    Musical rationale:
    - Even harmonics are perceived as "warmth" and "body"
    - Odd harmonics read as "edge" and "aggression"
    - Melodic trap usually wants warmth without harshness

    Implementation:
    - Add a small bias (asymmetry), saturate, remove DC offset.
    """
    bias = float(np.clip(asym, 0.0, 0.35))
    y = np.tanh((x + bias) * drive_lin) - np.tanh(bias * drive_lin)
    # Normalize roughly to keep gain stable across drive
    y /= (np.tanh(drive_lin) + 1e-6)
    return y.astype(np.float32)


def adaptive_harmonic_balancer(
    audio: np.ndarray,
    sr: int,
    bp_lo: float = 180.0,
    bp_hi: float = 4200.0,
    mix: float = 0.08,
    drive_db: float = 2.0,
    dyn_depth_db: float = 2.5,
    env_ms: float = 80.0,
    asym_major: float = 0.06,
    asym_minor: float = 0.12,
    key_seconds: float = 45.0,
    tension_sensitivity: float = 0.35,
    numtaps: int = 513,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Adaptive Harmonic Balancer (AHB)
    -------------------------------

    Innovative feature: *musically guided warmth*.

    1) Infer key + mode from the first N seconds (fast heuristic).
    2) Work in Mid/Side: enhance the MID (center) for vocal/melody cohesion.
    3) Band-pass the harmonic region and apply *even-harmonic* saturation.
    4) Dynamically modulate drive:
       - more enhancement when the band is quiet (upward richness)
       - reduce overall intensity if tonal density (entropy) is high

    Result:
    - Tonal coherence improves because the enhancement respects the track’s modal intent.
    - The master feels warmer, deeper, and more "spatially glued" without harshness.
    """
    if audio.ndim != 2 or audio.shape[1] != 2:
        return audio.astype(np.float32, copy=False), {"enabled": False}

    mix = float(np.clip(mix, 0.0, 1.0))
    if mix <= 0.0:
        return audio.astype(np.float32, copy=False), {"enabled": False}

    bp_lo = float(max(80.0, min(bp_lo, sr / 2 - 2000.0)))
    bp_hi = float(max(bp_lo + 800.0, min(bp_hi, sr / 2 - 50.0)))
    env_ms = float(max(10.0, env_ms))

    # --- Key/mode inference on a short excerpt for speed ---
    n = int(min(len(audio), sr * float(max(5.0, key_seconds))))
    mono_excerpt = to_mono(audio[:n])
    key, mode, conf, entropy = infer_key_mode(mono_excerpt, sr=sr)

    # Mode controls asymmetry (minor tends to benefit from more warmth)
    asym = float(asym_minor if mode == "minor" else asym_major)

    # Tonal density controls intensity (busy harmony -> keep subtle)
    # entropy in [0..1], higher means more uniformly spread chroma (less anchored)
    tension = float(np.clip(entropy, 0.0, 1.0))
    intensity_scale = float(np.clip(1.0 - tension_sensitivity * tension, 0.55, 1.0))

    mid, side = _ms_encode(audio)
    mid_st = np.stack([mid, mid], axis=1)

    # Band-pass the "harmonic core"
    hp = _design_fir_highpass(sr, cutoff_hz=bp_lo, numtaps=numtaps)
    bp = apply_fir_stereo(mid_st, hp)
    lp = _design_fir_lowpass(sr, cutoff_hz=bp_hi, numtaps=numtaps)
    bp = apply_fir_stereo(bp, lp)
    band = bp[:, 0]

    # Envelope follower to make drive dynamic (richer in quiet moments)
    win = int(max(8, sr * env_ms / 1000.0))
    kernel = np.ones(win, dtype=np.float32) / float(win)
    env = np.sqrt(np.convolve(band * band, kernel, mode="same") + 1e-12)
    env_norm = env / (np.max(env) + 1e-12)

    drive_curve_db = (float(drive_db) + float(dyn_depth_db) * (1.0 - env_norm)) * intensity_scale
    drive_curve_db = np.clip(drive_curve_db, 0.0, 12.0)
    drive_lin = (10.0 ** (drive_curve_db / 20.0)).astype(np.float32)

    sat = _even_harmonic_waveshaper(band, drive_lin, asym=asym)

    mid_out = mid + mix * sat
    out = _ms_decode(mid_out, side)

    info = {
        "enabled": True,
        "key": str(key),
        "mode": str(mode),
        "confidence": float(conf),
        "chroma_entropy": float(entropy),
        "intensity_scale": float(intensity_scale),
        "bp_lo": float(bp_lo),
        "bp_hi": float(bp_hi),
        "mix": float(mix),
        "drive_db": float(drive_db),
        "dyn_depth_db": float(dyn_depth_db),
        "asym_used": float(asym),
        "env_ms": float(env_ms),
        "key_seconds": float(key_seconds),
    }
    return out.astype(np.float32), info


# -----------------------------
# Air & Presence (de-ess + mid expressiveness)
# -----------------------------

def dynamic_deesser(
    audio: np.ndarray,
    sr: int,
    band_hp_hz: float = 6000.0,
    band_lp_hz: float = 11000.0,
    thresh: float = 0.12,
    max_reduction_db: float = 6.0,
    env_ms: float = 20.0,
    numtaps: int = 513,
) -> np.ndarray:
    """
    Simple dynamic de-esser.

    Musical rationale:
    - Sibilance spikes destroy "hi-fi" perception instantly.
    - We reduce only the sibilance band, preserving vocal presence.
    """
    if audio.ndim != 2 or audio.shape[1] != 2:
        return audio.astype(np.float32, copy=False)

    band_hp_hz = float(max(1000.0, min(band_hp_hz, sr / 2 - 500.0)))
    band_lp_hz = float(max(band_hp_hz + 1000.0, min(band_lp_hz, sr / 2 - 50.0)))
    max_reduction_db = float(max(0.0, max_reduction_db))
    env_ms = float(max(5.0, env_ms))
    thresh = float(max(1e-6, thresh))

    hp = _design_fir_highpass(sr, cutoff_hz=band_hp_hz, numtaps=numtaps)
    sib = apply_fir_stereo(audio, hp)
    lp = _design_fir_lowpass(sr, cutoff_hz=band_lp_hz, numtaps=numtaps)
    sib = apply_fir_stereo(sib, lp)

    mono_sib = to_mono(sib)

    win = int(max(8, sr * env_ms / 1000.0))
    kernel = np.ones(win, dtype=np.float32) / float(win)
    env = np.sqrt(np.convolve(mono_sib * mono_sib, kernel, mode="same") + 1e-12)

    env_max = float(np.max(env) + 1e-12)
    over = np.maximum(0.0, env - thresh) / env_max
    gr_db = -max_reduction_db * np.clip(over, 0.0, 1.0)
    g = undb(gr_db).astype(np.float32)[:, None]

    sib_att = sib * g
    return (audio - (sib - sib_att)).astype(np.float32)


def mid_presence_enhance(
    audio: np.ndarray,
    sr: int,
    bp_lo: float = 1000.0,
    bp_hi: float = 3800.0,
    mix: float = 0.10,
    drive_db: float = 2.0,
    dyn_depth_db: float = 3.0,
    env_ms: float = 60.0,
    numtaps: int = 513,
) -> np.ndarray:
    """
    Midrange expressiveness enhancer (M/S).

    Musical rationale:
    - Instead of boosting EQ (which can get harsh),
      saturation adds harmonics that feel like "presence" and "confidence".
    """
    if audio.ndim != 2 or audio.shape[1] != 2:
        return audio.astype(np.float32, copy=False)

    mix = float(np.clip(mix, 0.0, 1.0))
    if mix <= 0.0:
        return audio.astype(np.float32, copy=False)

    bp_lo = float(max(200.0, min(bp_lo, sr / 2 - 2000.0)))
    bp_hi = float(max(bp_lo + 800.0, min(bp_hi, sr / 2 - 50.0)))
    env_ms = float(max(10.0, env_ms))

    mid, side = _ms_encode(audio)
    mid_st = np.stack([mid, mid], axis=1)

    hp = _design_fir_highpass(sr, cutoff_hz=bp_lo, numtaps=numtaps)
    bp = apply_fir_stereo(mid_st, hp)
    lp = _design_fir_lowpass(sr, cutoff_hz=bp_hi, numtaps=numtaps)
    bp = apply_fir_stereo(bp, lp)

    band = bp[:, 0]

    win = int(max(8, sr * env_ms / 1000.0))
    kernel = np.ones(win, dtype=np.float32) / float(win)
    env = np.sqrt(np.convolve(band * band, kernel, mode="same") + 1e-12)
    env_norm = env / (np.max(env) + 1e-12)

    drive_curve_db = float(drive_db) + float(dyn_depth_db) * (1.0 - env_norm)
    drive_curve_db = np.clip(drive_curve_db, 0.0, 12.0)
    drive_lin = (10.0 ** (drive_curve_db / 20.0)).astype(np.float32)

    sat = np.tanh(band * drive_lin)
    sat /= (np.tanh(drive_lin) + 1e-6)

    mid_out = mid + mix * sat
    return _ms_decode(mid_out, side)


# -----------------------------
# Loudness match + limiting safety
# -----------------------------

def match_lufs(audio: np.ndarray, sr: int, target_lufs: float) -> Tuple[np.ndarray, float]:
    """
    Match integrated loudness (LUFS) by uniform gain (no added dynamics).
    Returns (audio_out, gain_db_applied).
    """
    x = ensure_finite(audio, "audio_for_lufs_match")
    mono = to_mono(x)
    try:
        cur = measure_lufs_integrated(mono, sr)
    except Exception as e:
        LOG.warning("LUFS measurement failed: %s", e)
        return x.astype(np.float32), 0.0

    if not np.isfinite(cur):
        LOG.warning("LUFS measurement invalid; skipping normalization.")
        return x.astype(np.float32), 0.0

    gain_db = float(target_lufs - cur)
    gain = float(10 ** (gain_db / 20.0))
    out = (x * gain).astype(np.float32)
    return out, gain_db


def _estimate_true_peak_lin(audio: np.ndarray, oversample: int = 4) -> float:
    """
    Estimate true peak by oversampling and measuring max abs.

    Why it matters:
    - Inter-sample peaks can clip DACs/encoders even if sample peak is safe.
    """
    x = ensure_finite(audio, "audio_for_true_peak")
    oversample = int(max(1, oversample))
    if oversample == 1:
        return float(np.max(np.abs(x)) + 1e-12)

    # Oversample along time axis.
    y = resample_poly(x, oversample, 1, axis=0)
    return float(np.max(np.abs(y)) + 1e-12)


def peak_limit(audio: np.ndarray, peak_dbfs_target: float = -1.0) -> Tuple[np.ndarray, float]:
    """
    Simple sample-peak limiter by uniform scaling (transparent).
    Returns (limited_audio, extra_gain_db_applied [<=0]).
    """
    x = ensure_finite(audio, "audio_for_peak_limit")
    target_lin = float(10 ** (peak_dbfs_target / 20.0))
    peak = float(np.max(np.abs(x)) + 1e-12)
    if peak <= target_lin:
        return x.astype(np.float32), 0.0
    g = target_lin / peak
    gain_db = float(20.0 * np.log10(max(g, 1e-12)))
    return (x * g).astype(np.float32), gain_db


def true_peak_limit(audio: np.ndarray, peak_dbfs_target: float = -1.0, oversample: int = 4) -> Tuple[np.ndarray, float]:
    """
    True-peak-safe limiter (still uniform scaling, but measured with oversampling).
    Returns (limited_audio, gain_db_applied [<=0]).
    """
    x = ensure_finite(audio, "audio_for_true_peak_limit")
    target_lin = float(10 ** (peak_dbfs_target / 20.0))
    tp = _estimate_true_peak_lin(x, oversample=oversample)
    if tp <= target_lin:
        return x.astype(np.float32), 0.0
    g = target_lin / tp
    gain_db = float(20.0 * np.log10(max(g, 1e-12)))
    return (x * g).astype(np.float32), gain_db


def tpdf_dither(audio: np.ndarray, bits: int = 24, amount: float = 1.0) -> np.ndarray:
    """
    Optional TPDF dither before writing integer PCM.

    Musical rationale:
    - Dither preserves low-level detail (reverb tails, fades) more gracefully.
    - Useful when exporting 24-bit masters.

    amount:
      1.0 ~= 1 LSB triangular distribution
    """
    x = ensure_finite(audio, "audio_for_dither")
    bits = int(np.clip(bits, 16, 32))
    # 24-bit PCM has 23 bits of magnitude + sign; LSB step approx:
    lsb = 1.0 / float(2 ** (bits - 1))
    noise = (np.random.rand(*x.shape).astype(np.float32) - np.random.rand(*x.shape).astype(np.float32))
    return (x + noise * lsb * float(amount)).astype(np.float32)


# -----------------------------
# Reporting
# -----------------------------

def metrics_to_dict(m: AudioMetrics) -> Dict:
    return {
        "sr": m.sr,
        "duration_s": m.duration_s,
        "lufs_i": m.lufs_i,
        "rms_dbfs": m.rms_dbfs,
        "peak_dbfs": m.peak_dbfs,
        "crest_db": m.crest_db,
        "spectral_centroid_hz": m.spectral_centroid_hz,
        "band_db": m.band_db,
    }


def write_report(
    report_path: Path,
    ref_path: Path,
    tgt_path: Path,
    out_path: Path,
    before: AudioMetrics,
    after: AudioMetrics,
    ref: AudioMetrics,
    eq_info: Dict[str, float],
    lufs_gain_db: float,
    limiter_gain_db: float,
    limiter_mode: str,
    compressor_used: bool,
    sub_align_info: Optional[Dict[str, float]] = None,
    luminance_info: Optional[Dict[str, float]] = None,
    sub_anchor_info: Optional[Dict[str, float]] = None,
    air_presence_info: Optional[Dict[str, float]] = None,
    harmonic_balancer_info: Optional[Dict[str, float]] = None,
) -> None:
    def fmt(d: Dict) -> str:
        return json.dumps(d, indent=2)

    lines: List[str] = []
    lines.append("# Enhancement_Report")
    lines.append("")
    lines.append("## Files")
    lines.append(f"- Reference: `{ref_path}`")
    lines.append(f"- Target: `{tgt_path}`")
    lines.append(f"- Output: `{out_path}`")
    lines.append("")
    lines.append("## Reference Metrics")
    lines.append("```json")
    lines.append(fmt(metrics_to_dict(ref)))
    lines.append("```")
    lines.append("")
    lines.append("## Target Metrics (before)")
    lines.append("```json")
    lines.append(fmt(metrics_to_dict(before)))
    lines.append("```")
    lines.append("")
    lines.append("## Enhanced Metrics (after)")
    lines.append("```json")
    lines.append(fmt(metrics_to_dict(after)))
    lines.append("```")
    lines.append("")
    lines.append("## Adjustments Applied")
    lines.append(
        "- Spectral/EQ matching FIR: numtaps={numtaps}, gain_range_db=[{mn:.2f}, {mx:.2f}], "
        "guardrails={gr}".format(
            numtaps=int(eq_info.get("numtaps", 0)),
            mn=eq_info.get("min_gain_db_applied", 0.0),
            mx=eq_info.get("max_gain_db_applied", 0.0),
            gr=eq_info.get("guardrails", False),
        )
    )

    if sub_align_info:
        lines.append(
            "- Sub-bass phase align: cutoff_hz={cutoff_hz:.1f}, lag_ms={lag_ms:.3f}, "
            "mono_strength={mono_strength:.2f}".format(**sub_align_info)
        )
    if sub_anchor_info:
        lines.append(
            "- Sub-Anchor: cutoff_hz={cutoff_hz:.1f}, threshold_db={comp_threshold_db:.1f}, ratio={ratio:.2f}, "
            "attack_ms={attack_ms:.1f}, release_ms={release_ms:.1f}, sat_mix={sat_mix:.2f}, sat_drive_db={sat_drive_db:.1f}".format(**sub_anchor_info)
        )
    if harmonic_balancer_info and harmonic_balancer_info.get("enabled"):
        lines.append(
            "- Adaptive Harmonic Balancer: key={key} {mode} (conf={confidence:.2f}), entropy={chroma_entropy:.2f}, "
            "mix={mix:.2f}, band=[{bp_lo:.0f}-{bp_hi:.0f}] Hz, asym={asym_used:.3f}, intensity_scale={intensity_scale:.2f}".format(**harmonic_balancer_info)
        )
    if luminance_info:
        lines.append(
            "- Harmonic luminance: hp_hz={hp_hz:.1f}, mix={mix:.3f}, drive_db={drive_db:.2f}, "
            "dyn_depth_db={dyn_depth_db:.2f}".format(**luminance_info)
        )
    if air_presence_info:
        lines.append(
            "- Air & Presence: deesser_hp_hz={deesser_hp_hz:.1f}, deesser_lp_hz={deesser_lp_hz:.1f}, "
            "deesser_thresh={deesser_thresh:.3f}, deesser_max_red_db={deesser_max_red_db:.1f}, "
            "presence_bp=[{presence_bp_lo:.1f}, {presence_bp_hi:.1f}], presence_mix={presence_mix:.2f}".format(**air_presence_info)
        )

    lines.append(f"- Compression used: {compressor_used}")
    lines.append(f"- LUFS normalization gain: {lufs_gain_db:.2f} dB")
    lines.append(f"- Limiter mode: {limiter_mode}")
    lines.append(f"- Limiter gain: {limiter_gain_db:.2f} dB")
    lines.append("")
    lines.append("## Notes")
    lines.append("- FIR EQ is linear-phase; extremely sharp transients can show slight pre-ringing.")
    lines.append("- True-peak limiting here is transparent uniform scaling with oversampled peak measurement.")
    lines.append("")

    report_path.write_text("/n".join(lines), encoding="utf-8")


# -----------------------------
# Main pipeline
# -----------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="AuralMind v5: reference-based mastering enhancer (LUFS + spectral match).")

    ap.add_argument("--reference", required=True, type=Path, help="Reference audio file (.wav/.mp3/...).")
    ap.add_argument("--target", required=True, type=Path, help="Target audio file to enhance (.wav/.mp3/...).")
    ap.add_argument("--out", default=Path("enhanced_target.wav"), type=Path, help="Output WAV path.")
    ap.add_argument("--report", default=Path("Enhancement_Report.md"), type=Path, help="Markdown report path.")
    ap.add_argument("--sr", type=int, default=44100, help="Working sample rate (Hz). Default: 44100.")

    # EQ matching
    ap.add_argument("--fir_taps", type=int, default=1025, help="FIR EQ length (odd). Default: 1025.")
    ap.add_argument("--max_eq_db", type=float, default=12.0, help="Max spectral boost/cut in dB. Default: 12.")
    ap.add_argument("--eq_smooth_hz", type=float, default=80.0, help="EQ delta smoothing (Hz). Default: 80.")
    ap.add_argument("--no_eq_guardrails", action="store_true", help="Disable safe EQ guardrails for subs/air.")

    # Sub alignment
    ap.add_argument("--enable_compression", action="store_true", help="Enable gentle compression if needed.")
    ap.add_argument("--no_sub_align", action="store_true", help="Disable sub-bass phase alignment.")
    ap.add_argument("--sub_align_cutoff_hz", type=float, default=120.0, help="Low-band cutoff for sub alignment (Hz).")
    ap.add_argument("--sub_align_max_ms", type=float, default=2.0, help="Max time shift for sub alignment (ms).")
    ap.add_argument("--sub_align_mono_strength", type=float, default=0.6, help="0..1, reduce low-end stereo width.")

    # Sub-anchor
    ap.add_argument("--no_sub_anchor", action="store_true", help="Disable Sub-Anchor low-end stabilizer.")
    ap.add_argument("--sub_anchor_cutoff_hz", type=float, default=120.0, help="Sub-Anchor split cutoff (Hz).")
    ap.add_argument("--sub_anchor_threshold_db", type=float, default=-24.0, help="Sub-Anchor compressor threshold (dB).")
    ap.add_argument("--sub_anchor_ratio", type=float, default=2.5, help="Sub-Anchor compressor ratio.")
    ap.add_argument("--sub_anchor_attack_ms", type=float, default=8.0, help="Sub-Anchor compressor attack (ms).")
    ap.add_argument("--sub_anchor_release_ms", type=float, default=160.0, help="Sub-Anchor compressor release (ms).")
    ap.add_argument("--sub_anchor_sat_mix", type=float, default=0.10, help="0..1 Sub-Anchor saturation mix.")
    ap.add_argument("--sub_anchor_sat_drive_db", type=float, default=3.0, help="Sub-Anchor saturation drive (dB).")

    # NEW: Adaptive Harmonic Balancer
    ap.add_argument("--no_harmonic_balancer", action="store_true", help="Disable Adaptive Harmonic Balancer.")
    ap.add_argument("--hb_bp_lo", type=float, default=180.0, help="Harmonic Balancer band-pass low cutoff (Hz).")
    ap.add_argument("--hb_bp_hi", type=float, default=4200.0, help="Harmonic Balancer band-pass high cutoff (Hz).")
    ap.add_argument("--hb_mix", type=float, default=0.08, help="0..1 Harmonic Balancer mix.")
    ap.add_argument("--hb_drive_db", type=float, default=2.0, help="Harmonic Balancer base drive (dB).")
    ap.add_argument("--hb_dyn_depth_db", type=float, default=2.5, help="Harmonic Balancer dynamic drive depth (dB).")
    ap.add_argument("--hb_env_ms", type=float, default=80.0, help="Harmonic Balancer envelope window (ms).")
    ap.add_argument("--hb_asym_major", type=float, default=0.06, help="Even-harmonic asymmetry amount (major).")
    ap.add_argument("--hb_asym_minor", type=float, default=0.12, help="Even-harmonic asymmetry amount (minor).")
    ap.add_argument("--hb_key_seconds", type=float, default=45.0, help="Seconds used to infer key/mode.")
    ap.add_argument("--hb_tension_sensitivity", type=float, default=0.35, help="0..1 tonal density -> intensity reduction.")

    # Luminance
    ap.add_argument("--no_luminance", action="store_true", help="Disable harmonic luminance sculptor.")
    ap.add_argument("--luminance_hp_hz", type=float, default=6500.0, help="High-pass cutoff for luminance (Hz).")
    ap.add_argument("--luminance_mix", type=float, default=0.08, help="0..1 mix for harmonic luminance.")
    ap.add_argument("--luminance_drive_db", type=float, default=2.0, help="Base drive for luminance (dB).")
    ap.add_argument("--luminance_dyn_depth_db", type=float, default=3.0, help="Dynamic drive range (dB).")

    # Air & Presence
    ap.add_argument("--no_air_presence", action="store_true", help="Disable Air & Presence (de-esser + mid expressiveness).")
    ap.add_argument("--deesser_hp_hz", type=float, default=6000.0, help="De-esser band high-pass (Hz).")
    ap.add_argument("--deesser_lp_hz", type=float, default=11000.0, help="De-esser band low-pass (Hz).")
    ap.add_argument("--deesser_thresh", type=float, default=0.12, help="De-esser envelope threshold (linear).")
    ap.add_argument("--deesser_max_reduction_db", type=float, default=6.0, help="De-esser max attenuation (dB).")
    ap.add_argument("--deesser_env_ms", type=float, default=20.0, help="De-esser envelope window (ms).")
    ap.add_argument("--presence_bp_lo", type=float, default=1000.0, help="Presence band-pass low cutoff (Hz).")
    ap.add_argument("--presence_bp_hi", type=float, default=3800.0, help="Presence band-pass high cutoff (Hz).")
    ap.add_argument("--presence_mix", type=float, default=0.10, help="0..1 presence enhancer mix.")
    ap.add_argument("--presence_drive_db", type=float, default=2.0, help="Presence enhancer base drive (dB).")
    ap.add_argument("--presence_dyn_depth_db", type=float, default=3.0, help="Presence enhancer dynamic drive range (dB).")
    ap.add_argument("--presence_env_ms", type=float, default=60.0, help="Presence enhancer envelope window (ms).")

    # Output safety
    ap.add_argument("--target_peak_dbfs", type=float, default=-1.0, help="Peak ceiling after processing. Default: -1 dBFS.")
    ap.add_argument("--limiter_mode", choices=["sample", "true_peak"], default="true_peak",
                    help="Limiter measurement mode. Default: true_peak.")
    ap.add_argument("--limiter_oversample", type=int, default=4, help="True-peak oversample factor. Default: 4.")
    ap.add_argument("--dither", action="store_true", help="Enable TPDF dither before PCM export.")
    ap.add_argument("--dither_amount", type=float, default=1.0, help="Dither amount in LSB units. Default: 1.0.")

    ap.add_argument("--log_level", default="INFO", help="Logging level (DEBUG, INFO, WARNING). Default: INFO.")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s %(message)s"
    )

    if not args.reference.exists():
        raise FileNotFoundError(f"Reference not found: {args.reference}")
    if not args.target.exists():
        raise FileNotFoundError(f"Target not found: {args.target}")
    if args.fir_taps < 129 or args.fir_taps % 2 == 0:
        raise ValueError("--fir_taps must be an odd integer >= 129 (e.g., 513, 1025, 2049, 4097).")

    sr = int(args.sr)

    # Load
    LOG.info("Loading reference audio: %s", args.reference)
    ref_audio, _ = load_audio_any_format(args.reference, sr=sr)

    LOG.info("Loading target audio: %s", args.target)
    tgt_audio, _ = load_audio_any_format(args.target, sr=sr)

    ref_audio = ensure_finite(ref_audio, "ref_audio")
    tgt_audio = ensure_finite(tgt_audio, "tgt_audio")

    # Analyze
    ref_metrics = compute_metrics(ref_audio, sr)
    tgt_before = compute_metrics(tgt_audio, sr)
    LOG.info("Reference LUFS: %.2f", ref_metrics.lufs_i)
    LOG.info("Target crest (before): %.2f dB", tgt_before.crest_db)

    # Build EQ match filter
    h, eq_info = design_match_fir(
        ref_mono=to_mono(ref_audio),
        tgt_mono=to_mono(tgt_audio),
        sr=sr,
        numtaps=args.fir_taps,
        max_gain_db=args.max_eq_db,
        smooth_hz=args.eq_smooth_hz,
        enable_guardrails=(not args.no_eq_guardrails),
    )

    # Apply EQ
    processed = apply_fir_stereo(tgt_audio, h)

    # Sub-bass align
    sub_align_info: Optional[Dict[str, float]] = None
    if not args.no_sub_align:
        processed, lag_ms = align_subbass_phase(
            processed,
            sr=sr,
            cutoff_hz=args.sub_align_cutoff_hz,
            max_shift_ms=args.sub_align_max_ms,
            mono_strength=args.sub_align_mono_strength,
            numtaps=513,
        )
        sub_align_info = {
            "cutoff_hz": float(args.sub_align_cutoff_hz),
            "lag_ms": float(lag_ms),
            "mono_strength": float(np.clip(args.sub_align_mono_strength, 0.0, 1.0)),
        }
        LOG.info("Sub-bass align applied (lag_ms=%.3f).", lag_ms)

    # Sub-anchor
    sub_anchor_info: Optional[Dict[str, float]] = None
    if not args.no_sub_anchor:
        processed = sub_anchor(
            processed,
            sr=sr,
            cutoff_hz=args.sub_anchor_cutoff_hz,
            comp_threshold_db=args.sub_anchor_threshold_db,
            ratio=args.sub_anchor_ratio,
            attack_ms=args.sub_anchor_attack_ms,
            release_ms=args.sub_anchor_release_ms,
            sat_mix=args.sub_anchor_sat_mix,
            sat_drive_db=args.sub_anchor_sat_drive_db,
            numtaps=513,
        )
        sub_anchor_info = {
            "cutoff_hz": float(args.sub_anchor_cutoff_hz),
            "comp_threshold_db": float(args.sub_anchor_threshold_db),
            "ratio": float(args.sub_anchor_ratio),
            "attack_ms": float(args.sub_anchor_attack_ms),
            "release_ms": float(args.sub_anchor_release_ms),
            "sat_mix": float(np.clip(args.sub_anchor_sat_mix, 0.0, 1.0)),
            "sat_drive_db": float(args.sub_anchor_sat_drive_db),
        }
        LOG.info("Sub-Anchor applied (cutoff=%.1f Hz).", args.sub_anchor_cutoff_hz)

    # NEW: Adaptive Harmonic Balancer
    harmonic_balancer_info: Optional[Dict[str, float]] = None
    if not args.no_harmonic_balancer:
        processed, harmonic_balancer_info = adaptive_harmonic_balancer(
            processed,
            sr=sr,
            bp_lo=args.hb_bp_lo,
            bp_hi=args.hb_bp_hi,
            mix=args.hb_mix,
            drive_db=args.hb_drive_db,
            dyn_depth_db=args.hb_dyn_depth_db,
            env_ms=args.hb_env_ms,
            asym_major=args.hb_asym_major,
            asym_minor=args.hb_asym_minor,
            key_seconds=args.hb_key_seconds,
            tension_sensitivity=args.hb_tension_sensitivity,
            numtaps=513,
        )
        if harmonic_balancer_info.get("enabled"):
            LOG.info(
                "Harmonic Balancer: %s %s (conf=%.2f, entropy=%.2f).",
                harmonic_balancer_info.get("key"),
                harmonic_balancer_info.get("mode"),
                harmonic_balancer_info.get("confidence"),
                harmonic_balancer_info.get("chroma_entropy"),
            )

    # Luminance (air sheen)
    luminance_info: Optional[Dict[str, float]] = None
    if not args.no_luminance:
        processed = apply_harmonic_luminance(
            processed,
            sr=sr,
            highpass_hz=args.luminance_hp_hz,
            mix=args.luminance_mix,
            drive_db=args.luminance_drive_db,
            dyn_depth_db=args.luminance_dyn_depth_db,
            numtaps=513,
            env_ms=50.0,
        )
        luminance_info = {
            "hp_hz": float(args.luminance_hp_hz),
            "mix": float(np.clip(args.luminance_mix, 0.0, 1.0)),
            "drive_db": float(args.luminance_drive_db),
            "dyn_depth_db": float(args.luminance_dyn_depth_db),
        }
        LOG.info("Luminance applied (hp=%.0f Hz, mix=%.3f).", args.luminance_hp_hz, args.luminance_mix)

    # Air & Presence
    air_presence_info: Optional[Dict[str, float]] = None
    if not args.no_air_presence:
        processed = dynamic_deesser(
            processed,
            sr=sr,
            band_hp_hz=args.deesser_hp_hz,
            band_lp_hz=args.deesser_lp_hz,
            thresh=args.deesser_thresh,
            max_reduction_db=args.deesser_max_reduction_db,
            env_ms=args.deesser_env_ms,
            numtaps=513,
        )
        processed = mid_presence_enhance(
            processed,
            sr=sr,
            bp_lo=args.presence_bp_lo,
            bp_hi=args.presence_bp_hi,
            mix=args.presence_mix,
            drive_db=args.presence_drive_db,
            dyn_depth_db=args.presence_dyn_depth_db,
            env_ms=args.presence_env_ms,
            numtaps=513,
        )
        air_presence_info = {
            "deesser_hp_hz": float(args.deesser_hp_hz),
            "deesser_lp_hz": float(args.deesser_lp_hz),
            "deesser_thresh": float(args.deesser_thresh),
            "deesser_max_red_db": float(args.deesser_max_reduction_db),
            "presence_bp_lo": float(args.presence_bp_lo),
            "presence_bp_hi": float(args.presence_bp_hi),
            "presence_mix": float(np.clip(args.presence_mix, 0.0, 1.0)),
        }
        LOG.info(
            "Air & Presence applied (de-ess %.0f-%.0f Hz, presence %.0f-%.0f Hz).",
            args.deesser_hp_hz, args.deesser_lp_hz, args.presence_bp_lo, args.presence_bp_hi
        )

    # Optional compression (only if crest is much higher than reference)
    compressor_used = False
    if args.enable_compression:
        proc_metrics_mid = compute_metrics(processed, sr)
        if proc_metrics_mid.crest_db > ref_metrics.crest_db + 2.0:
            processed = compress_broadband(processed, sr)
            compressor_used = True
            LOG.info("Compression applied (crest diff %.2f dB).", proc_metrics_mid.crest_db - ref_metrics.crest_db)

    # Loudness match
    processed, lufs_gain_db = match_lufs(processed, sr, target_lufs=ref_metrics.lufs_i)

    # Limiting safety
    limiter_mode = str(args.limiter_mode)
    if limiter_mode == "true_peak":
        processed, limiter_gain_db = true_peak_limit(processed, peak_dbfs_target=args.target_peak_dbfs, oversample=args.limiter_oversample)
    else:
        processed, limiter_gain_db = peak_limit(processed, peak_dbfs_target=args.target_peak_dbfs)

    # Optional dither before PCM export
    if args.dither:
        processed = tpdf_dither(processed, bits=24, amount=args.dither_amount)

    # Final metrics
    tgt_after = compute_metrics(processed, sr)

    # Export (24-bit PCM WAV)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out, processed, sr, subtype="PCM_24")

    # Report
    args.report.parent.mkdir(parents=True, exist_ok=True)
    write_report(
        report_path=args.report,
        ref_path=args.reference,
        tgt_path=args.target,
        out_path=args.out,
        before=tgt_before,
        after=tgt_after,
        ref=ref_metrics,
        eq_info=eq_info,
        lufs_gain_db=lufs_gain_db,
        limiter_gain_db=limiter_gain_db,
        limiter_mode=limiter_mode,
        compressor_used=compressor_used,
        sub_align_info=sub_align_info,
        luminance_info=luminance_info,
        sub_anchor_info=sub_anchor_info,
        air_presence_info=air_presence_info,
        harmonic_balancer_info=harmonic_balancer_info,
    )

    LOG.info("Wrote: %s", args.out)
    LOG.info("Report: %s", args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
