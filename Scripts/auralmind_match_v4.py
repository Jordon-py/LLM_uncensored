#!/usr/bin/env python3
"""
AuralMind - reference-based enhancement (LUFS + spectral/EQ matching + mild dynamics).

Usage:
  
  python "C:/Users/goku/LLM_uncensored/auralmind_match_v4.py" --reference "C:/Users/goku/Downloads/Lil Wayne_She Will.mp3" --target "C:/Users/goku/Downloads/I'm Him (9).wav" --out "C:/Users/goku/LLM_uncensored/out_v4/I'm Him (9)_v4_hifi.wav" --report "C:/Users/goku/LLM_uncensored/out_v4/Im_Him_2_v4_hifi_report.md" --sr 48000 --fir_taps 4097 --max_eq_db 9 --enable_compression --sub_align_cutoff_hz 120 --sub_align_max_ms 1.5 --sub_align_mono_strength 0.55 --sub_anchor_cutoff_hz 120 --sub_anchor_threshold_db -20 --sub_anchor_ratio 2.2 --sub_anchor_attack_ms 10 --sub_anchor_release_ms 180 --sub_anchor_sat_mix 0.06 --sub_anchor_sat_drive_db 2.0 --luminance_hp_hz 7200 --luminance_mix 0.06 --luminance_drive_db 1.5 --luminance_dyn_depth_db 2.0 --deesser_hp_hz 6500 --deesser_lp_hz 11500 --deesser_thresh 0.13 --deesser_max_reduction_db 4 --deesser_env_ms 25 --presence_bp_lo 800 --presence_bp_hi 3750 --presence_mix 0.07 --presence_drive_db 1.6 --presence_dyn_depth_db 2.0 --presence_env_ms 70 --target_peak_dbfs -1.0 --log_level INFO

  python auralmind_match_v4.py --reference "C:/Users/goku/Downloads/Lil Wayne_She Will.mp3" --target "C:/Users/goku/Downloads/Bigger picture .wav" --out "Bigger Picture_v5.wav
  
   Pipeline:
  1) Decode/normalize inputs to float32 stereo at --sr.
  2) Measure loudness and long-term spectrum.
  3) Design a linear-phase FIR EQ to match the reference spectrum.
  4) Apply the FIR EQ to the target.
  5) Sub-bass phase align + low-end mono control (optional).
  6) Sub-Anchor low-end stabilizer (optional).
  7) Harmonic luminance sculptor for smooth air (optional).
  8) Air & Presence: dynamic de-essing + midrange expressiveness (optional).
  9) Optional gentle compression if crest factor is too high.
 10) LUFS match to the reference and apply a safety peak limiter.
 11) Write output audio + Markdown report.
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
from typing import Dict, Tuple, Optional

import numpy as np
import soundfile as sf
import librosa
import pyloudnorm as pyln
from scipy.signal import firwin, firwin2, fftconvolve, savgol_filter

LOG = logging.getLogger("auralmind")


# -----------------------------
# IO utilities
# -----------------------------

def _require_ffmpeg() -> str:
    """
    Return the path to ffmpeg if installed, else raise a helpful error.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install ffmpeg or provide WAV inputs./n"
            "Windows: choco install ffmpeg/n"
            "macOS: brew install ffmpeg/n"
            "Linux: sudo apt-get install ffmpeg"
        )
    return ffmpeg


def _run(cmd: list[str]) -> None:
    LOG.debug("Running: %s", " ".join(cmd))
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}/n{p.stderr}")


def decode_with_ffmpeg_to_wav(input_path: Path, out_wav_path: Path, sr: int, channels: int = 2) -> None:
    """
    Decode any audio to float32 WAV at specified sample rate and channel count.
    """
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
      - soundfile for WAV
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


def ensure_finite(x: np.ndarray, name: str = "array") -> np.ndarray:
    """
    Replace NaN/inf with zeros and warn.
    """
    if not np.isfinite(x).all():
        LOG.warning("%s contained NaN/inf; replacing with zeros.", name)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x.astype(np.float32, copy=False)


def to_mono(audio: np.ndarray) -> np.ndarray:
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
    band_db: Dict[str, float]  # low/mid/high energy in dB (relative units)
    duration_s: float


def measure_lufs_integrated(mono: np.ndarray, sr: int) -> float:
    meter = pyln.Meter(sr)
    return float(meter.integrated_loudness(mono.astype(np.float64)))


def band_energy_db(mono: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Return rough band energies using STFT magnitude integration.
    Bands are heuristic and intended for reporting/diagnostics.
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
        "sub": band(20.0, 60.0),
        "low": band(60.0, 250.0),
        "mid": band(250.0, 2000.0),
        "high": band(2000.0, 12000.0),
        "air": band(12000.0, min(20000.0, sr / 2 - 1.0)),
    }


def spectral_centroid(mono: np.ndarray, sr: int) -> float:
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
    Compute long-term average magnitude spectrum in dB.
    Returns (freqs, spectrum_db).
    """
    S = np.abs(librosa.stft(mono, n_fft=n_fft, hop_length=n_fft // 8, window="hann")) + 1e-12
    mag = np.mean(S, axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    spec_db = db(mag)
    return freqs, spec_db


# -----------------------------
# Spectral match EQ (linear-phase FIR)
# -----------------------------

def design_match_fir(
    ref_mono: np.ndarray,
    tgt_mono: np.ndarray,
    sr: int,
    numtaps: int = 1025,
    max_gain_db: float = 12.0,
    smooth_hz: float = 80.0,
    bass_extra_db: float = 0.0,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Build a linear-phase FIR EQ that nudges target spectrum toward reference spectrum.

    Notes:
      - This is *not* a magic bullet; it is a wide-band correction.
      - Use max_gain_db to prevent overfitting weird resonances.
    """
    numtaps = int(numtaps)
    if numtaps % 2 == 0:
        numtaps += 1
    max_gain_db = float(max_gain_db)

    freqs, ref_db = average_spectrum_db(ref_mono, sr)
    _, tgt_db = average_spectrum_db(tgt_mono, sr)

    delta_db = ref_db - tgt_db

    # Smooth delta (frequency domain) to avoid narrow notches/boosts.
    df = float(freqs[1] - freqs[0])
    win = max(5, int(smooth_hz / max(df, 1e-9)))
    if win % 2 == 0:
        win += 1
    delta_smooth = savgol_filter(delta_db, window_length=min(win, len(delta_db) // 2 * 2 - 1), polyorder=3, mode="interp")

    # Clip correction to avoid crazy EQ moves.
    delta_smooth = np.clip(delta_smooth, -max_gain_db, max_gain_db)

    # Optional extra bass shelf (rarely needed; keep at 0 unless you know you want it).
    if bass_extra_db != 0.0:
        bass_mask = freqs < 90.0
        delta_smooth[bass_mask] = np.clip(delta_smooth[bass_mask] + bass_extra_db, -max_gain_db, max_gain_db)

    # Convert dB to linear magnitude response
    desired_mag = undb(delta_smooth)

    # FIR design with frequency sampling (firwin2).
    # Normalize frequency to [0,1] with 1 = Nyquist.
    nyq = sr / 2.0
    f_norm = np.clip(freqs / nyq, 0.0, 1.0)

    # firwin2 expects first=0 and last=1
    if f_norm[0] != 0.0:
        f_norm[0] = 0.0
    if f_norm[-1] != 1.0:
        f_norm[-1] = 1.0

    h = firwin2(numtaps, f_norm, desired_mag, window="hann")

    info = {
        "numtaps": float(numtaps),
        "max_gain_db": float(max_gain_db),
        "smooth_hz": float(smooth_hz),
        "bass_extra_db": float(bass_extra_db),
        "min_gain_db_applied": float(np.min(delta_smooth)),
        "max_gain_db_applied": float(np.max(delta_smooth)),
    }
    return h.astype(np.float32), info


def apply_fir_stereo(audio: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Apply FIR to stereo audio using FFT convolution.
    """
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


def _fractional_shift(x: np.ndarray, shift_samples: float) -> np.ndarray:
    """
    Fractional delay via linear interpolation (small shifts only).
    Positive shift delays the signal (moves it later).
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


def _estimate_delay_samples(a: np.ndarray, b: np.ndarray, sr: int, max_shift_ms: float = 2.0, decim: int = 8) -> int:
    """
    Estimate integer-sample delay between a and b using cross-correlation on decimated signal.
    Returns delay in original sample rate (integer).
    """
    max_shift_samp = int(sr * max_shift_ms / 1000.0)
    if max_shift_samp < 1:
        return 0

    # Decimate for speed
    ad = a[::decim]
    bd = b[::decim]
    maxd = max_shift_samp // decim

    if len(ad) < 16 or len(bd) < 16:
        return 0

    # Cross-correlation around 0 lag
    # We compute corr at lags [-maxd, maxd]
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
    Returns (processed_audio, lag_ms_applied).
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
        # Shift right to align to left
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

    drive_curve_db = drive_db + dyn_depth_db * (1.0 - env_norm)
    drive_curve_db = np.clip(drive_curve_db, 0.0, 12.0)
    drive_lin = (10.0 ** (drive_curve_db / 20.0)).astype(np.float32)
    drive_lin = drive_lin[:, None]

    sat = np.tanh(high * drive_lin)
    norm = np.tanh(drive_lin) + 1e-6
    sat = sat / norm

    return (audio + mix * sat).astype(np.float32)

# -----------------------------
# Trap enhancements (Sub-Anchor + Air/Presence)
# -----------------------------

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
    Trap-centric low-end stabilizer.

    Steps:
      - split sub band (<= cutoff_hz)
      - apply gentle envelope compression to stabilize 808 note swings
      - add tiny soft saturation to improve translation on small speakers
      - recombine with untouched highs

    This is intentionally conservative by default.
    """
    if audio.ndim != 2 or audio.shape[1] != 2:
        return audio.astype(np.float32, copy=False)

    cutoff_hz = float(max(20.0, min(cutoff_hz, 300.0)))
    sat_mix = float(np.clip(sat_mix, 0.0, 1.0))

    h_lp = _design_fir_lowpass(sr, cutoff_hz=cutoff_hz, numtaps=numtaps)
    sub = apply_fir_stereo(audio, h_lp)
    high = (audio - sub).astype(np.float32)

    # Sub-only compression
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
        # Tiny soft saturation on the sub band (kept stereo-safe)
        drive = float(10.0 ** (float(sat_drive_db) / 20.0))
        sat = np.tanh(sub_c * drive)
        sat /= (np.tanh(drive) + 1e-6)
        sub_out = (1.0 - sat_mix) * sub_c + sat_mix * sat

    return (sub_out + high).astype(np.float32)


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

    - Isolate sibilance band (HP -> LP)
    - Track an RMS-ish envelope
    - Apply dynamic attenuation to that band only

    This is deliberately simple and stable (no external DSP dependencies).
    """
    if audio.ndim != 2 or audio.shape[1] != 2:
        return audio.astype(np.float32, copy=False)

    band_hp_hz = float(max(1000.0, min(band_hp_hz, sr / 2 - 500.0)))
    band_lp_hz = float(max(band_hp_hz + 1000.0, min(band_lp_hz, sr / 2 - 50.0)))
    max_reduction_db = float(max(0.0, max_reduction_db))
    env_ms = float(max(5.0, env_ms))
    thresh = float(max(1e-6, thresh))

    # Band isolate
    hp = _design_fir_highpass(sr, cutoff_hz=band_hp_hz, numtaps=numtaps)
    sib = apply_fir_stereo(audio, hp)
    lp = _design_fir_lowpass(sr, cutoff_hz=band_lp_hz, numtaps=numtaps)
    sib = apply_fir_stereo(sib, lp)

    mono_sib = to_mono(sib)

    win = int(max(8, sr * env_ms / 1000.0))
    kernel = np.ones(win, dtype=np.float32) / float(win)
    env = np.sqrt(np.convolve(mono_sib * mono_sib, kernel, mode="same") + 1e-12)

    # Gain curve: above threshold -> reduce, capped at max_reduction_db
    # Normalize "over" relative to the band envelope's max so behavior is robust across levels.
    env_max = float(np.max(env) + 1e-12)
    over = np.maximum(0.0, env - thresh) / env_max
    gr_db = -max_reduction_db * np.clip(over, 0.0, 1.0)
    g = undb(gr_db).astype(np.float32)[:, None]

    sib_att = sib * g

    # Subtract the removed energy from the original (band-replacement in time-domain)
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

    - Encode Mid/Side
    - Band-pass the Mid channel (bp_lo..bp_hi)
    - Add *dynamic* soft saturation to the band (more when quiet)
    - Blend back into the Mid channel and decode

    Goal: keep vocals/melodies emotionally forward without harsh EQ boosts.
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

    # Work on mid channel as "stereo" for existing FIR helper.
    mid_st = np.stack([mid, mid], axis=1)

    hp = _design_fir_highpass(sr, cutoff_hz=bp_lo, numtaps=numtaps)
    bp = apply_fir_stereo(mid_st, hp)
    lp = _design_fir_lowpass(sr, cutoff_hz=bp_hi, numtaps=numtaps)
    bp = apply_fir_stereo(bp, lp)

    mono_bp = bp[:, 0]

    win = int(max(8, sr * env_ms / 1000.0))
    kernel = np.ones(win, dtype=np.float32) / float(win)
    env = np.sqrt(np.convolve(mono_bp * mono_bp, kernel, mode="same") + 1e-12)
    env_norm = env / (np.max(env) + 1e-12)

    # Dynamic drive: more enrichment when the band is quieter (upward expressiveness).
    drive_curve_db = float(drive_db) + float(dyn_depth_db) * (1.0 - env_norm)
    drive_curve_db = np.clip(drive_curve_db, 0.0, 12.0)
    drive_lin = (10.0 ** (drive_curve_db / 20.0)).astype(np.float32)

    sat = np.tanh(mono_bp * drive_lin)
    sat /= (np.tanh(drive_lin) + 1e-6)

    mid_out = mid + mix * sat
    return _ms_decode(mid_out, side)

# -----------------------------
# Dynamics (gentle compressor)
# -----------------------------

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
    Conservative settings by default.
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

    # Gain reduction curve in dB
    over_db = np.maximum(0.0, level_db - threshold_db)
    gr_db = over_db - (over_db / ratio)  # how much to reduce
    desired_gain_db = -gr_db

    # Smooth gain changes with hop-based attack/release (frame-step smoothing).
    tau_a = max(1e-4, attack_ms / 1000.0)
    tau_r = max(1e-4, release_ms / 1000.0)
    a_a = math.exp(-hop / (sr * tau_a))
    a_r = math.exp(-hop / (sr * tau_r))

    sm = np.zeros_like(desired_gain_db, dtype=np.float32)
    sm[0] = desired_gain_db[0]
    for i in range(1, len(sm)):
        if desired_gain_db[i] < sm[i - 1]:
            # more reduction: attack
            sm[i] = a_a * sm[i - 1] + (1.0 - a_a) * desired_gain_db[i]
        else:
            # less reduction: release
            sm[i] = a_r * sm[i - 1] + (1.0 - a_r) * desired_gain_db[i]

    # Interpolate smoothed gain to sample domain.
    gain_db = np.zeros(len(mono), dtype=np.float32)
    for idx, pos in enumerate(positions):
        gain_db[pos:pos + hop] = sm[idx]
    if positions[-1] + hop < len(gain_db):
        gain_db[positions[-1] + hop:] = sm[-1]

    gain_lin = undb(gain_db + makeup_db).astype(np.float32)

    if x.ndim == 1:
        y = x * gain_lin
        return y.astype(np.float32)
    yL = x[:, 0] * gain_lin
    yR = x[:, 1] * gain_lin
    return np.stack([yL, yR], axis=1).astype(np.float32)


# -----------------------------
# Loudness match + limiter
# -----------------------------

def match_lufs(audio: np.ndarray, sr: int, target_lufs: float) -> Tuple[np.ndarray, float]:
    """
    Match integrated loudness (LUFS) by uniform gain (no dynamics).
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
        return audio.astype(np.float32), 0.0

    # Apply same gain to stereo channels using loudness delta.
    gain_db = float(target_lufs - cur)
    gain = float(10 ** (gain_db / 20.0))
    out = (audio * gain).astype(np.float32)
    return out, gain_db


def peak_limit(audio: np.ndarray, peak_dbfs_target: float = -1.0) -> Tuple[np.ndarray, float]:
    """
    Simple peak limiter by uniform scaling (not true-peak).
    Returns (limited_audio, extra_gain_db_applied [<=0]).
    """
    x = ensure_finite(audio, "audio_for_peak_limit")
    target_lin = float(10 ** (peak_dbfs_target / 20.0))
    peak = float(np.max(np.abs(x)) + 1e-12)
    if peak <= 0:
        return x.astype(np.float32), 0.0
    if peak <= target_lin:
        return x.astype(np.float32), 0.0
    g = target_lin / peak
    gain_db = float(20.0 * np.log10(max(g, 1e-12)))
    return (x * g).astype(np.float32), gain_db


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
    compressor_used: bool,
    sub_align_info: Optional[Dict[str, float]] = None,
    luminance_info: Optional[Dict[str, float]] = None,
    sub_anchor_info: Optional[Dict[str, float]] = None,
    air_presence_info: Optional[Dict[str, float]] = None,
) -> None:
    def fmt(d: Dict) -> str:
        return json.dumps(d, indent=2)

    lines: list[str] = []
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
    lines.append(f"- Spectral/EQ matching FIR: numtaps={int(eq_info.get('numtaps', 0))}, "
                 f"gain_range_db=[{eq_info.get('min_gain_db_applied', 0):.2f}, {eq_info.get('max_gain_db_applied', 0):.2f}], "
                 f"bass_extra_db={eq_info.get('bass_extra_db', 0):.2f}")
    if sub_align_info:
        lines.append(
            "- Sub-bass phase align: cutoff_hz={cutoff_hz:.1f}, lag_ms={lag_ms:.3f}, "
            "mono_strength={mono_strength:.2f}, max_shift_ms={max_shift_ms:.2f}".format(**sub_align_info)
        )
    if luminance_info:
        lines.append(
            "- Harmonic luminance: hp_hz={hp_hz:.1f}, mix={mix:.3f}, drive_db={drive_db:.2f}, "
            "dyn_depth_db={dyn_depth_db:.2f}".format(**luminance_info)
        )
    if sub_anchor_info:
        lines.append(
            "- Sub-Anchor: cutoff_hz={cutoff_hz:.1f}, comp_threshold_db={comp_threshold_db:.1f}, ratio={ratio:.2f}, "
            "attack_ms={attack_ms:.1f}, release_ms={release_ms:.1f}, sat_mix={sat_mix:.2f}, sat_drive_db={sat_drive_db:.1f}".format(**sub_anchor_info)
        )
    if air_presence_info:
        lines.append(
            "- Air & Presence: deesser_hp_hz={deesser_hp_hz:.1f}, deesser_lp_hz={deesser_lp_hz:.1f}, "
            "deesser_thresh={deesser_thresh:.3f}, deesser_max_red_db={deesser_max_red_db:.1f}, "
            "presence_bp=[{presence_bp_lo:.1f}, {presence_bp_hi:.1f}], presence_mix={presence_mix:.2f}, "
            "presence_drive_db={presence_drive_db:.1f}, presence_dyn_depth_db={presence_dyn_depth_db:.1f}".format(**air_presence_info)
        )
    lines.append(f"- Compression used: {compressor_used}")
    lines.append(f"- LUFS normalization gain (approx): {lufs_gain_db:.2f} dB")
    lines.append(f"- Peak limiter gain: {limiter_gain_db:.2f} dB")
    lines.append("")
    lines.append("## Notes")
    lines.append("- Limiter is sample-peak (not true-peak). For strict broadcast specs, add true-peak limiting via oversampling.")
    lines.append("- Spectral EQ is linear-phase FIR; it may introduce latency and slight pre-ringing on sharp transients.")
    lines.append("")

    report_path.write_text("/n".join(lines), encoding="utf-8")


# -----------------------------
# Main pipeline
# -----------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="AuralMind: reference-based audio enhancement (LUFS + spectral match).")
    ap.add_argument("--reference", required=True, type=Path, help="Reference audio file (.wav/.mp3/...).")
    ap.add_argument("--target", required=True, type=Path, help="Target audio file to enhance (.wav/.mp3/...).")
    ap.add_argument("--out", default=Path("enhanced_target.wav"), type=Path, help="Output WAV path.")
    ap.add_argument("--report", default=Path("Enhancement_Report.md"), type=Path, help="Markdown report path.")
    ap.add_argument("--sr", type=int, default=44100, help="Working sample rate (Hz). Default: 44100.")
    ap.add_argument("--fir_taps", type=int, default=1025, help="FIR EQ length (odd). Default: 1025.")
    ap.add_argument("--max_eq_db", type=float, default=12.0, help="Max spectral boost/cut in dB. Default: 12.")
    ap.add_argument("--enable_compression", action="store_true", help="Enable gentle compression if needed.")
    ap.add_argument("--no_sub_align", action="store_true", help="Disable sub-bass phase alignment.")
    ap.add_argument("--sub_align_cutoff_hz", type=float, default=120.0, help="Low-band cutoff for sub alignment (Hz).")
    ap.add_argument("--sub_align_max_ms", type=float, default=2.0, help="Max time shift for sub alignment (ms).")
    ap.add_argument("--sub_align_mono_strength", type=float, default=0.6, help="0..1, reduce low-end stereo width.")
    ap.add_argument("--no_luminance", action="store_true", help="Disable harmonic luminance sculptor.")
    ap.add_argument("--luminance_hp_hz", type=float, default=6500.0, help="High-pass cutoff for luminance (Hz).")
    ap.add_argument("--luminance_mix", type=float, default=0.08, help="0..1 mix for harmonic luminance.")
    ap.add_argument("--luminance_drive_db", type=float, default=2.0, help="Base drive for luminance (dB).")
    ap.add_argument("--luminance_dyn_depth_db", type=float, default=3.0, help="Dynamic drive range (dB).")
    ap.add_argument("--no_sub_anchor", action="store_true", help="Disable Sub-Anchor low-end stabilizer.")
    ap.add_argument("--sub_anchor_cutoff_hz", type=float, default=120.0, help="Sub-Anchor split cutoff (Hz).")
    ap.add_argument("--sub_anchor_threshold_db", type=float, default=-24.0, help="Sub-Anchor sub-band compressor threshold (dB).")
    ap.add_argument("--sub_anchor_ratio", type=float, default=2.5, help="Sub-Anchor sub-band compressor ratio.")
    ap.add_argument("--sub_anchor_attack_ms", type=float, default=8.0, help="Sub-Anchor compressor attack (ms).")
    ap.add_argument("--sub_anchor_release_ms", type=float, default=160.0, help="Sub-Anchor compressor release (ms).")
    ap.add_argument("--sub_anchor_sat_mix", type=float, default=0.10, help="0..1 Sub-Anchor saturation mix.")
    ap.add_argument("--sub_anchor_sat_drive_db", type=float, default=3.0, help="Sub-Anchor saturation drive (dB).")
    ap.add_argument("--no_air_presence", action="store_true", help="Disable Air & Presence (de-esser + midrange expressiveness).")
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
    ap.add_argument("--target_peak_dbfs", type=float, default=-1.0, help="Peak ceiling after processing. Default: -1 dBFS.")
    ap.add_argument("--log_level", default="INFO", help="Logging level (DEBUG, INFO, WARNING). Default: INFO.")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    if not args.reference.exists():
        raise FileNotFoundError(f"Reference not found: {args.reference}")
    if not args.target.exists():
        raise FileNotFoundError(f"Target not found: {args.target}")
    if args.fir_taps < 129 or args.fir_taps % 2 == 0:
        raise ValueError("--fir_taps must be an odd integer >= 129 (e.g., 513, 1025, 2049).")

    sr = args.sr

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
    LOG.debug("Target crest (before): %.2f dB", tgt_before.crest_db)

    # Build EQ match filter (using mono for analysis)
    h, eq_info = design_match_fir(
        ref_mono=to_mono(ref_audio),
        tgt_mono=to_mono(tgt_audio),
        sr=sr,
        numtaps=args.fir_taps,
        max_gain_db=args.max_eq_db,
    )

    # Apply EQ
    processed = apply_fir_stereo(tgt_audio, h)

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
            "max_shift_ms": float(args.sub_align_max_ms),
        }
        LOG.info("Sub-bass align applied (lag_ms=%.3f).", lag_ms)

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
        LOG.info("Harmonic luminance applied (hp_hz=%.1f, mix=%.3f).", args.luminance_hp_hz, args.luminance_mix)

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
            "presence_drive_db": float(args.presence_drive_db),
            "presence_dyn_depth_db": float(args.presence_dyn_depth_db),
        }
        LOG.info(
            "Air & Presence applied (de-ess %.0f-%.0f Hz, presence %.0f-%.0f Hz).",
            args.deesser_hp_hz, args.deesser_lp_hz, args.presence_bp_lo, args.presence_bp_hi
        )

    # Optional compression (only if target crest is much higher than reference)
    compressor_used = False
    if args.enable_compression:
        proc_metrics_mid = compute_metrics(processed, sr)
        if proc_metrics_mid.crest_db > ref_metrics.crest_db + 2.0:
            processed = compress_broadband(processed, sr)
            compressor_used = True
            LOG.info("Compression applied (crest difference %.2f dB).", proc_metrics_mid.crest_db - ref_metrics.crest_db)

    # Loudness match to reference LUFS
    processed, lufs_gain_db = match_lufs(processed, sr, target_lufs=ref_metrics.lufs_i)

    # Peak safety
    processed, limiter_gain_db = peak_limit(processed, peak_dbfs_target=args.target_peak_dbfs)

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
        compressor_used=compressor_used,
        sub_align_info=sub_align_info,
        luminance_info=luminance_info,
        sub_anchor_info=sub_anchor_info,
        air_presence_info=air_presence_info,
    )

    LOG.info("Wrote: %s", args.out)
    LOG.info("Report: %s", args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
