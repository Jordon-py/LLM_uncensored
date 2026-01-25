#!/usr/bin/env python3
"""
AuralMind Match v6.0 (Expert) — Reference-Based Mastering for Melodic Trap
========================================================================

What this script does (musically):
----------------------------------
You give it:
  - a "reference" track (the sound you want)
  - a "target" track (your mix to master)

It outputs:
  - a mastered WAV with:
      • tonal coherence (spectral match EQ)
      • tight club-ready low end (sub alignment + sub-anchor)
      • smooth, modern air (luminance + de-ess)
      • controlled presence (mid harmonics)
      • melodic stereo warmth (frequency-dependent M/S imaging)
      • modern loudness handling (soft clipper + true peak ceiling)
      • stable final loudness (iterative LUFS finalization)

python auralmind_match_v6.py
  --reference "C:/Users/goku/Downloads/Brent Faiyaz - Pistachios [Official Video].wav" --target "C:/Users/goku/Downloads/Vegas - top teir (20).wav" --out "C:/Users/goku/LLM_uncensored/out_v4/Vegas Top Tier hifi.wav"
  --report "C:/path/out/TARGET_v6_wide_report.md"
  --sr 48000
  --fir_taps 4097 --max_eq_db 8 --eq_smooth_hz 90 --eq_phase minimum
  --stereo_side_hp_hz 180 --stereo_width_mid 1.09 --stereo_width_hi 1.26 --stereo_corr_min 0.00
  --clip_drive_db 2.2 --clip_mix 0.20 --clip_oversample 4
  --rhythm_amount 0.12
  --target_peak_dbfs -1.0 --limiter_oversample 4 --finalize_iters 3 --finalize_tol_lu 0.10
  --log_level INFO

----------------------------------
- Trap lives/dies by: 808 translation + kick snap + vocal forwardness + “wide but controlled” space.
- Many masters fail because they widen the wrong band, crush transients, or miss loudness after limiting.

Dependencies:
-------------
numpy, scipy, soundfile, librosa, pyloudnorm
ffmpeg recommended for MP3/M4A decode.

Notes:
------
This file performs real processing when executed normally.
Here, we only present the enhanced script (no audio rendering is performed in chat).
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

from scipy.signal import (
    firwin,
    firwin2,
    fftconvolve,
    savgol_filter,
    resample_poly,
)

LOG = logging.getLogger("auralmind")


# ============================================================
# 1) IO Utilities
# ============================================================

def _require_ffmpeg() -> str:
    """Return ffmpeg path if installed; raise helpful error if missing."""
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
    """Run subprocess and raise a clean error on failure."""
    LOG.debug("Running: %s", " ".join(cmd))
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}/n{p.stderr}")


def decode_with_ffmpeg_to_wav(input_path: Path, out_wav_path: Path, sr: int, channels: int = 2) -> None:
    """Decode any audio format to float32 WAV at chosen SR / channel count."""
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
    Load audio as float32 stereo at sample rate `sr`.
    - soundfile for PCM formats
    - ffmpeg temp decode for mp3/m4a/etc
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

    with tempfile.TemporaryDirectory() as td:
        tmp_wav = Path(td) / "tmp_decode.wav"
        decode_with_ffmpeg_to_wav(path, tmp_wav, sr=sr, channels=2)
        y, file_sr = sf.read(tmp_wav, always_2d=True)
        y = y.astype(np.float32)
        if file_sr != sr:
            y = librosa.resample(y.T, orig_sr=file_sr, target_sr=sr).T.astype(np.float32)
        return y, sr


# ============================================================
# 2) Core math helpers
# ============================================================

def ensure_finite(x: np.ndarray, name: str = "array") -> np.ndarray:
    """Replace NaN/Inf with zeros (safety)."""
    if not np.isfinite(x).all():
        LOG.warning("%s contained NaN/inf; replacing with zeros.", name)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x.astype(np.float32, copy=False)


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Stereo -> mono mid-sum."""
    if audio.ndim == 1:
        return audio.astype(np.float32, copy=False)
    return (0.5 * (audio[:, 0] + audio[:, 1])).astype(np.float32)


def db(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(x), eps))


def undb(dbv: np.ndarray) -> np.ndarray:
    return 10.0 ** (dbv / 20.0)


def rms_dbfs(mono: np.ndarray) -> float:
    r = float(np.sqrt(np.mean(mono * mono) + 1e-12))
    return float(db(np.array([r]))[0])


def peak_dbfs(audio: np.ndarray) -> float:
    p = float(np.max(np.abs(audio)) + 1e-12)
    return float(db(np.array([p]))[0])


def _ms_encode(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Mid/Side encode."""
    mid = 0.5 * (x[:, 0] + x[:, 1])
    side = 0.5 * (x[:, 0] - x[:, 1])
    return mid.astype(np.float32), side.astype(np.float32)


def _ms_decode(mid: np.ndarray, side: np.ndarray) -> np.ndarray:
    """Mid/Side decode."""
    return np.stack([mid + side, mid - side], axis=1).astype(np.float32)


def stereo_correlation(audio: np.ndarray) -> float:
    """
    Correlation proxy:
    +1 = mono identical
    0  = wide/uncorrelated
    <0 = phasey / risky mono collapse
    """
    if audio.ndim != 2 or audio.shape[1] != 2:
        return 1.0
    L = audio[:, 0].astype(np.float64)
    R = audio[:, 1].astype(np.float64)
    L -= np.mean(L)
    R -= np.mean(R)
    denom = (np.std(L) * np.std(R) + 1e-12)
    return float(np.mean(L * R) / denom)


# ============================================================
# 3) Analysis (LUFS, spectrum, diagnostics)
# ============================================================

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
    meter = pyln.Meter(sr)
    return float(meter.integrated_loudness(mono.astype(np.float64)))


def spectral_centroid(mono: np.ndarray, sr: int) -> float:
    c = librosa.feature.spectral_centroid(y=mono, sr=sr)
    return float(np.mean(c))


def band_energy_db(mono: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Rough band energies for reporting.
    These bands align with trap roles (sub/low/mid/high/air).
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


def compute_metrics(audio: np.ndarray, sr: int) -> AudioMetrics:
    x = ensure_finite(audio, "metrics_audio")
    mono = to_mono(x)
    lufs = measure_lufs_integrated(mono, sr)
    rmsv = rms_dbfs(mono)
    peakv = peak_dbfs(x)
    crest = float(peakv - rmsv)
    bands = band_energy_db(mono, sr)
    cent = spectral_centroid(mono, sr)
    dur = float(len(mono) / sr)
    return AudioMetrics(sr, lufs, rmsv, peakv, crest, cent, bands, dur)


def average_spectrum_db(mono: np.ndarray, sr: int, n_fft: int = 8192) -> Tuple[np.ndarray, np.ndarray]:
    """Long-term average magnitude spectrum (macro tone)."""
    S = np.abs(librosa.stft(mono, n_fft=n_fft, hop_length=n_fft // 8, window="hann")) + 1e-12
    mag = np.mean(S, axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    return freqs, db(mag)


# ============================================================
# 4) Spectral Match EQ (minimum-phase option)
# ============================================================

def _freq_guardrails_db(freqs: np.ndarray, delta_db: np.ndarray) -> np.ndarray:
    """
    Guardrails stop the match EQ from doing dangerous moves:
    - subs: avoid huge boosts (limiter stress)
    - highs: avoid harshness
    """
    out = delta_db.copy()
    out[freqs < 35.0] = np.clip(out[freqs < 35.0], -3.0, 3.0)
    out[(freqs >= 35.0) & (freqs < 90.0)] = np.clip(out[(freqs >= 35.0) & (freqs < 90.0)], -6.0, 6.0)
    out[freqs >= 8000.0] = np.clip(out[freqs >= 8000.0], -4.0, 4.0)
    return out


def minimum_phase_fir_from_mag(freqs_hz: np.ndarray, mag: np.ndarray, sr: int, numtaps: int, n_fft: int = 32768) -> np.ndarray:
    """
    Minimum-phase FIR reconstruction from magnitude curve using cepstrum.

    Why:
      Linear-phase FIR match can add pre-ringing → softer transient punch.
      Minimum-phase keeps the energy after the transient (more “impact”). :contentReference[oaicite:5]{index=5}
    """
    n_fft = int(max(4096, n_fft))
    if n_fft < 2 * numtaps:
        n_fft = 1 << int(np.ceil(np.log2(2 * numtaps)))

    nyq = sr / 2.0
    dense_f = np.linspace(0.0, nyq, n_fft // 2 + 1)

    mag_dense = np.interp(dense_f, freqs_hz, mag).astype(np.float64)
    mag_dense = np.maximum(mag_dense, 1e-6)

    log_mag = np.log(mag_dense)
    log_full = np.concatenate([log_mag, log_mag[-2:0:-1]])
    cep = np.fft.ifft(log_full).real

    # minimum-phase cepstrum liftering
    cep_min = np.zeros_like(cep)
    cep_min[0] = cep[0]
    cep_min[1:n_fft // 2] = 2.0 * cep[1:n_fft // 2]
    cep_min[n_fft // 2] = cep[n_fft // 2]

    spec_min = np.exp(np.fft.fft(cep_min))
    h = np.fft.ifft(spec_min).real
    h = h[:numtaps]

    # light window = smooth impulse truncation
    w = np.hanning(numtaps)
    h = (h * w).astype(np.float32)

    # DC normalize to avoid global level shifts
    dc = float(np.sum(h))
    if abs(dc) > 1e-6:
        h /= dc

    return h.astype(np.float32)


def design_match_fir(
    ref_mono: np.ndarray,
    tgt_mono: np.ndarray,
    sr: int,
    numtaps: int,
    max_gain_db: float,
    smooth_hz: float,
    enable_guardrails: bool,
    eq_phase: str,
    minphase_nfft: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Build match EQ FIR (linear or minimum-phase)."""
    if numtaps % 2 == 0:
        numtaps += 1

    freqs, ref_db = average_spectrum_db(ref_mono, sr)
    _, tgt_db = average_spectrum_db(tgt_mono, sr)

    delta_db = ref_db - tgt_db

    # Smooth avoids chasing tiny resonances
    df = float(freqs[1] - freqs[0])
    win = max(5, int(smooth_hz / max(df, 1e-9)))
    if win % 2 == 0:
        win += 1
    win = min(win, len(delta_db) - (len(delta_db) % 2 == 0))
    win = max(win, 5)

    delta_smooth = savgol_filter(delta_db, window_length=win, polyorder=3, mode="interp")
    delta_smooth = np.clip(delta_smooth, -max_gain_db, max_gain_db)

    if enable_guardrails:
        delta_smooth = _freq_guardrails_db(freqs, delta_smooth)

    desired_mag = undb(delta_smooth)

    eq_phase = eq_phase.lower().strip()
    if eq_phase == "linear":
        nyq = sr / 2.0
        f_norm = np.clip(freqs / nyq, 0.0, 1.0)
        f_norm[0] = 0.0
        f_norm[-1] = 1.0
        h = firwin2(numtaps, f_norm, desired_mag, window="hann").astype(np.float32)
    elif eq_phase == "minimum":
        h = minimum_phase_fir_from_mag(freqs, desired_mag, sr=sr, numtaps=numtaps, n_fft=minphase_nfft)
    else:
        raise ValueError("--eq_phase must be 'linear' or 'minimum'")

    info = {
        "numtaps": float(numtaps),
        "max_gain_db": float(max_gain_db),
        "smooth_hz": float(smooth_hz),
        "guardrails": bool(enable_guardrails),
        "min_gain_db_applied": float(np.min(delta_smooth)),
        "max_gain_db_applied": float(np.max(delta_smooth)),
        "eq_phase": eq_phase,
        "minphase_nfft": float(minphase_nfft),
    }
    return h, info


def apply_fir_stereo(audio: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Fast convolution FIR on stereo."""
    x = ensure_finite(audio, "fir_audio")
    yL = fftconvolve(x[:, 0], h, mode="same")
    yR = fftconvolve(x[:, 1], h, mode="same")
    return np.stack([yL, yR], axis=1).astype(np.float32)


def _design_fir_lowpass(sr: int, cutoff_hz: float, numtaps: int = 513) -> np.ndarray:
    cutoff_hz = float(np.clip(cutoff_hz, 10.0, sr / 2 - 10.0))
    return firwin(numtaps, cutoff=cutoff_hz, fs=sr, pass_zero=True, window="hann").astype(np.float32)


def _design_fir_highpass(sr: int, cutoff_hz: float, numtaps: int = 513) -> np.ndarray:
    cutoff_hz = float(np.clip(cutoff_hz, 10.0, sr / 2 - 10.0))
    return firwin(numtaps, cutoff=cutoff_hz, fs=sr, pass_zero=False, window="hann").astype(np.float32)


# ============================================================
# 5) Sub alignment + low-end mono (tight 808 translation)
# ============================================================

def _fractional_shift(x: np.ndarray, shift_samples: float) -> np.ndarray:
    """Fractional delay via linear interpolation."""
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


def _estimate_delay_samples(a: np.ndarray, b: np.ndarray, sr: int, max_shift_ms: float, decim: int = 8) -> int:
    """Estimate small integer lag via local cross-correlation."""
    max_shift_samp = int(sr * max_shift_ms / 1000.0)
    if max_shift_samp < 1:
        return 0

    ad = a[::decim]
    bd = b[::decim]
    maxd = max_shift_samp // decim
    if len(ad) < 64 or len(bd) < 64:
        return 0

    best_lag = 0
    best_val = -1e18
    for lag in range(-maxd, maxd + 1):
        if lag < 0:
            x1 = ad[-lag:]
            x2 = bd[:len(x1)]
        else:
            x1 = ad[:len(ad) - lag]
            x2 = bd[lag:lag + len(x1)]
        if len(x1) < 64:
            continue
        val = float(np.sum(x1 * x2))
        if val > best_val:
            best_val = val
            best_lag = lag

    return int(best_lag * decim)


def align_subbass_phase(
    audio: np.ndarray,
    sr: int,
    cutoff_hz: float,
    max_shift_ms: float,
    mono_strength: float,
) -> Tuple[np.ndarray, float]:
    """
    Align sub-band L/R and optionally reduce width under cutoff.
    """
    x = ensure_finite(audio, "sub_align_audio")

    h_lp = _design_fir_lowpass(sr, cutoff_hz=cutoff_hz, numtaps=513)
    low = apply_fir_stereo(x, h_lp)
    high = (x - low).astype(np.float32)

    L = low[:, 0]
    R = low[:, 1]
    lag_samples = _estimate_delay_samples(L, R, sr=sr, max_shift_ms=max_shift_ms, decim=8)

    if lag_samples != 0:
        R2 = _fractional_shift(R, shift_samples=float(lag_samples))
        low = np.stack([L, R2], axis=1).astype(np.float32)

    # monoize sub-band gradually
    mono_strength = float(np.clip(mono_strength, 0.0, 1.0))
    if mono_strength > 0.0:
        mid, side = _ms_encode(low)
        side *= (1.0 - mono_strength)
        low = _ms_decode(mid, side)

    lag_ms = float(lag_samples / sr * 1000.0)
    return (low + high).astype(np.float32), lag_ms


# ============================================================
# 6) Sub-Anchor (low-end stabilizer)
# ============================================================

def soft_saturate(x: np.ndarray, drive_db: float = 2.0) -> np.ndarray:
    """
    Gentle saturation.
    Musical effect: adds harmonics to help 808 “read” on small speakers.
    """
    drive = float(10 ** (drive_db / 20.0))
    y = np.tanh(x * drive)
    # compensate roughly for tanh level drop
    y /= np.tanh(drive + 1e-9)
    return y.astype(np.float32)


def sub_anchor(
    audio: np.ndarray,
    sr: int,
    cutoff_hz: float,
    threshold_db: float,
    ratio: float,
    attack_ms: float,
    release_ms: float,
    sat_mix: float,
    sat_drive_db: float,
) -> np.ndarray:
    """
    Compress + saturate only the sub band, leaving mids/highs clean.
    Trap rationale: tight low end = louder master with less pumping.
    """
    x = ensure_finite(audio, "sub_anchor_audio")

    h_lp = _design_fir_lowpass(sr, cutoff_hz=cutoff_hz, numtaps=513)
    sub = apply_fir_stereo(x, h_lp)
    rest = (x - sub).astype(np.float32)

    # detector on mono sub
    sub_m = to_mono(sub)
    env = np.abs(sub_m).astype(np.float32)

    # envelope smoothing
    atk = max(1e-4, attack_ms / 1000.0)
    rel = max(1e-4, release_ms / 1000.0)
    a_a = math.exp(-1.0 / (sr * atk))
    a_r = math.exp(-1.0 / (sr * rel))

    sm = np.zeros_like(env)
    sm[0] = env[0]
    for i in range(1, len(env)):
        if env[i] > sm[i - 1]:
            sm[i] = a_a * sm[i - 1] + (1 - a_a) * env[i]
        else:
            sm[i] = a_r * sm[i - 1] + (1 - a_r) * env[i]

    level_db = db(sm + 1e-9)
    over_db = np.maximum(0.0, level_db - threshold_db)
    gr_db = over_db - (over_db / max(1e-6, ratio))
    gain = undb(-gr_db).astype(np.float32)

    sub_c = sub * gain[:, None]

    # subtle harmonic “glue” on subs
    sat_mix = float(np.clip(sat_mix, 0.0, 1.0))
    if sat_mix > 0.0:
        sat = soft_saturate(sub_c, drive_db=sat_drive_db)
        sub_c = (1.0 - sat_mix) * sub_c + sat_mix * sat

    return (sub_c + rest).astype(np.float32)


# ============================================================
# 7) Air / Presence / De-ess (kept, simplified but effective)
# ============================================================

def bandpass_fir(sr: int, lo_hz: float, hi_hz: float, numtaps: int = 513) -> np.ndarray:
    lo = float(np.clip(lo_hz, 10.0, sr / 2 - 50.0))
    hi = float(np.clip(hi_hz, lo + 50.0, sr / 2 - 10.0))
    return firwin(numtaps, [lo, hi], fs=sr, pass_zero=False, window="hann").astype(np.float32)


def dynamic_band_shaper(
    audio: np.ndarray,
    sr: int,
    bp_lo: float,
    bp_hi: float,
    mix: float,
    drive_db: float,
    dyn_depth_db: float,
    env_ms: float,
) -> np.ndarray:
    """
    Generic dynamic harmonic enhancer for a band:
    - isolates a band (bandpass)
    - adds gentle saturation
    - uses envelope to prevent harsh buildup
    """
    x = ensure_finite(audio, "dyn_band_audio")
    mix = float(np.clip(mix, 0.0, 1.0))
    if mix <= 0.0:
        return x

    h_bp = bandpass_fir(sr, bp_lo, bp_hi, numtaps=513)
    band = apply_fir_stereo(x, h_bp)
    rest = (x - band).astype(np.float32)

    # band envelope -> dynamic depth control
    band_m = to_mono(band)
    env = np.abs(band_m).astype(np.float32)

    tau = max(1e-4, env_ms / 1000.0)
    a = math.exp(-1.0 / (sr * tau))
    sm = np.zeros_like(env)
    sm[0] = env[0]
    for i in range(1, len(env)):
        sm[i] = a * sm[i - 1] + (1 - a) * env[i]

    # high env => reduce effect to avoid harshness
    smn = sm / (np.percentile(sm, 95) + 1e-9)
    smn = np.clip(smn, 0.0, 1.0)

    depth = float(dyn_depth_db)
    dyn_db = -depth * smn
    dyn_gain = undb(dyn_db).astype(np.float32)

    shaped = soft_saturate(band * dyn_gain[:, None], drive_db=drive_db)
    band_out = (1.0 - mix) * band + mix * shaped

    return (band_out + rest).astype(np.float32)


def de_ess(
    audio: np.ndarray,
    sr: int,
    hp_hz: float,
    lp_hz: float,
    thresh: float,
    max_reduction_db: float,
    env_ms: float,
) -> np.ndarray:
    """
    De-esser:
    - isolates sibilant band
    - compresses it when it exceeds thresh
    """
    x = ensure_finite(audio, "deess_audio")
    h_bp = bandpass_fir(sr, hp_hz, lp_hz, numtaps=513)
    sib = apply_fir_stereo(x, h_bp)
    rest = (x - sib).astype(np.float32)

    s = to_mono(sib)
    env = np.abs(s).astype(np.float32)

    tau = max(1e-4, env_ms / 1000.0)
    a = math.exp(-1.0 / (sr * tau))
    sm = np.zeros_like(env)
    sm[0] = env[0]
    for i in range(1, len(env)):
        sm[i] = a * sm[i - 1] + (1 - a) * env[i]

    smn = sm / (np.percentile(sm, 95) + 1e-9)
    smn = np.clip(smn, 0.0, 2.0)

    over = np.maximum(0.0, smn - float(thresh))
    red_db = np.minimum(float(max_reduction_db), over * float(max_reduction_db))
    gain = undb(-red_db).astype(np.float32)

    sib_c = sib * gain[:, None]
    return (sib_c + rest).astype(np.float32)


# ============================================================
# 8) Enhancement Loop 1: Stereo Scene Sculptor + Translation Guard
# ============================================================

def stereo_scene_sculptor(
    audio: np.ndarray,
    sr: int,
    side_hp_hz: float,
    width_hi: float,
    width_mid: float,
    hi_cross_hz: float,
    mid_cross_hz: float,
    correlation_min: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Melodic stereo widening:
    - keep subs mono-safe by HPF on side
    - widen harmonics up high (air + melodic texture)
    - keep mid center stable (vocal clarity)

    Standard mastering concept:
      "boost highs on the side channel" to create width illusion :contentReference[oaicite:6]{index=6}
    """
    x = ensure_finite(audio, "stereo_sculpt_audio")
    mid, side = _ms_encode(x)

    # 1) side HPF → tight mono-safe low-end
    h_hp = _design_fir_highpass(sr, side_hp_hz, numtaps=513)
    side_hp = fftconvolve(side, h_hp, mode="same").astype(np.float32)

    # 2) split side into bands: low / mid / high
    h_lp_mid = _design_fir_lowpass(sr, mid_cross_hz, numtaps=513)
    side_low = fftconvolve(side_hp, h_lp_mid, mode="same").astype(np.float32)
    side_rest = side_hp - side_low

    h_lp_hi = _design_fir_lowpass(sr, hi_cross_hz, numtaps=513)
    side_mid = fftconvolve(side_rest, h_lp_hi, mode="same").astype(np.float32)
    side_hi = side_rest - side_mid

    # 3) apply width scaling
    width_mid = float(np.clip(width_mid, 0.8, 1.25))
    width_hi = float(np.clip(width_hi, 0.8, 1.40))

    side_new = side_low + (width_mid * side_mid) + (width_hi * side_hi)
    out = _ms_decode(mid, side_new)

    corr_before = stereo_correlation(x)
    corr_after = stereo_correlation(out)

    # Translation Guard: clamp width if correlation drops too low
    if corr_after < correlation_min:
        # bring side down until safe (simple clamp)
        target = float(np.clip(correlation_min, -0.2, 0.4))
        k = 0.85  # reduce sides
        side_new2 = side_new * k
        out2 = _ms_decode(mid, side_new2)
        corr_after2 = stereo_correlation(out2)

        info = {
            "corr_before": float(corr_before),
            "corr_after": float(corr_after),
            "corr_after_guard": float(corr_after2),
            "guard_applied": True,
            "width_mid": float(width_mid),
            "width_hi": float(width_hi),
            "side_hp_hz": float(side_hp_hz),
            "mid_cross_hz": float(mid_cross_hz),
            "hi_cross_hz": float(hi_cross_hz),
        }
        return out2.astype(np.float32), info

    info = {
        "corr_before": float(corr_before),
        "corr_after": float(corr_after),
        "guard_applied": False,
        "width_mid": float(width_mid),
        "width_hi": float(width_hi),
        "side_hp_hz": float(side_hp_hz),
        "mid_cross_hz": float(mid_cross_hz),
        "hi_cross_hz": float(hi_cross_hz),
    }
    return out.astype(np.float32), info


# ============================================================
# 9) Enhancement Loop 2: Soft Clipper + Crest-Aware Auto Glue
# ============================================================

def oversampled_soft_clip(
    audio: np.ndarray,
    drive_db: float,
    mix: float,
    oversample: int,
) -> np.ndarray:
    """
    Pre-limiter clipper:
    - shaves fast peaks before limiter
    - allows louder masters with fewer limiter artifacts

    Oversampling reduces aliasing for non-linear waveshaping.
    """
    x = ensure_finite(audio, "clip_audio")
    mix = float(np.clip(mix, 0.0, 1.0))
    if mix <= 0.0:
        return x

    os = int(max(1, oversample))
    drive = float(10 ** (drive_db / 20.0))

    y = x
    if os > 1:
        y = resample_poly(y, os, 1, axis=0).astype(np.float32)

    # tanh clip: smooth, “musical” saturation curve
    z = np.tanh(y * drive) / np.tanh(drive + 1e-9)

    if os > 1:
        z = resample_poly(z, 1, os, axis=0).astype(np.float32)

    return ((1.0 - mix) * x + mix * z).astype(np.float32)


def compress_broadband_glue(
    audio: np.ndarray,
    sr: int,
    threshold_db: float,
    ratio: float,
    attack_ms: float,
    release_ms: float,
    makeup_db: float = 0.0,
) -> np.ndarray:
    """
    Conservative broadband glue compressor.
    Kept intentionally gentle (mastering-safe).
    """
    x = ensure_finite(audio, "glue_audio")
    mono = to_mono(x)

    frame = int(0.02 * sr)
    hop = int(0.01 * sr)
    if frame < 32 or hop < 16:
        return x

    rms_vals = []
    positions = []
    for i in range(0, len(mono) - frame, hop):
        seg = mono[i:i + frame]
        rms_vals.append(float(np.sqrt(np.mean(seg * seg) + 1e-12)))
        positions.append(i)

    if not rms_vals:
        return x

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

    gain_db = np.zeros(len(mono), dtype=np.float32)
    for idx, pos in enumerate(positions):
        gain_db[pos:pos + hop] = sm[idx]
    if positions[-1] + hop < len(gain_db):
        gain_db[positions[-1] + hop:] = sm[-1]

    gain_lin = undb(gain_db + makeup_db).astype(np.float32)
    return np.stack([x[:, 0] * gain_lin, x[:, 1] * gain_lin], axis=1).astype(np.float32)


def auto_glue_from_crest(ref_crest_db: float, cur_crest_db: float) -> Optional[Dict[str, float]]:
    """
    Crest-aware compression settings:
    If the target is too dynamic vs reference, apply gentle glue.
    """
    diff = float(cur_crest_db - ref_crest_db)
    if diff < 1.5:
        return None  # already close

    # more crest difference -> slightly stronger glue
    # still mastering-safe: aim 1–2 dB GR, not mixbus smashing
    return {
        "threshold_db": -18.0 - min(4.0, diff),  # lower threshold slightly
        "ratio": 1.6 + min(0.6, diff / 6.0),
        "attack_ms": 12.0,
        "release_ms": 140.0,
        "makeup_db": 0.0,
    }


# ============================================================
# 10) Enhancement Loop 3: Rhythm Pulse + Harmonic Density Warmth
# ============================================================

def rhythm_pulse_enhancer(
    audio: np.ndarray,
    sr: int,
    amount: float,
    attack_band_hz: Tuple[float, float] = (2200.0, 9000.0),
    hop_length: int = 512,
) -> np.ndarray:
    """
    Rhythm enhancement (musical transient support):
    - detect onset strength (drum activity)
    - apply a tiny dynamic lift to the attack band during hits

    Effect:
      “drums feel clearer” without harsh static EQ.
    """
    x = ensure_finite(audio, "rhythm_audio")
    amount = float(np.clip(amount, 0.0, 0.25))
    if amount <= 0.0:
        return x

    mono = to_mono(x)
    onset = librosa.onset.onset_strength(y=mono, sr=sr, hop_length=hop_length)
    onset /= (np.percentile(onset, 95) + 1e-9)
    onset = np.clip(onset, 0.0, 1.0)

    # upsample onset envelope to sample rate
    env = np.repeat(onset, hop_length)[: len(mono)]
    if len(env) < len(mono):
        env = np.pad(env, (0, len(mono) - len(env)))

    # smooth a bit to avoid “flutter”
    win = int(0.010 * sr)  # 10ms smoothing
    if win > 8:
        k = np.ones(win, dtype=np.float32) / float(win)
        env = np.convolve(env.astype(np.float32), k, mode="same")

    # isolate attack band
    h_bp = bandpass_fir(sr, attack_band_hz[0], attack_band_hz[1], numtaps=513)
    atk = apply_fir_stereo(x, h_bp)
    rest = (x - atk).astype(np.float32)

    # gain curve: only small lift
    gain = (1.0 + amount * env).astype(np.float32)
    atk2 = atk * gain[:, None]
    return (atk2 + rest).astype(np.float32)


def harmonic_density_warmth_balancer(
    audio: np.ndarray,
    sr: int,
    base_sat_mix: float,
    sat_drive_db: float,
    max_extra_mix: float,
    n_fft: int = 2048,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Innovative adaptive warmth:
    - estimate harmonic vs percussive dominance (HPSS)
    - if harmonic density is high → allow more warmth/saturation
    - if drum-heavy → reduce warmth to prevent grit

    This is musically grounded:
      melodic sections benefit from harmonic depth;
      percussive density benefits from cleanliness.
    """
    x = ensure_finite(audio, "h_density_audio")
    base_sat_mix = float(np.clip(base_sat_mix, 0.0, 0.25))
    if base_sat_mix <= 0.0:
        return x, {"enabled": False}

    mono = to_mono(x)

    S = librosa.stft(mono, n_fft=n_fft, hop_length=n_fft // 4, window="hann")
    H, P = librosa.decompose.hpss(S)
    h_mono = librosa.istft(H, hop_length=n_fft // 4)
    p_mono = librosa.istft(P, hop_length=n_fft // 4)

    # ensure lengths match
    m = min(len(mono), len(h_mono), len(p_mono))
    mono = mono[:m]
    h_mono = h_mono[:m]
    p_mono = p_mono[:m]

    hr = float(np.sqrt(np.mean(h_mono * h_mono) + 1e-12))
    pr = float(np.sqrt(np.mean(p_mono * p_mono) + 1e-12))
    ratio = hr / (pr + 1e-12)

    # scale mix based on harmonic density
    # ratio ~1 => balanced; ratio >1.5 => melodic heavy
    extra = float(np.clip((math.log1p(ratio) - math.log1p(1.0)) * 0.12, 0.0, max_extra_mix))
    sat_mix = float(np.clip(base_sat_mix + extra, 0.0, 0.35))

    y = x.copy()
    sat = soft_saturate(y, drive_db=sat_drive_db)
    y = ((1.0 - sat_mix) * y + sat_mix * sat).astype(np.float32)

    info = {
        "enabled": True,
        "harmonic_rms": hr,
        "percussive_rms": pr,
        "harmonic_to_percussive_ratio": ratio,
        "base_sat_mix": base_sat_mix,
        "extra_sat_mix": extra,
        "final_sat_mix": sat_mix,
        "sat_drive_db": float(sat_drive_db),
    }
    return y, info


# ============================================================
# 11) Loudness + limiting (iterative finalize)
# ============================================================

def match_lufs(audio: np.ndarray, sr: int, target_lufs: float) -> Tuple[np.ndarray, float]:
    """Uniform gain to match integrated LUFS."""
    x = ensure_finite(audio, "lufs_audio")
    cur = measure_lufs_integrated(to_mono(x), sr)
    gain_db = float(target_lufs - cur)
    gain = float(10 ** (gain_db / 20.0))
    return (x * gain).astype(np.float32), gain_db


def _estimate_true_peak_lin(audio: np.ndarray, oversample: int = 4) -> float:
    """Estimate true peak by oversampling and measuring max abs."""
    x = ensure_finite(audio, "tp_audio")
    os = int(max(1, oversample))
    if os == 1:
        return float(np.max(np.abs(x)) + 1e-12)
    y = resample_poly(x, os, 1, axis=0)
    return float(np.max(np.abs(y)) + 1e-12)


def true_peak_limit(audio: np.ndarray, peak_dbfs_target: float, oversample: int) -> Tuple[np.ndarray, float]:
    """
    Transparent true-peak scaling limiter.
    True-peak ceiling (-1 dBTP) helps avoid inter-sample overs after encoding. :contentReference[oaicite:7]{index=7}
    """
    x = ensure_finite(audio, "tpl_audio")
    target_lin = float(10 ** (peak_dbfs_target / 20.0))
    tp = _estimate_true_peak_lin(x, oversample=oversample)
    if tp <= target_lin:
        return x, 0.0
    g = target_lin / tp
    gain_db = float(20.0 * np.log10(max(g, 1e-12)))
    return (x * g).astype(np.float32), gain_db


def finalize_lufs_and_ceiling(
    audio: np.ndarray,
    sr: int,
    target_lufs: float,
    target_peak_dbfs: float,
    oversample: int,
    iters: int,
    tol_lu: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Iterate (LUFS match → true peak ceiling) until we *land* on target loudness.

    This solves the classic:
      “I matched LUFS, but after limiting I'm quieter.”
    """
    x = ensure_finite(audio, "finalize_audio")

    iters = int(max(1, iters))
    tol_lu = float(max(0.02, tol_lu))
    total_lufs_gain_db = 0.0
    total_lim_gain_db = 0.0

    achieved = float("nan")
    for i in range(iters):
        x, g_db = match_lufs(x, sr, target_lufs)
        total_lufs_gain_db += float(g_db)

        x, lim_db = true_peak_limit(x, peak_dbfs_target=target_peak_dbfs, oversample=oversample)
        total_lim_gain_db += float(lim_db)

        achieved = measure_lufs_integrated(to_mono(x), sr)
        err = float(target_lufs - achieved)

        LOG.info("Finalize %d/%d: achieved %.2f LUFS (err %.2f), limiter_gain %.2f dB",
                 i + 1, iters, achieved, err, lim_db)

        if abs(err) <= tol_lu:
            break

        # If ceiling is the limiting factor, avoid endless push
        if lim_db < -2.5 and err > 0.25:
            LOG.info("Ceiling is constraining loudness; stopping to preserve dynamics.")
            break

    info = {
        "target_lufs": float(target_lufs),
        "achieved_lufs": float(achieved),
        "target_peak_dbfs": float(target_peak_dbfs),
        "total_lufs_gain_db": float(total_lufs_gain_db),
        "total_limiter_gain_db": float(total_lim_gain_db),
        "iters": int(iters),
        "tol_lu": float(tol_lu),
        "oversample": int(oversample),
    }
    return x.astype(np.float32), info


# ============================================================
# 12) Report
# ============================================================

def write_report(path: Path, payload: Dict) -> None:
    path.write_text("# Enhancement_Report/n/n```json/n" + json.dumps(payload, indent=2) + "/n```/n", encoding="utf-8")


# ============================================================
# 13) Main
# ============================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="AuralMind v6.0 Expert — reference-based mastering for melodic trap.")

    ap.add_argument("--reference", required=True, type=Path)
    ap.add_argument("--target", required=True, type=Path)
    ap.add_argument("--out", default=Path("enhanced_target_v6.wav"), type=Path)
    ap.add_argument("--report", default=Path("Enhancement_Report_v6.md"), type=Path)
    ap.add_argument("--sr", type=int, default=48000)

    # Match EQ
    ap.add_argument("--fir_taps", type=int, default=4097)
    ap.add_argument("--max_eq_db", type=float, default=8.0)
    ap.add_argument("--eq_smooth_hz", type=float, default=90.0)
    ap.add_argument("--no_eq_guardrails", action="store_true")
    ap.add_argument("--eq_phase", choices=["linear", "minimum"], default="minimum")
    ap.add_argument("--eq_minphase_nfft", type=int, default=32768)

    # Sub align
    ap.add_argument("--no_sub_align", action="store_true")
    ap.add_argument("--sub_align_cutoff_hz", type=float, default=120.0)
    ap.add_argument("--sub_align_max_ms", type=float, default=1.5)
    ap.add_argument("--sub_align_mono_strength", type=float, default=0.60)

    # Sub anchor
    ap.add_argument("--no_sub_anchor", action="store_true")
    ap.add_argument("--sub_anchor_cutoff_hz", type=float, default=120.0)
    ap.add_argument("--sub_anchor_threshold_db", type=float, default=-20.0)
    ap.add_argument("--sub_anchor_ratio", type=float, default=2.2)
    ap.add_argument("--sub_anchor_attack_ms", type=float, default=10.0)
    ap.add_argument("--sub_anchor_release_ms", type=float, default=180.0)
    ap.add_argument("--sub_anchor_sat_mix", type=float, default=0.06)
    ap.add_argument("--sub_anchor_sat_drive_db", type=float, default=2.0)

    # Air/presence/de-ess
    ap.add_argument("--no_luminance", action="store_true")
    ap.add_argument("--luminance_hp_hz", type=float, default=7200.0)
    ap.add_argument("--luminance_mix", type=float, default=0.06)
    ap.add_argument("--luminance_drive_db", type=float, default=1.5)
    ap.add_argument("--luminance_dyn_depth_db", type=float, default=2.0)

    ap.add_argument("--no_deesser", action="store_true")
    ap.add_argument("--deesser_hp_hz", type=float, default=6500.0)
    ap.add_argument("--deesser_lp_hz", type=float, default=11500.0)
    ap.add_argument("--deesser_thresh", type=float, default=0.13)
    ap.add_argument("--deesser_max_reduction_db", type=float, default=4.0)
    ap.add_argument("--deesser_env_ms", type=float, default=25.0)

    ap.add_argument("--no_presence", action="store_true")
    ap.add_argument("--presence_bp_lo", type=float, default=800.0)
    ap.add_argument("--presence_bp_hi", type=float, default=3750.0)
    ap.add_argument("--presence_mix", type=float, default=0.07)
    ap.add_argument("--presence_drive_db", type=float, default=1.6)
    ap.add_argument("--presence_dyn_depth_db", type=float, default=2.0)
    ap.add_argument("--presence_env_ms", type=float, default=70.0)

    # Enhancement Loop 1: Stereo sculptor
    ap.add_argument("--no_stereo_sculpt", action="store_true")
    ap.add_argument("--stereo_side_hp_hz", type=float, default=160.0)
    ap.add_argument("--stereo_width_mid", type=float, default=1.06)
    ap.add_argument("--stereo_width_hi", type=float, default=1.18)
    ap.add_argument("--stereo_mid_cross_hz", type=float, default=1600.0)
    ap.add_argument("--stereo_hi_cross_hz", type=float, default=6500.0)
    ap.add_argument("--stereo_corr_min", type=float, default=0.02)

    # Enhancement Loop 2: Clipper + glue
    ap.add_argument("--no_clipper", action="store_true")
    ap.add_argument("--clip_drive_db", type=float, default=2.0)
    ap.add_argument("--clip_mix", type=float, default=0.18)
    ap.add_argument("--clip_oversample", type=int, default=4)

    ap.add_argument("--enable_auto_glue", action="store_true")

    # Enhancement Loop 3: Rhythm + harmonic density
    ap.add_argument("--no_rhythm_pulse", action="store_true")
    ap.add_argument("--rhythm_amount", type=float, default=0.10)

    ap.add_argument("--no_harmonic_density", action="store_true")
    ap.add_argument("--hd_base_sat_mix", type=float, default=0.06)
    ap.add_argument("--hd_sat_drive_db", type=float, default=2.0)
    ap.add_argument("--hd_max_extra_mix", type=float, default=0.06)

    # Finalize
    ap.add_argument("--target_peak_dbfs", type=float, default=-1.0)
    ap.add_argument("--limiter_oversample", type=int, default=4)
    ap.add_argument("--finalize_iters", type=int, default=3)
    ap.add_argument("--finalize_tol_lu", type=float, default=0.10)

    ap.add_argument("--log_level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s %(message)s"
    )

    sr = int(args.sr)

    ref_audio, _ = load_audio_any_format(args.reference, sr=sr)
    tgt_audio, _ = load_audio_any_format(args.target, sr=sr)

    ref_audio = ensure_finite(ref_audio, "ref_audio")
    tgt_audio = ensure_finite(tgt_audio, "tgt_audio")

    ref_metrics = compute_metrics(ref_audio, sr)
    before_metrics = compute_metrics(tgt_audio, sr)

    # 1) Match EQ
    h, eq_info = design_match_fir(
        ref_mono=to_mono(ref_audio),
        tgt_mono=to_mono(tgt_audio),
        sr=sr,
        numtaps=args.fir_taps,
        max_gain_db=args.max_eq_db,
        smooth_hz=args.eq_smooth_hz,
        enable_guardrails=(not args.no_eq_guardrails),
        eq_phase=args.eq_phase,
        minphase_nfft=args.eq_minphase_nfft,
    )
    y = apply_fir_stereo(tgt_audio, h)

    # 2) Sub align
    sub_align_info = {"enabled": False}
    if not args.no_sub_align:
        y, lag_ms = align_subbass_phase(
            y,
            sr=sr,
            cutoff_hz=args.sub_align_cutoff_hz,
            max_shift_ms=args.sub_align_max_ms,
            mono_strength=args.sub_align_mono_strength,
        )
        sub_align_info = {"enabled": True, "lag_ms": float(lag_ms)}

    # 3) Sub anchor
    sub_anchor_info = {"enabled": False}
    if not args.no_sub_anchor:
        y = sub_anchor(
            y, sr=sr,
            cutoff_hz=args.sub_anchor_cutoff_hz,
            threshold_db=args.sub_anchor_threshold_db,
            ratio=args.sub_anchor_ratio,
            attack_ms=args.sub_anchor_attack_ms,
            release_ms=args.sub_anchor_release_ms,
            sat_mix=args.sub_anchor_sat_mix,
            sat_drive_db=args.sub_anchor_sat_drive_db,
        )
        sub_anchor_info = {"enabled": True}

    # 4) Presence (midrange expressiveness)
    if not args.no_presence:
        y = dynamic_band_shaper(
            y, sr=sr,
            bp_lo=args.presence_bp_lo,
            bp_hi=args.presence_bp_hi,
            mix=args.presence_mix,
            drive_db=args.presence_drive_db,
            dyn_depth_db=args.presence_dyn_depth_db,
            env_ms=args.presence_env_ms,
        )

    # 5) Luminance (smooth air)
    if not args.no_luminance:
        y = dynamic_band_shaper(
            y, sr=sr,
            bp_lo=args.luminance_hp_hz,
            bp_hi=min(20000.0, sr / 2 - 100.0),
            mix=args.luminance_mix,
            drive_db=args.luminance_drive_db,
            dyn_depth_db=args.luminance_dyn_depth_db,
            env_ms=40.0,
        )

    # 6) De-ess (protect from harsh top)
    if not args.no_deesser:
        y = de_ess(
            y, sr=sr,
            hp_hz=args.deesser_hp_hz,
            lp_hz=args.deesser_lp_hz,
            thresh=args.deesser_thresh,
            max_reduction_db=args.deesser_max_reduction_db,
            env_ms=args.deesser_env_ms,
        )

    # --- Enhancement Loop 1: stereo sculpt ---
    stereo_info = {"enabled": False}
    if not args.no_stereo_sculpt:
        y, stereo_info = stereo_scene_sculptor(
            y, sr=sr,
            side_hp_hz=args.stereo_side_hp_hz,
            width_hi=args.stereo_width_hi,
            width_mid=args.stereo_width_mid,
            hi_cross_hz=args.stereo_hi_cross_hz,
            mid_cross_hz=args.stereo_mid_cross_hz,
            correlation_min=args.stereo_corr_min,
        )

    # --- Enhancement Loop 2: clipper + auto glue ---
    if not args.no_clipper:
        y = oversampled_soft_clip(
            y,
            drive_db=args.clip_drive_db,
            mix=args.clip_mix,
            oversample=args.clip_oversample,
        )

    glue_info = {"enabled": False}
    if args.enable_auto_glue:
        cur_mid = compute_metrics(y, sr)
        settings = auto_glue_from_crest(ref_metrics.crest_db, cur_mid.crest_db)
        if settings:
            y = compress_broadband_glue(y, sr=sr, **settings)
            glue_info = {"enabled": True, **settings}

    # --- Enhancement Loop 3: rhythm + harmonic density ---
    if not args.no_rhythm_pulse:
        y = rhythm_pulse_enhancer(y, sr=sr, amount=args.rhythm_amount)

    hd_info = {"enabled": False}
    if not args.no_harmonic_density:
        y, hd_info = harmonic_density_warmth_balancer(
            y, sr=sr,
            base_sat_mix=args.hd_base_sat_mix,
            sat_drive_db=args.hd_sat_drive_db,
            max_extra_mix=args.hd_max_extra_mix,
        )

    # 7) Finalize loudness & true peak ceiling
    y, finalize_info = finalize_lufs_and_ceiling(
        y,
        sr=sr,
        target_lufs=ref_metrics.lufs_i,
        target_peak_dbfs=args.target_peak_dbfs,
        oversample=args.limiter_oversample,
        iters=args.finalize_iters,
        tol_lu=args.finalize_tol_lu,
    )

    after_metrics = compute_metrics(y, sr)

    # Write output
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out, y, sr, subtype="PCM_24")

    # Write report
    payload = {
        "files": {
            "reference": str(args.reference),
            "target": str(args.target),
            "out": str(args.out),
        },
        "metrics": {
            "reference": ref_metrics.__dict__,
            "before": before_metrics.__dict__,
            "after": after_metrics.__dict__,
        },
        "eq_info": eq_info,
        "sub_align": sub_align_info,
        "sub_anchor": sub_anchor_info,
        "stereo_sculpt": stereo_info,
        "auto_glue": glue_info,
        "harmonic_density": hd_info,
        "finalize": finalize_info,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    write_report(args.report, payload)

    LOG.info("Wrote output: %s", args.out)
    LOG.info("Wrote report: %s", args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
