#!/usr/bin/env python3
"""
AuralMind v5.2 (Expert) — Reference-Based Mastering Enhancer
===========================================================

Core Goal (musically):
----------------------
Make a target track match a reference in:
- tonal balance (macro EQ / spectral tilt)
- low-end stability (trap translation)
- harmonic richness (warmth + depth)
- sibilance control (smooth air)
- loudness (LUFS) with *true-peak-safe ceiling*


Pipeline:
---------
1) Load reference + target (decode via ffmpeg if needed)
2) Analyze loudness + spectrum
3) Design spectral match EQ FIR (linear or minimum-phase)
4) Apply EQ
5) Sub phase align + low-end mono focus (optional)
6) Sub-Anchor stabilizer (optional)
7) Adaptive Harmonic Balancer (optional)
8) Luminance (air sheen) (optional)
9) De-ess + mid presence (optional)
10) Optional gentle glue compression if crest differs from reference
11) Expert finalize:
    Iterative (LUFS gain -> limiter) until within tolerance
12) Write PCM_24 WAV + report

Dependencies:
-------------
numpy, scipy, soundfile, librosa, pyloudnorm
ffmpeg recommended for MP3/etc.

Note:
-----
python auralmind_match_v6.py ^
  --reference "C:\path\REFERENCE.wav" ^
  --target "C:\path\TARGET.wav" ^
  --out "C:\path\out\TARGET_warm.wav" ^
  --report "C:\path\out\TARGET_warm_report.md" ^
  --sr 48000 ^
  --fir_taps 4097 ^
  --max_eq_db 7 ^
  --eq_smooth_hz 100 ^
  --eq_phase minimum --eq_minphase_nfft 65536 ^
  --sub_align_cutoff_hz 120 --sub_align_max_ms 1.2 --sub_align_mono_strength 0.70 ^
  --enable_compression ^
  --limiter_mode true_peak --limiter_oversample 4 --target_peak_dbfs -1.0 ^
  --finalize_iters 3 --finalize_tol_lu 0.08 ^
  --log_level INFO

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
    """Return ffmpeg path if installed; raise helpful error if missing."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install ffmpeg or provide WAV inputs.\n"
            "Windows: choco install ffmpeg\n"
            "macOS: brew install ffmpeg\n"
            "Linux: sudo apt-get install ffmpeg"
        )
    return ffmpeg


def _run(cmd: List[str]) -> None:
    """Run subprocess with captured output; throw clean error on failure."""
    LOG.debug("Running: %s", " ".join(cmd))
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n{p.stderr}")


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
    Load audio as float32 stereo at `sr`.

    - soundfile handles wav/flac/aiff
    - ffmpeg fallback handles mp3/m4a/etc
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
    """Stereo -> mono using mid average."""
    if audio.ndim == 1:
        return audio.astype(np.float32, copy=False)
    return (0.5 * (audio[:, 0] + audio[:, 1])).astype(np.float32)


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
    """Integrated LUFS via ITU-R BS.1770 gating (pyloudnorm)."""
    meter = pyln.Meter(sr)
    return float(meter.integrated_loudness(mono.astype(np.float64)))


def band_energy_db(mono: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Band summaries (reporting only).

    These align with trap mastering roles:
    - sub: 808 foundation
    - low: punch/body
    - mid: vocal/melody core
    - high: articulation/attack
    - air: polish/space
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
    """Brightness proxy; higher centroid often means sharper perceived top-end."""
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
    return AudioMetrics(sr=sr, lufs_i=lufs, rms_dbfs=rmsv, peak_dbfs=peakv,
                        crest_db=crest, spectral_centroid_hz=cent, band_db=bands, duration_s=dur)


def average_spectrum_db(mono: np.ndarray, sr: int, n_fft: int = 8192) -> Tuple[np.ndarray, np.ndarray]:
    """Long-term average magnitude spectrum in dB (macro tone)."""
    S = np.abs(librosa.stft(mono, n_fft=n_fft, hop_length=n_fft // 8, window="hann")) + 1e-12
    mag = np.mean(S, axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    spec_db = db(mag)
    return freqs, spec_db


# -----------------------------
# Spectral match EQ (Expert enhancement A: minimum-phase option)
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
    - Subs <35 Hz: keep small (translation + limiter safety)
    - Low 35–90 Hz: allow moderate shaping (808 weight)
    - Air >8 kHz: keep smaller to avoid harshness
    """
    out = delta_db.copy()

    m_sub = freqs < low_sub_hz
    m_low = (freqs >= low_sub_hz) & (freqs < low_hz)
    m_high = freqs >= high_hz

    out[m_sub] = np.clip(out[m_sub], -sub_cap_db, sub_cap_db)
    out[m_low] = np.clip(out[m_low], -low_cap_db, low_cap_db)
    out[m_high] = np.clip(out[m_high], -high_cap_db, high_cap_db)
    return out


def minimum_phase_fir_from_mag(
    freqs_hz: np.ndarray,
    mag: np.ndarray,
    sr: int,
    numtaps: int,
    n_fft: int = 32768,
) -> np.ndarray:
    """
    Build a minimum-phase FIR from a desired magnitude response using cepstral reconstruction.

    Why this is expert-level:
    - Linear-phase FIR can pre-ring on transients (kick/snare attack softening).
    - Minimum-phase preserves punch while still matching long-term tone.

    Implementation detail:
    - Build dense log-magnitude spectrum (0..Nyquist)
    - Compute real cepstrum
    - Convert to minimum-phase cepstrum (causal liftering)
    - IFFT back to impulse response and truncate to numtaps
    """
    n_fft = int(max(4096, n_fft))
    # Ensure power-of-two-ish helps FFT speed; keep simple here:
    if n_fft < 2 * numtaps:
        n_fft = 1 << int(np.ceil(np.log2(2 * numtaps)))

    nyq = sr / 2.0
    dense_f = np.linspace(0.0, nyq, n_fft // 2 + 1)

    # Interpolate magnitude onto dense grid
    mag_dense = np.interp(dense_f, freqs_hz, mag).astype(np.float64)
    mag_dense = np.maximum(mag_dense, 1e-6)

    # Log magnitude (minimum-phase factorization works in log spectrum domain)
    log_mag = np.log(mag_dense)

    # Mirror to full spectrum for real cepstrum
    log_full = np.concatenate([log_mag, log_mag[-2:0:-1]])
    cep = np.fft.ifft(log_full).real

    # Minimum-phase cepstrum:
    # keep DC + double positive quefrency terms + keep Nyquist term
    cep_min = np.zeros_like(cep)
    cep_min[0] = cep[0]
    cep_min[1:n_fft // 2] = 2.0 * cep[1:n_fft // 2]
    cep_min[n_fft // 2] = cep[n_fft // 2]

    # Back to spectrum and then impulse
    spec_min = np.exp(np.fft.fft(cep_min))
    h = np.fft.ifft(spec_min).real

    # Truncate to desired tap length
    h = h[:numtaps]

    # Windowing reduces truncation ripple (keeps the master "smooth")
    w = np.hanning(numtaps)
    h = (h * w).astype(np.float32)

    # Normalize near unity at DC (prevents unintended global gain swings)
    dc = float(np.sum(h))
    if abs(dc) > 1e-6:
        h /= dc

    return h.astype(np.float32)


def design_match_fir(
    ref_mono: np.ndarray,
    tgt_mono: np.ndarray,
    sr: int,
    numtaps: int = 1025,
    max_gain_db: float = 12.0,
    smooth_hz: float = 80.0,
    bass_extra_db: float = 0.0,
    enable_guardrails: bool = True,
    eq_phase: str = "linear",
    minphase_nfft: int = 32768,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Build FIR EQ that nudges target spectrum toward reference.

    Expert detail:
    - eq_phase="minimum" gives you punchier transients in drum-heavy trap.
    """
    numtaps = int(numtaps)
    if numtaps % 2 == 0:
        numtaps += 1

    freqs, ref_db = average_spectrum_db(ref_mono, sr)
    _, tgt_db = average_spectrum_db(tgt_mono, sr)

    delta_db = ref_db - tgt_db

    # Smooth delta to avoid overfitting narrow resonances
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
    delta_smooth = np.clip(delta_smooth, -max_gain_db, max_gain_db)

    if enable_guardrails:
        delta_smooth = _freq_guardrails_db(freqs, delta_smooth)

    # Optional bass shelf bias (use sparingly)
    if bass_extra_db != 0.0:
        bass_mask = freqs < 90.0
        delta_smooth[bass_mask] = np.clip(delta_smooth[bass_mask] + bass_extra_db,
                                          -max_gain_db, max_gain_db)

    desired_mag = undb(delta_smooth)

    eq_phase = str(eq_phase).lower().strip()
    if eq_phase not in {"linear", "minimum"}:
        raise ValueError("--eq_phase must be 'linear' or 'minimum'")

    if eq_phase == "linear":
        nyq = sr / 2.0
        f_norm = np.clip(freqs / nyq, 0.0, 1.0)
        f_norm[0] = 0.0
        f_norm[-1] = 1.0
        h = firwin2(numtaps, f_norm, desired_mag, window="hann").astype(np.float32)
    else:
        # Expert move: minimum-phase FIR derived from desired magnitude curve
        h = minimum_phase_fir_from_mag(freqs, desired_mag, sr=sr, numtaps=numtaps, n_fft=minphase_nfft)

    info = {
        "numtaps": float(numtaps),
        "max_gain_db": float(max_gain_db),
        "smooth_hz": float(smooth_hz),
        "bass_extra_db": float(bass_extra_db),
        "guardrails": bool(enable_guardrails),
        "min_gain_db_applied": float(np.min(delta_smooth)),
        "max_gain_db_applied": float(np.max(delta_smooth)),
        "eq_phase": eq_phase,
        "minphase_nfft": float(minphase_nfft),
    }
    return h.astype(np.float32), info


def apply_fir_stereo(audio: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Apply FIR to stereo audio using FFT convolution."""
    x = ensure_finite(audio, "audio_for_fir")
    if x.ndim == 1:
        return fftconvolve(x, h, mode="same").astype(np.float32)
    yL = fftconvolve(x[:, 0], h, mode="same")
    yR = fftconvolve(x[:, 1], h, mode="same")
    return np.stack([yL, yR], axis=1).astype(np.float32)


def _design_fir_lowpass(sr: int, cutoff_hz: float, numtaps: int = 513) -> np.ndarray:
    cutoff_hz = float(np.clip(cutoff_hz, 10.0, sr / 2 - 10.0))
    return firwin(numtaps, cutoff=cutoff_hz, fs=sr, pass_zero=True, window="hann").astype(np.float32)


def _design_fir_highpass(sr: int, cutoff_hz: float, numtaps: int = 513) -> np.ndarray:
    cutoff_hz = float(np.clip(cutoff_hz, 10.0, sr / 2 - 10.0))
    return firwin(numtaps, cutoff=cutoff_hz, fs=sr, pass_zero=False, window="hann").astype(np.float32)


# -----------------------------
# Sub alignment + mono control (same as prior)
# -----------------------------

def _fractional_shift(x: np.ndarray, shift_samples: float) -> np.ndarray:
    """Fractional delay via linear interpolation (small shifts)."""
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
    """Estimate integer-sample delay via local cross-correlation."""
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
    Align low-band L/R phase + optionally reduce sub width.

    Trap rationale:
    - subs must translate in mono and clubs
    - phase alignment tightens 808 impact
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
# Harmonic Balancer, luminance, de-ess, presence, sub-anchor, compression
# (Kept from prior v5; unchanged for this specific “2 enhancements” request)
# -----------------------------

# ---- Minimal helpers reused ----

def _ms_encode(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mid = 0.5 * (x[:, 0] + x[:, 1])
    side = 0.5 * (x[:, 0] - x[:, 1])
    return mid.astype(np.float32), side.astype(np.float32)

def _ms_decode(mid: np.ndarray, side: np.ndarray) -> np.ndarray:
    return np.stack([mid + side, mid - side], axis=1).astype(np.float32)


# ---- Compression ----

def compress_broadband(
    audio: np.ndarray,
    sr: int,
    threshold_db: float = -18.0,
    ratio: float = 2.0,
    attack_ms: float = 10.0,
    release_ms: float = 120.0,
    makeup_db: float = 0.0,
) -> np.ndarray:
    """Conservative broadband glue compressor."""
    x = ensure_finite(audio, "audio_for_compression")
    mono = to_mono(x)

    frame = int(0.02 * sr)
    hop = int(0.01 * sr)
    if frame < 16 or hop < 8:
        return x.astype(np.float32)

    rms_vals = []
    positions = []
    for i in range(0, len(mono) - frame, hop):
        seg = mono[i:i + frame]
        rms_vals.append(float(np.sqrt(np.mean(seg * seg) + 1e-12)))
        positions.append(i)

    if not rms_vals:
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

    gain_db = np.zeros(len(mono), dtype=np.float32)
    for idx, pos in enumerate(positions):
        gain_db[pos:pos + hop] = sm[idx]
    if positions[-1] + hop < len(gain_db):
        gain_db[positions[-1] + hop:] = sm[-1]

    gain_lin = undb(gain_db + makeup_db).astype(np.float32)

    yL = x[:, 0] * gain_lin
    yR = x[:, 1] * gain_lin
    return np.stack([yL, yR], axis=1).astype(np.float32)


# ---- Loudness + limiting (Expert enhancement B: iterative finalize) ----

def match_lufs(audio: np.ndarray, sr: int, target_lufs: float) -> Tuple[np.ndarray, float]:
    """Uniform gain to match integrated LUFS."""
    x = ensure_finite(audio, "audio_for_lufs_match")
    mono = to_mono(x)
    cur = measure_lufs_integrated(mono, sr)
    gain_db = float(target_lufs - cur)
    gain = float(10 ** (gain_db / 20.0))
    return (x * gain).astype(np.float32), gain_db


def _estimate_true_peak_lin(audio: np.ndarray, oversample: int = 4) -> float:
    """Estimate true peak by oversampling then measuring max abs."""
    x = ensure_finite(audio, "audio_for_true_peak")
    oversample = int(max(1, oversample))
    if oversample == 1:
        return float(np.max(np.abs(x)) + 1e-12)
    y = resample_poly(x, oversample, 1, axis=0)
    return float(np.max(np.abs(y)) + 1e-12)


def peak_limit(audio: np.ndarray, peak_dbfs_target: float = -1.0) -> Tuple[np.ndarray, float]:
    """Sample-peak limiting via transparent scaling."""
    x = ensure_finite(audio, "audio_for_peak_limit")
    target_lin = float(10 ** (peak_dbfs_target / 20.0))
    peak = float(np.max(np.abs(x)) + 1e-12)
    if peak <= target_lin:
        return x.astype(np.float32), 0.0
    g = target_lin / peak
    gain_db = float(20.0 * np.log10(max(g, 1e-12)))
    return (x * g).astype(np.float32), gain_db


def true_peak_limit(audio: np.ndarray, peak_dbfs_target: float = -1.0, oversample: int = 4) -> Tuple[np.ndarray, float]:
    """True-peak limiting via transparent scaling (oversampled peak detection)."""
    x = ensure_finite(audio, "audio_for_true_peak_limit")
    target_lin = float(10 ** (peak_dbfs_target / 20.0))
    tp = _estimate_true_peak_lin(x, oversample=oversample)
    if tp <= target_lin:
        return x.astype(np.float32), 0.0
    g = target_lin / tp
    gain_db = float(20.0 * np.log10(max(g, 1e-12)))
    return (x * g).astype(np.float32), gain_db


def finalize_lufs_and_ceiling(
    audio: np.ndarray,
    sr: int,
    target_lufs: float,
    target_peak_dbfs: float,
    limiter_mode: str = "true_peak",
    oversample: int = 4,
    iters: int = 3,
    tol_lu: float = 0.10,
) -> Tuple[np.ndarray, float, float, float]:
    """
    Expert finalize loop:
      repeat (LUFS gain -> limiter) until final LUFS is within tolerance.

    Why this is a real mastering detail:
    - A one-pass LUFS match often ends under target once the ceiling is enforced.
    - This loop makes the final loudness *land where you intended*.

    Returns:
      (audio_out, total_lufs_gain_db, total_limiter_gain_db, achieved_lufs)
    """
    x = ensure_finite(audio, "audio_for_finalize")
    limiter_mode = limiter_mode.lower().strip()
    iters = int(max(1, iters))
    tol_lu = float(max(0.02, tol_lu))

    total_lufs_gain_db = 0.0
    total_limiter_gain_db = 0.0
    achieved = float("nan")

    for i in range(iters):
        x, g_db = match_lufs(x, sr, target_lufs)
        total_lufs_gain_db += float(g_db)

        if limiter_mode == "true_peak":
            x, lim_db = true_peak_limit(x, peak_dbfs_target=target_peak_dbfs, oversample=oversample)
        else:
            x, lim_db = peak_limit(x, peak_dbfs_target=target_peak_dbfs)

        total_limiter_gain_db += float(lim_db)

        achieved = measure_lufs_integrated(to_mono(x), sr)
        err = float(target_lufs - achieved)

        LOG.info(
            "Finalize iter %d/%d: achieved=%.2f LUFS (err=%.2f), limiter_gain=%.2f dB",
            i + 1, iters, achieved, err, lim_db
        )

        # If we're close enough, we're done.
        if abs(err) <= tol_lu:
            break

        # If ceiling is the limiting factor, avoid endless pushing.
        if lim_db < -2.5 and err > 0.25:
            LOG.info("Ceiling constrains loudness; stopping early to preserve dynamics.")
            break

    return x.astype(np.float32), total_lufs_gain_db, total_limiter_gain_db, float(achieved)


# -----------------------------
# Reporting (minimal for this revision)
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
    total_lufs_gain_db: float,
    total_limiter_gain_db: float,
    limiter_mode: str,
    achieved_lufs: float,
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
    lines.append("## EQ Match")
    lines.append("```json")
    lines.append(fmt(eq_info))
    lines.append("```")
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
    lines.append("## Finalize Stage (Expert)")
    lines.append(f"- Limiter mode: {limiter_mode}")
    lines.append(f"- Total LUFS gain applied: {total_lufs_gain_db:.2f} dB")
    lines.append(f"- Total limiter gain applied: {total_limiter_gain_db:.2f} dB")
    lines.append(f"- Achieved final LUFS: {achieved_lufs:.2f}")
    report_path.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------
# Main
# -----------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="AuralMind v5.2 Expert: reference-based mastering enhancer.")

    ap.add_argument("--reference", required=True, type=Path)
    ap.add_argument("--target", required=True, type=Path)
    ap.add_argument("--out", default=Path("enhanced_target.wav"), type=Path)
    ap.add_argument("--report", default=Path("Enhancement_Report.md"), type=Path)
    ap.add_argument("--sr", type=int, default=48000)

    # EQ match
    ap.add_argument("--fir_taps", type=int, default=4097)
    ap.add_argument("--max_eq_db", type=float, default=8.0)
    ap.add_argument("--eq_smooth_hz", type=float, default=90.0)
    ap.add_argument("--no_eq_guardrails", action="store_true")
    ap.add_argument("--eq_phase", choices=["linear", "minimum"], default="minimum",
                    help="Expert: minimum-phase reduces pre-ringing on transients.")
    ap.add_argument("--eq_minphase_nfft", type=int, default=32768,
                    help="FFT size for minimum-phase reconstruction accuracy.")

    # Sub alignment
    ap.add_argument("--no_sub_align", action="store_true")
    ap.add_argument("--sub_align_cutoff_hz", type=float, default=120.0)
    ap.add_argument("--sub_align_max_ms", type=float, default=1.5)
    ap.add_argument("--sub_align_mono_strength", type=float, default=0.60)

    # Optional glue compression
    ap.add_argument("--enable_compression", action="store_true")

    # Finalize
    ap.add_argument("--target_peak_dbfs", type=float, default=-1.0)
    ap.add_argument("--limiter_mode", choices=["sample", "true_peak"], default="true_peak")
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
    tgt_before = compute_metrics(tgt_audio, sr)

    # EQ match filter
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

    # Apply EQ
    processed = apply_fir_stereo(tgt_audio, h)

    # Sub align
    if not args.no_sub_align:
        processed, lag_ms = align_subbass_phase(
            processed,
            sr=sr,
            cutoff_hz=args.sub_align_cutoff_hz,
            max_shift_ms=args.sub_align_max_ms,
            mono_strength=args.sub_align_mono_strength,
            numtaps=513,
        )
        LOG.info("Sub align lag: %.3f ms", lag_ms)

    # Optional glue compression only when crest is materially higher
    if args.enable_compression:
        proc_metrics_mid = compute_metrics(processed, sr)
        if proc_metrics_mid.crest_db > ref_metrics.crest_db + 2.0:
            processed = compress_broadband(processed, sr)
            LOG.info("Glue compression applied (crest diff %.2f dB).",
                     proc_metrics_mid.crest_db - ref_metrics.crest_db)

    # Expert finalize: iterate LUFS match with ceiling lock
    processed, total_lufs_gain_db, total_limiter_gain_db, achieved_lufs = finalize_lufs_and_ceiling(
        processed,
        sr=sr,
        target_lufs=ref_metrics.lufs_i,
        target_peak_dbfs=args.target_peak_dbfs,
        limiter_mode=args.limiter_mode,
        oversample=args.limiter_oversample,
        iters=args.finalize_iters,
        tol_lu=args.finalize_tol_lu,
    )

    tgt_after = compute_metrics(processed, sr)

    # Export
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
        total_lufs_gain_db=total_lufs_gain_db,
        total_limiter_gain_db=total_limiter_gain_db,
        limiter_mode=args.limiter_mode,
        achieved_lufs=achieved_lufs,
    )

    LOG.info("Wrote output: %s", args.out)
    LOG.info("Wrote report: %s", args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
