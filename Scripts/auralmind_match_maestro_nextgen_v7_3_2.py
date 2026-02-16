#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AuralMind Match — Maestro v7.3 "StereoAI" (Expert-tier)
======================================================

Goal
----
Reference-based (or curve-based) mastering with *closed-loop* controls that protect
"pre-loudness openness" at loud playback, while still achieving competitive loudness.

What's new vs earlier generations
---------------------------------
1) Loudness Governor: Automatically backs off LUFS target if limiting would exceed a safe GR ceiling.
2) Mono-Sub v2: Note-aware cutoff (derived from detected sub fundamental) + adaptive mono mix.
3) Dynamic Masking EQ: Low-mid dip responds to masking ratio (220–360 Hz vs 2–6 kHz).
4) Stereo Image Enhancements:
   - Spatial Realism Enhancer (frequency-dependent M/S width + correlation guard)
   - NEW: Depth Distance Cue (energy-dependent HF tilt for front-to-back depth)
   - NEW: Depth-Adaptive CGMS MicroShift: micro-delay applied ONLY to SIDE high-band (>=2k),
          depth-adaptive to transients with a mono-compatibility guard.

Dependencies
------------
- numpy
- scipy
- soundfile

No librosa / numba required. (This script stays lightweight and portable.)

Usage
-----
Basic (curve-based master):
    python auralmind_match_maestro_nextgen_v7_3_stereoAI.py --target "song.wav" --out "song_master.wav"

Reference match:
    python auralmind_match_maestro_nextgen_v7_3_stereoAI.py --reference "ref.mp3" --target "song.wav" --out "song_master.wav"

Choose a preset:
    python auralmind_match_maestro_nextgen_v7_3_stereoAI.py --preset hi_fi_streaming --target "song.wav" --out "song_master.wav"

    python auralmind_match_maestro_nextgen_v7_3_2.py --preset hi_fi_streaming --target "C:\\Users\\goku\\Downloads\\Don't let me down  (10).wav" --out "DontLetMeDown.wav" 
Notes
-----
- Default sample rate is 48000 Hz (streaming-friendly, modern production workflows).
- True-peak limiting is approximated via oversampling peak detection + smooth gain.
  For mission-critical mastering, a dedicated TP limiter is still recommended, but this is robust enough for real work.
"""

from __future__ import annotations

import argparse
import os
import sys
import math
import json
import time
import logging
import faulthandler
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import soundfile as sf
import scipy
import scipy.signal as sps
from scipy.signal import fftconvolve, oaconvolve
from scipy.ndimage import maximum_filter1d

# Optional Demucs (HT-Demucs stem separation) — enabled by default, with graceful fallback
try:
    import torch  # type: ignore
    from demucs import pretrained  # type: ignore
    from demucs.apply import apply_model  # type: ignore
    _HAS_DEMUCS = True
except Exception:
    _HAS_DEMUCS = False

# Optional Numba for accelerated envelope follower
try:
    from numba import njit  # type: ignore
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

# Optional Joblib for batch parallel processing
try:
    from joblib import Parallel, delayed  # type: ignore
    _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False


sci = scipy.ndimage
"""
python auralmind_match_maestro_nextgen_v7_3_2.py \\\\
  --target "C:\\\\Users\\\\goku\\\\Downloads\\\\Vegas - top teir (20).wav" --reference "C:\\\\Users\\\\goku\\\\Downloads\\\\Brent Faiyaz - Pistachios [Official Video].mp3" --out "C:\\\\Users\\\\goku\\\\LLM_uncensored\\\\Scripts\\\\Mastered\\\\Vegas_top_teir__NEXTGEN_COMPETITIVE.wav" --report "C:\\\\Users\\\\goku\\\\LLM_uncensored\\\\Scripts\\\\Master\\\\Vegas_top_teir__NEXTGEN_COMPETITIVE.md"
  --log_level DEBUG \\\\
  --log_file "C:/path/to/auralmind.log" \\\\
  --progress_json "C:/path/to/progress.json" \\\\
  --watchdog_s 120
   --preset competitive_trap --demucs-device cpu --demucs-shifts 2 --demucs-overlap 0.25 --movement-amount 0.13 --hooklift-mix 0.26 --hooklift-percentile 78
"""
# ---------------------------
# Logging & progress infrastructure
# ---------------------------

LOG = logging.getLogger(__name__)


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
        force=True,  # IMPORTANT: override any earlier basicConfig
    )


def _enable_watchdog(seconds: int) -> None:
    """If a stage truly stalls, dump Python stack traces repeatedly to help pinpoint it."""
    if seconds and seconds > 0:
        faulthandler.enable(all_threads=True)
        faulthandler.dump_traceback_later(seconds, repeat=True)
        LOG.info("Watchdog enabled: dumping stack traces every %ss if stuck.", seconds)


@dataclass
class ProgressTick:
    stage: str
    pct: float
    detail: str
    t0: float
    last_update: float


class ProgressJSON:
    """Writes a JSON heartbeat file during long processing runs."""

    def __init__(self, path: Optional[str]):
        self.path = Path(path) if path else None
        self.t0 = time.perf_counter()
        self.stage = ""
        self.pct = 0.0
        self.detail = ""
        self._enabled = self.path is not None

    def update(self, stage: str, pct: float = 0.0, detail: str = "") -> None:
        if not self._enabled:
            return
        self.stage = stage
        self.pct = float(np.clip(pct, 0.0, 1.0))
        self.detail = detail
        payload = {
            "stage": self.stage,
            "pct": self.pct,
            "detail": self.detail,
            "elapsed_s": round(time.perf_counter() - self.t0, 3),
            "ts": time.time(),
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self.path)


@contextmanager
def stage(name: str, prog: Optional[ProgressJSON] = None):
    """Context manager that logs stage timing and updates progress JSON."""
    t0 = time.perf_counter()
    LOG.info("▶ %s", name)
    if prog:
        prog.update(name, 0.0, "start")
    try:
        yield
        if prog:
            prog.update(name, 1.0, "done")
    finally:
        dt = time.perf_counter() - t0
        LOG.info("✓ %s (%.2fs)", name, dt)


# ---------------------------
# Utility helpers
# ---------------------------

def excerpt_for_analysis(x: np.ndarray, sr: int, seconds_total: float = 45.0, segments: int = 3) -> np.ndarray:
    """
    Extract representative excerpts (start/middle/end) for faster spectral/key analysis.
    For melodic trap / tonal content, 3 segments of ~15s each captures the harmonic character
    without full-track STFT overhead.
    """
    x = np.asarray(x, dtype=np.float32)
    n = x.shape[0]
    if n <= 0:
        return x
    segs = int(max(1, segments))
    seg_len = int(max(2048, round(sr * (seconds_total / segs))))
    if n <= seg_len * segs:
        return x
    starts = np.linspace(0, n - seg_len, segs).astype(int)
    chunks = [x[s:s + seg_len] for s in starts]
    return np.concatenate(chunks, axis=0)


# Numba-accelerated envelope follower for de-esser (massive speedup on long files)
if _HAS_NUMBA:
    @njit(cache=True)
    def _env_follower(env: np.ndarray, att: float, rel: float) -> np.ndarray:
        out = np.empty_like(env, dtype=np.float32)
        e = 0.0
        for i in range(env.size):
            v = float(env[i])
            if v > e:
                e = v + att * (e - v)
            else:
                e = v + rel * (e - v)
            out[i] = e
        return out
else:
    # Pure Python fallback
    def _env_follower(env: np.ndarray, att: float, rel: float) -> np.ndarray:
        out = np.empty_like(env, dtype=np.float32)
        e = 0.0
        for i in range(env.size):
            v = float(env[i])
            if v > e:
                e = v + att * (e - v)
            else:
                e = v + rel * (e - v)
            out[i] = e
        return out


def key_confidence(det: Dict) -> float:
    """
    Crude confidence proxy from key detection profile similarity.
    For melodic trap, this prevents shimmer/glow from adding 'off' emphasis
    when the tonal center is uncertain - only lean into scale partials when stable.
    """
    a = float(det.get("score_major", 0.0))
    b = float(det.get("score_minor", 0.0))
    return float(np.clip(max(a, b), 0.0, 1.0))


def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))

def db_to_lin(db: float) -> float:
    return float(10.0 ** (db / 20.0))

def lin_to_db(x: float, eps: float = 1e-12) -> float:
    return float(20.0 * math.log10(max(abs(x), eps)))

def rms(x: np.ndarray, eps: float = 1e-12) -> float:
    return float(math.sqrt(np.mean(np.square(x), dtype=np.float64) + eps))

def peak(x: np.ndarray) -> float:
    return float(np.max(np.abs(x)))

def smoothstep(x: float, lo: float, hi: float) -> float:
    """0..1 smooth curve between lo and hi."""
    if hi <= lo:
        return 0.0
    t = clamp((x - lo) / (hi - lo), 0.0, 1.0)
    return float(t * t * (3.0 - 2.0 * t))

def smoothstep_vec(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Vectorized smoothstep for envelopes."""
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)

def _onepole_smooth(x: np.ndarray, sr: int, tc_ms: float) -> np.ndarray:
    """Simple one-pole smoother for control envelopes."""
    if tc_ms <= 0.0:
        return x.astype(np.float32)
    a = math.exp(-1.0 / (sr * (tc_ms / 1000.0) + 1e-12))
    b = np.array([1.0 - a], dtype=np.float32)
    a_coeff = np.array([1.0, -a], dtype=np.float32)
    return sps.lfilter(b, a_coeff, x).astype(np.float32)

def ensure_stereo(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return np.stack([y, y], axis=1)
    if y.shape[1] == 1:
        return np.repeat(y, 2, axis=1)
    return y


def to_mono(y: np.ndarray) -> np.ndarray:
    y = ensure_stereo(y).astype(np.float32)
    return 0.5 * (y[:, 0] + y[:, 1])

def resample_audio(y: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return y
    # Use polyphase for quality + speed
    g = math.gcd(sr_in, sr_out)
    up = sr_out // g
    down = sr_in // g
    return sps.resample_poly(y, up=up, down=down, axis=0).astype(np.float32)

@lru_cache(maxsize=128)
def _butter_highpass_cached(cut_hz: float, sr: int, order: int = 2):
    nyq = 0.5 * sr
    cut = max(1.0, float(cut_hz)) / nyq
    return sps.butter(order, cut, btype="highpass")

def butter_highpass(cut_hz: float, sr: int, order: int = 2):
    return _butter_highpass_cached(float(cut_hz), int(sr), int(order))

@lru_cache(maxsize=128)
def _butter_bandpass_cached(lo_hz: float, hi_hz: float, sr: int, order: int = 2):
    nyq = 0.5 * sr
    lo = max(1.0, float(lo_hz)) / nyq
    hi = min(nyq * 0.999, float(hi_hz)) / nyq
    if hi <= lo:
        hi = min(0.999, lo + 0.05)
    return sps.butter(order, [lo, hi], btype="bandpass")

def butter_bandpass(lo_hz: float, hi_hz: float, sr: int, order: int = 2):
    return _butter_bandpass_cached(float(lo_hz), float(hi_hz), int(sr), int(order))

def apply_iir(y: np.ndarray, b, a) -> np.ndarray:
    return sps.lfilter(b, a, y, axis=0).astype(np.float32)

def mid_side_encode(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    L = y[:, 0]
    R = y[:, 1]
    mid = 0.5 * (L + R)
    side = 0.5 * (L - R)
    return mid.astype(np.float32), side.astype(np.float32)

def mid_side_decode(mid: np.ndarray, side: np.ndarray) -> np.ndarray:
    L = mid + side
    R = mid - side
    return np.stack([L, R], axis=1).astype(np.float32)

def windowed_fft_mag(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """Return average magnitude spectrum (linear) across frames for mono x."""
    if x.ndim != 1:
        raise ValueError("windowed_fft_mag expects mono array.")
    x = np.asarray(x, dtype=np.float32)
    n = x.shape[0]
    if n <= 0:
        return np.zeros(n_fft//2 + 1, dtype=np.float32)

    # Match the original loop semantics (exclude the last full frame if start == n - n_fft).
    if n <= n_fft:
        n_frames = 1
    else:
        n_frames = 1 + max(0, (n - n_fft - 1) // hop)

    end_len = (n_frames - 1) * hop + n_fft
    if end_len > n:
        x = np.pad(x, (0, end_len - n))
    x = np.ascontiguousarray(x)

    shape = (n_frames, n_fft)
    strides = (x.strides[0] * hop, x.strides[0])
    frames = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

    win = np.hanning(n_fft).astype(np.float32)
    spec = np.fft.rfft(frames * win[None, :], axis=1)
    return np.mean(np.abs(spec), axis=0).astype(np.float32)


# ---------------------------
# Loudness (approx BS.1770)
# ---------------------------

def k_weighting_filter(sr: int):
    """
    Approximate K-weighting using a cascaded high-pass + shelving filter.
    This is not a full standard-validated implementation, but is stable and
    consistent for controlling relative loudness in this script.
    """
    # High-pass around 60 Hz (prevents sub-dominance in loudness estimate)
    b1, a1 = sps.butter(2, 60.0 / (0.5 * sr), btype="highpass")
    # Gentle high-shelf (~ +4 dB above ~1.5 kHz) to approximate K-weighting tilt
    # Use a biquad shelf design via RBJ cookbook.
    f0 = 1500.0
    gain_db = 4.0
    S = 1.0
    A = 10**(gain_db/40)
    w0 = 2*math.pi*f0/sr
    alpha = math.sin(w0)/2 * math.sqrt((A + 1/A)*(1/S - 1) + 2)
    cosw0 = math.cos(w0)

    b0 =    A*((A+1) + (A-1)*cosw0 + 2*math.sqrt(A)*alpha)
    b1s = -2*A*((A-1) + (A+1)*cosw0)
    b2 =    A*((A+1) + (A-1)*cosw0 - 2*math.sqrt(A)*alpha)
    a0 =        (A+1) - (A-1)*cosw0 + 2*math.sqrt(A)*alpha
    a1s =    2*((A-1) - (A+1)*cosw0)
    a2 =        (A+1) - (A-1)*cosw0 - 2*math.sqrt(A)*alpha

    bs = np.array([b0, b1s, b2]) / a0
    a_s = np.array([1.0, a1s/a0, a2/a0])
    return (b1, a1), (bs, a_s)

def integrated_loudness_lufs(y: np.ndarray, sr: int) -> float:
    """
    Approx integrated loudness:
    - K-weight filter (approx)
    - block energy in 400ms windows
    - absolute gate at -70 LUFS, relative gate at -10 LU below ungated mean
    """
    y = ensure_stereo(y)
    (b1, a1), (bs, a_s) = k_weighting_filter(sr)
    yk = apply_iir(apply_iir(y, b1, a1), bs, a_s)

    # Sum channels with weights (stereo: 1.0 each)
    mono = np.mean(yk, axis=1).astype(np.float32)

    block = int(0.400 * sr)
    hop = int(0.100 * sr)
    if block <= 0:
        return -100.0

    n = mono.shape[0]
    if n <= 0:
        return -100.0

    # Match the original loop semantics (exclude the last full frame if start == n - block).
    if n <= block:
        n_frames = 1
    else:
        n_frames = 1 + max(0, (n - block - 1) // hop)

    end_len = (n_frames - 1) * hop + block
    if end_len > n:
        mono = np.pad(mono, (0, end_len - n))
    mono = np.ascontiguousarray(mono)

    shape = (n_frames, block)
    strides = (mono.strides[0] * hop, mono.strides[0])
    frames = np.lib.stride_tricks.as_strided(mono, shape=shape, strides=strides)
    energies = np.mean(frames.astype(np.float64) ** 2, axis=1)
    if energies.size == 0:
        return -100.0

    # Convert to LUFS-ish: LUFS ~= -0.691 + 10*log10(mean_square)
    # The -0.691 is a common calibration constant; we use it for consistency.
    lufs_blocks = -0.691 + 10.0 * np.log10(np.maximum(energies, 1e-12))

    # absolute gate
    keep_abs = lufs_blocks > -70.0
    if not np.any(keep_abs):
        return -100.0
    lufs_abs_mean = float(np.mean(lufs_blocks[keep_abs]))

    # relative gate
    keep_rel = lufs_blocks > (lufs_abs_mean - 10.0)
    if not np.any(keep_rel):
        return lufs_abs_mean
    return float(np.mean(lufs_blocks[keep_rel]))

def apply_lufs_gain(y: np.ndarray, sr: int, target_lufs: float) -> Tuple[np.ndarray, float, float]:
    cur = integrated_loudness_lufs(y, sr)
    gain_db = target_lufs - cur
    g = db_to_lin(gain_db)
    return (y * g).astype(np.float32), cur, gain_db


# ---------------------------
# True-peak limiting (approx)
# ---------------------------

def true_peak_estimate(y: np.ndarray, sr: int, oversample: int = 4) -> float:
    if oversample <= 1:
        return peak(y)
    # oversample via polyphase for TP estimate
    y_os = sps.resample_poly(y, up=oversample, down=1, axis=0)
    return peak(y_os)

def limiter_smooth_gain(gains: np.ndarray, sr: int, attack_ms: float, release_ms: float) -> np.ndarray:
    atk = max(1, int(sr * attack_ms / 1000.0))
    rel = max(1, int(sr * release_ms / 1000.0))
    out = np.empty_like(gains)
    g = gains[0]
    for i, x in enumerate(gains):
        if x < g:  # need more reduction -> attack quickly
            g = g + (x - g) / atk
        else:      # release slowly
            g = g + (x - g) / rel
        out[i] = g
    return out

def true_peak_limiter(y: np.ndarray, sr: int, ceiling_dbfs: float = -1.0,
                      oversample: int = 4, attack_ms: float = 1.0, release_ms: float = 60.0) -> Tuple[np.ndarray, float]:
    """
    Approx TP limiter:
    - compute instantaneous peak envelope
    - derive gain to keep below ceiling
    - smooth gain
    """
    y = ensure_stereo(y).astype(np.float32)
    ceiling = db_to_lin(ceiling_dbfs)
    # peak envelope over short windows (1ms)
    win = max(8, int(sr * 0.001))
    env = np.zeros(len(y), dtype=np.float32)
    # max(|L|,|R|)
    inst = np.max(np.abs(y), axis=1).astype(np.float32)
    # moving max (fast)
    env = sci.maximum_filter1d(inst, size=win, mode="nearest")
    # gain to keep env under ceiling
    gains = np.minimum(1.0, ceiling / np.maximum(env, 1e-9)).astype(np.float32)
    gains = limiter_smooth_gain(gains, sr, attack_ms, release_ms).astype(np.float32)
    y_l = y[:, 0] * gains
    y_r = y[:, 1] * gains
    y2 = np.stack([y_l, y_r], axis=1).astype(np.float32)

    # Ensure true peak (oversampled) is under ceiling with one correction
    tp = true_peak_estimate(y2, sr, oversample=oversample)
    if tp > ceiling:
        corr = ceiling / max(tp, 1e-9)
        y2 = (y2 * corr).astype(np.float32)
        gains *= corr

    gr_db = lin_to_db(np.min(gains))
    return y2, gr_db


# ---------------------------
# Musical analysis (sub fundamental)
# ---------------------------

def estimate_sub_fundamental_hz(y: np.ndarray, sr: int,
                               lo_hz: float = 28.0, hi_hz: float = 85.0) -> Optional[float]:
    """
    Estimate 808/sub fundamental by scanning for strongest peak in [lo_hz, hi_hz]
    of the MID channel.
    """
    y = ensure_stereo(y)
    mid, _ = mid_side_encode(y)
    # bandpass to focus on sub
    b, a = butter_bandpass(lo_hz, hi_hz, sr, order=2)
    sub = sps.lfilter(b, a, mid).astype(np.float32)

    n = 1
    while n < len(sub):
        n *= 2
        if n >= 262144:
            break
    n = min(n, 262144)
    seg = sub[:n]
    if len(seg) < 4096:
        return None
    win = np.hanning(len(seg)).astype(np.float32)
    spec = np.fft.rfft(seg * win)
    mag = np.abs(spec)
    freqs = np.fft.rfftfreq(len(seg), 1.0 / sr)

    mask = (freqs >= lo_hz) & (freqs <= hi_hz)
    if not np.any(mask):
        return None
    idx = int(np.argmax(mag[mask]))
    peak_hz = float(freqs[mask][idx])
    if peak_hz <= 0:
        return None
    return peak_hz


# ---------------------------
# EQ design + match curve
# ---------------------------

def smooth_curve(y: np.ndarray, win_bins: int) -> np.ndarray:
    if win_bins <= 1:
        return y
    kernel = np.ones(win_bins, dtype=np.float32) / float(win_bins)
    return np.convolve(y, kernel, mode="same").astype(np.float32)

def design_fir_from_eq(freqs: np.ndarray, eq_db: np.ndarray, sr: int, taps: int) -> np.ndarray:
    """
    Design linear-phase FIR via frequency sampling.
    freqs: Hz, increasing (0..Nyq)
    eq_db: per freq
    """
    nyq = 0.5 * sr
    f = np.clip(freqs / nyq, 0.0, 1.0).astype(np.float64)
    g = (10.0 ** (eq_db / 20.0)).astype(np.float64)
    # Ensure endpoints exist
    if f[0] > 0.0:
        f = np.concatenate([[0.0], f])
        g = np.concatenate([[g[0]], g])
    if f[-1] < 1.0:
        f = np.concatenate([f, [1.0]])
        g = np.concatenate([g, [g[-1]]])
    fir = sps.firwin2(taps, f, g, window="hann").astype(np.float32)
    return fir

def apply_fir(y: np.ndarray, fir: np.ndarray) -> np.ndarray:
    """
    Apply FIR filter using overlap-add convolution for stable RAM usage on long signals.
    Falls back to fftconvolve if oaconvolve fails.
    """
    y = ensure_stereo(y)
    fir = fir.astype(np.float32)
    out = np.zeros_like(y, dtype=np.float32)
    try:
        # oaconvolve is overlap-add; more stable RAM usage on long signals
        for ch in range(2):
            out[:, ch] = oaconvolve(y[:, ch], fir, mode="same").astype(np.float32)
    except Exception:
        # fallback to fftconvolve
        for ch in range(2):
            out[:, ch] = fftconvolve(y[:, ch], fir, mode="same").astype(np.float32)
    return out

def build_target_curve(freqs: np.ndarray) -> np.ndarray:
    """
    Hi-fi trap translation target:
    - protect subs (but avoid boom)
    - dip low-mids a bit
    - presence lift
    - gentle air lift (not harshness)
    """
    f = freqs.astype(np.float32)
    eq = np.zeros_like(f, dtype=np.float32)

    # Sub tilt: +1.0 dB @ 45 Hz, 0 dB @ 90 Hz
    eq += np.interp(f, [20, 45, 90, 200], [0.2, 1.0, 0.0, 0.0]).astype(np.float32)

    # Low-mid control: -1.2 dB around 280 Hz
    eq += -1.2 * np.exp(-0.5 * ((np.log2(np.maximum(f, 1.0)/280.0))/0.45)**2).astype(np.float32)

    # Presence: +0.8 dB around 3.2 kHz
    eq += 0.8 * np.exp(-0.5 * ((np.log2(np.maximum(f, 1.0)/3200.0))/0.55)**2).astype(np.float32)

    # Air: +0.9 dB around 12 kHz (guarded later)
    eq += 0.9 * np.exp(-0.5 * ((np.log2(np.maximum(f, 1.0)/12000.0))/0.70)**2).astype(np.float32)

    return eq

def match_eq_curve(reference: Optional[np.ndarray], target: np.ndarray, sr: int,
                   max_eq_db: float, eq_smooth_hz: float,
                   match_strength: float, hi_factor: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build an EQ delta curve in dB across rfft bins.
    If reference is None: curve-based target (translation curve).
    """
    target = ensure_stereo(target)
    mid_t, _ = mid_side_encode(target)
    n_fft = 8192
    hop = 2048
    mag_t = windowed_fft_mag(mid_t, n_fft=n_fft, hop=hop) + 1e-9

    freqs = np.fft.rfftfreq(n_fft, 1.0/sr).astype(np.float32)

    if reference is not None:
        reference = ensure_stereo(reference)
        mid_r, _ = mid_side_encode(reference)
        mag_r = windowed_fft_mag(mid_r, n_fft=n_fft, hop=hop) + 1e-9
        delta_db = 20.0 * np.log10(mag_r) - 20.0 * np.log10(mag_t)
    else:
        # Want to steer target toward a translation curve (relative to current)
        desired = build_target_curve(freqs)
        # Apply as delta directly
        delta_db = desired

    delta_db = delta_db.astype(np.float32)

    # Smooth in frequency: convert smoothing Hz -> bins
    hz_per_bin = freqs[1] - freqs[0]
    win_bins = max(1, int(eq_smooth_hz / max(hz_per_bin, 1e-9)))
    delta_db = smooth_curve(delta_db, win_bins)

    # Frequency-dependent strength: reduce high band influence
    strength = match_strength * np.ones_like(delta_db, dtype=np.float32)
    strength *= np.interp(freqs, [0, 2000, 6000, 20000], [1.0, 1.0, hi_factor, hi_factor]).astype(np.float32)

    # Guardrails: do NOT allow large negative HF cuts (muffle risk).
    # Caps: below 200 Hz allow full range; 200–6k allow full; above 8k restrict cuts to -1.2 dB.
    hf_cut_cap = np.interp(freqs, [0, 6000, 8000, 20000], [-max_eq_db, -max_eq_db, -1.2, -1.2]).astype(np.float32)
    delta_db = np.maximum(delta_db, hf_cut_cap)

    # Hard cap overall
    delta_db = np.clip(delta_db, -max_eq_db, max_eq_db).astype(np.float32)

    eq_db = (delta_db * strength).astype(np.float32)
    return freqs, eq_db


# ---------------------------
# Dynamic masking EQ + De-ess
# ---------------------------

def dynamic_masking_eq(y: np.ndarray, sr: int, max_dip_db: float = 1.5) -> np.ndarray:
    """
    If low-mids mask the presence band, apply a gentle dynamic dip around ~300 Hz.
    (Static estimate over whole track; robust and safe.)
    """
    y = ensure_stereo(y)
    mid, _ = mid_side_encode(y)

    # measure low-mid and presence energies
    b1, a1 = butter_bandpass(220, 360, sr, order=2)
    b2, a2 = butter_bandpass(2000, 6000, sr, order=2)
    lm = sps.lfilter(b1, a1, mid)
    pr = sps.lfilter(b2, a2, mid)
    ratio = rms(lm) / max(rms(pr), 1e-9)  # >1 means masking risk

    # map ratio to dip amount
    amt = smoothstep(ratio, lo=0.95, hi=1.55)  # 0..1
    dip_db = -max_dip_db * amt

    if dip_db >= -0.01:
        return y

    # peaking filter (RBJ) at 300 Hz
    f0 = 300.0
    Q = 1.0
    A = 10**(dip_db/40)
    w0 = 2*math.pi*f0/sr
    alpha = math.sin(w0)/(2*Q)
    cosw0 = math.cos(w0)

    b0 = 1 + alpha*A
    b1p = -2*cosw0
    b2 = 1 - alpha*A
    a0 = 1 + alpha/A
    a1p = -2*cosw0
    a2 = 1 - alpha/A

    b = np.array([b0, b1p, b2]) / a0
    a = np.array([1.0, a1p/a0, a2/a0])
    return apply_iir(y, b, a)

def de_ess(y: np.ndarray, sr: int, band: Tuple[float,float]=(6000, 10000),
           threshold_db: float = -18.0, ratio: float = 3.0, mix: float = 0.55) -> np.ndarray:
    """
    Simple broadband de-esser: bandpass -> envelope -> gain reduction -> mix back.
    """
    y = ensure_stereo(y).astype(np.float32)
    mid, side = mid_side_encode(y)
    b, a = butter_bandpass(band[0], band[1], sr, order=2)
    s_band = sps.lfilter(b, a, mid).astype(np.float32)

    # envelope (RMS over 5ms)
    win = max(32, int(sr * 0.005))
    env = np.sqrt(sps.convolve(s_band**2, np.ones(win)/win, mode="same") + 1e-12).astype(np.float32)
    env_db = 20*np.log10(np.maximum(env, 1e-9)).astype(np.float32)

    # gain reduction when above threshold
    over = np.maximum(0.0, env_db - threshold_db)
    gr_db = -over * (1.0 - 1.0/ratio)
    gr = (10**(gr_db/20.0)).astype(np.float32)

    # apply to band component only
    s_band_out = s_band * gr
    mid_out = mid - mix*(s_band - s_band_out)  # reduce sibilance
    return mid_side_decode(mid_out, side)


# ---------------------------
# Harmonic glow (safe)
# ---------------------------

def harmonic_glow(y: np.ndarray, sr: int, band=(900, 3800), drive_db: float=1.0, mix: float=0.55) -> np.ndarray:
    y = ensure_stereo(y).astype(np.float32)
    mid, side = mid_side_encode(y)
    b, a = butter_bandpass(band[0], band[1], sr, order=2)
    x = sps.lfilter(b, a, mid).astype(np.float32)
    drive = db_to_lin(drive_db)
    sat = np.tanh(x * drive).astype(np.float32)
    mid2 = mid + mix*(sat - x)
    return mid_side_decode(mid2, side)


# ---------------------------
# Stereo enhancements
# ---------------------------

def corrcoef_band(y: np.ndarray, sr: int, lo: float, hi: float) -> float:
    y = ensure_stereo(y)
    b, a = butter_bandpass(lo, hi, sr, order=2)
    L = sps.lfilter(b, a, y[:,0]).astype(np.float32)
    R = sps.lfilter(b, a, y[:,1]).astype(np.float32)
    if rms(L) < 1e-6 or rms(R) < 1e-6:
        return 0.0
    c = np.corrcoef(L, R)[0,1]
    if np.isnan(c):
        return 0.0
    return float(c)

def spatial_realism_enhancer(y: np.ndarray, sr: int,
                            width_mid: float = 1.06, width_hi: float = 1.28,
                            mid_split_hz: float = 500.0, hi_split_hz: float = 2500.0,
                            corr_guard: float = 0.15) -> np.ndarray:
    """
    Frequency-dependent width scaling with correlation guard.
    - Mild width on mids (>= mid_split_hz)
    - More width on highs (>= hi_split_hz)
    - If correlation is already low (wide/phasey), reduce widening.
    """
    y = ensure_stereo(y).astype(np.float32)
    mid, side = mid_side_encode(y)

    # correlation in mid-high band
    corr = corrcoef_band(y, sr, 800, 6000)
    guard = smoothstep(corr, lo=corr_guard, hi=0.85)  # 0..1
    w_mid = 1.0 + (width_mid - 1.0) * guard
    w_hi  = 1.0 + (width_hi  - 1.0) * guard

    # split side into bands
    b1, a1 = butter_highpass(mid_split_hz, sr, order=2)
    b2, a2 = butter_highpass(hi_split_hz, sr, order=2)
    side_mid = sps.lfilter(b1, a1, side).astype(np.float32)
    side_hi  = sps.lfilter(b2, a2, side).astype(np.float32)

    side_lo = side - side_mid
    side_mid_only = side_mid - side_hi

    side_out = side_lo + side_mid_only * w_mid + side_hi * w_hi
    return mid_side_decode(mid, side_out)

def depth_distance_cue(
    y: np.ndarray,
    sr: int,
    hi_hz: float = 6500.0,
    max_cut_db: float = 1.2,
    side_air_db: float = 0.6,
    env_win_ms: float = 40.0,
    env_smooth_ms: float = 180.0,
    depth_smooth_ms: float = 120.0,
    corr_guard: float = 0.20,
) -> np.ndarray:
    """
    Depth Distance Cue:
      - Reduces high-band MID energy on low-energy segments (push back)
      - Adds a subtle high-band SIDE lift on tails (diffuse depth)
      - Correlation guard prevents widening when already phasey
    """
    y = ensure_stereo(y).astype(np.float32)
    mid, side = mid_side_encode(y)
    if not np.isfinite(mid).all():
        mid = np.nan_to_num(mid, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.isfinite(side).all():
        side = np.nan_to_num(side, nan=0.0, posinf=0.0, neginf=0.0)

    # Energy envelope (RMS) for depth mapping
    win = max(64, int(sr * env_win_ms / 1000.0))
    kernel = np.ones(win, dtype=np.float32) / float(win)
    env = np.sqrt(np.convolve(mid * mid, kernel, mode="same") + 1e-12).astype(np.float32)
    if not np.isfinite(env).all():
        env = np.nan_to_num(env, nan=0.0, posinf=0.0, neginf=0.0)
    env = _onepole_smooth(env, sr, env_smooth_ms)

    # Normalize using percentiles for stability across songs
    p_lo, p_hi = np.percentile(env, [10.0, 85.0]).astype(np.float32)
    denom = max(float(p_hi - p_lo), 1e-9)
    env_n = np.clip((env - float(p_lo)) / denom, 0.0, 1.0).astype(np.float32)

    # Depth = 1 for quiet tails, 0 for loud sections
    depth = 1.0 - smoothstep_vec(env_n, lo=0.10, hi=0.90)
    depth = _onepole_smooth(depth, sr, depth_smooth_ms)
    if not np.isfinite(depth).all():
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

    # High-band isolation
    b, a = butter_highpass(hi_hz, sr, order=2)
    mid_hi = sps.lfilter(b, a, mid).astype(np.float32)
    side_hi = sps.lfilter(b, a, side).astype(np.float32)

    # MID high-band attenuation on tails (front-to-back cue)
    cut_db = -float(max_cut_db) * depth
    mid_gain = (10.0 ** (cut_db / 20.0)).astype(np.float32)
    mid_out = mid + (mid_gain - 1.0) * mid_hi

    # SIDE high-band gentle lift on tails (diffuse depth), correlation-guarded
    corr = corrcoef_band(y, sr, hi_hz, 12000.0)
    guard = smoothstep(corr, lo=corr_guard, hi=0.85)  # 0..1
    side_gain_db = float(side_air_db) * depth * guard
    side_gain = (10.0 ** (side_gain_db / 20.0)).astype(np.float32)
    side_out = side + (side_gain - 1.0) * side_hi

    return mid_side_decode(mid_out.astype(np.float32), side_out.astype(np.float32))

def microshift_widen_side(y: np.ndarray, sr: int,
                          shift_ms: float = 0.22,
                          hi_split_hz: float = 2000.0,
                          mix: float = 0.18,
                          corr_guard: float = 0.20,
                          depth_range_ms: float = 0.18,
                          transient_win_ms: float = 5.0,
                          transient_slow_ms: float = 60.0,
                          depth_smooth_ms: float = 70.0) -> np.ndarray:
    y = ensure_stereo(y).astype(np.float32)
    
    # Ensure input is finite
    if not np.isfinite(y).all():
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    
    # correlation guard in high band
    corr = corrcoef_band(y, sr, hi_split_hz, 12000)
    guard = smoothstep(corr, lo=corr_guard, hi=0.90)  # 0..1

    base_mix = mix * guard
    if base_mix <= 1e-4:
        return y

    mid, side = mid_side_encode(y)
    if not np.isfinite(mid).all():
        mid = np.nan_to_num(mid, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.isfinite(side).all():
        side = np.nan_to_num(side, nan=0.0, posinf=0.0, neginf=0.0)

    # isolate SIDE high band
    b, a = butter_highpass(hi_split_hz, sr, order=2)
    side_hi = sps.lfilter(b, a, side).astype(np.float32)
    side_lo = side - side_hi

    # transient index from MID (short vs slow RMS), used to avoid smearing attacks
    win_s = max(32, int(sr * transient_win_ms / 1000.0))
    win_l = max(win_s + 1, int(sr * transient_slow_ms / 1000.0))
    
    # Ensure mid values are finite before computing envelopes
    mid_safe = np.nan_to_num(mid, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Compute envelopes with additional safety checks
    mid_squared = mid_safe * mid_safe
    if not np.isfinite(mid_squared).all():
        mid_squared = np.nan_to_num(mid_squared, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Short envelope
    env_s_raw = sps.convolve(mid_squared, np.ones(win_s) / win_s, mode="same")
    env_s = np.sqrt(np.maximum(env_s_raw + 1e-12, 0.0)).astype(np.float32)
    
    # Long envelope  
    env_l_raw = sps.convolve(mid_squared, np.ones(win_l) / win_l, mode="same")
    env_l = np.sqrt(np.maximum(env_l_raw + 1e-12, 0.0)).astype(np.float32)

    # Compute transient detection with safety
    transient = np.clip((env_s - env_l) / (env_l + 1e-9), 0.0, 1.0).astype(np.float32)
    depth = 1.0 - smoothstep_vec(transient, lo=0.08, hi=0.45)
    depth = _onepole_smooth(depth, sr, depth_smooth_ms)
    if not np.isfinite(depth).all():
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

    # fractional delay via linear interpolation (stable + cheap)
    shift_samp = (shift_ms + depth * depth_range_ms) * (sr / 1000.0)
    n = len(side_hi)
    idx = np.arange(n, dtype=np.float32)
    src = idx - shift_samp
    src0 = np.floor(src).astype(np.int64)
    frac = (src - src0).astype(np.float32)
    src0 = np.clip(src0, 0, n-1)
    src1 = np.clip(src0 + 1, 0, n-1)
    delayed = (1.0 - frac) * side_hi[src0] + frac * side_hi[src1]

    # mix delayed into side_hi (depth-adaptive)
    eff_mix = base_mix * depth
    side_hi_out = side_hi + eff_mix * delayed
    # normalize to avoid accidental level jumps in side band
    norm = max(1.0, rms(side_hi_out) / max(rms(side_hi), 1e-9))
    side_hi_out = (side_hi_out / norm).astype(np.float32)

    side_out = side_lo + side_hi_out
    return mid_side_decode(mid, side_out)


# ---------------------------
# Mono-Sub v2 (note-aware + adaptive)
# ---------------------------


# ---------------------------
# Movement (ported from v7.3) - section-aware width modulation + optional HookLift
# ---------------------------

def movement_automation(y: np.ndarray, sr: int, amount: float = 0.10) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Subtle, mastering-safe movement:
      - Modulates SIDE slightly based on MID energy envelope.
    """
    y = ensure_stereo(y).astype(np.float32)
    mid, side = mid_side_encode(y)

    env = np.abs(mid).astype(np.float32)
    win = int(max(16, round(sr * 0.03)))
    kernel = np.ones(win, dtype=np.float32) / float(win)
    env_s = np.convolve(env, kernel, mode="same").astype(np.float32)

    denom = float(np.max(env_s) - np.min(env_s) + 1e-12)
    env_n = (env_s - float(np.min(env_s))) / denom  # 0..1

    amt = float(clamp(amount, 0.0, 0.35))
    mod = (1.0 + amt * (env_n - 0.5)).astype(np.float32)

    side_out = (side * mod).astype(np.float32)
    return mid_side_decode(mid.astype(np.float32), side_out), {"enabled": True, "amount": amt}

def build_section_lift_mask(
    y: np.ndarray,
    sr: int,
    win_s: float = 0.80,
    percentile: float = 75.0,
    attack_s: float = 0.25,
    release_s: float = 0.90,
) -> np.ndarray:
    """Return a 0..1 envelope indicating high-energy sections (hooks/choruses)."""
    m = to_mono(y).astype(np.float32)
    win = int(max(256, round(sr * float(win_s))))
    kernel = np.ones(win, dtype=np.float32) / float(win)

    rms_env = np.sqrt(np.convolve(m * m, kernel, mode="same") + 1e-12).astype(np.float32)
    thr = np.percentile(rms_env, float(clamp(percentile, 50.0, 95.0)))

    target = (rms_env >= thr).astype(np.float32)

    a = math.exp(-1.0 / (sr * max(0.02, float(attack_s)) + 1e-12))
    r = math.exp(-1.0 / (sr * max(0.05, float(release_s)) + 1e-12))

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
    y: np.ndarray,
    sr: int,
    mix: float = 0.22,
    width_gain: float = 0.18,
    width_hp_hz: float = 1600.0,
    air_hz: float = 8500.0,
    air_gain: float = 0.14,
    shimmer_drive: float = 1.55,
    shimmer_mix: float = 0.35,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    HookLift - "bright chorus lift" without harshness.
      1) High-band SIDE boost (adds width in hooks)
      2) Gentle air shelf on MID
      3) Soft shimmer saturation on the air band
    """
    mix = float(clamp(mix, 0.0, 0.65))
    if mix <= 1e-6:
        return ensure_stereo(y).astype(np.float32), {"enabled": False}

    x = ensure_stereo(y).astype(np.float32)
    mid, side = mid_side_encode(x)

    # SIDE high-band lift (zero-phase)
    hp = float(clamp(width_hp_hz, 600.0, 6000.0))
    b, a = sps.butter(2, hp / (0.5 * sr), btype="high")
    side_hi = sps.filtfilt(b, a, side).astype(np.float32)
    side_boosted = (side + float(width_gain) * side_hi).astype(np.float32)

    # Air shelf approx: high-pass the MID and add back
    air_hz = float(clamp(air_hz, 4000.0, 16000.0))
    b2, a2 = sps.butter(2, air_hz / (0.5 * sr), btype="high")
    air = sps.filtfilt(b2, a2, mid).astype(np.float32)
    mid_air = (mid + float(air_gain) * air).astype(np.float32)

    # Shimmer: soft saturation on the air band
    drv = float(clamp(shimmer_drive, 1.0, 3.0))
    air_sat = np.tanh(air * drv).astype(np.float32)
    mid_air2 = (mid_air + float(shimmer_mix) * air_sat).astype(np.float32)

    lifted = mid_side_decode(mid_air2, side_boosted).astype(np.float32)
    y_out = (1.0 - mix) * x + mix * lifted
    return y_out.astype(np.float32), {"enabled": True, "mix": mix, "air_hz": air_hz, "width_gain": float(width_gain), "air_gain": float(air_gain)}

def mono_sub_v2(y: np.ndarray, sr: int,
                f0_hz: Optional[float],
                base_mix: float = 0.55) -> Tuple[np.ndarray, float, float]:
    """
    Note-aware cutoff + adaptive mono mix.
    Returns (y_out, cutoff_hz, mono_mix).
    """
    y = ensure_stereo(y).astype(np.float32)
    mid, side = mid_side_encode(y)

    if f0_hz is None:
        f0_hz = 55.0

    cutoff = clamp(1.95 * float(f0_hz), 72.0, 105.0)

    # instability metric: side/mid energy under cutoff
    b, a = butter_bandpass(25, cutoff, sr, order=2)
    low_mid = sps.lfilter(b, a, mid).astype(np.float32)
    low_side = sps.lfilter(b, a, side).astype(np.float32)
    ratio = rms(low_side) / max(rms(low_mid), 1e-9)

    # adaptive mix: only increase mono strength if low stereo is unstable
    # Typical range: 0.45–0.72 (not 0.85)
    add = 0.55 * smoothstep(ratio, lo=0.10, hi=0.35)
    mono_mix = clamp(0.45 + add, 0.45, 0.72)

    # apply: high-pass SIDE below cutoff (monoizing the sub)
    b_hp, a_hp = butter_highpass(cutoff, sr, order=2)
    side_hp = sps.lfilter(b_hp, a_hp, side).astype(np.float32)

    # blend: keep some original side for vibe but protect sub
    side_out = side_hp * (1.0 - mono_mix) + side * mono_mix
    return mid_side_decode(mid, side_out), cutoff, mono_mix


# ---------------------------

# ---------------------------
# Stem separation (HT-Demucs) - run early, then stem-aware pre-pass, then recombine
# ---------------------------

def gain_match_rms(y: np.ndarray, ref: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Scale y to match ref RMS (mono fold-down)."""
    y_m = rms(to_mono(y), eps=eps)
    r_m = rms(to_mono(ref), eps=eps)
    if y_m <= eps or r_m <= eps:
        return y.astype(np.float32)
    g = float(r_m / y_m)
    return (ensure_stereo(y) * g).astype(np.float32)

def demucs_separate_stems(
    y: np.ndarray,
    sr: int,
    *,
    model_name: str = "htdemucs",
    device: str = "cpu",
    split: bool = True,
    overlap: float = 0.25,
    shifts: int = 1,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Separate stereo audio into stems using Demucs (e.g., HT-Demucs).
    Returns (stems, info). Stems are np.float32 arrays shaped [N,2] at the original sr.
    """
    if not _HAS_DEMUCS:
        raise RuntimeError(
            "Demucs is not available. Install requirements: torch + demucs "
            "(e.g., pip install torch demucs)."
        )

    x = ensure_stereo(y).astype(np.float32)
    src_sr = int(sr)
    model_sr = 48000

    if src_sr != model_sr:
        x_rs = resample_audio(x, src_sr, model_sr)
    else:
        x_rs = x

    wav = torch.from_numpy(x_rs.T).unsqueeze(0)  # [1,2,T]
    dev = torch.device(device)
    wav = wav.to(dev)

    model = pretrained.get_model(model_name)
    model.to(dev)
    model.eval()

    kwargs = dict(split=bool(split), overlap=float(overlap), shifts=int(shifts))
    with torch.no_grad():
        try:
            sources = apply_model(model, wav, device=dev, progress=False, **kwargs)
        except TypeError:
            sources = apply_model(model, wav, device=dev, **kwargs)

    src_names = list(getattr(model, "sources", [])) or ["drums", "bass", "other", "vocals"]
    stems: Dict[str, np.ndarray] = {}
    for i, name in enumerate(src_names):
        s = sources[0, i].detach().cpu().numpy().T.astype(np.float32)  # [T, C]
        if src_sr != model_sr:
            s = resample_audio(s, model_sr, src_sr).astype(np.float32)
        # length align
        if s.shape[0] > x.shape[0]:
            s = s[: x.shape[0], :]
        elif s.shape[0] < x.shape[0]:
            pad = np.zeros((x.shape[0] - s.shape[0], 2), dtype=np.float32)
            s = np.concatenate([s, pad], axis=0)
        stems[name] = ensure_stereo(s).astype(np.float32)

    info = {
        "enabled": True,
        "model_name": model_name,
        "device": device,
        "split": bool(split),
        "overlap": float(overlap),
        "shifts": int(shifts),
        "model_sr": model_sr,
        "sr": src_sr,
        "sources": src_names,
    }
    return stems, info

def stem_pre_master_pass(stem: np.ndarray, sr: int, stem_name: str, preset: "Preset") -> np.ndarray:
    """
    Lightweight, stem-aware pre-pass:
      - Vocals: de-ess a touch earlier (pre EQ) to prevent sibilance boosts.
      - Bass: mono-sub stabilization is left to the master bus (safer).
      - Drums/Other: keep neutral to avoid Demucs artifacts compounding.
    """
    y = ensure_stereo(stem).astype(np.float32)

    # remove DC/rumble on each stem (Demucs can leak sub content into vocals/other)
    b, a = butter_highpass(25.0, sr, order=2)
    y = apply_iir(y, b, a)

    name = stem_name.lower().strip()
    if "vocal" in name and preset.enable_deess:
        y = de_ess(
            y, sr,
            threshold_db=float(preset.deess_threshold_db - 1.5),
            ratio=float(preset.deess_ratio),
            mix=float(clamp(preset.deess_mix + 0.10, 0.0, 0.85)),
        )
        if preset.enable_glow:
            y = harmonic_glow(
                y, sr,
                drive_db=float(preset.glow_drive_db * 0.80),
                mix=float(clamp(preset.glow_mix * 0.75, 0.0, 0.70)),
            )

    return y.astype(np.float32)

# Master pipeline
# ---------------------------

@dataclass
class Preset:
    name: str
    target_lufs: float = -12.3
    ceiling_dbfs: float = -1.0
    sr: int = 48000

    fir_taps: int = 4097
    match_strength: float = 0.62
    hi_factor: float = 0.75
    max_eq_db: float = 6.0
    eq_smooth_hz: float = 100.0

    enable_masking_eq: bool = True
    enable_deess: bool = True
    deess_threshold_db: float = -18.0
    deess_ratio: float = 3.0
    deess_mix: float = 0.55

    enable_glow: bool = True
    glow_drive_db: float = 0.9
    glow_mix: float = 0.55

    enable_mono_sub_v2: bool = True
    mono_sub_base_mix: float = 0.55  # actual mix becomes adaptive

    enable_spatial: bool = True
    width_mid: float = 1.06
    width_hi: float = 1.28

    enable_depth_cue: bool = True
    depth_hf_hz: float = 6500.0
    depth_max_cut_db: float = 1.2
    depth_side_air_db: float = 0.6
    depth_env_win_ms: float = 40.0
    depth_env_smooth_ms: float = 180.0
    depth_smooth_ms: float = 120.0
    depth_corr_guard: float = 0.20

    enable_microshift: bool = True
    microshift_ms: float = 0.22
    microshift_mix: float = 0.18

    limiter_oversample: int = 4
    limiter_attack_ms: float = 1.0
    limiter_release_ms: float = 60.0

    # HT-Demucs stem separation (run early)
    enable_stem_separation: bool = True
    demucs_model: str = "htdemucs"
    demucs_device: str = "cpu"
    demucs_split: bool = True
    demucs_overlap: float = 0.25
    demucs_shifts: int = 1

    # Movement + HookLift (section-aware)
    enable_movement: bool = True
    movement_amount: float = 0.10
    enable_hooklift: bool = True
    hooklift_auto: bool = True
    hooklift_auto_percentile: float = 75.0
    hooklift_mix: float = 0.22

    # Loudness governor
    governor_iters: int = 3
    governor_gr_limit_db: float = -1.2  # if min gain is <= -1.2 dB, back off target_lufs by step
    governor_step_db: float = -0.6      # reduce loudness target by 0.6 dB per iteration

def get_presets() -> Dict[str, Preset]:
    return {
        "hi_fi_streaming": Preset(
            name="hi_fi_streaming",
            target_lufs=-12.8,
            match_strength=0.63,
            hi_factor=0.78,
            max_eq_db=5.6,
            eq_smooth_hz=110.0,
            width_mid=1.05,
            width_hi=1.26,
            depth_max_cut_db=1.0,
            depth_side_air_db=0.50,
            microshift_ms=0.20,
            microshift_mix=0.16,
        ),
        "competitive_trap": Preset(
            name="competitive_trap",
            target_lufs=-11.0,
            match_strength=0.63,
            hi_factor=0.75,
            max_eq_db=6.0,
            eq_smooth_hz=95.0,
            movement_amount=0.13,
            width_mid=1.07,
            width_hi=1.30,
            depth_max_cut_db=0.80,
            depth_side_air_db=0.45,
            microshift_ms=0.24,
            microshift_mix=0.23,
            governor_gr_limit_db=-1.0,
        ),
        "melodic_trap": Preset(
            # Optimized for melodic trap: warm, tonal stability, controlled air
            name="melodic_trap",
            target_lufs=-13.0,          # Slightly louder than hi-fi but not crushing
            match_strength=0.63,        # Strong tonal matching
            hi_factor=0.72,             # Protect high-end from over-matching
            max_eq_db=5.0,              # Moderate EQ range
            eq_smooth_hz=105.0,         # Smooth transitions
            width_mid=1.04,             # Subtle mid widening (preserve vocal focus)
            width_hi=1.23,              # Controlled high-band width (melodic clarity)
            depth_max_cut_db=1.30,      # Slightly deeper tails for melodic space
            depth_side_air_db=0.50,     # Subtle diffuse side air
            microshift_ms=0.18,         # Less stereo shimmer (keep vocals centered)
            microshift_mix=0.14,        # Subtle widening
            glow_drive_db=1.3,          # More harmonic warmth for melodic content
            glow_mix=0.60,              # Stronger mid polish
            hooklift_mix=0.24,          # Moderate hook lift
            hooklift_auto_percentile=70.0,  # Broader hook detection for melodic builds
            movement_amount=0.12,       # Dynamic movement
            governor_gr_limit_db=-1.1,  # Protect dynamics
        ),
        "club_clean": Preset(
            name="club_clean",
            target_lufs=-10.4,
            match_strength=0.56,
            hi_factor=0.70,
            max_eq_db=6.0,
            eq_smooth_hz=90.0,
            width_mid=1.06,
            width_hi=1.28,
            depth_max_cut_db=1.10,
            depth_side_air_db=0.55,
            microshift_ms=0.26,
            microshift_mix=0.18,
            governor_gr_limit_db=-1.6,
        ),
    }

_W64_RIFF_GUID = bytes.fromhex("726966662E91CF11A5D628DB04C10000")


def _format_hint(path: Path) -> str:
    try:
        with path.open("rb") as f:
            header = f.read(16)
    except Exception:
        return "unknown"
    if len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WAVE":
        return "wav"
    if len(header) >= 12 and header[:4] == b"RF64" and header[8:12] == b"WAVE":
        return "rf64"
    if len(header) >= 16 and header[:16] == _W64_RIFF_GUID:
        return "w64"
    return "unknown"


def _require_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install ffmpeg or convert the input to standard WAV/FLAC."
        )
    return ffmpeg


def _run_cmd(cmd: List[str]) -> None:
    LOG.debug("Running command: %s", " ".join(cmd))
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            "Command failed:\\n"
            f"{' '.join(cmd)}\\n\\n"
            f"STDERR:\\n{p.stderr.strip()}"
        )


def _decode_with_ffmpeg_to_wav(input_path: Path, out_wav_path: Path, channels: int = 2) -> None:
    ffmpeg = _require_ffmpeg()
    cmd = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        str(channels),
        "-acodec",
        "pcm_f32le",
        str(out_wav_path),
    ]
    _run_cmd(cmd)


def _load_audio_with_ffmpeg(path: Path, fmt_hint: str) -> Tuple[np.ndarray, int]:
    try:
        with tempfile.TemporaryDirectory() as td:
            tmp_wav = Path(td) / "decoded.wav"
            _decode_with_ffmpeg_to_wav(path, tmp_wav, channels=2)
            y, sr = sf.read(tmp_wav, always_2d=True, dtype="float32")
            return y, int(sr)
    except Exception as exc:
        raise RuntimeError(
            f"SoundFile failed to read {path} (format hint: {fmt_hint}) and ffmpeg decode failed. "
            "Install ffmpeg or convert the file to standard WAV/FLAC."
        ) from exc


def load_audio(path: str) -> Tuple[np.ndarray, int]:
    try:
        y, sr = sf.read(path, always_2d=True, dtype="float32")
        return y, int(sr)
    except Exception as exc:
        fmt_hint = _format_hint(Path(path))
        LOG.info("SoundFile failed for %s (format hint: %s). Trying ffmpeg.", path, fmt_hint)
        return _load_audio_with_ffmpeg(Path(path), fmt_hint)

def write_audio(path: str, y: np.ndarray, sr: int):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    sf.write(path, y, sr)

def _list_audio_files(folder: Path) -> List[Path]:
    exts = {".wav", ".flac", ".aif", ".aiff", ".ogg", ".mp3", ".m4a"}
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(files, key=lambda p: p.name.lower())

def _chunked(items: List[Path], size: int) -> List[List[Path]]:
    size = max(1, int(size))
    return [items[i:i + size] for i in range(0, len(items), size)]

def _batch_worker(target_path: str, out_path: str, report_path: Optional[str],
                  preset: Preset, reference_path: Optional[str]) -> Dict[str, Any]:
    return master(target_path, out_path, preset, reference_path=reference_path, report_path=report_path, prog=None)

def master(target_path: str, out_path: str, preset: Preset,
           reference_path: Optional[str] = None,
           report_path: Optional[str] = None,
           prog: Optional[ProgressJSON] = None) -> Dict[str, Any]:

    t0 = time.time()

    with stage("load target", prog):
        y_t, sr_t = load_audio(target_path)
        y_t = ensure_stereo(y_t)
        if sr_t != preset.sr:
            y_t = resample_audio(y_t, sr_t, preset.sr)
            sr_t = preset.sr

    y_r = None
    if reference_path:
        with stage("load reference", prog):
            y_r, sr_r = load_audio(reference_path)
            y_r = ensure_stereo(y_r)
            if sr_r != preset.sr:
                y_r = resample_audio(y_r, sr_r, preset.sr)

    # Safety HPF (DC + rumble)
    b, a = butter_highpass(20.0, sr_t, order=2)
    y = apply_iir(y_t, b, a)

    # ---------------------------------------------------------------------
    # HT-Demucs stem separation (EARLY) + stem-aware pre-pass + recombine
    # ---------------------------------------------------------------------
    stems_info: Dict[str, Any] = {"enabled": False}
    if preset.enable_stem_separation:
        if not _HAS_DEMUCS:
            stems_info = {"enabled": False, "reason": "demucs_not_installed"}
        else:
            with stage("demucs separation", prog):
                try:
                    pre_stem_ref = y.copy()
                    stems, stems_info = demucs_separate_stems(
                        y, sr_t,
                        model_name=str(preset.demucs_model),
                        device=str(preset.demucs_device),
                        split=bool(preset.demucs_split),
                        overlap=float(preset.demucs_overlap),
                        shifts=int(preset.demucs_shifts),
                    )

                    stems_pp: Dict[str, np.ndarray] = {}
                    for s_name, s_audio in stems.items():
                        stems_pp[s_name] = stem_pre_master_pass(s_audio, sr_t, s_name, preset)

                    y_stem = np.zeros_like(pre_stem_ref, dtype=np.float32)
                    for s_audio in stems_pp.values():
                        y_stem += ensure_stereo(s_audio).astype(np.float32)

                    y = gain_match_rms(y_stem, pre_stem_ref)

                except Exception as e:
                    stems_info = {"enabled": False, "error": str(e)}

    # Musical analysis: sub f0
    f0 = estimate_sub_fundamental_hz(y, sr_t)

    # Mono-Sub v2
    mono_cut = None
    mono_mix = None
    if preset.enable_mono_sub_v2:
        y, mono_cut, mono_mix = mono_sub_v2(y, sr_t, f0, base_mix=preset.mono_sub_base_mix)

    # Match EQ (reference or translation curve)
    freqs, eq_db = match_eq_curve(
        reference=y_r, target=y, sr=sr_t,
        max_eq_db=preset.max_eq_db,
        eq_smooth_hz=preset.eq_smooth_hz,
        match_strength=preset.match_strength,
        hi_factor=preset.hi_factor
    )
    fir = design_fir_from_eq(freqs, eq_db, sr_t, preset.fir_taps)
    y = apply_fir(y, fir)

    # Dynamic masking EQ
    if preset.enable_masking_eq:
        y = dynamic_masking_eq(y, sr_t, max_dip_db=1.5)

    # De-ess (protect harshness without killing air)
    if preset.enable_deess:
        y = de_ess(y, sr_t, threshold_db=preset.deess_threshold_db, ratio=preset.deess_ratio, mix=preset.deess_mix)

    # Harmonic glow (midrange polish)
    if preset.enable_glow:
        y = harmonic_glow(y, sr_t, drive_db=preset.glow_drive_db, mix=preset.glow_mix)

    # Stereo: spatial realism enhancer
    if preset.enable_spatial:
        y = spatial_realism_enhancer(y, sr_t, width_mid=preset.width_mid, width_hi=preset.width_hi)

    # Stereo: depth distance cue (front-to-back realism)
    if preset.enable_depth_cue:
        y = depth_distance_cue(
            y, sr_t,
            hi_hz=preset.depth_hf_hz,
            max_cut_db=preset.depth_max_cut_db,
            side_air_db=preset.depth_side_air_db,
            env_win_ms=preset.depth_env_win_ms,
            env_smooth_ms=preset.depth_env_smooth_ms,
            depth_smooth_ms=preset.depth_smooth_ms,
            corr_guard=preset.depth_corr_guard,
        )

    # Stereo: NEW depth-adaptive microshift CGMS
    if preset.enable_microshift:
        y = microshift_widen_side(y, sr_t, shift_ms=preset.microshift_ms, mix=preset.microshift_mix)

    # ---------------------------------------------------------------------
    # Movement + HookLift (section-aware)
    # ---------------------------------------------------------------------
    movement_info: Dict[str, Any] = {"enabled": False}
    hooklift_info: Dict[str, Any] = {"enabled": False}

    if preset.enable_movement:
        y, movement_info = movement_automation(y, sr_t, amount=float(preset.movement_amount))

    if preset.enable_hooklift:
        if bool(preset.hooklift_auto):
            mask = build_section_lift_mask(
                y, sr_t,
                percentile=float(preset.hooklift_auto_percentile),
            )
            lifted, hinfo = hooklift(y, sr_t, mix=float(preset.hooklift_mix))
            mask_col = mask[:, None]
            y = (1.0 - mask_col) * y + mask_col * lifted
            hooklift_info = {**hinfo, "auto": True, "auto_percentile": float(preset.hooklift_auto_percentile)}
        else:
            y, hooklift_info = hooklift(y, sr_t, mix=float(preset.hooklift_mix))

    # Loudness Governor + final true-peak limiter
    governor_target = preset.target_lufs
    final_gr_db = 0.0
    pre_lufs = integrated_loudness_lufs(y, sr_t)

    for it in range(preset.governor_iters):
        y_norm, cur_lufs, gain_db = apply_lufs_gain(y, sr_t, governor_target)
        y_lim, gr_db = true_peak_limiter(
            y_norm, sr_t,
            ceiling_dbfs=preset.ceiling_dbfs,
            oversample=preset.limiter_oversample,
            attack_ms=preset.limiter_attack_ms,
            release_ms=preset.limiter_release_ms
        )
        final_gr_db = gr_db

        # If too much reduction, back off loudness target and try again
        if gr_db <= preset.governor_gr_limit_db:
            governor_target += preset.governor_step_db  # more negative step -> lower loudness
            continue

        y = y_lim
        break
    else:
        # If loop never broke, keep last limited audio
        y = y_lim

    post_lufs = integrated_loudness_lufs(y, sr_t)
    tp = lin_to_db(true_peak_estimate(y, sr_t, oversample=preset.limiter_oversample) + 1e-12)

    write_audio(out_path, y, sr_t)

    result = {
        "preset": preset.name,
        "sr": sr_t,
        "target_lufs_requested": preset.target_lufs,
        "target_lufs_after_governor": governor_target,
        "lufs_pre": float(pre_lufs),
        "lufs_post": float(post_lufs),
        "true_peak_dbfs": float(tp),
        "limiter_min_gain_db": float(final_gr_db),
        "sub_f0_hz": float(f0) if f0 is not None else None,
        "mono_sub_cutoff_hz": float(mono_cut) if mono_cut is not None else None,
        "mono_sub_mix": float(mono_mix) if mono_mix is not None else None,

"movement": movement_info,
"hooklift": hooklift_info,
"stems": stems_info,
        "runtime_sec": float(time.time() - t0),
        "out_path": out_path,
    }

    if report_path:
        os.makedirs(os.path.dirname(os.path.abspath(report_path)) or ".", exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# AuralMind Maestro v7.3 StereoAI — Report\\n\\n")
            f.write("## Summary\\n")
            f.write(f"- Preset: **{result['preset']}**\\n")
            f.write(f"- Sample rate: **{result['sr']} Hz**\\n")
            f.write(f"- LUFS (pre): **{result['lufs_pre']:.2f}**\\n")
            f.write(f"- LUFS (post): **{result['lufs_post']:.2f}**\\n")
            f.write(f"- True peak (approx): **{result['true_peak_dbfs']:.2f} dBFS**\\n")
            f.write(f"- Limiter min gain (approx GR): **{result['limiter_min_gain_db']:.2f} dB**\\n\\n")

            f.write("## Low-end / music theory anchors\\n")
            f.write(f"- Estimated sub fundamental f0: **{result['sub_f0_hz']} Hz**\\n")
            f.write(f"- Mono-sub v2 cutoff: **{result['mono_sub_cutoff_hz']} Hz**\\n")
            f.write(f"- Mono-sub v2 adaptive mix: **{result['mono_sub_mix']}**\\n\\n")

            f.write("## Stereo enhancements\\n")
            f.write("- Spatial Realism Enhancer: frequency-dependent width + correlation guard\\n")
            f.write("- NEW Depth Distance Cue: energy-dependent HF tilt for front-to-back depth\\n")
            f.write("- NEW Depth-Adaptive CGMS MicroShift: micro-delay on SIDE high-band, transient-aware + correlation-guarded\\n\\n")
            f.write("## Movement / HookLift\\n")
            f.write(f"- Movement enabled: **{result['movement'].get('enabled', False)}** (amount={result['movement'].get('amount', None)})\\n")
            f.write(f"- HookLift enabled: **{result['hooklift'].get('enabled', False)}** (mix={result['hooklift'].get('mix', None)})\\n")
            if result['hooklift'].get('auto', False):
                f.write(f"  - Auto mask percentile: **{result['hooklift'].get('auto_percentile', None)}**\\n")
            f.write("\\n")

            f.write("## Stem separation (HT-Demucs)\\n")
            f.write(f"- Enabled: **{result['stems'].get('enabled', False)}**\\n")
            if result['stems'].get('enabled', False):
                f.write(f"- Model: **{result['stems'].get('model_name', None)}**\\n")
                f.write(f"- Sources: **{result['stems'].get('sources', None)}**\\n")
            else:
                if 'reason' in result['stems']:
                    f.write(f"- Reason: **{result['stems'].get('reason', None)}**\\n")
                if 'error' in result['stems']:
                    f.write(f"- Error: **{result['stems'].get('error', None)}**\\n")
            f.write("\\n")

            f.write("## Loudness Governor\\n")
            f.write(f"- Requested target LUFS: **{preset.target_lufs}**\\n")
            f.write(f"- Governor final target LUFS: **{result['target_lufs_after_governor']}**\\n")
            f.write("  If limiter GR exceeded the ceiling, the governor backed off the LUFS target.\\n\\n")

            f.write("## JSON dump\\n")
            f.write("```json\\n")
            f.write(json.dumps(result, indent=2))
            f.write("\\n```\\n")

    return result


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AuralMind Maestro v7.3 StereoAI — Expert-tier mastering script")
    p.add_argument("--target", required=False, help="Path to target audio (wav/flac/aiff/ogg).")
    p.add_argument("--reference", default=None, help="Optional reference audio for match EQ.")
    p.add_argument("--out", required=False, help="Output mastered wav path.")
    p.add_argument("--report", default=None, help="Optional report markdown output.")
    p.add_argument("--preset", default="hi_fi_streaming", choices=list(get_presets().keys()), help="Preset name.")
    p.add_argument("--batch_dir", default=None, help="Batch mode: directory of target files.")
    p.add_argument("--out_dir", default=None, help="Batch mode: output directory.")
    p.add_argument("--report_dir", default=None, help="Batch mode: report directory (optional).")
    p.add_argument("--out_suffix", default=None, help="Batch mode: suffix for output/report files.")
    p.add_argument("--batch_size", type=int, default=2, help="Batch mode: files per chunk.")
    p.add_argument("--batch_jobs", type=int, default=4, help="Batch mode: parallel jobs per chunk.")
    p.add_argument("--no-stems", action="store_true", help="Disable HT-Demucs stem separation (otherwise enabled by preset).")
    p.add_argument("--demucs-model", default="htdemucs", help="Demucs model name (default from preset, e.g., htdemucs).")
    p.add_argument("--demucs-device", default="cpu", help="Demucs device: cpu or cuda (if available).")
    p.add_argument("--demucs-overlap", type=float, default=0.23, help="Demucs overlap (0.0-0.99).")
    p.add_argument("--demucs-no-split", action="store_true", help="Disable split processing inside Demucs (faster, more RAM).")
    p.add_argument("--demucs-shifts", type=int, default=2, help="Demucs shifts (quality vs speed).")

    p.add_argument("--no-movement", action="store_true", help="Disable movement automation (otherwise enabled by preset).")
    p.add_argument("--movement-amount", type=float, default=None, help="Movement amount (0.0-0.35).")

    p.add_argument("--no-hooklift", action="store_true", help="Disable HookLift (otherwise enabled by preset).")
    p.add_argument("--hooklift-mix", type=float, default=None, help="HookLift wet mix (0.0-0.65).")
    p.add_argument("--hooklift-no-auto", action="store_true", help="Disable auto mask; apply HookLift across the full track.")
    p.add_argument("--hooklift-percentile", type=float, default=None, help="Auto mask percentile (50-95).")

    # Logging & debugging
    p.add_argument("--log_level", default="DEBUG", help="Logging level (DEBUG/INFO/WARNING/ERROR).")
    p.add_argument("--log_file", default=None, help="Optional log file path.")
    p.add_argument("--progress_json", default=None, help="Path for progress JSON heartbeat.")
    p.add_argument("--watchdog_s", type=int, default=0, help="Watchdog stack dump interval (seconds), 0=disabled.")
    return p

def main():
    args = build_arg_parser().parse_args()

    # Initialize logging & watchdog
    _setup_logging(args.log_level, args.log_file)
    if args.watchdog_s and args.watchdog_s > 0:
        _enable_watchdog(args.watchdog_s)

    # Initialize progress tracker
    prog = ProgressJSON(args.progress_json)

    presets = get_presets()
    preset = presets[args.preset]

    updates: Dict[str, Any] = {}

    # Stem separation overrides
    if args.no_stems:
        updates['enable_stem_separation'] = False
    if args.demucs_model is not None:
        updates['demucs_model'] = args.demucs_model
    if args.demucs_device is not None:
        updates['demucs_device'] = args.demucs_device
    if args.demucs_overlap is not None:
        updates['demucs_overlap'] = float(args.demucs_overlap)
    if args.demucs_no_split:
        updates['demucs_split'] = False
    if args.demucs_shifts is not None:
        updates['demucs_shifts'] = int(args.demucs_shifts)

    # Movement overrides
    if args.no_movement:
        updates['enable_movement'] = False
    if args.movement_amount is not None:
        updates['movement_amount'] = float(args.movement_amount)

    # HookLift overrides
    if args.no_hooklift:
        updates['enable_hooklift'] = False
    if args.hooklift_mix is not None:
        updates['hooklift_mix'] = float(args.hooklift_mix)
    if args.hooklift_no_auto:
        updates['hooklift_auto'] = False
    if args.hooklift_percentile is not None:
        updates['hooklift_auto_percentile'] = float(args.hooklift_percentile)

    if updates:
        preset = replace(preset, **updates)

    # Batch mode
    if args.batch_dir:
        if not _HAS_JOBLIB:
            raise RuntimeError("Joblib not installed. Run: pip install joblib")
        if not args.out_dir:
            raise ValueError("--out_dir is required when using --batch_dir.")

        batch_dir = Path(args.batch_dir)
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if args.report_dir:
            report_dir = Path(args.report_dir)
        else:
            default_report = out_dir.parent / "Master"
            report_dir = default_report if default_report.exists() else out_dir
        report_dir.mkdir(parents=True, exist_ok=True)

        out_suffix = args.out_suffix if args.out_suffix is not None else f"__{preset.name}"

        files = _list_audio_files(batch_dir)
        if not files:
            raise ValueError(f"No audio files found in: {batch_dir}")

        chunks = _chunked(files, args.batch_size)
        results: List[Dict[str, Any]] = []
        for chunk in chunks:
            jobs = []
            for p in chunk:
                stem = p.stem
                out_path = out_dir / f"{stem}{out_suffix}.wav"
                report_path = report_dir / f"{stem}{out_suffix}.md"
                jobs.append((str(p), str(out_path), str(report_path)))

            batch_res = Parallel(
                n_jobs=int(max(1, args.batch_jobs)),
                backend="threading"
            )(delayed(_batch_worker)(t, o, r, preset, args.reference) for (t, o, r) in jobs)
            results.extend(batch_res)

        print(json.dumps({"batch_count": len(results), "results": results}, indent=2))
        return

    # Single file mode
    if not args.target or not args.out:
        raise ValueError("Single-file mode requires --target and --out.")

    res = master(args.target, args.out, preset, reference_path=args.reference, report_path=args.report, prog=prog)
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
