#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AuralMind Match — Maestro v7.3 "expert" (Expert-tier)
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
   - NEW: Correlation-Guarded MicroShift (CGMS): micro-delay applied ONLY to SIDE high-band (>=2k)
          with a mono-compatibility guard to avoid phasey collapse.

Dependencies
------------
- numpy
- scipy
- soundfile

No librosa / numba required. (This script stays lightweight and portable.)

Usage
-----
Basic (curve-based master):
    python auralmind_match_maestro_nextgen_v7_3_expert.py --target "song.wav" --out "song_master.wav"

Reference match:
    python auralmind_match_maestro_nextgen_v7_3_expert.py --reference "ref.mp3" --target "song.wav" --out "song_master.wav"

Choose a preset:
    python auralmind_match_maestro_nextgen_v7_3_expert.py --preset hi_fi_streaming --target "song.wav" --out "song_master.wav"

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
from dataclasses import dataclass, replace
from typing import Optional, Tuple, Dict, Any, Union

import numpy as np
import soundfile as sf
import scipy
import scipy.signal as sps
from scipy.signal import fftconvolve
from scipy.ndimage import maximum_filter1d

# Optional Demucs (HT-Demucs stem separation) — enabled by default, with graceful fallback
try:
    import torch  # type: ignore
    from demucs import pretrained  # type: ignore
    from demucs.apply import apply_model  # type: ignore
    _HAS_DEMUCS = True
except Exception:
    _HAS_DEMUCS = False


sci = scipy.ndimage

# ---------------------------
# Utility helpers
# ---------------------------

def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))

def db_to_lin(db: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    if np.isscalar(db):
        return float(10.0 ** (float(db) / 20.0))
    arr = np.asarray(db, dtype=np.float32)
    return (10.0 ** (arr / 20.0)).astype(np.float32)

def lin_to_db(x: Union[float, np.ndarray], eps: float = 1e-12) -> Union[float, np.ndarray]:
    if np.isscalar(x):
        return float(20.0 * math.log10(max(abs(float(x)), eps)))
    arr = np.asarray(x, dtype=np.float32)
    eps_val = np.float32(eps)
    return (np.float32(20.0) * np.log10(np.maximum(np.abs(arr), eps_val))).astype(np.float32)

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

def butter_highpass(cut_hz: float, sr: int, order: int = 2):
    nyq = 0.5 * sr
    cut = max(1.0, cut_hz) / nyq
    return sps.butter(order, cut, btype="highpass")

def butter_bandpass(lo_hz: float, hi_hz: float, sr: int, order: int = 2):
    nyq = 0.5 * sr
    lo = max(1.0, lo_hz) / nyq
    hi = min(nyq * 0.999, hi_hz) / nyq
    if hi <= lo:
        hi = min(0.999, lo + 0.05)
    return sps.butter(order, [lo, hi], btype="bandpass")

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
    win = np.hanning(n_fft).astype(np.float32)
    mags = []
    for start in range(0, max(1, len(x) - n_fft), hop):
        frame = x[start:start+n_fft]
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        spec = np.fft.rfft(frame * win)
        mags.append(np.abs(spec))
    if not mags:
        return np.zeros(n_fft//2 + 1, dtype=np.float32)
    return np.mean(np.stack(mags, axis=0), axis=0).astype(np.float32)


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
    mono = np.mean(yk, axis=1)

    block = int(0.400 * sr)
    hop = int(0.100 * sr)
    if block <= 0:
        return -100.0

    energies = []
    for i in range(0, max(1, len(mono) - block), hop):
        seg = mono[i:i+block]
        if len(seg) < block:
            seg = np.pad(seg, (0, block - len(seg)))
        e = np.mean(seg.astype(np.float64)**2)
        energies.append(e)

    energies = np.array(energies, dtype=np.float64)
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




def analyze_track_features(y: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Lightweight analysis for auto-tuning and reporting.
    All metrics are approximate but stable.
    """
    y = ensure_stereo(y).astype(np.float32)
    lufs = float(integrated_loudness_lufs(y, sr))
    tp_db = float(lin_to_db(true_peak_estimate(y, sr, oversample=4) + 1e-12))

    peak_db = float(lin_to_db(np.max(np.abs(y)) + 1e-12))
    rms = np.sqrt(np.mean(y.astype(np.float64) ** 2))
    rms_db = float(lin_to_db(float(rms) + 1e-12))
    crest_db = float(peak_db - rms_db)

    corr_hi = float(corrcoef_band(y, sr, 2000.0, 12000.0))
    corr_lo = float(corrcoef_band(y, sr, 20.0, 200.0))

    # Simple spectral centroid (Hz) for tone brightness proxy
    mono = np.mean(y, axis=1).astype(np.float32)
    n = min(len(mono), sr * 8)  # cap work
    if n < 2048:
        centroid = 0.0
    else:
        x = mono[:n] * np.hanning(n).astype(np.float32)
        mag = np.abs(np.fft.rfft(x))
        freqs = np.fft.rfftfreq(n, d=1.0 / sr)
        centroid = float((freqs @ mag) / (np.sum(mag) + 1e-12))

    return {
        "lufs": lufs,
        "tp_dbfs": tp_db,
        "peak_dbfs": peak_db,
        "rms_dbfs": rms_db,
        "crest_db": crest_db,
        "corr_hi": corr_hi,
        "corr_lo": corr_lo,
        "centroid_hz": float(centroid),
    }


def auto_select_preset_name(features: Dict[str, float]) -> str:
    """
    Pick one of the built-in presets based on dynamics/brightness.
    """
    crest = float(features.get("crest_db", 10.0))
    lufs = float(features.get("lufs", -16.0))
    centroid = float(features.get("centroid_hz", 2500.0))

    # High crest -> preserve dynamics (hi-fi)
    if crest >= 12.0:
        return "hi_fi_streaming"

    # Already loud and dense -> club-clean style
    if lufs >= -12.0 or crest <= 8.5:
        return "club_clean"

    # Otherwise: competitive trap (balanced loudness + punch)
    if centroid >= 2800.0:
        return "competitive_trap"

    return "competitive_trap"


def auto_tune_preset(
    base_preset: "Preset",
    target_features: Dict[str, float],
    ref_features: Optional[Dict[str, float]] = None,
) -> Tuple["Preset", Dict[str, Any]]:
    """
    Expert tier auto-tune:
    - optionally nudges target LUFS toward reference (clamped)
    - adapts governor GR limit based on crest (preserve dynamics)
    """
    tuned = base_preset
    info: Dict[str, Any] = {"enabled": True, "base": base_preset.name}

    crest = float(target_features.get("crest_db", 10.0))
    target_lufs = float(base_preset.target_lufs)

    if ref_features is not None:
        ref_lufs = float(ref_features.get("lufs", target_lufs))
        # match "feel" but keep safe for streaming (do not chase extreme loudness)
        target_lufs = float(clamp(ref_lufs + 0.4, -14.5, -10.2))
        info["ref_lufs"] = ref_lufs

    # Dynamic GR ceiling: high crest => allow *less* limiting
    if crest >= 12.0:
        gr_limit = -0.9
    elif crest <= 8.5:
        gr_limit = -2.2
    else:
        gr_limit = float(base_preset.governor_gr_limit_db)

    tuned = tuned.__class__(**{**tuned.__dict__, "target_lufs": target_lufs, "governor_gr_limit_db": gr_limit})
    info["target_lufs"] = target_lufs
    info["gr_limit_db"] = gr_limit
    info["crest_db"] = crest
    return tuned, info


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


def softclip_oversampled(
    y: np.ndarray,
    sr: int,
    pre_db_below_ceiling: float = 0.6,
    ceiling_dbfs: float = -1.0,
    drive_db: float = 1.2,
    mix: float = 0.25,
    oversample: int = 4,
) -> np.ndarray:
    """
    Mastering-safe pre-limiter soft clip (oversampled).

    Purpose:
    - shave transient spikes *before* the limiter to preserve openness/punch
    - reduce limiter GR requirements at the same integrated loudness

    Controls:
    - pre_db_below_ceiling: start soft-clipping slightly below ceiling
    - drive_db: increases saturation intensity
    - mix: wet/dry blend
    """
    y = ensure_stereo(y).astype(np.float32)
    mix = float(clamp(mix, 0.0, 1.0))
    if mix <= 0.0 or oversample < 2:
        return y

    ceiling = db_to_lin(ceiling_dbfs)
    thresh = ceiling * db_to_lin(-abs(pre_db_below_ceiling))

    # Oversample -> softclip -> downsample
    up = sps.resample_poly(y, oversample, 1, axis=0).astype(np.float32)

    # Drive and clip curve (tanh soft clip), scaled to keep unity-ish under threshold
    drive = db_to_lin(drive_db)
    x = up * drive

    # Normalize threshold into driven domain
    t = max(thresh * drive, 1e-6)

    # Soft clip: linear below t, tanh above (smooth knee)
    mag = np.abs(x)
    sgn = np.sign(x)
    above = mag > t
    y_sc = x.copy()
    # compress above threshold; knee width proportional to threshold
    k = 0.35 * t + 1e-9
    y_sc[above] = (sgn[above] * (t + k * np.tanh((mag[above] - t) / k))).astype(np.float32)

    # Back out drive
    y_sc = (y_sc / max(drive, 1e-9)).astype(np.float32)

    down = sps.resample_poly(y_sc, 1, oversample, axis=0).astype(np.float32)

    # Align length (polyphase may be off by 1)
    down = down[:len(y)]
    return (y * (1.0 - mix) + down * mix).astype(np.float32)


def true_peak_limiter_v2(
    y: np.ndarray,
    sr: int,
    ceiling_dbfs: float = -1.0,
    oversample: int = 4,
    lookahead_ms: float = 3.0,
    attack_ms: float = 0.6,
    release_ms: float = 80.0,
    stereo_link: float = 0.92,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    TP limiter v2 (expert):
    - lookahead peak envelope via max-filter window
    - stereo linking (mid-heavy) to reduce image pumping
    - one-pass ISP correction via oversampled peak estimate
    - returns stats dict with min gain + avg GR and final TP

    NOTE: lookahead uses a centered max-filter; effective lookahead ~ lookahead_ms/2.
    """
    y = ensure_stereo(y).astype(np.float32)
    ceiling = db_to_lin(ceiling_dbfs)

    inst_l = np.abs(y[:, 0]).astype(np.float32)
    inst_r = np.abs(y[:, 1]).astype(np.float32)

    # Link: blend max(L,R) with mid proxy to reduce stereo pumping
    inst_max = np.maximum(inst_l, inst_r)
    mid = np.abs(0.5 * (y[:, 0] + y[:, 1])).astype(np.float32)
    inst = (stereo_link * inst_max + (1.0 - stereo_link) * mid).astype(np.float32)

    win = max(16, int(sr * (lookahead_ms / 1000.0)))
    env = maximum_filter1d(inst, size=win, mode="nearest").astype(np.float32)

    raw_g = np.minimum(1.0, ceiling / np.maximum(env, 1e-9)).astype(np.float32)
    g = limiter_smooth_gain(raw_g, sr, attack_ms, release_ms).astype(np.float32)

    y2 = (y * g[:, None]).astype(np.float32)

    # ISP correction (single correction pass)
    tp_lin = true_peak_estimate(y2, sr, oversample=oversample)
    if tp_lin > ceiling:
        corr = ceiling / max(tp_lin, 1e-9)
        y2 = (y2 * corr).astype(np.float32)
        g = (g * corr).astype(np.float32)
        tp_lin = true_peak_estimate(y2, sr, oversample=oversample)

    min_gain = float(np.min(g))
    stats = {
        "min_gain_db": float(lin_to_db(min_gain)),
        "avg_gr_db": float(lin_to_db(float(np.mean(g)) + 1e-12)),
        "tp_dbfs": float(lin_to_db(tp_lin + 1e-12)),
        "ceiling_dbfs": float(ceiling_dbfs),
    }
    return y2, stats


def peak_control_chain(
    y: np.ndarray,
    sr: int,
    preset: "Preset",
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Final peak control chain:
    1) optional oversampled soft clip (pre-limiter)
    2) TP limiter (v1 or v2)
    """
    y = ensure_stereo(y).astype(np.float32)

    if getattr(preset, "enable_softclip", True):
        y = softclip_oversampled(
            y, sr,
            pre_db_below_ceiling=float(getattr(preset, "softclip_pre_db_below_ceiling", 0.6)),
            ceiling_dbfs=float(preset.ceiling_dbfs),
            drive_db=float(getattr(preset, "softclip_drive_db", 1.2)),
            mix=float(getattr(preset, "softclip_mix", 0.25)),
            oversample=int(preset.limiter_oversample),
        )

    mode = str(getattr(preset, "limiter_mode", "v2")).lower()
    if mode == "v1":
        y2, gr_db = true_peak_limiter(
            y, sr,
            ceiling_dbfs=float(preset.ceiling_dbfs),
            oversample=int(preset.limiter_oversample),
            attack_ms=float(preset.limiter_attack_ms),
            release_ms=float(preset.limiter_release_ms),
        )
        stats = {
            "min_gain_db": float(gr_db),
            "tp_dbfs": float(lin_to_db(true_peak_estimate(y2, sr, oversample=int(preset.limiter_oversample)) + 1e-12)),
            "ceiling_dbfs": float(preset.ceiling_dbfs),
            "mode": 1.0,
        }
        return y2, stats

    y2, st = true_peak_limiter_v2(
        y, sr,
        ceiling_dbfs=float(preset.ceiling_dbfs),
        oversample=int(preset.limiter_oversample),
        lookahead_ms=float(getattr(preset, "limiter_lookahead_ms", 3.0)),
        attack_ms=float(preset.limiter_attack_ms),
        release_ms=float(preset.limiter_release_ms),
        stereo_link=float(getattr(preset, "limiter_stereo_link", 0.92)),
    )
    st["mode"] = 2.0
    return y2, st



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

def next_pow2(n: int) -> int:
    """Small helper: next power-of-two >= n."""
    if n <= 1:
        return 1
    return 1 << (int(n - 1).bit_length())


_FIR_CACHE: Dict[Tuple[int, int, float], np.ndarray] = {}


def apply_fir(y: np.ndarray, fir: np.ndarray) -> np.ndarray:
    """
    Apply an FIR to stereo audio.

    Expert upgrade:
    - Auto-selects between full fftconvolve (small/short) and a streaming overlap-save engine (long files).
    - Caches the FIR frequency response for the streaming path to reduce repeated FFT work.
    """
    y = ensure_stereo(y).astype(np.float32)
    fir = np.asarray(fir, dtype=np.float32)

    # Heuristic: streaming wins for long material or large tap counts.
    n = int(len(y))
    taps = int(len(fir))
    if n < 1:
        return y

    # "same" fftconvolve is fine for short clips.
    if (n < 48000 * 20) and (taps <= 4097):
        out = np.zeros_like(y, dtype=np.float32)
        for ch in range(2):
            out[:, ch] = fftconvolve(y[:, ch], fir, mode="same").astype(np.float32)
        return out

    # Streaming overlap-save convolution (low peak RAM)
    return apply_fir_streaming_overlap_save(y, fir)


def apply_fir_streaming_overlap_save(
    y: np.ndarray,
    fir: np.ndarray,
    block_pow2: int = 17,
) -> np.ndarray:
    """
    Overlap-save FFT convolution (streaming).

    block_pow2=17 -> 131072-sample base block.
    Effective FFT size will be next_pow2(block + taps - 1).
    """
    y = ensure_stereo(y).astype(np.float32)
    fir = np.asarray(fir, dtype=np.float32)

    n = int(len(y))
    taps = int(len(fir))
    if n < 1 or taps < 2:
        return y

    block = int(2 ** int(block_pow2))
    overlap = taps - 1
    nfft = next_pow2(block + overlap)

    # Cache key: (taps, nfft, fir_rms) keeps cache stable without hashing full array.
    key = (taps, nfft, float(np.sqrt(np.mean(fir.astype(np.float64) ** 2)) + 1e-12))
    H = _FIR_CACHE.get(key)
    if H is None:
        H = np.fft.rfft(fir, n=nfft).astype(np.complex64)
        _FIR_CACHE[key] = H

    out = np.zeros((n, 2), dtype=np.float32)

    # Prime overlap buffer with zeros
    prev = np.zeros((overlap, 2), dtype=np.float32)

    i = 0
    while i < n:
        chunk = y[i:i + block]
        # pad last chunk
        if len(chunk) < block:
            pad = np.zeros((block - len(chunk), 2), dtype=np.float32)
            chunk = np.vstack([chunk, pad])

        x = np.vstack([prev, chunk])  # shape: overlap+block, 2
        X = np.fft.rfft(x, n=nfft, axis=0).astype(np.complex64)
        Y = (X * H[:, None]).astype(np.complex64)
        y_time = np.fft.irfft(Y, n=nfft, axis=0).astype(np.float32)

        # discard the overlap (overlap-save) -> keep 'block' samples
        y_valid = y_time[overlap:overlap + block]

        # write to output (trim for final partial)
        write_len = min(block, n - i)
        out[i:i + write_len] = y_valid[:write_len]

        # update overlap buffer with last overlap samples of current input (NOT output)
        prev = x[-overlap:].copy()
        i += block

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

def microshift_widen_side(y: np.ndarray, sr: int,
                          shift_ms: float = 0.22,
                          hi_split_hz: float = 2000.0,
                          mix: float = 0.18,
                          corr_guard: float = 0.20) -> np.ndarray:
    """
    NEW Stereo Enhancement: Correlation-Guarded MicroShift (CGMS)
    - Applies a tiny delay (microshift) ONLY to the SIDE high band (>= hi_split_hz).
    - This increases perceived width/air without wrecking mono compatibility.
    - Guard: if the band is already wide/phasey (low correlation), reduce or disable.

    Why it helps your "preLoudnorm sounds better" issue:
    - It restores spaciousness *without* needing HF boosts that can turn harsh.
    - It reduces the subjective "blanket" effect created when limiting collapses micro-detail.
    """
    y = ensure_stereo(y).astype(np.float32)
    # correlation guard in high band
    corr = corrcoef_band(y, sr, hi_split_hz, 12000)
    guard = smoothstep(corr, lo=corr_guard, hi=0.90)  # 0..1

    eff_mix = mix * guard
    if eff_mix <= 1e-4:
        return y

    mid, side = mid_side_encode(y)

    # isolate SIDE high band
    b, a = butter_highpass(hi_split_hz, sr, order=2)
    side_hi = sps.lfilter(b, a, side).astype(np.float32)
    side_lo = side - side_hi

    # fractional delay via linear interpolation (stable + cheap)
    shift_samp = (shift_ms / 1000.0) * sr
    n = len(side_hi)
    idx = np.arange(n, dtype=np.float32)
    src = idx - shift_samp
    src0 = np.floor(src).astype(np.int64)
    frac = (src - src0).astype(np.float32)
    src0 = np.clip(src0, 0, n-1)
    src1 = np.clip(src0 + 1, 0, n-1)
    delayed = (1.0 - frac) * side_hi[src0] + frac * side_hi[src1]

    # mix delayed into side_hi
    side_hi_out = side_hi + eff_mix * delayed
    # normalize to avoid accidental level jumps in side band
    norm = max(1.0, rms(side_hi_out) / max(rms(side_hi), 1e-9))
    side_hi_out = (side_hi_out / norm).astype(np.float32)

    side_out = side_lo + side_hi_out
    return mid_side_decode(mid, side_out)


# ---------------------------
# Mono-Sub v2 (note-aware + adaptive)
# ---------------------------



def microdetail_recovery_side_high(
    y: np.ndarray,
    sr: int,
    band_lo_hz: float = 2500.0,
    band_hi_hz: float = 12000.0,
    threshold_db: float = -34.0,
    max_boost_db: float = 3.5,
    amount: float = 0.22,
    mix: float = 0.65,
    attack_ms: float = 12.0,
    release_ms: float = 160.0,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Micro-detail recovery (expert):
    - upward micro-compression on SIDE high-band (2.5k–12k by default)
    - correlation guard: reduces effect if the band is already very wide/phasey
    - designed to recover perceived air/detail *without* brute-force EQ boosts

    Returns (audio, info).
    """
    y = ensure_stereo(y).astype(np.float32)
    amount = float(clamp(amount, 0.0, 0.8))
    mix = float(clamp(mix, 0.0, 1.0))
    if amount <= 0.0 or mix <= 0.0:
        return y, {"enabled": 0.0}

    # Correlation guard: if already very wide (low corr), reduce effect.
    corr = float(corrcoef_band(y, sr, band_lo_hz, band_hi_hz))
    guard = float(smoothstep(corr, 0.10, 0.35))  # 0 -> very wide, 1 -> fairly mono/solid
    eff_amt = amount * guard
    if eff_amt <= 1e-6:
        return y, {"enabled": 0.0, "corr": corr, "guard": guard}

    mid, side = mid_side_encode(y)
    b, a = butter_bandpass(band_lo_hz, band_hi_hz, sr, order=2)
    side_band = sps.lfilter(b, a, side).astype(np.float32)

    env = np.abs(side_band).astype(np.float32)
    # Smooth envelope quickly (3ms) to avoid chatter
    env = maximum_filter1d(env, size=max(8, int(sr * 0.003)), mode="nearest").astype(np.float32)
    lvl_db = lin_to_db(env + 1e-9).astype(np.float32)

    # Upward comp target: lift quiet details toward threshold.
    boost_db = np.clip((threshold_db - lvl_db) * eff_amt, 0.0, max_boost_db).astype(np.float32)
    target_gain = db_to_lin(boost_db).astype(np.float32)

    # Dual-time smoothing (fast-ish up, slower down) to avoid pumping.
    atk = max(1, int(sr * attack_ms / 1000.0))
    rel = max(1, int(sr * release_ms / 1000.0))
    g = target_gain[0]
    g_s = np.empty_like(target_gain)
    for i, x in enumerate(target_gain):
        if x > g:  # increasing gain -> attack
            g = g + (x - g) / atk
        else:      # decreasing gain -> release
            g = g + (x - g) / rel
        g_s[i] = g

    side_band2 = (side_band * g_s).astype(np.float32)
    side2 = (side + (side_band2 - side_band) * mix).astype(np.float32)

    y2 = mid_side_decode(mid, side2).astype(np.float32)
    info = {
        "enabled": 1.0,
        "corr": corr,
        "guard": guard,
        "eff_amount": float(eff_amt),
        "threshold_db": float(threshold_db),
        "max_boost_db": float(max_boost_db),
        "mix": float(mix),
    }
    return y2, info


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
    model_sr = 44100

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

    enable_microshift: bool = True
    microshift_ms: float = 0.22
    microshift_mix: float = 0.18

    limiter_oversample: int = 4
    limiter_attack_ms: float = 1.0
    limiter_release_ms: float = 60.0
    limiter_mode: str = "v2"
    limiter_lookahead_ms: float = 3.0
    limiter_stereo_link: float = 0.92

    enable_softclip: bool = True
    softclip_pre_db_below_ceiling: float = 0.6
    softclip_drive_db: float = 1.2
    softclip_mix: float = 0.25

    # FIR streaming (match-EQ)
    fir_streaming: str = "auto"   # auto | on | off
    fir_block_pow2: int = 17      # 2^17 = 131072 samples per block

    # Micro-detail recovery (SIDE high-band upward micro-comp)
    enable_microdetail: bool = True
    microdetail_amount: float = 0.22
    microdetail_threshold_db: float = -34.0
    microdetail_max_boost_db: float = 3.5
    microdetail_band_lo_hz: float = 2500.0
    microdetail_band_hi_hz: float = 12000.0
    microdetail_mix: float = 0.65

    # Governor v2 (binary search)
    governor_search_steps: int = 11
    governor_allow_above_db: float = 0.0

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
            match_strength=0.68,
            hi_factor=0.78,
            max_eq_db=5.6,
            eq_smooth_hz=110.0,
            width_mid=1.05,
            width_hi=1.26,
            microshift_ms=0.20,
            microshift_mix=0.16,
        ),
        "competitive_trap": Preset(
            name="competitive_trap",
            target_lufs=-11.4,
            match_strength=0.62,
            hi_factor=0.75,
            max_eq_db=6.2,
            eq_smooth_hz=95.0,
            width_mid=1.07,
            width_hi=1.30,
            microshift_ms=0.24,
            microshift_mix=0.20,
            governor_gr_limit_db=-1.4,
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
            microshift_ms=0.26,
            microshift_mix=0.18,
            governor_gr_limit_db=-1.6,
        ),
    }

def load_audio(path: str) -> Tuple[np.ndarray, int]:
    y, sr = sf.read(path, always_2d=True)
    y = y.astype(np.float32)
    return y, int(sr)

def write_audio(path: str, y: np.ndarray, sr: int):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    sf.write(path, y, sr)

def master(target_path: str, out_path: str, preset: Preset,
           reference_path: Optional[str] = None,
           report_path: Optional[str] = None) -> Dict[str, Any]:

    t0 = time.time()

    y_t, sr_t = load_audio(target_path)
    y_t = ensure_stereo(y_t)
    if sr_t != preset.sr:
        y_t = resample_audio(y_t, sr_t, preset.sr)
        sr_t = preset.sr

    y_r = None
    if reference_path:
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
    fir_mode = str(getattr(preset, "fir_streaming", "auto")).lower()
    if fir_mode == "off":
        out = np.zeros_like(y, dtype=np.float32)
        for ch in range(2):
            out[:, ch] = fftconvolve(y[:, ch], fir, mode="same").astype(np.float32)
        y = out
    elif fir_mode == "on":
        y = apply_fir_streaming_overlap_save(y, fir, block_pow2=int(getattr(preset, "fir_block_pow2", 17)))
    else:
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

    # Stereo: NEW microshift CGMS
    if preset.enable_microshift:
        y = microshift_widen_side(y, sr_t, shift_ms=preset.microshift_ms, mix=preset.microshift_mix)

    microdetail_info: Dict[str, Any] = {"enabled": False}
    if getattr(preset, "enable_microdetail", False):
        y, md = microdetail_recovery_side_high(
            y, sr_t,
            band_lo_hz=float(getattr(preset, "microdetail_band_lo_hz", 2500.0)),
            band_hi_hz=float(getattr(preset, "microdetail_band_hi_hz", 12000.0)),
            threshold_db=float(getattr(preset, "microdetail_threshold_db", -34.0)),
            max_boost_db=float(getattr(preset, "microdetail_max_boost_db", 3.5)),
            amount=float(getattr(preset, "microdetail_amount", 0.22)),
            mix=float(getattr(preset, "microdetail_mix", 0.65)),
        )
        microdetail_info = md

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

    # Loudness Governor v2 (binary search) + final peak control chain (softclip + TP limiter)
    pre_lufs = integrated_loudness_lufs(y, sr_t)

    def _render_at(target_lufs: float) -> Tuple[np.ndarray, Dict[str, float]]:
        y_norm, cur_lufs, gain_db = apply_lufs_gain(y, sr_t, target_lufs)
        y_lim, lim_stats = peak_control_chain(y_norm, sr_t, preset)
        post = integrated_loudness_lufs(y_lim, sr_t)
        lim_stats = {
            **lim_stats,
            "target_lufs": float(target_lufs),
            "pre_lufs": float(cur_lufs),
            "post_lufs": float(post),
            "gain_db": float(gain_db),
        }
        return y_lim, lim_stats

    allow_above = float(getattr(preset, "governor_allow_above_db", 0.0))
    high = float(preset.target_lufs + allow_above)
    low = float(preset.target_lufs + preset.governor_step_db * max(1, int(preset.governor_iters)))
    if low > high:
        low, high = high, low

    best_audio: Optional[np.ndarray] = None
    best_stats: Optional[Dict[str, float]] = None
    lo, hi = low, high
    steps = int(getattr(preset, "governor_search_steps", 11))

    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        cand_audio, cand_stats = _render_at(mid)

        ok_gr = float(cand_stats.get("min_gain_db", -999.0)) > float(preset.governor_gr_limit_db)
        ok_tp = float(cand_stats.get("tp_dbfs", 0.0)) <= float(preset.ceiling_dbfs + 0.10)

        if ok_gr and ok_tp:
            best_audio, best_stats = cand_audio, cand_stats
            lo = mid  # try louder (closer to high)
        else:
            hi = mid  # back off

    if best_audio is None or best_stats is None:
        best_audio, best_stats = _render_at(low)

    y = best_audio
    governor_target = float(best_stats.get("target_lufs", preset.target_lufs))
    final_gr_db = float(best_stats.get("min_gain_db", 0.0))
    post_lufs = float(best_stats.get("post_lufs", integrated_loudness_lufs(y, sr_t)))
    tp = float(best_stats.get("tp_dbfs", lin_to_db(true_peak_estimate(y, sr_t, oversample=preset.limiter_oversample) + 1e-12)))

    write_audio(out_path, y, sr_t)

    result = {
        "preset": preset.name,
        "sr": sr_t,
        "target_lufs_requested": preset.target_lufs,
        "governor_target_lufs": float(governor_target),
        "governor_steps": int(steps),
        "governor_gr_limit_db": float(preset.governor_gr_limit_db),
        "lufs_pre": float(pre_lufs),
        "lufs_post": float(post_lufs),
        "true_peak_dbfs": float(tp),
        "limiter_mode": str(getattr(preset, "limiter_mode", "v2")),
        "limiter_min_gain_db": float(final_gr_db),
        "limiter_avg_gr_db": float(best_stats.get("avg_gr_db", 0.0)) if "best_stats" in locals() and best_stats is not None else None,
        "sub_f0_hz": float(f0) if f0 is not None else None,
        "mono_sub_cutoff_hz": float(mono_cut) if mono_cut is not None else None,
        "mono_sub_mix": float(mono_mix) if mono_mix is not None else None,

"microdetail": microdetail_info,
        "microdetail": microdetail_info,
        "movement": movement_info,
        "hooklift": hooklift_info,
        "stems": stems_info,
        "runtime_sec": float(time.time() - t0),
        "out_path": out_path,
    }

    if report_path:
        os.makedirs(os.path.dirname(os.path.abspath(report_path)) or ".", exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# AuralMind Maestro v7.3 expert — Report\n\n")
            f.write("## Summary\n")
            f.write(f"- Preset: **{result['preset']}**\n")
            f.write(f"- Sample rate: **{result['sr']} Hz**\n")
            f.write(f"- LUFS (pre): **{result['lufs_pre']:.2f}**\n")
            f.write(f"- LUFS (post): **{result['lufs_post']:.2f}**\n")
            f.write(f"- True peak (approx): **{result['true_peak_dbfs']:.2f} dBFS**\n")
            f.write(f"- Limiter min gain (approx GR): **{result['limiter_min_gain_db']:.2f} dB**\n\n")

            f.write("## Low-end / music theory anchors\n")
            f.write(f"- Estimated sub fundamental f0: **{result['sub_f0_hz']} Hz**\n")
            f.write(f"- Mono-sub v2 cutoff: **{result['mono_sub_cutoff_hz']} Hz**\n")
            f.write(f"- Mono-sub v2 adaptive mix: **{result['mono_sub_mix']}**\n\n")

            f.write("## Stereo enhancements\n")
            f.write("- Spatial Realism Enhancer: frequency-dependent width + correlation guard\n")
            f.write("- NEW CGMS MicroShift: micro-delay applied to SIDE high-band only, correlation-guarded\n\n")
            f.write(f"- MicroDetail recovery: **{result['microdetail'].get('enabled', False)}**")
            if result['microdetail'].get('enabled', False):
                f.write(f" (corr={result['microdetail'].get('corr', None)}, eff_amount={result['microdetail'].get('eff_amount', None)})\n\n")
            else:
                f.write("\n\n")
            f.write("## Movement / HookLift\n")
            f.write(f"- Movement enabled: **{result['movement'].get('enabled', False)}** (amount={result['movement'].get('amount', None)})\n")
            f.write(f"- HookLift enabled: **{result['hooklift'].get('enabled', False)}** (mix={result['hooklift'].get('mix', None)})\n")
            if result['hooklift'].get('auto', False):
                f.write(f"  - Auto mask percentile: **{result['hooklift'].get('auto_percentile', None)}**\n")
            f.write("\n")

            f.write("## Stem separation (HT-Demucs)\n")
            f.write(f"- Enabled: **{result['stems'].get('enabled', False)}**\n")
            if result['stems'].get('enabled', False):
                f.write(f"- Model: **{result['stems'].get('model_name', None)}**\n")
                f.write(f"- Sources: **{result['stems'].get('sources', None)}**\n")
            else:
                if 'reason' in result['stems']:
                    f.write(f"- Reason: **{result['stems'].get('reason', None)}**\n")
                if 'error' in result['stems']:
                    f.write(f"- Error: **{result['stems'].get('error', None)}**\n")
            f.write("\n")

            f.write("## Loudness Governor\n")
            f.write(f"- Requested target LUFS: **{preset.target_lufs}**\n")
            f.write(f"- Governor final target LUFS: **{result['governor_target_lufs']}**\n")
            f.write(f"- Governor steps: **{result['governor_steps']}** (binary search)\n")
            f.write(f"- Limiter mode: **{result['limiter_mode']}**\n")
            if result.get('limiter_avg_gr_db') is not None:
                f.write(f"- Limiter avg gain (dB): **{result['limiter_avg_gr_db']:.2f}** (closer to 0 = less overall limiting)\n")
            f.write("  If limiter GR exceeded the ceiling, the governor backed off the LUFS target.\n\n")

            f.write("## JSON dump\n")
            f.write("```json\n")
            f.write(json.dumps(result, indent=2))
            f.write("\n```\n")

    return result


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AuralMind Maestro v7.3 expert — Expert-tier mastering script")
    p.add_argument("--target", required=True, help="Path to target audio (wav/flac/aiff/ogg).")
    p.add_argument("--reference", default=None, help="Optional reference audio for match EQ.")
    p.add_argument("--out", required=True, help="Output mastered wav path.")
    p.add_argument("--report", default=None, help="Optional report markdown output.")
    p.add_argument("--preset", default="hi_fi_streaming", choices=list(get_presets().keys()), help="Preset name.")
    p.add_argument("--no-stems", action="store_true", help="Disable HT-Demucs stem separation (otherwise enabled by preset).")
    p.add_argument("--demucs-model", default=None, help="Demucs model name (default from preset, e.g., htdemucs).")
    p.add_argument("--demucs-device", default=None, help="Demucs device: cpu or cuda (if available).")
    p.add_argument("--demucs-overlap", type=float, default=None, help="Demucs overlap (0.0-0.99).")
    p.add_argument("--demucs-no-split", action="store_true", help="Disable split processing inside Demucs (faster, more RAM).")
    p.add_argument("--demucs-shifts", type=int, default=None, help="Demucs shifts (quality vs speed).")

    p.add_argument("--no-movement", action="store_true", help="Disable movement automation (otherwise enabled by preset).")
    p.add_argument("--movement-amount", type=float, default=None, help="Movement amount (0.0-0.35).")

    p.add_argument("--no-hooklift", action="store_true", help="Disable HookLift (otherwise enabled by preset).")
    p.add_argument("--hooklift-mix", type=float, default=None, help="HookLift wet mix (0.0-0.65).")
    p.add_argument("--hooklift-no-auto", action="store_true", help="Disable auto mask; apply HookLift across the full track.")
    p.add_argument("--hooklift-percentile", type=float, default=None, help="Auto mask percentile (50-95).")

    # Expert overrides / QoL
    p.add_argument("--auto", action="store_true",
                   help="Analyze the target (and reference if provided) and auto-tune preset + safe mastering parameters.")
    p.add_argument("--target-lufs", type=float, default=None,
                   help="Override preset target LUFS (integrated). Example: -12.0")
    p.add_argument("--ceiling", type=float, default=None,
                   help="Override limiter ceiling (dBFS). Example: -1.0 (recommended for streaming)")
    p.add_argument("--limiter", choices=["v1", "v2"], default=None,
                   help="Limiter engine override (v2 is more transparent / stable).")
    p.add_argument("--no-softclip", action="store_true",
                   help="Disable pre-limiter oversampled soft clip stage.")
    p.add_argument("--fir-stream", choices=["auto", "on", "off"], default=None,
                   help="Match-EQ FIR application mode. auto=heuristic, on=overlap-save streaming, off=full fftconvolve.")
    p.add_argument("--fir-block-pow2", type=int, default=None,
                   help="Block size as pow2 for FIR streaming (e.g., 17 => 131072 samples).")
    p.add_argument("--microdetail", action="store_true",
                   help="Force-enable MicroDetail recovery (SIDE high-band upward micro-comp).")
    p.add_argument("--no-microdetail", action="store_true",
                   help="Disable MicroDetail recovery.")

    return p

def main():
    args = build_arg_parser().parse_args()
    presets = get_presets()
    preset = presets[args.preset]

    # Auto-tune (expert): pick preset + safe loudness/GR constraints from audio features
    auto_info: Dict[str, Any] = {"enabled": False}
    if args.auto:
        y_t, sr_t = load_audio(args.target)
        tf = analyze_track_features(y_t, sr_t)
        rf = None
        if args.reference:
            y_r, sr_r = load_audio(args.reference)
            rf = analyze_track_features(y_r, sr_r)
        name = auto_select_preset_name(tf)
        preset = presets.get(name, preset)
        preset, auto_info = auto_tune_preset(preset, tf, rf)

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
    # Expert overrides
    if args.target_lufs is not None:
        updates["target_lufs"] = float(args.target_lufs)
    if args.ceiling is not None:
        updates["ceiling_dbfs"] = float(args.ceiling)
    if args.limiter is not None:
        updates["limiter_mode"] = str(args.limiter)
    if args.no_softclip:
        updates["enable_softclip"] = False
    if args.fir_stream is not None:
        updates["fir_streaming"] = str(args.fir_stream)
    if args.fir_block_pow2 is not None:
        updates["fir_block_pow2"] = int(args.fir_block_pow2)
    if args.microdetail:
        updates["enable_microdetail"] = True
    if args.no_microdetail:
        updates["enable_microdetail"] = False


    if updates:
        preset = replace(preset, **updates)

    res = master(args.target, args.out, preset, reference_path=args.reference, report_path=args.report)
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
