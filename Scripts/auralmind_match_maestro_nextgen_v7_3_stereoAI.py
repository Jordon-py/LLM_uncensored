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
    python auralmind_match_maestro_nextgen_v7_3_stereoAI.py --target "song.wav" --out "song_master.wav"

Reference match:
    python auralmind_match_maestro_nextgen_v7_3_stereoAI.py --reference "ref.mp3" --target "song.wav" --out "song_master.wav"

Choose a preset:
    python auralmind_match_maestro_nextgen_v7_3_stereoAI.py --preset hi_fi_streaming --target "song.wav" --out "song_master.wav"

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
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import soundfile as sf
import scipy
import scipy.signal as sps
from scipy.signal import fftconvolve
from scipy.ndimage import maximum_filter1d

sci = scipy.ndimage

# ---------------------------
# Utility helpers
# ---------------------------

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

def ensure_stereo(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return np.stack([y, y], axis=1)
    if y.shape[1] == 1:
        return np.repeat(y, 2, axis=1)
    return y

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
    y = ensure_stereo(y)
    out = np.zeros_like(y, dtype=np.float32)
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

    # Stereo: NEW microshift CGMS
    if preset.enable_microshift:
        y = microshift_widen_side(y, sr_t, shift_ms=preset.microshift_ms, mix=preset.microshift_mix)

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
        "runtime_sec": float(time.time() - t0),
        "out_path": out_path,
    }

    if report_path:
        os.makedirs(os.path.dirname(os.path.abspath(report_path)) or ".", exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# AuralMind Maestro v7.3 StereoAI — Report\n\n")
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

            f.write("## Loudness Governor\n")
            f.write(f"- Requested target LUFS: **{preset.target_lufs}**\n")
            f.write(f"- Governor final target LUFS: **{result['target_lufs_after_governor']}**\n")
            f.write("  If limiter GR exceeded the ceiling, the governor backed off the LUFS target.\n\n")

            f.write("## JSON dump\n")
            f.write("```json\n")
            f.write(json.dumps(result, indent=2))
            f.write("\n```\n")

    return result


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AuralMind Maestro v7.3 StereoAI — Expert-tier mastering script")
    p.add_argument("--target", required=True, help="Path to target audio (wav/flac/aiff/ogg).")
    p.add_argument("--reference", default=None, help="Optional reference audio for match EQ.")
    p.add_argument("--out", required=True, help="Output mastered wav path.")
    p.add_argument("--report", default=None, help="Optional report markdown output.")
    p.add_argument("--preset", default="hi_fi_streaming", choices=list(get_presets().keys()), help="Preset name.")
    return p

def main():
    args = build_arg_parser().parse_args()
    presets = get_presets()
    preset = presets[args.preset]
    res = master(args.target, args.out, preset, reference_path=args.reference, report_path=args.report)
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
