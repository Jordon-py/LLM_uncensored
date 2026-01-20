"""
Melodic Trap Mastering Suite v7.0 (2026 Pro Upgrade)
---------------------------------------------------
Adds 3 major upgrades to the v6 chain:

(1) SIDE-Only De-Esser (6–11 kHz)
    - Prevents "air shelf width" from turning into sizzling cymbals/sibilance.
    - Operates ONLY on the SIDE channel high band.
    - Uses STFT-band level detection + attack/release gain reduction.

(2) 808 Stability Module (Sub-Glue)
    - Sub-band saturation (for density + consistent bass perception)
    - Mono anchor (keeps sub stable in mono playback)
    - Low-band gain reduction (tightens 808 so it stays loud without bloom)

(3) ORIGINAL MUSIC-THEORY TECHNIQUE:
    Scale-Locked Stereo Harmonic Widening (SLSHW)
    - Estimates scale degrees from chroma statistics.
    - Widen ONLY bins mapped to the inferred scale degrees.
    - Reduces width for out-of-scale bins (where sibilance/noise often lives).
    - Result: wide melodic content, controlled harshness, clearer vocal center.

Dependencies:
  pip install numpy librosa soundfile
Optional:
  pip install pyloudnorm scipy

Notes:
python melodic_trap_master_v7.py --in "C:/Users/goku/Downloads/FaceTime (12).wav" --out "FaceTime master_air_v7.wav" --preset "Trap Air"

python melodic_trap_master_v7.py --in "C:/Users/goku/Downloads/I'm Him (14).wav" --out "I'm Him master_soul_v7.wav" --preset "Trap Soul"

"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf

# Optional loudness (best path)
try:
    import pyloudnorm as pyln
except Exception:
    pyln = None

# Optional oversampling helper
try:
    from scipy.signal import resample_poly
except Exception:
    resample_poly = None


# -----------------------------
# Parameter configuration
# -----------------------------

class MasterParams:
    def __init__(self, preset: str = "Trap Air"):
        # Loudness / ceiling
        self.target_lufs = -11.0
        self.true_peak_dbfs = -1.0

        # STFT config
        self.n_fft = 2048
        self.hop_length = 512
        self.win_length = 2048

        # Harmonic resonator
        self.enable_harmonic_resonator = True
        self.key_center_hz = 433.0
        self.harmonic_gain_db = 2.5
        self.harmonic_q = 18.0

        # Exciter (air)
        self.enable_exciter = True
        self.exciter_strength = 0.85
        self.exciter_start_hz = 8500.0

        # Psychoacoustic clarity tilt (normalized & limited)
        self.enable_psycho_eq = True
        self.sensitivity_center_hz = 3500.0
        self.weighting_strength = 0.55
        self.max_eq_db = 4.5

        # Transient control (HPSS-based)
        self.enable_transient_punch = True
        self.perc_boost_db = 1.5
        self.harm_atten_db = 0.2

        # Resonance tamer
        self.enable_resonance_tamer = True
        self.res_tame_strength = 0.45
        self.res_tame_bins = 9

        # M/S imaging + mono low-end
        self.enable_mid_side = True
        self.mid_gain_db = -0.3
        self.side_gain_db = 1.3
        self.mono_below_hz = 120.0
        self.mono_transition_hz = 60.0
        self.side_air_shelf_db = 0.8
        self.side_air_start_hz = 10000.0

        # NEW: Scale-Locked Stereo Harmonic Widening (SLSHW)
        self.enable_scale_locked_width = True
        self.scale_width_start_hz = 220.0
        self.scale_width_end_hz = 11000.0
        self.scale_width_in_scale_db = 0.8     # widen in-scale bins
        self.scale_width_out_scale_db = -0.35  # narrow out-of-scale bins slightly

        # NEW: SIDE de-esser (high band only)
        self.enable_side_deesser = True
        self.deess_low_hz = 6000.0
        self.deess_high_hz = 11400.0
        self.deess_ratio = 4.0
        self.deess_attack_ms = 2.0
        self.deess_release_ms = 70.0
        self.deess_max_gr_db = 6.0
        self.deess_auto_threshold = True
        self.deess_threshold_db = -28.0        # used only if auto_threshold=False

        # NEW: 808 Stabilizer (Sub-Glue)
        self.enable_808_stabilizer = True
        self.sub_band_hz = 130.0
        self.sub_sat_drive = 1.65
        self.sub_sat_mix = 0.40
        self.sub_comp_ratio = 3.2
        self.sub_comp_attack_ms = 10.0
        self.sub_comp_release_ms = 140.0
        self.sub_comp_max_gr_db = 5.5
        self.sub_comp_auto_threshold = True
        self.sub_comp_threshold_db = -22.0     # fallback if auto off

        # Final dynamics
        self.enable_soft_clip = True
        self.soft_clip_drive = 1.3
        self.enable_true_peak_guard = True
        self.oversample_factor = 4

        # Presets
        if preset == "Trap Air":
            self.harmonic_gain_db = 2.3
            self.exciter_strength = 0.69
            self.side_gain_db = 1.6
            self.perc_boost_db = 1.6
            self.side_air_shelf_db = 1.1

            # Air preset benefits from stronger protection against sizzle:
            self.deess_max_gr_db = 7.0

            # More “expensive wide” feel from SLSHW:
            self.scale_width_in_scale_db = .90
            self.scale_width_out_scale_db = -0.45

        elif preset == "Trap Soul":
            self.harmonic_gain_db = 2.0
            self.exciter_strength = 0.55
            self.side_gain_db = 1.0
            self.perc_boost_db = 1.2
            self.side_air_shelf_db = 0.5

            # Softer de-ess and narrower scale widening:
            self.deess_max_gr_db = 6.9
            self.scale_width_in_scale_db = 0.65
            self.scale_width_out_scale_db = -0.25


# -----------------------------
# Utility helpers (stereo-safe)
# -----------------------------

def ensure_stereo(audio: np.ndarray) -> np.ndarray:
    """
    Returns audio shape (n_samples, 2).
    librosa.load(mono=False) returns shape (n_channels, n_samples).
    """
    audio = np.asarray(audio)

    if audio.ndim == 1:
        return np.stack([audio, audio], axis=-1)

    if audio.shape[0] == 2 and audio.shape[1] > 2:
        return audio.T

    if audio.shape[1] == 2:
        return audio

    raise ValueError(f"Unexpected audio shape: {audio.shape}")


def db_to_lin(db: float) -> float:
    return 10 ** (db / 20.0)


def lin_to_db(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(x, eps))


def mid_side_split(stereo: np.ndarray):
    L = stereo[:, 0]
    R = stereo[:, 1]
    mid = 0.5 * (L + R)
    side = 0.5 * (L - R)
    return mid, side


def mid_side_merge(mid: np.ndarray, side: np.ndarray) -> np.ndarray:
    L = mid + side
    R = mid - side
    return np.stack([L, R], axis=-1)


def stft_multi(stereo: np.ndarray, p: MasterParams) -> np.ndarray:
    """
    Returns STFT shape: (2, n_freq, n_frames)
    """
    out = []
    for ch in range(2):
        S = librosa.stft(
            stereo[:, ch],
            n_fft=p.n_fft,
            hop_length=p.hop_length,
            win_length=p.win_length,
            window="hann",
        )
        out.append(S)
    return np.stack(out, axis=0)


def istft_multi(S: np.ndarray, p: MasterParams, length: int) -> np.ndarray:
    """
    S shape: (2, n_freq, n_frames)
    Returns stereo shape: (n_samples, 2)
    """
    out = []
    for ch in range(2):
        y = librosa.istft(
            S[ch],
            hop_length=p.hop_length,
            win_length=p.win_length,
            window="hann",
            length=length,
        )
        out.append(y)
    return np.stack(out, axis=-1)


def smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def ar_coeffs(sr: int, hop_length: int, attack_ms: float, release_ms: float):
    """
    Converts attack/release milliseconds into envelope smoothing coefficients
    per STFT frame.
    """
    frame_rate = sr / hop_length
    a = np.exp(-1.0 / (max(attack_ms, 0.1) * 1e-3 * frame_rate))
    r = np.exp(-1.0 / (max(release_ms, 1.0) * 1e-3 * frame_rate))
    return float(a), float(r)


# -----------------------------
# Music theory inference (scale degrees)
# -----------------------------

def estimate_scale_pitch_classes(stereo: np.ndarray, sr: int) -> set[int]:
    """
    Estimates a "scale fingerprint" by selecting the most active pitch classes.
    This is not perfect key detection, but it's stable enough for SLSHW.

    Returns:
      pitch classes as ints 0..11 (0=C, 9=A, etc.)
    """
    mono = np.mean(stereo, axis=1)
    chroma = librosa.feature.chroma_stft(y=mono, sr=sr)
    pc_energy = chroma.mean(axis=1)

    # Pick top 7 pitch classes as a "scale-like" set
    top = np.argsort(pc_energy)[-7:]
    return set(int(x) for x in top)


def pitch_class_for_freq(freqs: np.ndarray) -> np.ndarray:
    """
    Map FFT bin frequencies to pitch classes using 12-TET around A4=440.
    For freqs<=0, returns -1.
    """
    pcs = np.full_like(freqs, fill_value=-1, dtype=int)
    mask = freqs > 0
    midi = 69.0 + 12.0 * np.log2(freqs[mask] / 432.0)
    midi_round = np.round(midi).astype(int)
    pcs[mask] = midi_round % 12
    return pcs


# -----------------------------
# Key-aware harmonic resonator (unchanged from v6)
# -----------------------------

def estimate_key_center_hz(stereo: np.ndarray, sr: int) -> float:
    mono = np.mean(stereo, axis=1)
    chroma = librosa.feature.chroma_stft(y=mono, sr=sr)
    pitch_class = int(np.argmax(chroma.mean(axis=1)))
    midi_root = 48 + pitch_class  # around octave 3
    hz = 433.0 * (2.0 ** ((midi_root - 69) / 12.0))
    return float(hz)


def harmonic_resonator(stereo: np.ndarray, sr: int, p: MasterParams) -> np.ndarray:
    length = stereo.shape[0]
    S = stft_multi(stereo, p)
    mag = np.abs(S)
    phase = np.angle(S)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=p.n_fft)
    key_hz = estimate_key_center_hz(stereo, sr)

    gain_lin = db_to_lin(p.harmonic_gain_db)

    for h in range(2, 6):
        target = key_hz * h
        if target >= freqs[-1]:
            continue

        bw = target / max(p.harmonic_q, 1e-6)
        g = np.exp(-0.5 * ((freqs - target) / max(bw, 1.0)) ** 2)
        mag *= (1.0 + (gain_lin - 1.0) * g[None, :, None])

    S2 = mag * np.exp(1j * phase)
    return istft_multi(S2, p, length=length)


# -----------------------------
# Controlled HF exciter (unchanged from v6)
# -----------------------------

def harmonic_exciter(stereo: np.ndarray, sr: int, p: MasterParams) -> np.ndarray:
    length = stereo.shape[0]
    S = stft_multi(stereo, p)
    mag = np.abs(S)
    phase = np.angle(S)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=p.n_fft)

    hf = freqs >= p.exciter_start_hz
    if not np.any(hf):
        return stereo

    mag_hf = mag * hf[None, :, None]
    S_hf = mag_hf * np.exp(1j * phase)
    hf_audio = istft_multi(S_hf, p, length=length)

    drive = 1.0 + 1.5 * p.exciter_strength
    sat = np.tanh(hf_audio * drive)

    mix = 0.10 * p.exciter_strength
    return stereo + mix * sat


# -----------------------------
# Psychoacoustic clarity tilt (unchanged from v6)
# -----------------------------

def psychoacoustic_weighted_eq(stereo: np.ndarray, sr: int, p: MasterParams) -> np.ndarray:
    length = stereo.shape[0]
    S = stft_multi(stereo, p)
    mag = np.abs(S)
    phase = np.angle(S)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=p.n_fft)

    sigma = 1500.0
    w = 1.0 + p.weighting_strength * np.exp(-((freqs - p.sensitivity_center_hz) ** 2) / (2.0 * sigma ** 2))
    w = w / max(np.mean(w), 1e-9)

    w_db = lin_to_db(w)
    w_db = np.clip(w_db, -p.max_eq_db, p.max_eq_db)
    w_lin = db_to_lin(w_db)

    mag2 = mag * w_lin[None, :, None]
    S2 = mag2 * np.exp(1j * phase)
    return istft_multi(S2, p, length=length)


# -----------------------------
# HPSS transient punch (unchanged from v6)
# -----------------------------

def transient_punch_hpss(stereo: np.ndarray, p: MasterParams) -> np.ndarray:
    out = np.zeros_like(stereo)
    for ch in range(2):
        y = stereo[:, ch]
        y_h, y_p = librosa.effects.hpss(y)
        y_h *= db_to_lin(-p.harm_atten_db)
        y_p *= db_to_lin(p.perc_boost_db)
        out[:, ch] = y_h + y_p
    return out


# -----------------------------
# Resonance tamer (unchanged from v6)
# -----------------------------

def resonance_tamer(stereo: np.ndarray, sr: int, p: MasterParams) -> np.ndarray:
    length = stereo.shape[0]
    S = stft_multi(stereo, p)
    mag = np.abs(S)
    phase = np.angle(S)

    k = max(int(p.res_tame_bins), 3)
    if k % 2 == 0:
        k += 1

    pad = k // 2
    mag_pad = np.pad(mag, ((0, 0), (pad, pad), (0, 0)), mode="edge")

    smooth = np.zeros_like(mag)
    for i in range(mag.shape[1]):
        smooth[:, i, :] = np.mean(mag_pad[:, i:i + k, :], axis=1)

    mag2 = (1.0 - p.res_tame_strength) * mag + p.res_tame_strength * smooth
    S2 = mag2 * np.exp(1j * phase)
    return istft_multi(S2, p, length=length)


# -----------------------------
# M/S cone width + ORIGINAL technique: SLSHW
# -----------------------------

def ms_cone_width_and_slshw(stereo: np.ndarray, sr: int, p: MasterParams) -> np.ndarray:
    """
    Stage: M/S foundation + mono low-end + optional side air shelf.
    PLUS: Scale-Locked Stereo Harmonic Widening (SLSHW) [ORIGINAL TECHNIQUE]
      - Estimate scale pitch classes (top 7 chroma bins)
      - For SIDE magnitude spectrum:
          widen bins that map to scale degrees
          narrow bins that don't
      - Restrict to a useful musical band so we don't widen sub or hiss.
    """
    mid, side = mid_side_split(stereo)
    mid *= db_to_lin(p.mid_gain_db)
    side *= db_to_lin(p.side_gain_db)

    # SIDE shaping in STFT domain
    S_side = librosa.stft(side, n_fft=p.n_fft, hop_length=p.hop_length, win_length=p.win_length, window="hann")
    mag = np.abs(S_side)
    phase = np.angle(S_side)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=p.n_fft)

    # (A) Mono low-end: attenuate SIDE below cutoff with smooth transition
    cutoff = p.mono_below_hz
    trans = p.mono_transition_hz
    lo = cutoff - trans
    hi = cutoff + trans
    x = (freqs - lo) / max((hi - lo), 1e-9)
    keep = smoothstep(x)  # 0 below lo, 1 above hi
    mag2 = mag * keep[:, None]

    # (B) Optional: Side air shelf for perceived width
    if p.side_air_shelf_db != 0.0:
        air = freqs >= p.side_air_start_hz
        mag2[air, :] *= db_to_lin(p.side_air_shelf_db)

    # (C) ORIGINAL: Scale-Locked Stereo Harmonic Widening (SLSHW)
    if p.enable_scale_locked_width:
        scale_pcs = estimate_scale_pitch_classes(stereo, sr)
        pcs = pitch_class_for_freq(freqs)

        band = (freqs >= p.scale_width_start_hz) & (freqs <= p.scale_width_end_hz) & (pcs >= 0)
        in_scale = np.zeros_like(freqs, dtype=bool)
        in_scale[band] = np.isin(pcs[band], np.array(sorted(scale_pcs), dtype=int))

        # in-scale gets slight extra width; out-of-scale gets slight narrowing
        widen_lin = db_to_lin(p.scale_width_in_scale_db)
        narrow_lin = db_to_lin(p.scale_width_out_scale_db)

        slshw_gain = np.ones_like(freqs, dtype=np.float32)
        slshw_gain[band] = np.where(in_scale[band], widen_lin, narrow_lin)

        # Apply to SIDE magnitude only (musically: widen tonal content, tame noisy content)
        mag2 *= slshw_gain[:, None]

    S2 = mag2 * np.exp(1j * phase)
    side2 = librosa.istft(S2, hop_length=p.hop_length, win_length=p.win_length, window="hann", length=len(side))

    return mid_side_merge(mid, side2)


# -----------------------------
# NEW: 808 Stability Module (Sub-Glue)
# -----------------------------

def stabilize_808_sub_glue(stereo: np.ndarray, sr: int, p: MasterParams) -> np.ndarray:
    """
    Tightens and stabilizes the 808 while preserving loudness.

    Steps:
      1) Work on MID channel (where 808 should live for mono stability)
      2) Extract sub band (<= sub_band_hz) via STFT mask
      3) Add sub saturation (density & perceived loudness)
      4) Apply low-band gain reduction (sub compressor behavior) per-frame
      5) Recombine back into stereo with original SIDE

    This avoids the common issue:
      "sub gets wide + blurry" -> collapses badly in mono or overpowers mix.
    """
    mid, side = mid_side_split(stereo)
    length = len(mid)

    # STFT mid
    S_mid = librosa.stft(mid, n_fft=p.n_fft, hop_length=p.hop_length, win_length=p.win_length, window="hann")
    mag = np.abs(S_mid)
    phase = np.angle(S_mid)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=p.n_fft)

    sub = freqs <= p.sub_band_hz
    if not np.any(sub):
        return stereo

    # --- (1) Reconstruct sub band audio
    mag_sub = mag * sub[:, None]
    S_sub = mag_sub * np.exp(1j * phase)
    sub_audio = librosa.istft(S_sub, hop_length=p.hop_length, win_length=p.win_length, window="hann", length=length)

    # --- (2) Sub saturation (tanh) + mix
    sat = np.tanh(sub_audio * p.sub_sat_drive)
    sub_sat = (1.0 - p.sub_sat_mix) * sub_audio + p.sub_sat_mix * sat

    # --- (3) Replace sub bins using STFT of saturated sub
    S_sub_sat = librosa.stft(sub_sat, n_fft=p.n_fft, hop_length=p.hop_length, win_length=p.win_length, window="hann")
    mag_sub_sat = np.abs(S_sub_sat)

    mag2 = mag.copy()
    mag2[sub, :] = mag_sub_sat[sub, :]

    # --- (4) Low-band gain reduction (sub compression) in STFT domain
    # Level per frame from sub magnitudes
    level = np.sqrt(np.mean((mag2[sub, :] ** 2), axis=0) + 1e-12)
    level_db = lin_to_db(level)

    # Auto threshold: set slightly below high-percentile sub energy
    if p.sub_comp_auto_threshold:
        thr = float(np.percentile(level_db, 80.0) - 3.0)
    else:
        thr = float(p.sub_comp_threshold_db)

    ratio = float(max(p.sub_comp_ratio, 1.0))
    over = np.maximum(level_db - thr, 0.0)
    gr_db_raw = over * (1.0 - 1.0 / ratio)
    gr_db_raw = np.clip(gr_db_raw, 0.0, p.sub_comp_max_gr_db)

    # Attack/release smoothing across frames
    a, r = ar_coeffs(sr, p.hop_length, p.sub_comp_attack_ms, p.sub_comp_release_ms)
    gr_db = np.zeros_like(gr_db_raw)

    for i in range(len(gr_db_raw)):
        x = gr_db_raw[i]
        prev = gr_db[i - 1] if i > 0 else 0.0
        if x > prev:
            gr_db[i] = a * prev + (1.0 - a) * x
        else:
            gr_db[i] = r * prev + (1.0 - r) * x

    gain = db_to_lin(-gr_db)
    mag2[sub, :] *= gain[None, :]

    S2 = mag2 * np.exp(1j * phase)
    mid2 = librosa.istft(S2, hop_length=p.hop_length, win_length=p.win_length, window="hann", length=length)

    return mid_side_merge(mid2, side)


# -----------------------------
# NEW: SIDE-only De-esser (high band)
# -----------------------------

def side_deesser(stereo: np.ndarray, sr: int, p: MasterParams) -> np.ndarray:
    """
    SIDE-only de-esser for 6–11 kHz:
      - Detect high-band SIDE level per frame
      - Apply gain reduction ONLY on those frequency bins
      - Attack/release smoothing prevents pumping
    """
    mid, side = mid_side_split(stereo)
    length = len(side)

    S = librosa.stft(side, n_fft=p.n_fft, hop_length=p.hop_length, win_length=p.win_length, window="hann")
    mag = np.abs(S)
    phase = np.angle(S)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=p.n_fft)

    band = (freqs >= p.deess_low_hz) & (freqs <= p.deess_high_hz)
    if not np.any(band):
        return stereo

    # Band level per frame
    band_mag = mag[band, :]
    level = np.sqrt(np.mean((band_mag ** 2), axis=0) + 1e-12)
    level_db = lin_to_db(level)

    # Auto threshold: robust, adapts per track
    if p.deess_auto_threshold:
        thr = float(np.percentile(level_db, 85.0) - 2.5)
    else:
        thr = float(p.deess_threshold_db)

    ratio = float(max(p.deess_ratio, 1.0))
    over = np.maximum(level_db - thr, 0.0)
    gr_db_raw = over * (1.0 - 1.0 / ratio)
    gr_db_raw = np.clip(gr_db_raw, 0.0, p.deess_max_gr_db)

    # Smooth GR
    a, r = ar_coeffs(sr, p.hop_length, p.deess_attack_ms, p.deess_release_ms)
    gr_db = np.zeros_like(gr_db_raw)

    for i in range(len(gr_db_raw)):
        x = gr_db_raw[i]
        prev = gr_db[i - 1] if i > 0 else 0.0
        if x > prev:
            gr_db[i] = a * prev + (1.0 - a) * x
        else:
            gr_db[i] = r * prev + (1.0 - r) * x

    gain = db_to_lin(-gr_db)

    # Apply only on the SIDE high band
    mag2 = mag.copy()
    mag2[band, :] *= gain[None, :]

    S2 = mag2 * np.exp(1j * phase)
    side2 = librosa.istft(S2, hop_length=p.hop_length, win_length=p.win_length, window="hann", length=length)

    return mid_side_merge(mid, side2)


# -----------------------------
# Loudness + true peak guard
# -----------------------------

def measure_lufs(stereo: np.ndarray, sr: int) -> float:
    mono = np.mean(stereo, axis=1)
    if pyln is not None:
        meter = pyln.Meter(sr)
        return float(meter.integrated_loudness(mono))
    rms = np.sqrt(np.mean(mono**2) + 1e-12)
    return float(20.0 * np.log10(max(rms, 1e-9)))


def apply_loudness_target(stereo: np.ndarray, sr: int, p: MasterParams) -> np.ndarray:
    current = measure_lufs(stereo, sr)
    gain_db = p.target_lufs - current
    return stereo * db_to_lin(gain_db)


def true_peak_dbfs(stereo: np.ndarray, sr: int, oversample: int = 4) -> float:
    peak = float(np.max(np.abs(stereo)))
    if oversample <= 1 or resample_poly is None:
        return float(20.0 * np.log10(max(peak, 1e-12)))

    peaks = []
    for ch in range(2):
        y = stereo[:, ch]
        y_os = resample_poly(y, oversample, 1)
        peaks.append(np.max(np.abs(y_os)))
    tp = float(np.max(peaks))
    return float(20.0 * np.log10(max(tp, 1e-12)))


def soft_clip(stereo: np.ndarray, drive: float = 1.2) -> np.ndarray:
    return np.tanh(stereo * drive) / np.tanh(drive)


def enforce_true_peak_ceiling(stereo: np.ndarray, sr: int, p: MasterParams) -> np.ndarray:
    tp = true_peak_dbfs(stereo, sr, oversample=p.oversample_factor)
    if tp <= p.true_peak_dbfs + 1e-6:
        return stereo
    diff_db = p.true_peak_dbfs - tp
    return stereo * db_to_lin(diff_db)


# -----------------------------
# Main chain
# -----------------------------

def apply_mastering(audio: np.ndarray, sr: int, p: MasterParams) -> np.ndarray:
    stereo = ensure_stereo(audio)

    # Stage 0: M/S foundation + mono low-end + ORIGINAL SLSHW
    if p.enable_mid_side:
        stereo = ms_cone_width_and_slshw(stereo, sr, p)

    # Stage 1: 808 Stabilizer (Sub-Glue)
    if p.enable_808_stabilizer:
        stereo = stabilize_808_sub_glue(stereo, sr, p)

    # Stage 2: Harmonic lift
    if p.enable_harmonic_resonator:
        stereo = harmonic_resonator(stereo, sr, p)

    # Stage 3: Air
    if p.enable_exciter:
        stereo = harmonic_exciter(stereo, sr, p)

    # Stage 4: Clarity tilt
    if p.enable_psycho_eq:
        stereo = psychoacoustic_weighted_eq(stereo, sr, p)

    # Stage 5: Punch
    if p.enable_transient_punch:
        stereo = transient_punch_hpss(stereo, p)

    # Stage 6: Resonance control
    if p.enable_resonance_tamer:
        stereo = resonance_tamer(stereo, sr, p)

    # Stage 7: SIDE De-esser (protects the new width + air shelf)
    if p.enable_side_deesser:
        stereo = side_deesser(stereo, sr, p)

    # Stage 8: Loudness + density + peak safety
    stereo = apply_loudness_target(stereo, sr, p)

    if p.enable_soft_clip:
        stereo = soft_clip(stereo, drive=p.soft_clip_drive)

    if p.enable_true_peak_guard:
        stereo = enforce_true_peak_ceiling(stereo, sr, p)

    return stereo


# -----------------------------
# File processing
# -----------------------------

_LAST_NUMBER_RE = re.compile(r"(\d+)(?!.*\d)")


def auto_versioned_output_path(input_path: str, out_path_or_dir: str, tail_letters: int = 14) -> str:
    in_path = Path(input_path)
    out_path = Path(out_path_or_dir)
    out_dir = out_path.parent if out_path.suffix else out_path

    letters_only = "".join(ch for ch in in_path.stem if ch.isalpha())
    name_tail = (letters_only[-tail_letters:] if len(letters_only) > tail_letters else letters_only).rstrip(" .")
    if not name_tail:
        name_tail = "master"

    m = _LAST_NUMBER_RE.search(in_path.stem)
    version = int(m.group(1)) + 1 if m else 0

    while True:
        candidate = out_dir / f"{name_tail}{version}.wav"
        if not candidate.exists():
            return str(candidate)
        version += 1


def process_file(input_path: str, output_path: str, preset: str = "Trap Air"):
    p = MasterParams(preset=preset)

    output_path = auto_versioned_output_path(input_path, output_path)

    audio, sr = librosa.load(input_path, mono=False, sr=48000)
    stereo = ensure_stereo(audio)

    mastered = apply_mastering(stereo, sr, p)
    sf.write(output_path, mastered, sr)

    print(f"Mastering complete. Preset={preset}  SR={sr}")
    print(f"Approx LUFS={measure_lufs(mastered, sr):.2f}")
    print(f"TruePeak(dBFS)≈{true_peak_dbfs(mastered, sr, oversample=p.oversample_factor):.2f}")


def build_cli():
    ap = argparse.ArgumentParser(description="Melodic Trap Mastering Suite v7.0")
    ap.add_argument("--in", dest="inp", required=True, help="Input audio path (wav/mp3)")
    ap.add_argument("--out", dest="out", required=True, help="Output dir or placeholder path (filename auto-derived)")
    ap.add_argument("--preset", default="Trap Air", choices=["Trap Air", "Trap Soul"])
    return ap


if __name__ == "__main__":
    args = build_cli().parse_args()
    process_file(args.inp, args.out, preset=args.preset)
