"""
Melodic Trap Mastering Suite v7.1 FIXED (2026 Pro Upgrade + Stage Profiler)
---------------------------------------------------------------------------
Fixes:
  - Broadcasting bug in spectral curve multiplication:
      mag shape = (2, n_freq, n_frames)
      curve shape = (n_freq,)
    Correct broadcast: curve[None, :, None]   ✅

Enhancements (kept):
  - GrooveLock rhythm enhancer (beat-synchronous pocket control)
  - Pedalboard final chain (compressor + limiter) if installed
  - Anti-aliased soft clip fallback to reduce "firework crackle"
  - JSON stage profiler for debugging

Best-practice upgrade:
  - librosa.load res_type="soxr_hq" when supported (you already installed soxr)

Usage:
  python melodic_trap_master_v7_1_FIXED.py --in "C:/path/song.wav" --out "C:/path/out.wav" --preset "Trap Soul"
"""

from __future__ import annotations

import argparse
import re
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf

# Optional libraries
try:
    import pyloudnorm as pyln
except Exception:
    pyln = None

try:
    from pedalboard import Pedalboard, Compressor, Limiter
except Exception:
    Pedalboard = None

try:
    from scipy.signal import resample_poly
except Exception:
    resample_poly = None

LOG="C:/Users/goku/Music/music_logs"
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

        # Harmonic exciter
        self.enable_exciter = True
        self.exciter_strength = 0.65
        self.exciter_hz = 8000.0

        # Psycho-EQ
        self.enable_psycho_eq = True
        self.low_shelf_db = 0.8
        self.high_shelf_db = 1.0
        self.presence_db = 0.8

        # Transient punch (HPSS)
        self.enable_transient_punch = True
        self.perc_boost_db = 1.4

        # Resonance tamer
        self.enable_resonance_tamer = True
        self.resonance_q = 6.0
        self.resonance_cut_db = -1.6

        # M/S width and mono safety
        self.enable_mid_side = True
        self.side_gain_db = 1.2
        self.sub_mono_hz = 120.0

        # Side de-esser
        self.enable_side_deesser = True
        self.deess_low_hz = 6000.0
        self.deess_high_hz = 11000.0
        self.deess_threshold_db = -22.0
        self.deess_max_gr_db = 6.5
        self.deess_attack_ms = 5.0
        self.deess_release_ms = 70.0

        # 808 Stabilizer
        self.enable_808_stabilizer = True
        self.sub_glue_cutoff_hz = 140.0
        self.sub_glue_ratio = 3.4
        self.sub_glue_threshold_db = -27.0
        self.sub_sat_drive = 5.0
        self.sub_sat_mix = 0.38
        self.sub_mono_strength = 0.88

        # SLSHW (Scale-Locked Stereo Harmonic Widening)
        self.enable_slshw = True
        self.scale_width_in_scale_db = 0.75
        self.scale_width_out_scale_db = -0.20

        # GrooveLock rhythm enhancer
        self.enable_groove_lock = True
        self.groove_lock_strength = 0.32
        self.groove_lock_min_gain = 0.90
        self.groove_lock_sigma = 0.08  # beat-smearing

        # Peak safety
        self.enable_true_peak_guard = True
        self.enable_soft_clip = True
        self.soft_clip_drive = 1.25
        self.oversample_factor = 4

        # Debug / analysis outputs
        self.enable_json_log = False
        self.debug_save_stage_wavs = False
        self.debug_stage_wav_dir = "debug_stages"
        self.debug_stage_wav_max_seconds = 0.0  # 0 = full length

        # Sound quality guardrails
        self.enable_aa_soft_clip = True

        # Presets
        if preset == "Trap Air":
            self.harmonic_gain_db = 2.3
            self.exciter_strength = 0.69
            self.side_gain_db = 1.6
            self.perc_boost_db = 1.6
            self.high_shelf_db = 1.1
            self.deess_max_gr_db = 7.0
            self.scale_width_in_scale_db = 0.90
            self.scale_width_out_scale_db = -0.25

        if preset == "Trap Soul":
            self.harmonic_gain_db = 2.0
            self.exciter_strength = 0.55
            self.side_gain_db = 1.0
            self.perc_boost_db = 1.2
            self.high_shelf_db = 0.5
            self.deess_max_gr_db = 6.9
            self.scale_width_in_scale_db = 0.65
            self.scale_width_out_scale_db = -0.25


# -----------------------------
# Utility helpers
# -----------------------------

def _sanitize_filename(name: str) -> str:
    keep = []
    for ch in name.strip():
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        elif ch in (" ", ".", "/"):
            keep.append("_")
    out = "".join(keep).strip("_")
    return out[:80] if out else "stage"


def db_to_lin(db: float | np.ndarray) -> float | np.ndarray:
    """Convert dB to linear; supports scalars and arrays."""
    out = 10 ** (np.asarray(db) / 20.0)
    if np.ndim(out) == 0:
        return float(out)
    return out


def lin_to_db(x: float | np.ndarray) -> float | np.ndarray:
    return 20.0 * np.log10(np.maximum(x, 1e-12))


def ensure_stereo(audio: np.ndarray) -> np.ndarray:
    """Ensure audio is (n_samples, 2)."""
    if audio.ndim == 1:
        return np.stack([audio, audio], axis=-1)
    if audio.shape[0] == 2 and audio.shape[1] != 2:
        return audio.T
    if audio.shape[1] == 1:
        return np.repeat(audio, 2, axis=1)
    return audio


def mid_side_split(stereo: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    L = stereo[:, 0]
    R = stereo[:, 1]
    mid = 0.5 * (L + R)
    side = 0.5 * (L - R)
    return mid.astype(np.float32), side.astype(np.float32)


def mid_side_merge(mid: np.ndarray, side: np.ndarray) -> np.ndarray:
    L = mid + side
    R = mid - side
    return np.stack([L, R], axis=-1).astype(np.float32)


def stft_multi(stereo: np.ndarray, p: MasterParams) -> np.ndarray:
    """STFT for stereo channels. Returns complex array shape (2, n_freq, n_frames)."""
    S = []
    for ch in range(2):
        S.append(
            librosa.stft(
                stereo[:, ch],
                n_fft=p.n_fft,
                hop_length=p.hop_length,
                win_length=p.win_length,
                window="hann",
            )
        )
    return np.stack(S, axis=0)


def istft_multi(S: np.ndarray, p: MasterParams, length: int) -> np.ndarray:
    out = []
    for ch in range(2):
        out.append(
            librosa.istft(
                S[ch],
                hop_length=p.hop_length,
                win_length=p.win_length,
                window="hann",
                length=length,
            )
        )
    return np.stack(out, axis=-1).astype(np.float32)


def smoothstep(x: np.ndarray, edge0: float, edge1: float) -> np.ndarray:
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# -----------------------------
# Debug / Profiling
# -----------------------------

def _peak_dbfs(stereo: np.ndarray) -> float:
    pk = float(np.max(np.abs(stereo)))
    return float(lin_to_db(max(pk, 1e-12)))


def _rms_dbfs(stereo: np.ndarray) -> float:
    mono = np.mean(stereo, axis=1)
    rms = float(np.sqrt(np.mean(mono**2) + 1e-12))
    return float(lin_to_db(max(rms, 1e-12)))


def _crest_db(stereo: np.ndarray) -> float:
    return float(_peak_dbfs(stereo) - _rms_dbfs(stereo))


def _stereo_corr(stereo: np.ndarray) -> float:
    L = stereo[:, 0]
    R = stereo[:, 1]
    if np.std(L) < 1e-9 or np.std(R) < 1e-9:
        return 0.0
    try:
        c = float(np.corrcoef(L, R)[0, 1])
        if np.isnan(c) or np.isinf(c):
            return 0.0
        return max(-1.0, min(1.0, c))
    except Exception:
        return 0.0


def _ms_width_db(stereo: np.ndarray) -> float:
    mid, side = mid_side_split(stereo)
    mid_rms = float(np.sqrt(np.mean(mid**2) + 1e-12))
    side_rms = float(np.sqrt(np.mean(side**2) + 1e-12))
    return float(lin_to_db(side_rms + 1e-12) - lin_to_db(mid_rms + 1e-12))


def _band_rms_mono(x: np.ndarray, sr: int, lo_hz: float, hi_hz: float, n_fft: int = 2048) -> float:
    try:
        S = librosa.stft(x, n_fft=n_fft, hop_length=n_fft // 4, win_length=n_fft, window="hann")
        mag = np.abs(S)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        band = (freqs >= lo_hz) & (freqs <= hi_hz)
        band_mag = mag[band, :]
        val = float(np.sqrt(np.mean(band_mag**2) + 1e-12))
        return val
    except Exception:
        return 0.0


def _crackle_score(stereo: np.ndarray, sr: int) -> float:
    mono = np.mean(stereo, axis=1).astype(np.float32)
    hi = _band_rms_mono(mono, sr, 6000.0, min(20000.0, sr * 0.49), n_fft=1024)
    d2 = np.diff(mono, n=2)
    spike = float(np.mean(np.abs(d2))) if len(d2) else 0.0
    return float((hi * 0.15) + (spike * 12.0))


def measure_lufs(stereo: np.ndarray, sr: int) -> float:
    mono = np.mean(stereo, axis=1)
    if pyln is not None:
        meter = pyln.Meter(sr)
        return float(meter.integrated_loudness(mono))
    rms = np.sqrt(np.mean(mono**2) + 1e-12)
    return float(20.0 * np.log10(max(rms, 1e-9)))


def true_peak_dbfs(stereo: np.ndarray, sr: int, oversample: int = 4) -> float:
    peak = float(np.max(np.abs(stereo)))
    if resample_poly is None or oversample <= 1:
        return float(lin_to_db(max(peak, 1e-12)))

    pk = 0.0
    for ch in range(2):
        x = stereo[:, ch].astype(np.float32)
        up = resample_poly(x, up=oversample, down=1)
        pk = max(pk, float(np.max(np.abs(up))))
    return float(lin_to_db(max(pk, 1e-12)))


def analyze_audio_stage(stereo: np.ndarray, sr: int, p: MasterParams, *, extra: dict | None = None) -> dict:
    tp = float(true_peak_dbfs(stereo, sr, oversample=p.oversample_factor))
    lufs = float(measure_lufs(stereo, sr))

    mid, side = mid_side_split(stereo)
    side_hi = _band_rms_mono(side.astype(np.float32), sr, p.deess_low_hz, p.deess_high_hz, n_fft=1024)
    clipped = int(np.sum(np.abs(stereo) >= 0.999))

    metrics = {
        "sr": int(sr),
        "lufs": float(lufs),
        "peak_dbfs": float(_peak_dbfs(stereo)),
        "true_peak_dbfs": float(tp),
        "rms_dbfs": float(_rms_dbfs(stereo)),
        "crest_db": float(_crest_db(stereo)),
        "stereo_corr": float(_stereo_corr(stereo)),
        "ms_width_db": float(_ms_width_db(stereo)),
        "side_highband_rms": float(side_hi),
        "clip_sample_count": int(clipped),
        "crackle_score": float(_crackle_score(stereo, sr)),
    }

    if extra:
        metrics.update(extra)

    warnings = []
    if tp > p.true_peak_dbfs + 0.25:
        warnings.append(f"true_peak_exceeds_ceiling({tp:.2f} dBFS)")
    if clipped > 0:
        warnings.append(f"hard_clipping_samples({clipped})")
    if metrics["crackle_score"] > 0.20:
        warnings.append("crackle_score_high")
    if warnings:
        metrics["warnings"] = warnings

    return metrics


class StageProfiler:
    def __init__(self, sr: int, preset: str):
        self.sr = int(sr)
        self.preset = preset
        self.started_at_utc = datetime.utcnow().isoformat() + "Z"
        self.events: list[dict] = []

    def record(self, stage: str, stereo: np.ndarray, sr: int, p: MasterParams, *, extra: dict | None = None):
        self.events.append({
            "t": time.time(),
            "stage": stage,
            "metrics": analyze_audio_stage(stereo, sr, p, extra=extra),
        })

    def to_report(self) -> dict:
        return {
            "script": "Melodic Trap Mastering Suite v7.1 FIXED",
            "preset": self.preset,
            "sr": self.sr,
            "started_at_utc": self.started_at_utc,
            "events": self.events,
        }


# -----------------------------
# Sound quality: anti-aliased soft clip
# -----------------------------

def soft_clip(stereo: np.ndarray, drive: float = 1.2) -> np.ndarray:
    return np.tanh(stereo * drive).astype(np.float32)


def anti_aliased_soft_clip(stereo: np.ndarray, sr: int, drive: float = 1.3, oversample: int = 4) -> np.ndarray:
    if resample_poly is None or oversample <= 1:
        return soft_clip(stereo, drive=drive)

    out = np.zeros_like(stereo, dtype=np.float32)

    for ch in range(2):
        x = stereo[:, ch].astype(np.float32)
        up = resample_poly(x, up=oversample, down=1).astype(np.float32)
        y = np.tanh(up * drive).astype(np.float32)
        dn = resample_poly(y, up=1, down=oversample).astype(np.float32)

        if len(dn) < len(x):
            dn = np.pad(dn, (0, len(x) - len(dn)), mode="edge")
        elif len(dn) > len(x):
            dn = dn[:len(x)]

        out[:, ch] = dn

    return out.astype(np.float32)


def enforce_true_peak_ceiling(stereo: np.ndarray, sr: int, p: MasterParams) -> np.ndarray:
    tp = true_peak_dbfs(stereo, sr, oversample=p.oversample_factor)
    if tp <= p.true_peak_dbfs:
        return stereo.astype(np.float32)
    diff = p.true_peak_dbfs - tp
    return (stereo * db_to_lin(diff)).astype(np.float32)


def apply_loudness_target(stereo: np.ndarray, sr: int, p: MasterParams) -> np.ndarray:
    current = measure_lufs(stereo, sr)
    gain_db = p.target_lufs - current
    return stereo * db_to_lin(gain_db)


# -----------------------------
# Music-theory helpers
# -----------------------------

def estimate_scale_pitch_classes(chroma: np.ndarray) -> list[int]:
    energy = chroma.mean(axis=1)
    idx = np.argsort(energy)[::-1]
    return sorted(list(map(int, idx[:7])))


def pitch_class_for_freq(freq: float, key_center_hz: float = 440.0) -> int:
    if freq <= 0:
        return 0
    n = 12.0 * np.log2(freq / key_center_hz)
    pc = int(np.round(n)) % 12
    return pc


def estimate_key_center_hz(stereo: np.ndarray, sr: int) -> float:
    mono = np.mean(stereo, axis=1)
    chroma = librosa.feature.chroma_stft(y=mono, sr=sr, n_fft=2048, hop_length=512)
    energy = chroma.mean(axis=1)
    root_pc = int(np.argmax(energy))
    return float(440.0 * (2 ** ((root_pc - 9) / 12.0)))


# -----------------------------
# Mastering blocks (FIXED broadcast)
# -----------------------------

def harmonic_resonator(stereo: np.ndarray, sr: int, p: MasterParams) -> np.ndarray:
    length = stereo.shape[0]
    S = stft_multi(stereo, p)
    mag = np.abs(S)
    phase = np.angle(S)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=p.n_fft)

    key = p.key_center_hz
    if key <= 0:
        key = estimate_key_center_hz(stereo, sr)

    gains = np.zeros_like(freqs, dtype=np.float32)
    for k in range(1, 8):
        f = key * k
        if f > sr * 0.45:
            break
        width = key * 0.07 * k
        bump = np.exp(-0.5 * ((freqs - f) / max(width, 1.0)) ** 2)
        gains += bump

    gains = gains / (np.max(gains) + 1e-12)
    lift = db_to_lin(p.harmonic_gain_db) - 1.0
    curve = (1.0 + lift * gains).astype(np.float32)

    # ✅ FIXED BROADCAST: curve maps to frequency axis (axis=1)
    mag2 = mag * curve[None, :, None]

    S2 = mag2 * np.exp(1j * phase)
    out = istft_multi(S2, p, length=length)
    return out.astype(np.float32)


def harmonic_exciter(stereo: np.ndarray, sr: int, p: MasterParams) -> np.ndarray:
    y = stereo.copy().astype(np.float32)
    alpha = np.exp(-2.0 * np.pi * p.exciter_hz / sr)
    hp = y.copy()
    hp[1:, :] = hp[1:, :] - alpha * hp[:-1, :]
    shaped = np.tanh(hp * (1.0 + 6.0 * p.exciter_strength))
    y = y + shaped * (0.12 + 0.18 * p.exciter_strength)
    return y.astype(np.float32)


def psychoacoustic_weighted_eq(stereo: np.ndarray, sr: int, p: MasterParams) -> np.ndarray:
    length = stereo.shape[0]
    S = stft_multi(stereo, p)
    mag = np.abs(S)
    phase = np.angle(S)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=p.n_fft)

    low = smoothstep(freqs, 20.0, 150.0) * (1.0 - smoothstep(freqs, 150.0, 300.0))
    high = smoothstep(freqs, 3500.0, 9000.0)
    presence = smoothstep(freqs, 900.0, 2500.0) * (1.0 - smoothstep(freqs, 2500.0, 4500.0))

    curve = (
        1.0
        + (db_to_lin(p.low_shelf_db) - 1.0) * low
        + (db_to_lin(p.presence_db) - 1.0) * presence
        + (db_to_lin(p.high_shelf_db) - 1.0) * high
    ).astype(np.float32)

    # ✅ FIXED BROADCAST
    mag2 = mag * curve[None, :, None]

    S2 = mag2 * np.exp(1j * phase)
    out = istft_multi(S2, p, length=length)
    return out.astype(np.float32)


def transient_punch_hpss(stereo: np.ndarray, p: MasterParams) -> np.ndarray:
    y = stereo.copy().astype(np.float32)
    for ch in range(2):
        harm, perc = librosa.effects.hpss(y[:, ch])
        y[:, ch] = harm + perc * db_to_lin(p.perc_boost_db)
    return y.astype(np.float32)


def resonance_tamer(stereo: np.ndarray, sr: int, p: MasterParams) -> np.ndarray:
    length = stereo.shape[0]
    S = stft_multi(stereo, p)
    mag = np.abs(S)
    phase = np.angle(S)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=p.n_fft)

    center = 3800.0
    width = center / max(p.resonance_q, 1.0)
    notch = np.exp(-0.5 * ((freqs - center) / max(width, 40.0)) ** 2).astype(np.float32)

    curve = (1.0 + (db_to_lin(p.resonance_cut_db) - 1.0) * notch).astype(np.float32)

    # ✅ FIXED BROADCAST
    mag2 = mag * curve[None, :, None]

    S2 = mag2 * np.exp(1j * phase)
    out = istft_multi(S2, p, length=length)
    return out.astype(np.float32)


def ms_cone_width_and_slshw(stereo: np.ndarray, sr: int, p: MasterParams) -> np.ndarray:
    mid, side = mid_side_split(stereo)
    length = len(mid)

    S_side = librosa.stft(side, n_fft=p.n_fft, hop_length=p.hop_length, win_length=p.win_length, window="hann")
    mag = np.abs(S_side)
    phase = np.angle(S_side)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=p.n_fft)

    low = (freqs <= p.sub_mono_hz).astype(np.float32)
    low_curve = 1.0 - 0.95 * low
    mag2 = mag * low_curve[:, None]
    S2 = mag2 * np.exp(1j * phase)
    side2 = librosa.istft(S2, hop_length=p.hop_length, win_length=p.win_length, window="hann", length=length)

    side2 = side2 * db_to_lin(p.side_gain_db)
    stereo2 = mid_side_merge(mid, side2)

    if not p.enable_slshw:
        return stereo2.astype(np.float32)

    mono = np.mean(stereo2, axis=1)
    chroma = librosa.feature.chroma_stft(y=mono, sr=sr, n_fft=2048, hop_length=512)
    in_scale = set(estimate_scale_pitch_classes(chroma))

    key_center = p.key_center_hz if p.key_center_hz > 0 else 440.0
    pcs = np.array([pitch_class_for_freq(float(f), key_center_hz=key_center) for f in freqs], dtype=int)
    mask_in = np.array([1.0 if pc in in_scale else 0.0 for pc in pcs], dtype=np.float32)

    width_in = db_to_lin(p.scale_width_in_scale_db)
    width_out = db_to_lin(p.scale_width_out_scale_db)

    mid, side = mid_side_split(stereo2)
    Sm = librosa.stft(mid, n_fft=p.n_fft, hop_length=p.hop_length, win_length=p.win_length, window="hann")
    Ss = librosa.stft(side, n_fft=p.n_fft, hop_length=p.hop_length, win_length=p.win_length, window="hann")

    mag_s = np.abs(Ss)
    ph_s = np.angle(Ss)

    gain = (width_out + (width_in - width_out) * mask_in).astype(np.float32)
    mag_s2 = mag_s * gain[:, None]
    Ss2 = mag_s2 * np.exp(1j * ph_s)
    side3 = librosa.istft(Ss2, hop_length=p.hop_length, win_length=p.win_length, window="hann", length=length)

    return mid_side_merge(mid, side3).astype(np.float32)


def stabilize_808_sub_glue(stereo: np.ndarray, sr: int, p: MasterParams) -> np.ndarray:
    mid, side = mid_side_split(stereo)
    length = stereo.shape[0]

    S = stft_multi(stereo, p)
    mag = np.abs(S)
    phase = np.angle(S)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=p.n_fft)

    sub = (freqs <= p.sub_glue_cutoff_hz).astype(np.float32)

    Sm = librosa.stft(mid, n_fft=p.n_fft, hop_length=p.hop_length, win_length=p.win_length, window="hann")
    Ss = librosa.stft(side, n_fft=p.n_fft, hop_length=p.hop_length, win_length=p.win_length, window="hann")

    mag_m = np.abs(Sm)
    ph_m = np.angle(Sm)
    mag_s = np.abs(Ss)
    ph_s = np.angle(Ss)

    mag_m_sub = mag_m * sub[:, None]
    mag_s_sub = mag_s * sub[:, None]

    mid_sub = librosa.istft(mag_m_sub * np.exp(1j * ph_m), hop_length=p.hop_length, win_length=p.win_length, window="hann", length=length)
    side_sub = librosa.istft(mag_s_sub * np.exp(1j * ph_s), hop_length=p.hop_length, win_length=p.win_length, window="hann", length=length)

    sub_st = mid_side_merge(mid_sub, side_sub)
    sat = np.tanh(sub_st * db_to_lin(p.sub_sat_drive))
    sub_sat = (1.0 - p.sub_sat_mix) * sub_st + p.sub_sat_mix * sat

    mono_sub = np.mean(sub_sat, axis=1)
    sub_rms = np.sqrt(np.mean(mono_sub**2) + 1e-12)
    thresh = db_to_lin(p.sub_glue_threshold_db)

    if sub_rms > thresh:
        over = sub_rms / max(thresh, 1e-12)
        gr = over ** (-(p.sub_glue_ratio - 1.0) / p.sub_glue_ratio)
    else:
        gr = 1.0

    sub_sat *= float(gr)

    m, s = mid_side_split(sub_sat)
    s *= float(p.sub_mono_strength)
    sub_final = mid_side_merge(m, s)

    S_sub = stft_multi(sub_final, p)
    mag_sub = np.abs(S_sub)
    ph_sub = np.angle(S_sub)

    mag2 = mag.copy()
    for ch in range(2):
        mag2[ch, sub.astype(bool), :] = mag_sub[ch, sub.astype(bool), :]

    S2 = mag2 * np.exp(1j * phase)
    out = istft_multi(S2, p, length=length)
    return out.astype(np.float32)


def side_deesser(stereo: np.ndarray, sr: int, p: MasterParams) -> tuple[np.ndarray, dict]:
    mid, side = mid_side_split(stereo)
    length = len(side)

    S = librosa.stft(side, n_fft=p.n_fft, hop_length=p.hop_length, win_length=p.win_length, window="hann")
    mag = np.abs(S)
    phase = np.angle(S)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=p.n_fft)
    band = (freqs >= p.deess_low_hz) & (freqs <= p.deess_high_hz)

    band_energy = np.sqrt(np.mean(mag[band, :] ** 2, axis=0) + 1e-12)
    band_db = lin_to_db(band_energy)

    over = band_db - p.deess_threshold_db
    over = np.maximum(over, 0.0)

    gr = np.minimum(over, p.deess_max_gr_db)
    gain = db_to_lin(-gr)

    frame_rate = sr / p.hop_length
    aA = np.exp(-1.0 / max(frame_rate * (p.deess_attack_ms / 1000.0), 1e-6))
    aR = np.exp(-1.0 / max(frame_rate * (p.deess_release_ms / 1000.0), 1e-6))

    g_smooth = np.ones_like(gain, dtype=np.float32)
    g = 1.0
    for i in range(len(gain)):
        target = float(gain[i])
        if target < g:
            g = aA * g + (1.0 - aA) * target
        else:
            g = aR * g + (1.0 - aR) * target
        g_smooth[i] = g

    gain = g_smooth

    mag2 = mag.copy()
    mag2[band, :] *= gain[None, :]

    S2 = mag2 * np.exp(1j * phase)
    side2 = librosa.istft(S2, hop_length=p.hop_length, win_length=p.win_length, window="hann", length=length)

    gr_db = -lin_to_db(np.clip(gain, 1e-6, 1.0))
    deess_metrics = {
        "avg_gr_db": float(np.mean(gr_db)),
        "max_gr_db": float(np.max(gr_db)),
        "band_hz": [float(p.deess_low_hz), float(p.deess_high_hz)],
    }
    return mid_side_merge(mid, side2), deess_metrics


def groove_lock_rhythm_enhancer(stereo: np.ndarray, sr: int, p: MasterParams) -> tuple[np.ndarray, dict]:
    if not p.enable_groove_lock:
        return stereo.astype(np.float32), {"enabled": False}

    mono = np.mean(stereo, axis=1).astype(np.float32)

    try:
        tempo, beats = librosa.beat.beat_track(y=mono, sr=sr)
        beats = librosa.frames_to_samples(beats, hop_length=512)
    except Exception:
        return stereo.astype(np.float32), {"enabled": False, "reason": "beat_track_failed"}

    if len(beats) < 4:
        return stereo.astype(np.float32), {"enabled": False, "reason": "too_few_beats"}

    env = np.ones(len(mono), dtype=np.float32)
    strength = float(np.clip(p.groove_lock_strength, 0.0, 1.0))
    gmin = float(np.clip(p.groove_lock_min_gain, 0.75, 1.0))
    sigma = float(max(p.groove_lock_sigma, 0.01))

    for i in range(len(beats) - 1):
        a = int(beats[i])
        b = int(beats[i + 1])
        seg_len = max(b - a, 1)
        ph = np.linspace(0.0, 1.0, seg_len, endpoint=False, dtype=np.float32)
        duck = 1.0 - (1.0 - gmin) * (np.sin(np.pi * ph) ** 2)
        seg = env[a:b]
        env[a:b] = (1.0 - strength) * seg + strength * duck[:len(seg)]

    beat_len = int(np.median(np.diff(beats)))
    k = int(max(5, beat_len * sigma))
    if k % 2 == 0:
        k += 1
    half = k // 2
    x = np.arange(-half, half + 1)
    kernel = np.exp(-0.5 * (x / max(half * 0.35, 1.0)) ** 2).astype(np.float32)
    kernel /= np.sum(kernel) + 1e-12

    env_smooth = np.convolve(env, kernel, mode="same").astype(np.float32)

    out = stereo.copy().astype(np.float32)
    out[:, 0] *= env_smooth
    out[:, 1] *= env_smooth

    tempo_value = float(np.atleast_1d(tempo)[0])
    metrics = {
        "enabled": True,
        "tempo_bpm": tempo_value,
        "groove_strength": float(strength),
        "min_gain": float(gmin),
        "beat_count": int(len(beats)),
    }
    return out, metrics


def pedalboard_final_chain(stereo: np.ndarray, sr: int, p: MasterParams) -> tuple[np.ndarray, dict]:
    if Pedalboard is None:
        return stereo.astype(np.float32), {"enabled": False, "reason": "pedalboard_not_installed"}
    try:
        comp = Compressor(threshold_db=-18.0, ratio=1.35, attack_ms=12.0, release_ms=120.0)
        lim = Limiter(threshold_db=float(p.true_peak_dbfs))
        board = Pedalboard([comp, lim])
        out = board(stereo.T, sr)
        out = out.T.astype(np.float32)
        return out, {"enabled": True}
    except Exception as e:
        return stereo.astype(np.float32), {"enabled": False, "reason": f"pedalboard_failed({e})"}


# -----------------------------
# Output path logic (FIX: respect explicit filename)
# -----------------------------

_LAST_NUMBER_RE = re.compile(r"(/d+)(?!.*/d)")


def auto_versioned_output_path(input_path: str, out_path_or_dir: str, tail_letters: int = 14) -> str:
    in_path = Path(input_path)
    out_path = Path(out_path_or_dir)

    # ✅ If user passed a real file path, respect it
    if out_path.suffix.lower() in {".wav", ".flac", ".aiff", ".aif"}:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not out_path.exists():
            return str(out_path)

        stem = out_path.stem
        for i in range(1, 1000):
            candidate = out_path.with_name(f"{stem}_v{i:03d}{out_path.suffix}")
            if not candidate.exists():
                return str(candidate)
        raise RuntimeError("Too many versioned outputs already exist for this filename.")

    # Otherwise treat as directory
    out_dir = out_path
    out_dir.mkdir(parents=True, exist_ok=True)

    letters_only = "".join(ch for ch in in_path.stem if ch.isalpha())
    name_tail = (letters_only[-tail_letters:] if len(letters_only) > tail_letters else letters_only).rstrip(" .")
    if not name_tail:
        name_tail = "master"

    existing = [p for p in out_dir.glob(f"{name_tail}*.wav")]
    max_n = 0
    for pth in existing:
        m = _LAST_NUMBER_RE.search(pth.stem)
        if m:
            max_n = max(max_n, int(m.group(1)))

    next_n = max_n + 1
    return str(out_dir / f"{name_tail}{next_n:03d}.wav")


# -----------------------------
# Mastering chain
# -----------------------------

def apply_mastering(audio: np.ndarray, sr: int, p: MasterParams, *, preset: str = "Trap Air") -> tuple[np.ndarray, dict | None]:
    stereo = ensure_stereo(audio)

    profiler: StageProfiler | None = None
    if getattr(p, "enable_json_log", False) or getattr(p, "debug_save_stage_wavs", False):
        profiler = StageProfiler(sr=sr, preset=preset)

    def _record(stage_name: str, *, extra: dict | None = None):
        nonlocal stereo
        if profiler is not None:
            profiler.record(stage_name, stereo, sr, p, extra=extra)
        if getattr(p, "debug_save_stage_wavs", False):
            out_dir = Path(getattr(p, "debug_stage_wav_dir", "."))
            out_dir.mkdir(parents=True, exist_ok=True)
            fn = out_dir / f"{_sanitize_filename(stage_name)}.wav"
            sf.write(str(fn), stereo.astype(np.float32), sr)

    _record("00_input")

    groove_metrics: dict = {"enabled": False}
    pb_metrics: dict = {"enabled": False}
    deess_metrics: dict = {"enabled": False}

    if p.enable_mid_side:
        stereo = ms_cone_width_and_slshw(stereo, sr, p)
    _record("01_ms_slshw")

    if p.enable_808_stabilizer:
        stereo = stabilize_808_sub_glue(stereo, sr, p)
    _record("02_808_stabilizer")

    if p.enable_harmonic_resonator:
        stereo = harmonic_resonator(stereo, sr, p)
    _record("03_harmonic_resonator")

    if p.enable_exciter:
        stereo = harmonic_exciter(stereo, sr, p)
    _record("04_exciter")

    if p.enable_psycho_eq:
        stereo = psychoacoustic_weighted_eq(stereo, sr, p)
    _record("05_psycho_eq")

    if p.enable_transient_punch:
        stereo = transient_punch_hpss(stereo, p)
    _record("06_transient_punch")

    stereo, groove_metrics = groove_lock_rhythm_enhancer(stereo, sr, p)
    _record("07_groove_lock", extra=groove_metrics)

    if p.enable_resonance_tamer:
        stereo = resonance_tamer(stereo, sr, p)
    _record("08_resonance_tamer")

    if p.enable_side_deesser:
        stereo, deess_metrics = side_deesser(stereo, sr, p)
        deess_metrics["enabled"] = True
    _record("09_side_deesser", extra=deess_metrics if deess_metrics.get("enabled") else None)

    stereo = apply_loudness_target(stereo, sr, p)
    _record("10_loudness_target")

    stereo, pb_metrics = pedalboard_final_chain(stereo, sr, p)
    _record("11_pedalboard_final_chain", extra=pb_metrics)

    if p.enable_soft_clip and not pb_metrics.get("enabled", False):
        if getattr(p, "enable_aa_soft_clip", True):
            stereo = anti_aliased_soft_clip(stereo, sr, drive=p.soft_clip_drive, oversample=p.oversample_factor)
        else:
            stereo = soft_clip(stereo, drive=p.soft_clip_drive)
    _record("12_soft_clip_fallback")

    if p.enable_true_peak_guard:
        stereo = enforce_true_peak_ceiling(stereo, sr, p)
    _record("13_true_peak_guard_output")

    return stereo, (profiler.to_report() if profiler is not None else None)


# -----------------------------
# File processing
# -----------------------------

def process_file(
    input_path: str,
    output_path: str,
    preset: str = "Trap Air",
    *,
    log_json: str | None = None,
    debug_wavs: bool = False,
    debug_dir: str = "debug_stages",
):
    p = MasterParams(preset=preset)

    output_path_final = auto_versioned_output_path(input_path, output_path)

    # ✅ Best practice: use soxr_hq resampling when supported
    try:
        audio, sr = librosa.load(
            input_path,
            mono=False,
            sr=48000,
            res_type="soxr_hq",
        )
    except TypeError:
        audio, sr = librosa.load(input_path, mono=False, sr=48000)

    stereo = ensure_stereo(audio)

    p.enable_json_log = bool(log_json)
    p.debug_save_stage_wavs = bool(debug_wavs)
    p.debug_stage_wav_dir = str(debug_dir)

    mastered, report = apply_mastering(stereo, sr, p, preset=preset)
    sf.write(output_path_final, mastered, sr)

    print(f"Mastering complete. Preset={preset}  SR={sr}")
    print(f"Approx LUFS={measure_lufs(mastered, sr):.2f}")
    print(f"TruePeak(dBFS)~{true_peak_dbfs(mastered, sr, oversample=p.oversample_factor):.2f}")
    print(f"Output: {output_path_final}")

    if log_json and report is not None:
        outp = Path(log_json)
        if outp.suffix.lower() != ".json":
            outp.mkdir(parents=True, exist_ok=True)
            outp = outp / f"{Path(output_path_final).stem}_stage_log.json"
        else:
            outp.parent.mkdir(parents=True, exist_ok=True)

        outp.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Stage log: {outp}")

    if debug_wavs:
        print(f"Stage WAVs: {Path(debug_dir).resolve()}")


def build_cli():
    ap = argparse.ArgumentParser(description="Melodic Trap Mastering Suite v7.1 FIXED")
    ap.add_argument("--in", dest="inp", required=True, help="Input audio path (wav/mp3)")
    ap.add_argument("--out", dest="out", required=True, help="Output dir OR output file path .wav")
    ap.add_argument("--preset", default="Trap Air", choices=["Trap Air", "Trap Soul"])
    ap.add_argument("--log_json", default=LOG, help="Write stage-by-stage JSON metrics log to this file or directory")
    ap.add_argument("--debug_wavs", action="store_true", help="Export intermediate stage WAVs")
    ap.add_argument("--debug_dir", default="debug_stages", help="Directory for intermediate stage WAV exports")
    return ap


if __name__ == "__main__":
    args = build_cli().parse_args()
    process_file(
        args.inp,
        args.out,
        preset=args.preset,
        log_json=args.log_json,
        debug_wavs=args.debug_wavs,
        debug_dir=args.debug_dir,
    )
