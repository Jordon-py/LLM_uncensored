from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import librosa

try:
    import pyloudnorm as pyln  # type: ignore
except Exception:
    pyln = None

try:
    from scipy.signal import resample_poly
except Exception:
    resample_poly = None

from .dsp_utils import ensure_stereo
from .music_theory import estimate_key_pitch_class

KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _db_to_lin(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def _true_peak(stereo: np.ndarray, oversample: int = 4) -> float:
    peak = float(np.max(np.abs(stereo)) + 1e-9)
    if oversample <= 1:
        return peak
    if resample_poly is None:
        return peak
    try:
        os = resample_poly(stereo, oversample, 1, axis=0)
    except TypeError:
        left = resample_poly(stereo[:, 0], oversample, 1)
        right = resample_poly(stereo[:, 1], oversample, 1)
        os = np.stack([left, right], axis=-1)
    return float(np.max(np.abs(os)) + 1e-9)


class MasteringMetricsLogger:
    """Collects and stores mastering metrics after each render."""

    def __init__(self, log_path: str | Path = "mastering_log.json"):
        self.log_path = Path(log_path)
        self.logs = self._load()

    def _load(self) -> list[dict]:
        if self.log_path.exists():
            try:
                return json.loads(self.log_path.read_text(encoding="utf-8"))
            except Exception:
                return []
        return []

    def _write(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text(json.dumps(self.logs, indent=2), encoding="utf-8")

    def _infer_mode(self, mono: np.ndarray, sr: int) -> str:
        chroma = librosa.feature.chroma_stft(y=mono, sr=sr)
        profile = chroma.mean(axis=1)
        major = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], dtype=float)
        minor = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0], dtype=float)
        major_score = float(np.dot(profile, major))
        minor_score = float(np.dot(profile, minor))
        return "Minor" if minor_score > major_score else "Major"

    def _true_peak(self, stereo: np.ndarray, oversample: int = 4) -> float:
        return _true_peak(stereo, oversample=oversample)

    def analyze(self, audio: np.ndarray, sr: int, name: str = "render") -> None:
        stereo = ensure_stereo(audio)
        mono = np.mean(stereo, axis=1)
        meter = pyln.Meter(sr) if pyln is not None else None

        loudness = float(meter.integrated_loudness(mono)) if meter else float(20.0 * np.log10(np.sqrt(np.mean(mono**2)) + 1e-9))
        peak = float(np.max(np.abs(stereo)))
        true_peak = self._true_peak(stereo, oversample=4)
        rms = float(np.sqrt(np.mean(mono**2)))
        crest = float(peak / rms) if rms > 0 else 0.0
        tempo, _ = librosa.beat.beat_track(y=mono, sr=sr)
        tempo_val = float(np.atleast_1d(tempo)[0]) if tempo is not None else 0.0
        corr = float(np.corrcoef(stereo[:, 0], stereo[:, 1])[0, 1]) if stereo.ndim == 2 else 1.0
        key_pc = estimate_key_pitch_class(stereo, sr)
        mode = self._infer_mode(mono, sr)
        key_name = f"{KEY_NAMES[key_pc]} {mode}"

        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "render_name": name,
            "metrics": {
                "peak_dbfs": float(20.0 * np.log10(peak + 1e-9)),
                "true_peak_dbfs": float(20.0 * np.log10(true_peak + 1e-9)),
                "integrated_lufs": loudness,
                "rms_db": float(20.0 * np.log10(rms + 1e-9)),
                "crest_factor": crest,
                "tempo_bpm": tempo_val,
                "key_estimate": key_name,
                "stereo_correlation": corr,
            },
        }
        self.logs.append(entry)
        self._write()


class PeakRiskLogger:
    """Logs peak/clip events to help locate crackle risk."""

    def __init__(self, log_path: str | Path = "peak_risk_log.json"):
        self.log_path = Path(log_path)
        self.logs = self._load()

    def _load(self) -> list[dict]:
        if self.log_path.exists():
            try:
                return json.loads(self.log_path.read_text(encoding="utf-8"))
            except Exception:
                return []
        return []

    def _write(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text(json.dumps(self.logs, indent=2), encoding="utf-8")

    def analyze(self, audio: np.ndarray, sr: int, name: str = "render") -> None:
        stereo = ensure_stereo(audio)
        peak = float(np.max(np.abs(stereo)) + 1e-9)
        true_peak = _true_peak(stereo, oversample=4)

        thresh_hot = _db_to_lin(-0.1)
        thresh_clip = _db_to_lin(0.0)
        mags = np.max(np.abs(stereo), axis=1)
        hot_idx = np.where(mags >= thresh_hot)[0]
        clip_idx = np.where(mags >= thresh_clip)[0]

        hot_times = (hot_idx[:8] / sr).tolist()
        clip_times = (clip_idx[:8] / sr).tolist()

        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "render_name": name,
            "metrics": {
                "sample_peak_dbfs": float(20.0 * np.log10(peak)),
                "true_peak_dbfs": float(20.0 * np.log10(true_peak)),
                "hot_sample_count": int(hot_idx.size),
                "clip_sample_count": int(clip_idx.size),
                "hot_sample_times_sec": hot_times,
                "clip_sample_times_sec": clip_times,
            },
        }
        self.logs.append(entry)
        self._write()


class TransientStressLogger:
    """Logs transient stress via spectral flux stats."""

    def __init__(self, log_path: str | Path = "transient_stress_log.json"):
        self.log_path = Path(log_path)
        self.logs = self._load()

    def _load(self) -> list[dict]:
        if self.log_path.exists():
            try:
                return json.loads(self.log_path.read_text(encoding="utf-8"))
            except Exception:
                return []
        return []

    def _write(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text(json.dumps(self.logs, indent=2), encoding="utf-8")

    def analyze(self, audio: np.ndarray, sr: int, name: str = "render") -> None:
        stereo = ensure_stereo(audio)
        mono = np.mean(stereo, axis=1)
        flux = librosa.onset.onset_strength(y=mono, sr=sr)
        flux_mean = float(np.mean(flux)) if flux.size else 0.0
        flux_p95 = float(np.percentile(flux, 95)) if flux.size else 0.0
        flux_max = float(np.max(flux)) if flux.size else 0.0

        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "render_name": name,
            "metrics": {
                "flux_mean": flux_mean,
                "flux_p95": flux_p95,
                "flux_max": flux_max,
            },
        }
        self.logs.append(entry)
        self._write()
