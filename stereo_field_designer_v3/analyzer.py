from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import librosa

from .dsp_utils import rms, mid_side_split
from .metrics_logger import MasteringMetricsLogger, PeakRiskLogger, TransientStressLogger

try:
    import pyloudnorm as pyln  # type: ignore
except Exception:
    pyln = None


@dataclass
class AnalysisResult:
    lufs: float
    correlation: float
    mid_rms: float
    side_rms: float
    width_index: float
    spectral_centroid: float
    stereo_emotion_index: float


class CorrelationMeter:
    """Pearson correlation for stereo coherence."""

    def measure(self, stereo: np.ndarray) -> float:
        left = stereo[:, 0]
        right = stereo[:, 1]
        denom = (np.linalg.norm(left) * np.linalg.norm(right)) + 1e-9
        return float(np.sum(left * right) / denom)


class EnergyBalancer:
    """Measure mid/side energy and a width index."""

    def measure(self, stereo: np.ndarray) -> tuple[float, float, float]:
        mid, side = mid_side_split(stereo)
        mid_rms = rms(mid)
        side_rms = rms(side)
        width_index = float(side_rms / (mid_rms + 1e-9))
        return mid_rms, side_rms, width_index


class FeatureExtractor:
    """Extracts loudness and spectral descriptors."""

    def spectral_centroid(self, mono: np.ndarray, sr: int) -> float:
        centroid = librosa.feature.spectral_centroid(y=mono, sr=sr).mean()
        return float(centroid)

    def lufs(self, mono: np.ndarray, sr: int) -> float:
        if pyln is not None:
            meter = pyln.Meter(sr)
            return float(meter.integrated_loudness(mono))
        rms_val = rms(mono)
        return float(20.0 * np.log10(max(rms_val, 1e-9)))


class Analyzer:
    """Analyzer for stereo integrity and musical context cues."""

    def __init__(self, log_path: str | None = None):
        self.correlation_meter = CorrelationMeter()
        self.energy_balancer = EnergyBalancer()
        self.features = FeatureExtractor()
        self.logger = MasteringMetricsLogger(log_path=log_path or "mastering_log.json")
        self.peak_logger = PeakRiskLogger(log_path="peak_risk_log.json")
        self.transient_logger = TransientStressLogger(log_path="transient_stress_log.json")

    def stereo_emotion_index(self, width_index: float, centroid: float, correlation: float) -> float:
        width = np.clip(width_index / 1.2, 0.0, 1.0)
        air = np.clip((centroid - 500.0) / 5000.0, 0.0, 1.0)
        focus = np.clip((correlation + 1.0) / 2.0, 0.0, 1.0)
        return float(0.4 * width + 0.3 * air + 0.3 * focus)

    def analyze(self, stereo: np.ndarray, sr: int) -> AnalysisResult:
        mono = np.mean(stereo, axis=1)
        correlation = self.correlation_meter.measure(stereo)
        mid_rms, side_rms, width_index = self.energy_balancer.measure(stereo)
        centroid = self.features.spectral_centroid(mono, sr)
        lufs = self.features.lufs(mono, sr)
        emotion = self.stereo_emotion_index(width_index, centroid, correlation)
        return AnalysisResult(
            lufs=lufs,
            correlation=correlation,
            mid_rms=mid_rms,
            side_rms=side_rms,
            width_index=width_index,
            spectral_centroid=centroid,
            stereo_emotion_index=emotion,
        )

    def log_metrics(self, stereo: np.ndarray, sr: int, name: str = "render") -> None:
        if self.logger:
            self.logger.analyze(stereo, sr, name=name)
        if self.peak_logger:
            self.peak_logger.analyze(stereo, sr, name=name)
        if self.transient_logger:
            self.transient_logger.analyze(stereo, sr, name=name)
