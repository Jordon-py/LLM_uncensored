from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .dsp_utils import peak_dbfs, rms


DEFAULT_PRESETS: dict[str, dict[str, Any]] = {
    "Wide Dream": {
        "width_db": 2.2,
        "motion_depth": 0.25,
        "morph_strength": 0.4,
        "air_strength": 0.6,
        "warmth_strength": 0.45,
        "mono_guard_hz": 130.0,
        "mono_guard_slope_hz": 80.0,
        "target_lufs": -13.0,
        "soft_clip_drive": 1.2,
        "soft_clip_mix": 0.7,
    },
    "Club Energy": {
        "width_db": 1.6,
        "motion_depth": 0.4,
        "morph_strength": 0.55,
        "air_strength": 0.7,
        "warmth_strength": 0.35,
        "mono_guard_hz": 140.0,
        "mono_guard_slope_hz": 90.0,
        "soft_clip_drive": 1.2,
        "soft_clip_mix": 0.7,
    },
    "Lo-Fi Warmth": {
        "width_db": 0.8,
        "motion_depth": 0.2,
        "morph_strength": 0.25,
        "air_strength": 0.35,
        "warmth_strength": 0.7,
        "mono_guard_hz": 150.0,
        "mono_guard_slope_hz": 110.0,
        "soft_clip_drive": 1.2,
        "soft_clip_mix": 0.7,
    },
    "Neon Skyline": {
        "width_db": 2.6,
        "motion_depth": 0.4,
        "morph_strength": 0.6,
        "air_strength": 0.75,
        "warmth_strength": 0.3,
        "mono_guard_hz": 120.0,
        "mono_guard_slope_hz": 70.0,
        "psycho_strength": 0.8,
        "phase_degrees": 14.0,
        "hrtf_amount": 0.45,
        "correlation_guard": 0.2,
        "target_lufs": -10.5,
        "soft_clip_drive": 1.2,
        "soft_clip_mix": 0.7,
    },
    "Hi Fidelity": {
        "width_db": 1.8,
        "motion_depth": 0.2,
        "morph_strength": 0.35,
        "air_strength": 0.55,
        "warmth_strength": 0.5,
        "mono_guard_hz": 140.0,
        "mono_guard_slope_hz": 90.0,
        "psycho_strength": 0.6,
        "phase_degrees": 10.0,
        "hrtf_amount": 0.3,
        "correlation_guard": 0.2,
        "target_lufs": -13.0,
        "soft_clip_drive": 1.15,
        "soft_clip_mix": 0.55,
        "crest_ratio": 1.4,
        "crest_threshold_dbfs": -17.0,
    },
}


class ConfigManager:
    """Load/save presets and configs for Stereo Field Designer v3."""

    def __init__(self, config_path: str | Path | None = None, presets_path: str | Path | None = None):
        root = Path(__file__).resolve().parent
        self.config_path = Path(config_path) if config_path else root / "config.json"
        self.presets_path = Path(presets_path) if presets_path else root / "presets.json"

    def load_presets(self) -> dict[str, dict[str, Any]]:
        if self.presets_path.exists():
            presets = json.loads(self.presets_path.read_text(encoding="utf-8"))
            merged = dict(presets)
            for name, values in DEFAULT_PRESETS.items():
                merged.setdefault(name, values)
            if merged != presets:
                self.presets_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
            return merged
        self.presets_path.write_text(json.dumps(DEFAULT_PRESETS, indent=2), encoding="utf-8")
        return dict(DEFAULT_PRESETS)

    def save_presets(self, presets: dict[str, dict[str, Any]]) -> None:
        self.presets_path.write_text(json.dumps(presets, indent=2), encoding="utf-8")

    def list_presets(self) -> list[str]:
        return sorted(self.load_presets().keys())

    def get_preset(self, name: str) -> dict[str, Any]:
        presets = self.load_presets()
        return dict(presets.get(name, DEFAULT_PRESETS["Wide Dream"]))


class GPUManager:
    """Detect and route optional GPU backends (CuPy for arrays, Torch for morphing)."""

    def __init__(self, preferred: str = "auto"):
        self.backend = "numpy"
        self.xp = np
        self.torch = None
        self.torch_device = None
        self._init_backends(preferred)

    def _init_backends(self, preferred: str) -> None:
        preferred = preferred.lower()
        if preferred in {"cupy", "auto"}:
            try:
                import cupy  # type: ignore

                self.backend = "cupy"
                self.xp = cupy
                return
            except Exception:
                pass
        self.backend = "numpy"
        self.xp = np
        if preferred in {"torch", "auto"}:
            try:
                import torch  # type: ignore

                self.torch = torch
                if torch.cuda.is_available():
                    self.torch_device = torch.device("cuda")
                else:
                    self.torch_device = torch.device("cpu")
            except Exception:
                self.torch = None
                self.torch_device = None

    def to_numpy(self, arr):
        if self.backend == "cupy":
            return self.xp.asnumpy(arr)
        return np.asarray(arr)

    def array(self, data, dtype=None):
        return self.xp.asarray(data, dtype=dtype)


@dataclass
class TestResult:
    ok: bool
    message: str


class TestSuite:
    """Lightweight stability tests to keep stereo processing safe."""

    @staticmethod
    def generate_example(sr: int = 48000, seconds: float = 2.0) -> np.ndarray:
        t = np.linspace(0.0, seconds, int(sr * seconds), endpoint=False)
        left = 0.5 * np.sin(2.0 * np.pi * 220.0 * t) + 0.2 * np.sin(2.0 * np.pi * 880.0 * t)
        right = 0.5 * np.sin(2.0 * np.pi * 220.0 * t + 0.15) + 0.2 * np.sin(2.0 * np.pi * 880.0 * t)
        noise = 0.02 * np.random.randn(t.size)
        return np.stack([left + noise, right + noise], axis=-1).astype(np.float32)

    @staticmethod
    def assert_stable(stereo: np.ndarray, peak_limit_dbfs: float = 3.0) -> TestResult:
        if not np.isfinite(stereo).all():
            return TestResult(False, "Non-finite samples detected.")
        peak = peak_dbfs(stereo)
        if peak > peak_limit_dbfs:
            return TestResult(False, f"Peak exceeds {peak_limit_dbfs:.1f} dBFS ({peak:.2f}).")
        return TestResult(True, "Signal is finite and within expected peak range.")

    @staticmethod
    def summarize_energy(stereo: np.ndarray) -> str:
        left_rms = rms(stereo[:, 0])
        right_rms = rms(stereo[:, 1])
        return f"RMS L={left_rms:.4f} R={right_rms:.4f}"
