from __future__ import annotations

import numpy as np


def ensure_stereo(audio: np.ndarray) -> np.ndarray:
    """Return audio as shape (n_samples, 2)."""
    audio = np.asarray(audio)
    if audio.ndim == 1:
        return np.stack([audio, audio], axis=-1)
    if audio.shape[0] == 2 and audio.shape[1] > 2:
        return audio.T
    if audio.shape[1] == 2:
        return audio
    raise ValueError(f"Unexpected audio shape: {audio.shape}")


def db_to_lin(db: float | np.ndarray) -> np.ndarray:
    return np.asarray(10.0 ** (np.asarray(db) / 20.0))


def lin_to_db(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(x, eps))


def mid_side_split(stereo: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = stereo[:, 0]
    right = stereo[:, 1]
    mid = 0.5 * (left + right)
    side = 0.5 * (left - right)
    return mid, side


def mid_side_merge(mid: np.ndarray, side: np.ndarray) -> np.ndarray:
    left = mid + side
    right = mid - side
    return np.stack([left, right], axis=-1)


def smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def rms(x: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.sqrt(np.mean(x * x) + eps))


def peak_dbfs(x: np.ndarray, eps: float = 1e-12) -> float:
    peak = float(np.max(np.abs(x)) + eps)
    return float(20.0 * np.log10(peak))
