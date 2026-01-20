from __future__ import annotations

import numpy as np
import librosa


def estimate_scale_pitch_classes(stereo: np.ndarray, sr: int) -> set[int]:
    """Return 7 most active pitch classes as a scale proxy."""
    mono = np.mean(stereo, axis=1)
    chroma = librosa.feature.chroma_stft(y=mono, sr=sr)
    pc_energy = chroma.mean(axis=1)
    top = np.argsort(pc_energy)[-7:]
    return set(int(x) for x in top)


def estimate_key_pitch_class(stereo: np.ndarray, sr: int) -> int:
    """Return dominant pitch class as key center estimate."""
    mono = np.mean(stereo, axis=1)
    chroma = librosa.feature.chroma_stft(y=mono, sr=sr)
    return int(np.argmax(chroma.mean(axis=1)))


def pitch_class_for_freq(freqs: np.ndarray, reference_hz: float = 440.0) -> np.ndarray:
    """Map frequencies to pitch classes using 12-TET around A4=reference_hz."""
    pcs = np.full_like(freqs, fill_value=-1, dtype=int)
    mask = freqs > 0
    midi = 69.0 + 12.0 * np.log2(freqs[mask] / reference_hz)
    midi_round = np.round(midi).astype(int)
    pcs[mask] = midi_round % 12
    return pcs
