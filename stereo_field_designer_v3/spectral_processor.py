from __future__ import annotations

import numpy as np
import librosa

from .dsp_utils import db_to_lin, mid_side_merge, mid_side_split, smoothstep


class FFTHandler:
    """STFT/I-STFT wrapper to keep FFT settings centralized."""

    def __init__(self, n_fft: int = 2048, hop_length: int = 512, win_length: int = 2048):
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)

    def stft(self, signal: np.ndarray) -> np.ndarray:
        return librosa.stft(
            signal,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window="hann",
        )

    def istft(self, spectrum: np.ndarray, length: int) -> np.ndarray:
        return librosa.istft(
            spectrum,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window="hann",
            length=length,
        )

    def freqs(self, sr: int) -> np.ndarray:
        return librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)


class PsychoacousticModel:
    """Ear-centric weighting curves for widening and HRTF color."""

    @staticmethod
    def bark_scale(freqs: np.ndarray) -> np.ndarray:
        return 13.0 * np.arctan(0.00076 * freqs) + 3.5 * np.arctan((freqs / 7500.0) ** 2)

    def ear_weight(self, freqs: np.ndarray) -> np.ndarray:
        bark = self.bark_scale(freqs)
        # Emphasize 2-6 kHz where spatial cues are most audible.
        target = 14.0
        width = 7.0
        weight = np.exp(-0.5 * ((bark - target) / width) ** 2)
        return weight / (np.max(weight) + 1e-9)

    def hrtf_ild(self, freqs: np.ndarray, amount: float) -> np.ndarray:
        # ILD rises with frequency; keep it subtle for mono safety.
        curve = (freqs / (freqs + 1200.0)) ** 0.7
        ild_db = amount * 2.5 * curve
        return db_to_lin(ild_db)


class PhaseManipulator:
    """Phase rotations for psychoacoustic width with mono-safe angles."""

    @staticmethod
    def rotate_phase(spectrum: np.ndarray, phase_curve: np.ndarray) -> np.ndarray:
        # Psychoacoustic note: tiny frequency-dependent phase rotations can
        # widen perception while avoiding mono collapse if kept under ~20 deg.
        return spectrum * np.exp(1j * phase_curve[:, None])


class SpectralProcessor:
    """Spectral-domain psychoacoustic widening + HRTF tinting."""

    def __init__(self, fft: FFTHandler, model: PsychoacousticModel | None = None):
        self.fft = fft
        self.model = model or PsychoacousticModel()
        self.phase = PhaseManipulator()

    def _mono_mask(self, freqs: np.ndarray, mono_hz: float, slope_hz: float) -> np.ndarray:
        lo = max(mono_hz - slope_hz, 10.0)
        hi = mono_hz + slope_hz
        x = (freqs - lo) / max(hi - lo, 1e-6)
        return smoothstep(x)

    def psychoacoustic_widen(
        self,
        stereo: np.ndarray,
        sr: int,
        strength: float,
        phase_degrees: float,
        hrtf_amount: float,
        mono_guard_hz: float,
        mono_guard_slope_hz: float,
    ) -> np.ndarray:
        mid, side = mid_side_split(stereo)
        length = len(side)

        spectrum = self.fft.stft(side)
        mag = np.abs(spectrum)
        phase = np.angle(spectrum)
        freqs = self.fft.freqs(sr)

        ear_weight = self.model.ear_weight(freqs)
        phase_curve = np.deg2rad(phase_degrees) * ear_weight * strength
        # Phase-stability note: phase rotation is applied to SIDE only, and
        # low-frequencies remain mono to preserve punch in collapse.
        rotated = mag * np.exp(1j * (phase + phase_curve[:, None]))

        hrtf_gain = self.model.hrtf_ild(freqs, amount=hrtf_amount)
        widened = rotated * hrtf_gain[:, None]

        if mono_guard_hz > 0.0:
            mask = self._mono_mask(freqs, mono_guard_hz, mono_guard_slope_hz)
            widened = widened * mask[:, None]

        side_out = self.fft.istft(widened, length=length)
        return mid_side_merge(mid, side_out)
