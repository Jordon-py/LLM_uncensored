from __future__ import annotations

import numpy as np
import librosa

from .dsp_utils import db_to_lin, lin_to_db, mid_side_merge, mid_side_split, smoothstep
from .spectral_processor import FFTHandler
from .system_utils import GPUManager
from .music_theory import pitch_class_for_freq


class AdaptiveWidthCurve:
    """Adaptive width based on spectral energy distribution."""

    def compute_db(self, mono: np.ndarray, sr: int, air: float, warmth: float) -> float:
        centroid = librosa.feature.spectral_centroid(y=mono, sr=sr).mean()
        tilt = np.clip((centroid - 500.0) / 4500.0, 0.0, 1.0)
        # Air leans wider on bright material; warmth nudges width inward.
        return float((air - warmth) * 1.2 * (tilt - 0.5))


class TemporalModulator:
    """Tempo-synced width motion for rhythmic stereo animation."""

    def estimate_tempo(self, mono: np.ndarray, sr: int, start_bpm=123) -> float:
        tempo, _ = librosa.beat.beat_track(y=mono, sr=sr, start_bpm=start_bpm)
        tempo_val = float(np.atleast_1d(tempo)[0])
        return tempo_val if tempo_val > 0 else float(start_bpm)

    def modulation_curve(self, n_frames: int, sr: int, hop_length: int, tempo: float, depth: float) -> np.ndarray:
        seconds_per_frame = hop_length / sr
        t = np.arange(n_frames) * seconds_per_frame
        # Tempo-synced sine LFO; depth controls stereo motion intensity.
        phase = 2.0 * np.pi * (tempo / 60.0) * t
        lfo = 0.5 + 0.5 * np.sin(phase)
        curve = 1.0 + depth * (lfo - 0.5) * 2.0
        return curve.astype(np.float32)


class NeuralMorpher:
    """Neural-inspired spectral morphing (optional Torch backend)."""

    def __init__(self, bands: int = 64, hidden: int = 32, gpu: GPUManager | None = None, seed: int = 7):
        self.bands = int(bands)
        self.hidden = int(hidden)
        self.gpu = gpu or GPUManager(preferred="auto")
        rng = np.random.RandomState(seed)
        self.w1 = rng.randn(self.hidden, self.bands).astype(np.float32) * 0.15
        self.b1 = rng.randn(self.hidden).astype(np.float32) * 0.05
        self.w2 = rng.randn(self.bands, self.hidden).astype(np.float32) * 0.15
        self.b2 = rng.randn(self.bands).astype(np.float32) * 0.05

        if self.gpu.torch is not None:
            torch = self.gpu.torch
            device = self.gpu.torch_device
            self.w1_t = torch.tensor(self.w1, device=device)
            self.b1_t = torch.tensor(self.b1, device=device)
            self.w2_t = torch.tensor(self.w2, device=device)
            self.b2_t = torch.tensor(self.b2, device=device)
        elif self.gpu.backend == "cupy":
            xp = self.gpu.xp
            self.w1_c = xp.asarray(self.w1)
            self.b1_c = xp.asarray(self.b1)
            self.w2_c = xp.asarray(self.w2)
            self.b2_c = xp.asarray(self.b2)

    def load_weights(self, path: str) -> None:
        """Load custom morph weights from a .npz file (w1, b1, w2, b2)."""
        data = np.load(path)
        self.w1 = data["w1"].astype(np.float32)
        self.b1 = data["b1"].astype(np.float32)
        self.w2 = data["w2"].astype(np.float32)
        self.b2 = data["b2"].astype(np.float32)

        if self.gpu.torch is not None:
            torch = self.gpu.torch
            device = self.gpu.torch_device
            self.w1_t = torch.tensor(self.w1, device=device)
            self.b1_t = torch.tensor(self.b1, device=device)
            self.w2_t = torch.tensor(self.w2, device=device)
            self.b2_t = torch.tensor(self.b2, device=device)
        elif self.gpu.backend == "cupy":
            xp = self.gpu.xp
            self.w1_c = xp.asarray(self.w1)
            self.b1_c = xp.asarray(self.b1)
            self.w2_c = xp.asarray(self.w2)
            self.b2_c = xp.asarray(self.b2)

    def morph_curve(self, mag: np.ndarray, strength: float) -> np.ndarray:
        mag_mean = np.mean(mag, axis=1)
        x = np.log1p(mag_mean)

        band_axis = np.linspace(0.0, 1.0, x.size)
        target_axis = np.linspace(0.0, 1.0, self.bands)
        x_bands = np.interp(target_axis, band_axis, x).astype(np.float32)

        if self.gpu.torch is not None:
            torch = self.gpu.torch
            x_t = torch.tensor(x_bands, device=self.gpu.torch_device)
            h = torch.tanh(self.w1_t @ x_t + self.b1_t)
            y = torch.tanh(self.w2_t @ h + self.b2_t)
            y_np = y.detach().cpu().numpy()
        elif self.gpu.backend == "cupy":
            xp = self.gpu.xp
            x_c = xp.asarray(x_bands)
            h = xp.tanh(self.w1_c @ x_c + self.b1_c)
            y = xp.tanh(self.w2_c @ h + self.b2_c)
            y_np = xp.asnumpy(y)
        else:
            h = np.tanh(self.w1 @ x_bands + self.b1)
            y_np = np.tanh(self.w2 @ h + self.b2)

        curve = np.interp(band_axis, target_axis, y_np)
        # Keep morph gentle to avoid phasey artifacts in mono.
        return np.clip(1.0 + strength * 0.4 * curve, 0.7, 1.3)


class WidthController:
    """Mid/Side width control with adaptive, neural, and tempo-synced layers."""

    def __init__(
        self,
        fft: FFTHandler,
        modulator: TemporalModulator | None = None,
        adaptive: AdaptiveWidthCurve | None = None,
        morpher: NeuralMorpher | None = None,
    ):
        self.fft = fft
        self.modulator = modulator or TemporalModulator()
        self.adaptive = adaptive or AdaptiveWidthCurve()
        self.morpher = morpher or NeuralMorpher()

    def _mono_mask(self, freqs: np.ndarray, mono_hz: float, slope_hz: float) -> np.ndarray:
        lo = max(mono_hz - slope_hz, 10.0)
        hi = mono_hz + slope_hz
        x = (freqs - lo) / max(hi - lo, 1e-6)
        return smoothstep(x)

    def _coherence_gain(
        self,
        stereo: np.ndarray,
        n_frames: int,
        frame_length: int,
        hop_length: int,
        threshold: float,
        smoothing: float = 0.6,
    ) -> np.ndarray:
        left = stereo[:, 0]
        right = stereo[:, 1]
        if left.size < frame_length:
            return np.ones(n_frames, dtype=np.float32)
        frames_l = librosa.util.frame(left, frame_length=frame_length, hop_length=hop_length).T
        frames_r = librosa.util.frame(right, frame_length=frame_length, hop_length=hop_length).T

        denom = (np.linalg.norm(frames_l, axis=1) * np.linalg.norm(frames_r, axis=1)) + 1e-9
        corr = np.sum(frames_l * frames_r, axis=1) / denom

        corr = corr[:n_frames] if corr.size >= n_frames else np.pad(corr, (0, n_frames - corr.size), mode="edge")
        gain = np.ones_like(corr)
        bad = corr < threshold
        gain[bad] = np.clip((corr[bad] + 1.0) / (threshold + 1.0), 0.2, 1.0)

        # One-pole smoothing to prevent pumping on rapid correlation swings.
        for i in range(1, gain.size):
            gain[i] = smoothing * gain[i - 1] + (1.0 - smoothing) * gain[i]
        return gain

    def process(
        self,
        stereo: np.ndarray,
        sr: int,
        width_db: float,
        motion_depth: float,
        morph_strength: float,
        air_strength: float,
        warmth_strength: float,
        mono_guard_hz: float,
        mono_guard_slope_hz: float,
        correlation_guard: float,
        tempo: float | None = None,
        enable_scale_lock: bool = True,
        scale_pcs: set[int] | None = None,
        scale_in_db: float = 0.6,
        scale_out_db: float = -0.35,
        scale_band_low_hz: float = 180.0,
        scale_band_high_hz: float = 10000.0,
    ) -> np.ndarray:
        mid, side = mid_side_split(stereo)
        length = len(side)

        spectrum = self.fft.stft(side)
        mag = np.abs(spectrum)
        phase = np.angle(spectrum)
        freqs = self.fft.freqs(sr)

        if mono_guard_hz > 0.0:
            mask = self._mono_mask(freqs, mono_guard_hz, mono_guard_slope_hz)
            mag = mag * mask[:, None]

        base_width = db_to_lin(width_db + self.adaptive.compute_db(np.mean(stereo, axis=1), sr, air_strength, warmth_strength))
        mag = mag * base_width

        morph_curve = self.morpher.morph_curve(mag, morph_strength)
        mag = mag * morph_curve[:, None]

        if tempo is None:
            tempo = self.modulator.estimate_tempo(np.mean(stereo, axis=1), sr)
        motion = self.modulator.modulation_curve(mag.shape[1], sr, self.fft.hop_length, tempo, motion_depth)
        mag = mag * motion[None, :]

        if enable_scale_lock and scale_pcs:
            pcs = pitch_class_for_freq(freqs)
            band = (freqs >= scale_band_low_hz) & (freqs <= scale_band_high_hz) & (pcs >= 0)
            scale_array = np.array(sorted(scale_pcs), dtype=int)
            in_scale = np.isin(pcs, scale_array)
            gain_db = np.zeros_like(freqs, dtype=np.float32)
            gain_db[band & in_scale] = scale_in_db
            gain_db[band & (~in_scale)] = scale_out_db
            gain_db = np.convolve(gain_db, np.array([0.2, 0.6, 0.2], dtype=np.float32), mode="same")
            mag = mag * db_to_lin(gain_db)[:, None]

        coherence_gain = self._coherence_gain(
            stereo,
            n_frames=mag.shape[1],
            frame_length=self.fft.win_length,
            hop_length=self.fft.hop_length,
            threshold=correlation_guard,
        )
        # Phase-stability note: correlation guard reduces side energy when
        # correlation dips, preventing harsh collapses in mono playback.
        mag = mag * coherence_gain[None, :]

        side_out = self.fft.istft(mag * np.exp(1j * phase), length=length)
        return mid_side_merge(mid, side_out)
