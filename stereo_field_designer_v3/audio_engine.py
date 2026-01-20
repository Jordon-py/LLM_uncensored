from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf

try:
    from scipy.signal import resample_poly
except Exception:
    resample_poly = None

from .analyzer import AnalysisResult, Analyzer
from .dsp_utils import db_to_lin, ensure_stereo, lin_to_db, mid_side_merge, mid_side_split
from .music_theory import estimate_key_pitch_class, estimate_scale_pitch_classes, pitch_class_for_freq
from .spectral_processor import FFTHandler, PsychoacousticModel, SpectralProcessor
from .system_utils import GPUManager
from .width_controller import NeuralMorpher, TemporalModulator, WidthController


@dataclass
class StereoFieldConfig:
    sample_rate: int = 48000
    n_fft: int = 2048
    hop_length: int = 512
    win_length: int = 2048

    width_db: float = 1.6
    motion_depth: float = 0.25
    morph_strength: float = 0.4
    air_strength: float = 0.6
    warmth_strength: float = 0.45

    psycho_strength: float = 0.7
    phase_degrees: float = 12.0
    hrtf_amount: float = 0.35

    mono_guard_hz: float = 120.0
    mono_guard_slope_hz: float = 70.0
    correlation_guard: float = 0.15
    enable_scale_locked_width: bool = True
    scale_width_in_db: float = 0.6
    scale_width_out_db: float = -0.35
    scale_band_low_hz: float = 180.0
    scale_band_high_hz: float = 10000.0
    enable_tonal_anchor: bool = True
    anchor_gain_db: float = 1.2
    anchor_third_gain_db: float = 0.6
    anchor_band_low_hz: float = 180.0
    anchor_band_high_hz: float = 4500.0

    enable_psycho_widen: bool = True
    enable_mode_tint: bool = True

    target_lufs: float = -12.0
    max_lufs_gain_db: float = 12.0
    loudness_headroom_db: float = 1.0
    bypass_processing: bool = False
    enable_crest_compressor: bool = True
    crest_threshold_dbfs: float = -18.0
    crest_ratio: float = 1.6
    crest_attack_ms: float = 8.0
    crest_release_ms: float = 140.0
    crest_knee_db: float = 4.0
    crest_window_ms: float = 30.0
    crest_hop_ms: float = 10.0
    enable_soft_clip: bool = True
    soft_clip_drive: float = 1.2
    soft_clip_mix: float = 0.7
    soft_clip_oversample: int = 4
    enable_limiter: bool = True
    limiter_threshold_dbfs: float = -3.0
    limiter_attack_ms: float = 1.5
    limiter_release_ms: float = 110.0
    limiter_lookahead_ms: float = 3.0
    limiter_knee_db: float = 3.5
    post_lufs_trim_db: float = 10.0
    output_peak_dbfs: float = -1.0
    true_peak_oversample: int = 2


class AudioEngine:
    """Core DSP engine for Stereo Field Designer v3."""

    def __init__(self, config: StereoFieldConfig, gpu: GPUManager | None = None):
        self.config = config
        self.gpu = gpu or GPUManager(preferred="auto")

        self.fft = FFTHandler(config.n_fft, config.hop_length, config.win_length)
        self.spectral = SpectralProcessor(self.fft, PsychoacousticModel())
        self.width = WidthController(
            self.fft,
            modulator=TemporalModulator(),
            morpher=NeuralMorpher(gpu=self.gpu),
        )
        self.analyzer = Analyzer()

    def _infer_mode(self, stereo: np.ndarray, sr: int) -> str:
        # Simple major/minor inference from chroma energy.
        mono = np.mean(stereo, axis=1)
        chroma = librosa.feature.chroma_stft(y=mono, sr=sr)
        profile = chroma.mean(axis=1)
        major = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], dtype=float)
        minor = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0], dtype=float)
        major_score = float(np.dot(profile, major))
        minor_score = float(np.dot(profile, minor))
        return "minor" if minor_score > major_score else "major"

    def _mode_tint(self, mode: str, air: float, warmth: float) -> tuple[float, float]:
        if mode == "minor":
            return air * 0.9, min(warmth * 1.1, 1.0)
        return air, warmth

    def _apply_tonal_anchor(self, stereo: np.ndarray, sr: int, key_pc: int, mode: str | None) -> np.ndarray:
        if key_pc is None:
            return stereo
        mid, side = mid_side_split(stereo)
        length = len(mid)

        spectrum = self.fft.stft(mid)
        mag = np.abs(spectrum)
        phase = np.angle(spectrum)
        freqs = self.fft.freqs(sr)
        pcs = pitch_class_for_freq(freqs)

        band = (freqs >= self.config.anchor_band_low_hz) & (freqs <= self.config.anchor_band_high_hz) & (pcs >= 0)
        root = int(key_pc % 12)
        fifth = int((key_pc + 7) % 12)
        third = int((key_pc + (3 if mode == "minor" else 4)) % 12)

        gain_db = np.zeros_like(freqs, dtype=np.float32)
        gain_db[band & (pcs == root)] += self.config.anchor_gain_db
        gain_db[band & (pcs == fifth)] += self.config.anchor_gain_db * 0.7
        gain_db[band & (pcs == third)] += self.config.anchor_third_gain_db

        kernel = np.array([0.2, 0.6, 0.2], dtype=np.float32)
        gain_db = np.convolve(gain_db, kernel, mode="same")
        gain = db_to_lin(gain_db)

        anchored = mag * gain[:, None] * np.exp(1j * phase)
        mid_out = self.fft.istft(anchored, length=length)
        # Music-theory note: tonal anchor reinforces root/third/fifth to keep harmony centered.
        return mid_side_merge(mid_out, side)

    def _limit_peak(self, stereo: np.ndarray) -> np.ndarray:
        peak = self._estimate_true_peak(stereo, self.config.true_peak_oversample)
        target = float(db_to_lin(self.config.output_peak_dbfs))
        if peak <= target:
            return stereo
        return stereo * (target / peak)

    def _apply_loudness_target(self, stereo: np.ndarray, sr: int) -> np.ndarray:
        mono = np.mean(stereo, axis=1)
        current_lufs = self.analyzer.features.lufs(mono, sr)
        gain_db = np.clip(self.config.target_lufs - current_lufs, -self.config.max_lufs_gain_db, self.config.max_lufs_gain_db)
        if not self.config.enable_limiter:
            peak_db = float(lin_to_db(self._estimate_true_peak(stereo, self.config.true_peak_oversample)))
            max_gain_db = self.config.output_peak_dbfs - peak_db - self.config.loudness_headroom_db
            gain_db = min(gain_db, max_gain_db)
        return stereo * db_to_lin(gain_db)

    def _apply_soft_clip(self, stereo: np.ndarray) -> np.ndarray:
        drive = float(np.clip(self.config.soft_clip_drive, 1.0, 6.0))
        mix = float(np.clip(self.config.soft_clip_mix, 0.0, 1.0))
        # Phase-stability note: soft clipping shapes peaks symmetrically to preserve stereo balance.
        if self.config.soft_clip_oversample > 1:
            os_factor = int(self.config.soft_clip_oversample)
            up = self._resample_stereo(stereo, os_factor, 1)
            clipped = np.tanh(up * drive) / np.tanh(drive)
            up = (1.0 - mix) * up + mix * clipped
            down = self._resample_stereo(up, 1, os_factor)
            if down.shape[0] > stereo.shape[0]:
                down = down[: stereo.shape[0], :]
            elif down.shape[0] < stereo.shape[0]:
                pad = stereo.shape[0] - down.shape[0]
                down = np.pad(down, ((0, pad), (0, 0)), mode="edge")
            return down.astype(np.float32)
        clipped = np.tanh(stereo * drive) / np.tanh(drive)
        return (1.0 - mix) * stereo + mix * clipped

    def _resample_stereo(self, audio: np.ndarray, up: int, down: int) -> np.ndarray:
        if up == down:
            return audio
        if resample_poly is not None:
            try:
                return resample_poly(audio, up, down, axis=0)
            except TypeError:
                left = resample_poly(audio[:, 0], up, down)
                right = resample_poly(audio[:, 1], up, down)
                return np.stack([left, right], axis=-1)
        target_sr = self.config.sample_rate * (up / down)
        left = librosa.resample(audio[:, 0], orig_sr=self.config.sample_rate, target_sr=target_sr)
        right = librosa.resample(audio[:, 1], orig_sr=self.config.sample_rate, target_sr=target_sr)
        return np.stack([left, right], axis=-1)

    def _estimate_true_peak(self, stereo: np.ndarray, oversample: int) -> float:
        peak = float(np.max(np.abs(stereo)) + 1e-9)
        if oversample <= 1:
            return peak
        os = self._resample_stereo(stereo, oversample, 1)
        return float(np.max(np.abs(os)) + 1e-9)

    def _peak_limiter(self, stereo: np.ndarray, sr: int) -> np.ndarray:
        threshold = float(db_to_lin(self.config.limiter_threshold_dbfs))
        if threshold <= 0.0:
            return stereo
        attack = float(np.exp(-1.0 / (max(self.config.limiter_attack_ms, 0.1) * 1e-3 * sr)))
        release = float(np.exp(-1.0 / (max(self.config.limiter_release_ms, 1.0) * 1e-3 * sr)))
        lookahead = int(max(0, self.config.limiter_lookahead_ms) * 1e-3 * sr)
        knee = float(max(self.config.limiter_knee_db, 0.0))
        n = stereo.shape[0]
        env = 0.0
        gain = np.ones(n, dtype=np.float32)

        padded = np.pad(stereo, ((lookahead, 0), (0, 0)), mode="edge") if lookahead > 0 else stereo
        for i in range(n):
            idx = i + lookahead
            x = max(abs(float(padded[idx, 0])), abs(float(padded[idx, 1])))
            if x > env:
                env = attack * env + (1.0 - attack) * x
            else:
                env = release * env + (1.0 - release) * x

            if env > threshold:
                if knee > 0.0:
                    env_db = 20.0 * np.log10(max(env, 1e-9))
                    over_db = env_db - self.config.limiter_threshold_dbfs
                    if over_db < knee:
                        gain_db = -(over_db * over_db) / (2.0 * knee)
                    else:
                        gain_db = -over_db + (knee * 0.5)
                    gain[i] = float(db_to_lin(gain_db))
                else:
                    gain[i] = threshold / max(env, 1e-9)

        # Quality note: lookahead + soft knee reduces crackle on sharp transients.
        return stereo * gain[:, None]

    def _apply_post_lufs_trim(self, stereo: np.ndarray, sr: int) -> np.ndarray:
        mono = np.mean(stereo, axis=1)
        current_lufs = self.analyzer.features.lufs(mono, sr)
        gain_db = self.config.target_lufs - current_lufs
        gain_db = float(np.clip(gain_db, -self.config.post_lufs_trim_db, self.config.post_lufs_trim_db))
        return stereo * db_to_lin(gain_db)

    def _crest_compress(self, stereo: np.ndarray, sr: int) -> np.ndarray:
        ratio = float(max(self.config.crest_ratio, 1.0))
        knee = float(max(self.config.crest_knee_db, 0.0))
        frame_length = max(int(sr * self.config.crest_window_ms * 1e-3), 256)
        hop_length = max(int(sr * self.config.crest_hop_ms * 1e-3), 128)

        mono = np.mean(stereo, axis=1)
        rms_frames = librosa.feature.rms(
            y=mono,
            frame_length=frame_length,
            hop_length=hop_length,
            center=True,
        )[0]
        rms_frames = np.maximum(rms_frames, 1e-9)
        rms_db = 20.0 * np.log10(rms_frames)
        over_db = rms_db - self.config.crest_threshold_dbfs

        gain_db = np.zeros_like(over_db, dtype=np.float32)
        if knee > 0.0:
            in_knee = (over_db > 0.0) & (over_db < knee)
            hard = over_db >= knee
            gain_db[in_knee] = -(over_db[in_knee] ** 2) / (2.0 * knee) * (1.0 - 1.0 / ratio)
            gain_db[hard] = -over_db[hard] * (1.0 - 1.0 / ratio)
        else:
            active = over_db > 0.0
            gain_db[active] = -over_db[active] * (1.0 - 1.0 / ratio)

        frame_samples = librosa.frames_to_samples(np.arange(len(gain_db)), hop_length=hop_length)
        sample_idx = np.arange(len(mono))
        gain_db_samples = np.interp(sample_idx, frame_samples, gain_db, left=gain_db[0], right=gain_db[-1])
        gain = db_to_lin(gain_db_samples.astype(np.float32))

        # DSP note: RMS-based crest compression avoids micro-transient crackle.
        return stereo * gain[:, None]

    def _finalize_loudness(self, stereo: np.ndarray, sr: int) -> np.ndarray:
        for _ in range(3):
            mono = np.mean(stereo, axis=1)
            current_lufs = self.analyzer.features.lufs(mono, sr)
            delta_db = float(self.config.target_lufs - current_lufs)
            if abs(delta_db) < 0.15:
                break
            step_db = float(np.clip(delta_db * 0.9, -4.0, 4.0))
            stereo = stereo * db_to_lin(step_db)
            if self.config.enable_soft_clip:
                stereo = self._apply_soft_clip(stereo)
            if self.config.enable_limiter:
                stereo = self._peak_limiter(stereo, sr)
            stereo = self._limit_peak(stereo)

        mono = np.mean(stereo, axis=1)
        current_lufs = self.analyzer.features.lufs(mono, sr)
        if current_lufs > self.config.target_lufs + 0.1:
            down_db = float(self.config.target_lufs - current_lufs)
            stereo = stereo * db_to_lin(down_db)
            if self.config.enable_limiter:
                stereo = self._peak_limiter(stereo, sr)
            stereo = self._limit_peak(stereo)
        elif current_lufs < self.config.target_lufs - 0.4 and self.config.enable_limiter:
            up_db = float(np.clip(self.config.target_lufs - current_lufs, 0.0, 1.5))
            stereo = stereo * db_to_lin(up_db)
            stereo = self._peak_limiter(stereo, sr)
            stereo = self._limit_peak(stereo)
        return stereo

    def process(self, audio: np.ndarray, sr: int | None = None) -> tuple[np.ndarray, AnalysisResult]:
        sr = sr or self.config.sample_rate
        stereo = ensure_stereo(audio)
        if self.config.bypass_processing:
            analysis = self.analyzer.analyze(stereo, sr)
            return stereo, analysis

        mode = None
        if self.config.enable_mode_tint or self.config.enable_tonal_anchor:
            mode = self._infer_mode(stereo, sr)
        air = self.config.air_strength
        warmth = self.config.warmth_strength
        if self.config.enable_mode_tint:
            air, warmth = self._mode_tint(mode, air, warmth)

        scale_pcs = None
        key_pc = None
        if self.config.enable_scale_locked_width or self.config.enable_tonal_anchor:
            scale_pcs = estimate_scale_pitch_classes(stereo, sr)
            key_pc = estimate_key_pitch_class(stereo, sr)

        if self.config.enable_psycho_widen:
            stereo = self.spectral.psychoacoustic_widen(
                stereo,
                sr=sr,
                strength=self.config.psycho_strength,
                phase_degrees=self.config.phase_degrees,
                hrtf_amount=self.config.hrtf_amount,
                mono_guard_hz=self.config.mono_guard_hz,
                mono_guard_slope_hz=self.config.mono_guard_slope_hz,
            )

        if self.config.enable_tonal_anchor and key_pc is not None:
            stereo = self._apply_tonal_anchor(stereo, sr, key_pc, mode)

        stereo = self.width.process(
            stereo,
            sr=sr,
            width_db=self.config.width_db,
            motion_depth=self.config.motion_depth,
            morph_strength=self.config.morph_strength,
            air_strength=air,
            warmth_strength=warmth,
            mono_guard_hz=self.config.mono_guard_hz,
            mono_guard_slope_hz=self.config.mono_guard_slope_hz,
            correlation_guard=self.config.correlation_guard,
            enable_scale_lock=self.config.enable_scale_locked_width,
            scale_pcs=scale_pcs,
            scale_in_db=self.config.scale_width_in_db,
            scale_out_db=self.config.scale_width_out_db,
            scale_band_low_hz=self.config.scale_band_low_hz,
            scale_band_high_hz=self.config.scale_band_high_hz,
        )

        if not self.config.enable_limiter:
            stereo = self._apply_loudness_target(stereo, sr)
        if self.config.enable_crest_compressor:
            stereo = self._crest_compress(stereo, sr)
        if self.config.enable_soft_clip:
            stereo = self._apply_soft_clip(stereo)
        if self.config.enable_limiter:
            stereo = self._peak_limiter(stereo, sr)
        stereo = self._finalize_loudness(stereo, sr)
        analysis = self.analyzer.analyze(stereo, sr)
        return stereo, analysis

    def load_audio(self, path: str | Path) -> tuple[np.ndarray, int]:
        audio, sr = librosa.load(path, sr=self.config.sample_rate, mono=False)
        return ensure_stereo(audio), sr

    def process_file(self, input_path: str | Path, output_path: str | Path) -> AnalysisResult:
        audio, sr = self.load_audio(input_path)
        processed, analysis = self.process(audio, sr=sr)
        render_name = Path(output_path).stem
        self.analyzer.log_metrics(processed, sr, name=render_name)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), processed, sr)
        return analysis
