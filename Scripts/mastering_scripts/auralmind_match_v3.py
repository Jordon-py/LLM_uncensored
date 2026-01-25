#!/usr/bin/env python3
"""
AuralMind v3.0 - The "Prism" Melodic Trap Enhancer
==================================================
Beyond Matching: Uses Harmonic Synthesis to create "HD" Air.

NEW FEATURES (v3):
1. Prism Air Exciter: Generates synthetic high-end shimmer for the side channels.
   - RESULT: A track that sounds wider and "higher resolution" than the reference.
   
CORE FEATURES (v2):
- Mono-Sub Processing (Tight Lows)
- Multiband Saturation (Warm Mids)
- Soft-Clipping (Commercial Loudness)
- Spectral Matching (Tonal Balance)

USAGE:
python auralmind_match_v3.py --reference "C:/Users/goku/Downloads/Brent Faiyaz - Pistachios [Official Video].wav" --target "C:/Users/goku/Downloads/SOMEBODY.wav" --out "master_v4.wav"

PIPELINE:
  1) Decode/normalize inputs to float32 stereo at --sr.
  2) Spectral match (EQ) to the reference.
  3) Low-end mono imaging and multiband saturation.
  4) Prism air exciter for side-channel shimmer.
  5) Soft-clip and loudness match with safety ceiling.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf
import librosa
import pyloudnorm as pyln
from scipy.signal import firwin2, fftconvolve, savgol_filter, butter, filtfilt

LOG = logging.getLogger(__name__)
DIRECT_READ_EXTS = {
    ".wav",
    ".wave",
    ".flac",
    ".ogg",
    ".oga",
    ".aiff",
    ".aif",
    ".aifc",
    ".caf",
    ".w64",
}

# -----------------------------
# UTILITIES & IO
# -----------------------------

def _require_ffmpeg(ffmpeg_path: Path | str | None = None) -> str:
    if ffmpeg_path:
        candidate = Path(ffmpeg_path)
        if candidate.is_dir():
            raise RuntimeError(f"ffmpeg path is a directory, not an executable: {candidate}")
        if not candidate.exists():
            raise RuntimeError(f"ffmpeg not found at: {candidate}")
        return str(candidate)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found on PATH. Please install ffmpeg or pass --ffmpeg.")
    return ffmpeg

def _run(cmd: list[str]) -> None:
    LOG.debug("Running command: %s", " ".join(cmd))
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        cmd_str = subprocess.list2cmdline(cmd)
        msg_lines = [f"Command failed ({result.returncode}): {cmd_str}"]
        if result.stdout:
            msg_lines.append("stdout:/n" + result.stdout.strip())
        if result.stderr:
            msg_lines.append("stderr:/n" + result.stderr.strip())
        raise RuntimeError("/n".join(msg_lines))

def decode_with_ffmpeg_to_wav(
    input_path: Path,
    out_wav_path: Path,
    sr: int,
    ffmpeg_path: Path | str | None = None,
) -> None:
    ffmpeg = _require_ffmpeg(ffmpeg_path)
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "2",
        "-ar",
        str(sr),
        "-acodec",
        "pcm_f32le",
        str(out_wav_path),
    ]
    _run(cmd)

def _validate_input_path(path: Path, label: str) -> Path:
    path = path.expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} path is not a file: {path}")
    return path

def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    resampled_channels = [
        librosa.resample(audio[:, ch], orig_sr=orig_sr, target_sr=target_sr)
        for ch in range(audio.shape[1])
    ]
    min_len = min(len(ch) for ch in resampled_channels)
    resampled = np.stack([ch[:min_len] for ch in resampled_channels], axis=1)
    return resampled.astype(np.float32, copy=False)

def ensure_stereo(audio: np.ndarray) -> np.ndarray:
    if audio.ndim != 2:
        raise ValueError("Expected audio with shape (n_samples, n_channels).")
    if audio.shape[1] == 1:
        return np.repeat(audio, 2, axis=1)
    if audio.shape[1] > 2:
        mono = np.mean(audio, axis=1, keepdims=True)
        return np.repeat(mono, 2, axis=1)
    return audio

def load_audio_any_format(
    path: Path,
    sr: int,
    ffmpeg_path: Path | str | None = None,
    label: str = "Input",
) -> Tuple[np.ndarray, int]:
    path = _validate_input_path(path, label)
    direct_error = None

    if path.suffix.lower() in DIRECT_READ_EXTS:
        try:
            audio, file_sr = sf.read(path, dtype="float32", always_2d=True)
            if file_sr != sr:
                audio = resample_audio(audio, file_sr, sr)
                file_sr = sr
            audio = ensure_finite(audio)
            return audio, file_sr
        except Exception as exc:
            direct_error = str(exc)

    ffmpeg_error = None
    try:
        with tempfile.TemporaryDirectory() as td:
            tmp_wav = Path(td) / "decoded.wav"
            decode_with_ffmpeg_to_wav(path, tmp_wav, sr=sr, ffmpeg_path=ffmpeg_path)
            audio, file_sr = sf.read(tmp_wav, dtype="float32", always_2d=True)
            audio = ensure_finite(audio)
            return audio, file_sr
    except Exception as exc:
        ffmpeg_error = str(exc)

    try:
        audio, file_sr = librosa.load(path, sr=sr, mono=False)
        if audio.ndim == 1:
            audio = audio[:, None]
        else:
            audio = audio.T
        audio = audio.astype(np.float32, copy=False)
        audio = ensure_finite(audio)
        return audio, file_sr
    except Exception as exc:
        msg_lines = [f"Audio decode failed for: {path}"]
        if direct_error:
            msg_lines.append(f"soundfile error: {direct_error}")
        if ffmpeg_error:
            msg_lines.append(f"ffmpeg error: {ffmpeg_error}")
        msg_lines.append(f"librosa error: {exc}")
        raise RuntimeError("/n".join(msg_lines)) from exc

def ensure_finite(x: np.ndarray) -> np.ndarray:
    if not np.isfinite(x).all():
        x = np.nan_to_num(x)
    return x

def to_mono(x: np.ndarray) -> np.ndarray:
    return np.mean(x, axis=1)

def db(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(eps, x))

def undb(x_db: np.ndarray) -> np.ndarray:
    return 10.0 ** (x_db / 20.0)


# -----------------------------
# DSP: CORE FILTERS
# -----------------------------

def butter_lowpass(cutoff: float, fs: int, order: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    return butter(order, normal_cutoff, btype='low', analog=False)

def butter_highpass(cutoff: float, fs: int, order: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    return butter(order, normal_cutoff, btype='high', analog=False)


# -----------------------------
# DSP: ENHANCEMENT MODULES
# -----------------------------

def apply_mono_sub(audio: np.ndarray, sr: int, cutoff_hz: float = 120.0) -> np.ndarray:
    """Forces low frequencies to mono for tight 808s."""
    b_lo, a_lo = butter_lowpass(cutoff_hz, sr, order=4)
    b_hi, a_hi = butter_highpass(cutoff_hz, sr, order=4)

    lows = np.zeros_like(audio)
    highs = np.zeros_like(audio)
    
    for ch in range(audio.shape[1]):
        lows[:, ch] = filtfilt(b_lo, a_lo, audio[:, ch])
        highs[:, ch] = filtfilt(b_hi, a_hi, audio[:, ch])

    lows_mono = np.mean(lows, axis=1, keepdims=True)
    lows_mono_stereo = np.tile(lows_mono, (1, 2))

    return lows_mono_stereo + highs


def soft_clip_func(x: np.ndarray, drive: float = 1.0) -> np.ndarray:
    """Tanh soft clipper."""
    return np.tanh(x * drive)

def apply_multiband_saturation(audio: np.ndarray, sr: int) -> np.ndarray:
    """Adds warmth to mids and thickness to lows."""
    nyq = 0.5 * sr
    b_l, a_l = butter(4, 200.0 / nyq, btype='low')
    b_h, a_h = butter(4, 4000.0 / nyq, btype='high')
    
    low_band = np.zeros_like(audio)
    high_band = np.zeros_like(audio)
    
    for ch in range(audio.shape[1]):
        low_band[:, ch] = filtfilt(b_l, a_l, audio[:, ch])
        high_band[:, ch] = filtfilt(b_h, a_h, audio[:, ch])
        
    mid_band = audio - (low_band + high_band)

    # Apply tailored saturation
    low_sat = soft_clip_func(low_band, drive=1.1)
    mid_sat = soft_clip_func(mid_band, drive=1.3) # Warmth focus
    high_sat = soft_clip_func(high_band, drive=1.0) # Clean highs

    return low_sat + mid_sat + high_sat


def apply_prism_air_exciter(audio: np.ndarray, sr: int, drive: float = 0.2) -> np.ndarray:
    """
    EDUCATIONAL: PRISM AIR EXCITER
    This is the "Magic" feature.
    1. Extracts High-Mids (2.5kHz+).
    2. Generates Harmonics (Distortion).
    3. Pushes this "fizz" to the SIDE channels only.
    
    Result: A wider, shimmering stereo image that adds 'expensive' detail 
    without muddying the center vocals/kick.
    """
    # 1. Isolate High Frequencies (The "Source" for excitation)
    b_hp, a_hp = butter_highpass(2500.0, sr, order=2)
    highs = np.zeros_like(audio)
    for ch in range(audio.shape[1]):
        highs[:, ch] = filtfilt(b_hp, a_hp, audio[:, ch])

    # 2. Generate Harmonics (Excitation)
    # We use absolute value + tanh to create even & odd harmonics (Rich texture)
    harmonics = np.tanh(np.abs(highs) * 2.0) - np.abs(np.tanh(highs))
    
    # 3. Filter the Harmonics
    # We only want the "Air" (above 5kHz), not the grit in the mids.
    b_air, a_air = butter_highpass(5000.0, sr, order=2)
    air_shimmer = np.zeros_like(harmonics)
    for ch in range(audio.shape[1]):
        air_shimmer[:, ch] = filtfilt(b_air, a_air, harmonics[:, ch])

    # 4. Mid-Side Processing: Add shimmer ONLY to Side
    # Left = Mid + Side; Right = Mid - Side
    # We will add the 'air_shimmer' essentially as a widening layer.
    
    # Simple stereo expansion technique:
    # Invert phase of right channel shimmer and mix low level.
    # L_new = L + (Shimmer * drive)
    # R_new = R - (Shimmer * drive)
    
    # Taking the mono sum of the shimmer ensures we add purely correlated harmonic content
    # that we then decorrelate by phase flipping, creating width.
    shimmer_mono = np.mean(air_shimmer, axis=1)
    
    out = audio.copy()
    out[:, 0] += shimmer_mono * drive      # Left Add
    out[:, 1] -= shimmer_mono * drive      # Right Subtract (creates width)
    
    return out


def apply_master_soft_clipper(audio: np.ndarray, threshold_db: float = -2.0) -> np.ndarray:
    """Final stage transient shaver."""
    thresh_lin = 10.0 ** (threshold_db / 20.0)
    out = audio.copy()
    # Hard drive into Tanh for "Trap" sound
    return np.tanh(out / thresh_lin) * thresh_lin


# -----------------------------
# SPECTRAL & LOUDNESS
# -----------------------------

@dataclass
class AudioMetrics:
    lufs_i: float
    crest_db: float

def compute_metrics(audio: np.ndarray, sr: int) -> AudioMetrics:
    if audio.size == 0 or audio.shape[0] == 0:
        LOG.warning("Metrics requested for empty audio.")
        return AudioMetrics(float("-inf"), 0.0)
    mono = to_mono(audio)
    meter = pyln.Meter(sr)
    lufs = float(meter.integrated_loudness(mono))
    if not np.isfinite(lufs):
        lufs = float("-inf")
    rms = float(np.sqrt(np.mean(mono**2)))
    peak = float(np.max(np.abs(mono)))
    if peak <= 0.0 or rms <= 0.0:
        crest = 0.0
    else:
        crest = float(20 * np.log10(peak / (rms + 1e-9)))
    return AudioMetrics(lufs, crest)

def design_match_fir(ref: np.ndarray, tgt: np.ndarray, sr: int, taps: int = 1025) -> np.ndarray:
    """Matches tonal balance."""
    def get_spec(x):
        S = np.abs(librosa.stft(x, n_fft=4096))
        return np.mean(S, axis=1)

    ref_spec = db(get_spec(ref))
    tgt_spec = db(get_spec(tgt))
    diff = ref_spec - tgt_spec
    diff_smooth = savgol_filter(diff, 101, 3)
    # Enhancement: Increased range from +/- 6dB to +/- 12dB for better matching
    diff_smooth = np.clip(diff_smooth, -12.0, 12.0)
    
    freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
    
    # firwin2 with fs=sr expects frequencies in Hz from 0 to fs/2
    freqs[0] = 0.0
    freqs[-1] = sr / 2.0
    
    gain = undb(diff_smooth)
    gain = np.nan_to_num(gain)
    
    h = firwin2(taps, freqs, gain, fs=sr)
    return h

def apply_fir(audio: np.ndarray, h: np.ndarray) -> np.ndarray:
    out = np.zeros_like(audio)
    for ch in range(audio.shape[1]):
        out[:, ch] = fftconvolve(audio[:, ch], h, mode='same')
    return out

def match_lufs_and_limit(audio: np.ndarray, sr: int, target_lufs: float) -> np.ndarray:
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(to_mono(audio))
    if not np.isfinite(loudness):
        LOG.warning("LUFS measurement invalid; skipping normalization.")
        return audio.astype(np.float32, copy=False)
    gain_db = target_lufs - loudness
    audio_norm = audio * (10 ** (gain_db / 20.0))
    ceiling = 10 ** (-0.1 / 20)
    return np.clip(audio_norm, -ceiling, ceiling)


# -----------------------------
# MAIN PIPELINE
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="AuralMind v3 - Prism Enhancer")
    parser.add_argument("--reference", required=True, type=Path, default=Path("C:/Users/goku/Downloads/Lil Wayne - She Will ft. Drake.mp3"))
    parser.add_argument("--target", required=True, type=Path, default=Path("C:/Users/goku/Downloads/FM Vegas - Consistent.wav"))
    parser.add_argument(
        "--ffmpeg",
        type=Path,
        default=None,
        help="Optional path to ffmpeg executable if not on PATH.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output WAV path. If omitted, saves next to target as enhanced_<target>.wav.",
    )
    parser.add_argument("--sr", type=int, default=48000)
    parser.add_argument("--log_level", default="INFO", help="Logging level (DEBUG, INFO, WARNING). Default: INFO.")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    if args.out is None:
        args.out = args.target.with_name(f"enhanced_{args.target.stem}.wav")
    args.out = args.out.expanduser()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    LOG.info("Loading audio (SR=%s)...", args.sr)
    LOG.info("Reference: %s", args.reference)
    LOG.info("Target: %s", args.target)
    LOG.info("Output: %s", args.out)
    ref_audio, _ = load_audio_any_format(args.reference, args.sr, args.ffmpeg, label="Reference")
    tgt_audio, _ = load_audio_any_format(args.target, args.sr, args.ffmpeg, label="Target")
    ref_audio = ensure_stereo(ref_audio)
    tgt_audio = ensure_stereo(tgt_audio)

    ref_metrics = compute_metrics(ref_audio, args.sr)
    LOG.info("Reference goal: %.1f LUFS", ref_metrics.lufs_i)

    # 1. SPECTRAL MATCHING (The "Clone" phase)
    LOG.info("1. Matching Reference Tone (EQ)...")
    h = design_match_fir(to_mono(ref_audio), to_mono(tgt_audio), args.sr)
    processed = apply_fir(tgt_audio, h)

    # 2. MONO-SUB IMAGING (The "Foundation" phase)
    LOG.info("2. Tightening Low-End (Mono-Sub)...")
    processed = apply_mono_sub(processed, args.sr)

    # 3. MULTIBAND SATURATION (The "Warmth" phase)
    LOG.info("3. Applying Multiband Saturation...")
    processed = apply_multiband_saturation(processed, args.sr)

    # 4. PRISM AIR EXCITER (The "Better Than Reference" phase)
    # This adds the 3D high-end that simple EQ cannot achieve.
    LOG.info("4. Synthesizing Prism Air (Side-Channel Excitation)...")
    processed = apply_prism_air_exciter(processed, args.sr, drive=0.15)

    # 5. SOFT CLIPPER & LIMITING (The "Loudness" phase)
    LOG.info("5. Soft-Clipping & Limiting...")
    processed = apply_master_soft_clipper(processed, threshold_db=-2.0)
    final_audio = match_lufs_and_limit(processed, args.sr, ref_metrics.lufs_i)

    sf.write(args.out, final_audio, args.sr, subtype="PCM_24")
    LOG.info("Done. Prism master saved to: %s", args.out)


if __name__ == "__main__":
    main()
