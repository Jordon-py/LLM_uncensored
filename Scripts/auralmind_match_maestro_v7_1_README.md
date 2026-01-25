# AuralMind Match Maestro v7.1

## Overview

A fast, preset-driven mastering script for melodic trap that matches a reference track and outputs a clean, loud, and controlled master. v7.1 focuses on batch support, fixed musical defaults, and simple CLI usage.

## Highlights

- Batch or single-file mastering
- Reference match EQ with minimum-phase FIR
- Mandatory-by-default stems (Demucs) for clearer matching
- Mono sub anchor for low-end translation
- HookLift: adaptive widening and shimmer for hook energy
- Loudness normalization and true-peak limiting
- Markdown report with JSON payload

## Requirements

- Python 3.9+
- Core libs: numpy, soundfile, librosa, scipy, pyloudnorm
- Optional: tqdm (progress), numba (faster limiter)
- Stems (default): torch + demucs

## Install

Example:

```powershell
pip install numpy soundfile librosa scipy pyloudnorm
pip install torch demucs
pip install tqdm numba
```

## Quick Start

Single target:

```powershell
python Scripts/auralmind_match_maestro_v7_1.py --preset cinematic_punch --reference "ref.mp3" --target "song.wav"
```

Batch folder:

```powershell
python Scripts/auralmind_match_maestro_v7_1.py --preset airy_streaming --reference "ref.mp3" --target_dir "C:/songs" --out_dir "C:/masters" --report_dir "C:/masters/reports"
```

## CLI Options

- `--preset` Preset name. Choices: `balanced_v7`, `cinematic_punch`, `airy_streaming`, `loud_club`
- `--reference` Path to reference audio (required)
- `--target` Path to single target file
- `--target_dir` Path to folder of target files
- `--out` Output file path for single target
- `--out_dir` Output folder for batch
- `--report` Report file path for single target
- `--report_dir` Report folder for batch
- `--sr` Target sample rate (default 48000)
- `--disable_stems` Disable Demucs stems
- `--disable_mono_sub` Disable mono sub anchor
- `--disable_hooklift` Disable HookLift

Notes:

- Use `--out` and `--report` only for single-target runs.
- For batch runs, use `--out_dir` and `--report_dir`.

## Presets

Presets only adjust match strength:

- `balanced_v7`: 0.69
- `cinematic_punch`: 0.85
- `airy_streaming`: 0.78
- `loud_club`: 0.80

## Processing Chain (v7.1)

1. Load reference + target, resample to `--sr`
2. Optional Demucs stems; sum stems into a new target mix
3. Match EQ (minimum-phase FIR, 4097 taps, max +/-7 dB, smooth 90 Hz)
4. Mono Sub Anchor (120 Hz)
5. HookLift (widened highs + air shimmer)
6. Finalize: loudness normalize to -11 LUFS, true-peak limit to -1 dBFS

## Outputs and Report

- Output default: `<target_stem>__Maestro_v7_1.wav` in the target folder
- Report default: `<target_stem>__Maestro_v7_1_Report.md` in the same folder
- Report contains JSON with paths, metrics (LUFS/peaks), and module info

## Troubleshooting

- Missing Demucs: install `torch` and `demucs`, or run with `--disable_stems`
- MP3 load errors: the loader falls back to librosa automatically
- Slower runs: install `numba` for faster limiter smoothing

## Supported Formats

wav, flac, aiff, aif, mp3, m4a, ogg, opus
