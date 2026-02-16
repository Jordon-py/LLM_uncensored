# AuralMind Maestro v7.3 StereoAI — Report

## Summary
- Preset: **melodic_trap**
- Sample rate: **48000 Hz**
- LUFS (pre): **-19.63**
- LUFS (post): **-19.34**
- True peak (approx): **-1.00 dBFS**
- Limiter min gain (approx GR): **-13.20 dB**

## Low-end / music theory anchors
- Estimated sub fundamental f0: **76.72119140625 Hz**
- Mono-sub v2 cutoff: **105.0 Hz**
- Mono-sub v2 adaptive mix: **0.45**

## Stereo enhancements
- Spatial Realism Enhancer: frequency-dependent width + correlation guard
- NEW Depth Distance Cue: energy-dependent HF tilt for front-to-back depth
- NEW Depth-Adaptive CGMS MicroShift: micro-delay on SIDE high-band, transient-aware + correlation-guarded

## Movement / HookLift
- Movement enabled: **True** (amount=0.13)
- HookLift enabled: **True** (mix=0.2)
  - Auto mask percentile: **70.0**

## Stem separation (HT-Demucs)
- Enabled: **True**
- Model: **htdemucs**
- Sources: **['drums', 'bass', 'other', 'vocals']**

## Loudness Governor
- Requested target LUFS: **-12.0**
- Governor final target LUFS: **-13.799999999999999**
  If limiter GR exceeded the ceiling, the governor backed off the LUFS target.

## JSON dump
```json
{
  "preset": "melodic_trap",
  "sr": 48000,
  "target_lufs_requested": -12.0,
  "target_lufs_after_governor": -13.799999999999999,
  "lufs_pre": -19.630526821757222,
  "lufs_post": -19.337466755240573,
  "true_peak_dbfs": -1.0000002900129947,
  "limiter_min_gain_db": -13.200486155248656,
  "sub_f0_hz": 76.72119140625,
  "mono_sub_cutoff_hz": 105.0,
  "mono_sub_mix": 0.45,
  "movement": {
    "enabled": true,
    "amount": 0.13
  },
  "hooklift": {
    "enabled": true,
    "mix": 0.2,
    "air_hz": 8500.0,
    "width_gain": 0.18,
    "air_gain": 0.14,
    "auto": true,
    "auto_percentile": 70.0
  },
  "stems": {
    "enabled": true,
    "model_name": "htdemucs",
    "device": "cpu",
    "split": true,
    "overlap": 0.25,
    "shifts": 1,
    "model_sr": 44100,
    "sr": 48000,
    "sources": [
      "drums",
      "bass",
      "other",
      "vocals"
    ]
  },
  "runtime_sec": 806.4407403469086,
  "out_path": "C:\\Users\\goku\\LLM_uncensored\\Scripts\\Mastered\\FaceTime (10)__melodic_trap.wav"
}
```
