# AuralMind Maestro v7.3 StereoAI — Report

## Summary
- Preset: **melodic_trap**
- Sample rate: **48000 Hz**
- LUFS (pre): **-16.27**
- LUFS (post): **-17.23**
- True peak (approx): **-1.00 dBFS**
- Limiter min gain (approx GR): **-8.64 dB**

## Low-end / music theory anchors
- Estimated sub fundamental f0: **51.81884765625 Hz**
- Mono-sub v2 cutoff: **101.0467529296875 Hz**
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
  "lufs_pre": -16.26754780084546,
  "lufs_post": -17.226734348736443,
  "true_peak_dbfs": -1.000000870903717,
  "limiter_min_gain_db": -8.644888754203098,
  "sub_f0_hz": 51.81884765625,
  "mono_sub_cutoff_hz": 101.0467529296875,
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
  "runtime_sec": 678.0888316631317,
  "out_path": "C:\\Users\\goku\\LLM_uncensored\\Scripts\\Mastered\\enhanced_SLIDE N  EXTENDO (2)__melodic_trap.wav"
}
```
