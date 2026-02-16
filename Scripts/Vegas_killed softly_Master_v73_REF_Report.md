# AuralMind Maestro v7.3 StereoAI — Report

## Summary
- Preset: **competitive_trap**
- Sample rate: **48000 Hz**
- LUFS (pre): **-17.21**
- LUFS (post): **-17.93**
- True peak (approx): **-1.00 dBFS**
- Limiter min gain (approx GR): **-10.25 dB**

## Low-end / music theory anchors
- Estimated sub fundamental f0: **80.01708984375 Hz**
- Mono-sub v2 cutoff: **105.0 Hz**
- Mono-sub v2 adaptive mix: **0.45**

## Stereo enhancements
- Spatial Realism Enhancer: frequency-dependent width + correlation guard
- NEW CGMS MicroShift: micro-delay applied to SIDE high-band only, correlation-guarded

## Loudness Governor
- Requested target LUFS: **-11.4**
- Governor final target LUFS: **-13.2**
  If limiter GR exceeded the ceiling, the governor backed off the LUFS target.

## JSON dump
```json
{
  "preset": "competitive_trap",
  "sr": 48000,
  "target_lufs_requested": -11.4,
  "target_lufs_after_governor": -13.2,
  "lufs_pre": -17.214146597087705,
  "lufs_post": -17.931469902168303,
  "true_peak_dbfs": -1.000000870903717,
  "limiter_min_gain_db": -10.24571507614268,
  "sub_f0_hz": 80.01708984375,
  "mono_sub_cutoff_hz": 105.0,
  "mono_sub_mix": 0.45,
  "runtime_sec": 49.96136808395386,
  "out_path": "Killed Softly_Master_v273_REF.wav"
}
```
