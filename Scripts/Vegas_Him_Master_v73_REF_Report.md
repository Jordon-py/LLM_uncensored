# AuralMind Maestro v7.3 StereoAI — Report

## Summary
- Preset: **competitive_trap**
- Sample rate: **48000 Hz**
- LUFS (pre): **-16.12**
- LUFS (post): **-17.39**
- True peak (approx): **-1.00 dBFS**
- Limiter min gain (approx GR): **-9.45 dB**

## Low-end / music theory anchors
- Estimated sub fundamental f0: **69.76318359375 Hz**
- Mono-sub v2 cutoff: **105.0 Hz**
- Mono-sub v2 adaptive mix: **0.5947847479966307**

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
  "lufs_pre": -16.119388355485917,
  "lufs_post": -17.390201808491703,
  "true_peak_dbfs": -0.9999979664504948,
  "limiter_min_gain_db": -9.449661794327117,
  "sub_f0_hz": 69.76318359375,
  "mono_sub_cutoff_hz": 105.0,
  "mono_sub_mix": 0.5947847479966307,
  "runtime_sec": 92.79176998138428,
  "out_path": "Consistant_Master_v273_REF.wav"
}
```
