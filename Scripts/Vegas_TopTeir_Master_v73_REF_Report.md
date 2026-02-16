# AuralMind Maestro v7.3 StereoAI — Report

## Summary
- Preset: **competitive_trap**
- Sample rate: **48000 Hz**
- LUFS (pre): **-17.57**
- LUFS (post): **-17.91**
- True peak (approx): **-1.00 dBFS**
- Limiter min gain (approx GR): **-13.81 dB**

## Low-end / music theory anchors
- Estimated sub fundamental f0: **44.49462890625 Hz**
- Mono-sub v2 cutoff: **86.7645263671875 Hz**
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
  "lufs_pre": -17.574333282144845,
  "lufs_post": -17.910905724042294,
  "true_peak_dbfs": -1.0000002900129947,
  "limiter_min_gain_db": -13.805685843757363,
  "sub_f0_hz": 44.49462890625,
  "mono_sub_cutoff_hz": 86.7645263671875,
  "mono_sub_mix": 0.45,
  "runtime_sec": 43.62110948562622,
  "out_path": "Vegas_TopTeir_Master_v73_REF.wav"
}
```
