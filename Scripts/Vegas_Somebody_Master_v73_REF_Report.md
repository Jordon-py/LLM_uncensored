# AuralMind Maestro v7.3 StereoAI — Report

## Summary
- Preset: **competitive_trap**
- Sample rate: **48000 Hz**
- LUFS (pre): **-16.57**
- LUFS (post): **-16.00**
- True peak (approx): **-1.00 dBFS**
- Limiter min gain (approx GR): **-6.83 dB**

## Low-end / music theory anchors
- Estimated sub fundamental f0: **77.2705078125 Hz**
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
  "lufs_pre": -16.5693104144106,
  "lufs_post": -15.995890826946878,
  "true_peak_dbfs": -0.9999997091223114,
  "limiter_min_gain_db": -6.827355998980224,
  "sub_f0_hz": 77.2705078125,
  "mono_sub_cutoff_hz": 105.0,
  "mono_sub_mix": 0.45,
  "runtime_sec": 32.404244899749756,
  "out_path": "Somebody_Master_v273_REF.wav"
}
```
