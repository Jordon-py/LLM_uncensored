# AuralMind Maestro v7.3 StereoAI — Report

## Summary
- Preset: **competitive_trap**
- Sample rate: **48000 Hz**
- LUFS (pre): **-17.08**
- LUFS (post): **-20.14**
- True peak (approx): **-1.00 dBFS**
- Limiter min gain (approx GR): **-14.00 dB**

## Low-end / music theory anchors
- Estimated sub fundamental f0: **46.5087890625 Hz**
- Mono-sub v2 cutoff: **90.692138671875 Hz**
- Mono-sub v2 adaptive mix: **0.4506319594726076**

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
  "lufs_pre": -17.075803166010616,
  "lufs_post": -20.140966317598817,
  "true_peak_dbfs": -1.0000002900129947,
  "limiter_min_gain_db": -14.00483842510872,
  "sub_f0_hz": 46.5087890625,
  "mono_sub_cutoff_hz": 90.692138671875,
  "mono_sub_mix": 0.4506319594726076,
  "runtime_sec": 41.59775495529175,
  "out_path": "MissingYou_Master_v273_REF.wav"
}
```
