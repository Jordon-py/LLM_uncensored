# AuralMind Maestro v7.3 StereoAI — Report

## Summary
- Preset: **competitive_trap**
- Sample rate: **48000 Hz**
- LUFS (pre): **-16.81**
- LUFS (post): **-18.31**
- True peak (approx): **-1.00 dBFS**
- Limiter min gain (approx GR): **-11.43 dB**

## Low-end / music theory anchors
- Estimated sub fundamental f0: **46.5087890625 Hz**
- Mono-sub v2 cutoff: **90.692138671875 Hz**
- Mono-sub v2 adaptive mix: **0.45062875365306454**

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
  "lufs_pre": -16.812083569691023,
  "lufs_post": -18.309582361487582,
  "true_peak_dbfs": -0.9999991282316669,
  "limiter_min_gain_db": -11.43139077947662,
  "sub_f0_hz": 46.5087890625,
  "mono_sub_cutoff_hz": 90.692138671875,
  "mono_sub_mix": 0.45062875365306454,
  "runtime_sec": 50.225698709487915,
  "out_path": "Fire_Master_v273_REF.wav"
}
```
