# AuralMind Maestro v7.3 StereoAI — Report

## Summary
- Preset: **competitive_trap**
- Sample rate: **48000 Hz**
- LUFS (pre): **-19.44**
- LUFS (post): **-18.19**
- True peak (approx): **-1.00 dBFS**
- Limiter min gain (approx GR): **-11.99 dB**

## Low-end / music theory anchors
- Estimated sub fundamental f0: **76.72119140625 Hz**
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
  "lufs_pre": -19.437066800879972,
  "lufs_post": -18.191299691732386,
  "true_peak_dbfs": -1.000000870903717,
  "limiter_min_gain_db": -11.988342378794922,
  "sub_f0_hz": 76.72119140625,
  "mono_sub_cutoff_hz": 105.0,
  "mono_sub_mix": 0.45,
  "runtime_sec": 51.65965962409973,
  "out_path": "FaceTime_Master_v273_REF.wav"
}
```
