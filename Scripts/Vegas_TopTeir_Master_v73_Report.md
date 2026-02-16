# AuralMind Maestro v7.3 StereoAI — Report

## Summary
- Preset: **hi_fi_streaming**
- Sample rate: **48000 Hz**
- LUFS (pre): **-20.59**
- LUFS (post): **-18.52**
- True peak (approx): **-1.00 dBFS**
- Limiter min gain (approx GR): **-11.72 dB**

## Low-end / music theory anchors
- Estimated sub fundamental f0: **44.49462890625 Hz**
- Mono-sub v2 cutoff: **86.7645263671875 Hz**
- Mono-sub v2 adaptive mix: **0.45**

## Stereo enhancements
- Spatial Realism Enhancer: frequency-dependent width + correlation guard
- NEW CGMS MicroShift: micro-delay applied to SIDE high-band only, correlation-guarded

## Loudness Governor
- Requested target LUFS: **-12.8**
- Governor final target LUFS: **-14.6**
  If limiter GR exceeded the ceiling, the governor backed off the LUFS target.

## JSON dump
```json
{
  "preset": "hi_fi_streaming",
  "sr": 48000,
  "target_lufs_requested": -12.8,
  "target_lufs_after_governor": -14.6,
  "lufs_pre": -20.58523821296647,
  "lufs_post": -18.51540069323292,
  "true_peak_dbfs": -1.000000870903717,
  "limiter_min_gain_db": -11.724926276005077,
  "sub_f0_hz": 44.49462890625,
  "mono_sub_cutoff_hz": 86.7645263671875,
  "mono_sub_mix": 0.45,
  "runtime_sec": 42.84449291229248,
  "out_path": "Vegas_TopTeir_Master_v73.wav"
}
```
