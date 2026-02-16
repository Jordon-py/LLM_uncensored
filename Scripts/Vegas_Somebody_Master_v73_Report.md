# AuralMind Maestro v7.3 StereoAI — Report

## Summary
- Preset: **hi_fi_streaming**
- Sample rate: **48000 Hz**
- LUFS (pre): **-15.84**
- LUFS (post): **-16.34**
- True peak (approx): **-1.00 dBFS**
- Limiter min gain (approx GR): **-4.26 dB**

## Low-end / music theory anchors
- Estimated sub fundamental f0: **77.63671875 Hz**
- Mono-sub v2 cutoff: **105.0 Hz**
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
  "lufs_pre": -15.842711532822094,
  "lufs_post": -16.339656680118583,
  "true_peak_dbfs": -0.9999979664504948,
  "limiter_min_gain_db": -4.261005135367998,
  "sub_f0_hz": 77.63671875,
  "mono_sub_cutoff_hz": 105.0,
  "mono_sub_mix": 0.45,
  "runtime_sec": 31.22237777709961,
  "out_path": "Somebody_Master_v73.wav"
}
```
