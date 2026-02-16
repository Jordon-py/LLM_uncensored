# AuralMind Maestro v7.3 StereoAI — Report

## Summary
- Preset: **melodic_trap**
- Sample rate: **48000 Hz**
- LUFS (pre): **-17.44**
- LUFS (post): **-18.52**
- True peak (approx): **-1.00 dBFS**
- Limiter min gain (approx GR): **-13.78 dB**

## Low-end / music theory anchors
- Estimated sub fundamental f0: **72.32666015625 Hz**
- Mono-sub v2 cutoff: **105.0 Hz**
- Mono-sub v2 adaptive mix: **0.45**

## Stereo enhancements
- Spatial Realism Enhancer: frequency-dependent width + correlation guard
- NEW CGMS MicroShift: micro-delay applied to SIDE high-band only, correlation-guarded

## Movement / HookLift
- Movement enabled: **True** (amount=0.12)
- HookLift enabled: **True** (mix=0.24)
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
  "lufs_pre": -17.441919256969765,
  "lufs_post": -18.520751085413018,
  "true_peak_dbfs": -0.9999997091223114,
  "limiter_min_gain_db": -13.778635558413965,
  "sub_f0_hz": 72.32666015625,
  "mono_sub_cutoff_hz": 105.0,
  "mono_sub_mix": 0.45,
  "movement": {
    "enabled": true,
    "amount": 0.12
  },
  "hooklift": {
    "enabled": true,
    "mix": 0.24,
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
    "shifts": 2,
    "model_sr": 44100,
    "sr": 48000,
    "sources": [
      "drums",
      "bass",
      "other",
      "vocals"
    ]
  },
  "runtime_sec": 657.6331095695496,
  "out_path": "C:\\Users\\goku\\LLM_uncensored\\Scripts\\Mastered\\Vegas_top_teir__MELODIC_TRAP.wav"
}
```
