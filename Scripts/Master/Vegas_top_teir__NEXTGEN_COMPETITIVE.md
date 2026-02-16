# AuralMind Maestro v7.3 StereoAI — Report

## Summary
- Preset: **competitive_trap**
- Sample rate: **48000 Hz**
- LUFS (pre): **-17.51**
- LUFS (post): **-18.24**
- True peak (approx): **-1.00 dBFS**
- Limiter min gain (approx GR): **-14.56 dB**

## Low-end / music theory anchors
- Estimated sub fundamental f0: **35.888671875 Hz**
- Mono-sub v2 cutoff: **72.0 Hz**
- Mono-sub v2 adaptive mix: **0.45**

## Stereo enhancements
- Spatial Realism Enhancer: frequency-dependent width + correlation guard
- NEW CGMS MicroShift: micro-delay applied to SIDE high-band only, correlation-guarded

## Movement / HookLift
- Movement enabled: **True** (amount=0.13)
- HookLift enabled: **True** (mix=0.26)
  - Auto mask percentile: **78.0**

## Stem separation (HT-Demucs)
- Enabled: **True**
- Model: **htdemucs**
- Sources: **['drums', 'bass', 'other', 'vocals']**

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
  "lufs_pre": -17.507979330105176,
  "lufs_post": -18.242987900672883,
  "true_peak_dbfs": -0.9999997091223114,
  "limiter_min_gain_db": -14.560664467225005,
  "sub_f0_hz": 35.888671875,
  "mono_sub_cutoff_hz": 72.0,
  "mono_sub_mix": 0.45,
  "movement": {
    "enabled": true,
    "amount": 0.13
  },
  "hooklift": {
    "enabled": true,
    "mix": 0.26,
    "air_hz": 8500.0,
    "width_gain": 0.18,
    "air_gain": 0.14,
    "auto": true,
    "auto_percentile": 78.0
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
  "runtime_sec": 757.8196659088135,
  "out_path": "C:\\Users\\goku\\LLM_uncensored\\Scripts\\Mastered\\Vegas_top_teir__NEXTGEN_COMPETITIVE.wav"
}
```
