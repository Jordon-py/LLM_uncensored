# AuralMind Maestro v7.3 StereoAI — Report

## Summary
- Preset: **melodic_trap**
- Sample rate: **48000 Hz**
- LUFS (pre): **-16.93**
- LUFS (post): **-17.20**
- True peak (approx): **-1.00 dBFS**
- Limiter min gain (approx GR): **-10.23 dB**

## Low-end / music theory anchors
- Estimated sub fundamental f0: **46.5087890625 Hz**
- Mono-sub v2 cutoff: **90.692138671875 Hz**
- Mono-sub v2 adaptive mix: **0.45151652598257397**

## Stereo enhancements
- Spatial Realism Enhancer: frequency-dependent width + correlation guard
- NEW CGMS MicroShift: micro-delay applied to SIDE high-band only, correlation-guarded

## Movement / HookLift
- Movement enabled: **True** (amount=0.16)
- HookLift enabled: **True** (mix=0.2)
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
  "lufs_pre": -16.929468493726333,
  "lufs_post": -17.195873420724197,
  "true_peak_dbfs": -1.0000002900129947,
  "limiter_min_gain_db": -10.230407040244751,
  "sub_f0_hz": 46.5087890625,
  "mono_sub_cutoff_hz": 90.692138671875,
  "mono_sub_mix": 0.45151652598257397,
  "movement": {
    "enabled": true,
    "amount": 0.16
  },
  "hooklift": {
    "enabled": true,
    "mix": 0.2,
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
    "shifts": 1,
    "model_sr": 44100,
    "sr": 48000,
    "sources": [
      "drums",
      "bass",
      "other",
      "vocals"
    ]
  },
  "runtime_sec": 380.3114185333252,
  "out_path": "C:\\Users\\goku\\LLM_uncensored\\Scripts\\Mastered\\Vegas_fireee__HIFI.wav"
}
```
