# AuralMind Maestro v7.3 StereoAI — Report

## Summary
- Preset: **hi_fi_streaming**
- Sample rate: **48000 Hz**
- LUFS (pre): **-17.55**
- LUFS (post): **-18.88**
- True peak (approx): **-1.00 dBFS**
- Limiter min gain (approx GR): **-12.77 dB**

## Low-end / music theory anchors
- Estimated sub fundamental f0: **43.02978515625 Hz**
- Mono-sub v2 cutoff: **83.9080810546875 Hz**
- Mono-sub v2 adaptive mix: **0.45**

## Stereo enhancements
- Spatial Realism Enhancer: frequency-dependent width + correlation guard
- NEW CGMS MicroShift: micro-delay applied to SIDE high-band only, correlation-guarded

## Movement / HookLift
- Movement enabled: **True** (amount=0.08)
- HookLift enabled: **True** (mix=0.2)
  - Auto mask percentile: **75.0**

## Stem separation (HT-Demucs)
- Enabled: **True**
- Model: **htdemucs**
- Sources: **['drums', 'bass', 'other', 'vocals']**

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
  "lufs_pre": -17.549890041314125,
  "lufs_post": -18.878403181266414,
  "true_peak_dbfs": -1.0000002900129947,
  "limiter_min_gain_db": -12.773157188544884,
  "sub_f0_hz": 43.02978515625,
  "mono_sub_cutoff_hz": 83.9080810546875,
  "mono_sub_mix": 0.45,
  "movement": {
    "enabled": true,
    "amount": 0.08
  },
  "hooklift": {
    "enabled": true,
    "mix": 0.2,
    "air_hz": 8500.0,
    "width_gain": 0.18,
    "air_gain": 0.14,
    "auto": true,
    "auto_percentile": 75.0
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
  "runtime_sec": 376.03327775001526,
  "out_path": "C:\\Users\\goku\\LLM_uncensored\\Scripts\\Mastered\\Vegas_top_teir__HIFI.wav"
}
```
