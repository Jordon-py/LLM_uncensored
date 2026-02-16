import numpy as np
import soundfile as sf
from pathlib import Path
import subprocess
import time
import os

def create_test_audio(dir_path):
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    
    sr = 48000
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Reference: Balanced spectrum (white noise + some low end)
    ref = np.random.normal(0, 0.1, t.shape)
    # Add some sub
    ref += 0.2 * np.sin(2 * np.pi * 50 * t)
    
    # Target: Dull sweep
    target = 0.5 * np.sin(2 * np.pi * np.geomspace(100, 5000, len(t)) * t)
    
    ref_path = dir_path / "test_ref.wav"
    tgt_path = dir_path / "test_tgt.wav"
    tgt2_path = dir_path / "test_tgt2.wav"
    
    sf.write(ref_path, np.stack([ref, ref], axis=1), sr)
    sf.write(tgt_path, np.stack([target, target], axis=1), sr)
    sf.write(tgt2_path, np.stack([target, target], axis=1), sr)
    
    return ref_path, tgt_path, tgt2_path

def main():
    test_dir = Path("c:/Users/goku/LLM_uncensored/Scripts/test_maestro")
    ref_path, tgt_path, tgt2_path = create_test_audio(test_dir)
    
    script_path = "c:/Users/goku/LLM_uncensored/Scripts/auralmind_match_maestro_v7_3.py"
    
    print("--- Running v7.3 Single File (Enhancements ON) ---")
    out_path = test_dir / "out.wav"
    rep_path = test_dir / "report.md"
    
    cmd = [
        "python", script_path,
        "--reference", str(ref_path),
        "--target", str(tgt_path),
        "--out", str(out_path),
        "--report", str(rep_path),
        "--enable_dynamic_sidechain",
        "--enable_transient_air",
        "--allow_no_stems",
        "--workers", "1"
    ]
    
    start = time.time()
    subprocess.run(cmd, check=True)
    print(f"Single file took: {time.time() - start:.2f}s")
    
    print("\n--- Running v7.3 Batch Mode (Testing Cache) ---")
    batch_out = test_dir / "batch_out"
    batch_rep = test_dir / "batch_reports"
    
    cmd_batch = [
        "python", script_path,
        "--reference", str(ref_path),
        "--target_dir", str(test_dir),
        "--out_dir", str(batch_out),
        "--report_dir", str(batch_rep),
        "--allow_no_stems",
        "--workers", "2",
        "--allow-concurrent-stems"
    ]
    
    start = time.time()
    subprocess.run(cmd_batch, check=True)
    print(f"Batch mode (2 files) took: {time.time() - start:.2f}s")

if __name__ == "__main__":
    main()
