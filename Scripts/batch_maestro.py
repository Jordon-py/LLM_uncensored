#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Batch Maestro Runner
--------------------
Runs auralmind_match_maestro_v7_0.py across a folder of target audio files
or a provided list of target file paths.

Design goals:
- Zero quoting headaches (uses subprocess args list, not shell strings)
- Auto output + report naming per target
- Progress visibility + logging
- Robust filtering for audio file extensions

Usage (PowerShell examples):
1) Folder mode:
   python batch_maestro.py `
     --maestro "C:\\Users\\goku\\LLM_uncensored\\auralmind_match_maestro_v7_0.py" `
     --reference "C:\\Users\\goku\\Downloads\\Brent Faiyaz - Pistachios [Official Video].mp3" `
     --targets-dir "C:\\Users\\goku\\LLM_uncensored\\Scripts\\notMastered\\test" `
     --out-dir "C:\\Users\\goku\\LLM_uncensored\\Scripts\\Mastered" `
     --report-dir "C:\\Users\\goku\\LLM_uncensored\\Scripts\\Mastered_reports" `
     --preset PUNCH

2) Explicit list mode:
   python batch_maestro.py --maestro "...\auralmind_match_maestro_v7_0.py" --reference "...\ref.mp3" `
     --targets "...\song1.wav" "...\song2.wav" --out-dir "...\out" --report-dir "...\reports" --preset AIRY

Optional:
- Install progress bar:
  pip install tqdm
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional

# Optional progress bar
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None


AUDIO_EXTS = {
    ".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".aiff", ".aif", ".wma"
}


def slugify_stem(p: Path) -> str:
    """
    Make filenames safe + consistent for output naming.
    Keeps it readable, avoids weird symbols.
    """
    s = p.stem.strip()
    # Replace common separators with underscores
    s = s.replace(" ", "_").replace("-", "_")
    # Remove characters that are annoying on Windows paths
    keep = []
    for ch in s:
        if ch.isalnum() or ch in {"_", "."}:
            keep.append(ch)
        else:
            keep.append("_")
    # Collapse multiple underscores
    out = "".join(keep)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def iter_targets_from_dir(targets_dir: Path, recursive: bool) -> List[Path]:
    """
    Collect audio targets from a directory, filtered by extension.
    """
    if not targets_dir.exists():
        raise FileNotFoundError(f"targets-dir not found: {targets_dir}")

    files: Iterable[Path]
    if recursive:
        files = targets_dir.rglob("*")
    else:
        files = targets_dir.glob("*")

    targets = [p for p in files if p.is_file() and p.suffix.lower() in AUDIO_EXTS]
    targets.sort(key=lambda x: x.name.lower())
    return targets


def build_command(
    maestro_script: Path,
    reference: Path,
    target: Path,
    out_path: Path,
    report_path: Path,
    preset: str,
    extra_args: Optional[List[str]] = None,
) -> List[str]:
    """
    Build subprocess command as an args list (safe on Windows, no manual quoting).
    """
    cmd = [
        sys.executable,
        str(maestro_script),
        "--reference", str(reference),
        "--target", str(target),
        "--out", str(out_path),
        "--report", str(report_path),
    ]

    # If/when you add --preset into Maestro, this becomes active:
    # cmd += ["--preset", preset]

    # Until then, we inject your preset as a parameter bundle in extra_args.
    # You can map "PUNCH/AIRY/LOUD" to the exact flag set below.
    if extra_args:
        cmd += extra_args

    return cmd


def preset_to_args(preset: str) -> List[str]:
    """
    Map one word -> your exact tuning flags.
    This lets you run:
      --preset PUNCH
    while still calling v7.0 (which currently has no --preset arg).
    """
    p = preset.strip().upper()

    # Common mandatory defaults you requested
    base = [
        "--enable_stems", "--demucs_model", "htdemucs", "--demucs_device", "cpu",
        "--target_peak_dbfs", "-1.0",
    ]

    if p == "PUNCH":
        return base + [
            "--target_lufs", "-11.0",
            "--fir_taps", "2049",
            "--match_strength", "0.85",
            "--max_eq_db", "7",
            "--eq_smooth_hz", "80",
            "--enable_key_glow", "--glow_gain_db", "0.85", "--glow_mix", "0.55",
            "--enable_spatial", "--stereo_width_mid", "1.07", "--stereo_width_hi", "1.25",
            "--enable_movement", "--rhythm_amount", "0.12",
            "--enable_transient_restore", "--attack_restore_db", "1.2", "--attack_restore_mix", "0.60",
        ]

    if p == "AIRY":
        return base + [
            "--target_lufs", "-11.0",
            "--match_strength", "0.78",
            "--max_eq_db", "6",
            "--eq_smooth_hz", "95",
            "--match_strength_hi_factor", "0.72",
            "--enable_key_glow", "--glow_gain_db", "0.95", "--glow_mix", "0.60",
            "--enable_spatial", "--stereo_width_mid", "1.06", "--stereo_width_hi", "1.28",
            "--enable_movement", "--rhythm_amount", "0.10",
        ]

    if p == "LOUD":
        return base + [
            "--target_lufs", "-10.2",
            "--clip_drive_db", "2.6",
            "--clip_mix", "0.25",
            "--tp_oversample", "8",
            "--finalize_iters", "4",
            "--ceiling_chase_strength", "1.15",
            "--enable_transient_restore", "--attack_restore_db", "1.0", "--attack_restore_mix", "0.55",
        ]

    raise ValueError(f"Unknown preset: {preset}. Use PUNCH | AIRY | LOUD")


def run_batch(
    maestro_script: Path,
    reference: Path,
    targets: List[Path],
    out_dir: Path,
    report_dir: Path,
    preset: str,
    dry_run: bool,
    continue_on_error: bool,
    logger: logging.Logger,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    extra_args = preset_to_args(preset)
    total = len(targets)

    iterator = targets
    if tqdm is not None:
        iterator = tqdm(targets, desc=f"Maestro {preset}", unit="file")  # type: ignore

    failures = 0

    for i, target in enumerate(iterator, start=1):
        safe_name = slugify_stem(target)
        out_path = out_dir / f"{safe_name}__{preset.upper()}_v7_0.wav"
        report_path = report_dir / f"{safe_name}__{preset.upper()}_v7_0_Report.md"

        cmd = build_command(
            maestro_script=maestro_script,
            reference=reference,
            target=target,
            out_path=out_path,
            report_path=report_path,
            preset=preset,
            extra_args=extra_args,
        )

        logger.info("(%d/%d) Target: %s", i, total, target)
        logger.info("Out: %s", out_path)
        logger.info("Report: %s", report_path)
        logger.debug("CMD: %s", " ".join(cmd))

        if dry_run:
            print("DRY RUN:", " ".join(cmd))
            continue

        proc = subprocess.run(cmd, capture_output=True, text=True)

        if proc.returncode != 0:
            failures += 1
            logger.error("FAILED (%s): exit=%d", target, proc.returncode)
            logger.error("STDERR:\n%s", proc.stderr.strip())
            logger.error("STDOUT:\n%s", proc.stdout.strip())

            if not continue_on_error:
                raise SystemExit(f"Batch stopped due to failure on: {target}")

        else:
            logger.info("OK: %s", target)

    if failures:
        logger.warning("Batch finished with %d failure(s).", failures)
    else:
        logger.info("Batch finished successfully (0 failures).")


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch runner for auralmind_match_maestro_v7_0.py")
    ap.add_argument("--maestro", required=True, type=str, help="Path to auralmind_match_maestro_v7_0.py")
    ap.add_argument("--reference", required=True, type=str, help="Reference audio path")
    ap.add_argument("--targets-dir", type=str, default="", help="Folder containing target audio files")
    ap.add_argument("--targets", nargs="*", default=[], help="Explicit list of target file paths")
    ap.add_argument("--out-dir", required=True, type=str, help="Output folder for mastered .wav files")
    ap.add_argument("--report-dir", required=True, type=str, help="Output folder for report .md files")
    ap.add_argument("--preset", type=str, default="PUNCH", help="PUNCH | AIRY | LOUD")
    ap.add_argument("--recursive", action="store_true", help="Scan targets-dir recursively")
    ap.add_argument("--dry-run", action="store_true", help="Print commands without running")
    ap.add_argument("--continue-on-error", action="store_true", help="Keep going if a file fails")
    ap.add_argument("--log-file", type=str, default="batch_maestro.log", help="Log file name/path")
    ap.add_argument("--log-level", type=str, default="INFO", help="DEBUG | INFO | WARNING | ERROR")
    args = ap.parse_args()

    # Logging setup
    logger = logging.getLogger("batch_maestro")
    logger.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")

    fh = logging.FileHandler(args.log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    maestro_script = Path(args.maestro).expanduser().resolve()
    reference = Path(args.reference).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    report_dir = Path(args.report_dir).expanduser().resolve()

    if not maestro_script.exists():
        raise FileNotFoundError(f"Maestro script not found: {maestro_script}")
    if not reference.exists():
        raise FileNotFoundError(f"Reference not found: {reference}")

    targets: List[Path] = []
    if args.targets_dir:
        targets += iter_targets_from_dir(Path(args.targets_dir).expanduser().resolve(), recursive=args.recursive)
    if args.targets:
        targets += [Path(p).expanduser().resolve() for p in args.targets]

    # De-dup by full path
    targets = sorted(list({t: None for t in targets}.keys()), key=lambda x: x.name.lower())

    if not targets:
        raise SystemExit("No targets found. Provide --targets-dir or --targets list.")

    run_batch(
        maestro_script=maestro_script,
        reference=reference,
        targets=targets,
        out_dir=out_dir,
        report_dir=report_dir,
        preset=args.preset,
        dry_run=args.dry_run,
        continue_on_error=args.continue_on_error,
        logger=logger,
    )


if __name__ == "__main__":
    main()
