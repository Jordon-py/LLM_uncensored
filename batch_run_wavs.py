from __future__ import annotations
"""python batch_run_wavs.py
  --folder "C:/Users/goku/Downloads/"
  --script "C:/Users/goku/LLM_uncensored/melodic_trap_master_v7.py"
  --recursive
  --out_dir "C:/Users/iProg/Desktop/"
  --preset "Hi Fidelity"
  --limit 3
"""
import argparse
import subprocess
import sys
from pathlib import Path


def find_wavs(folder: Path, recursive: bool) -> list[Path]:
    """
    Returns a sorted list of .wav files inside folder.

    If recursive=True, searches subfolders too.
    """
    if recursive:
        wavs = folder.rglob("*.wav")
    else:
        wavs = folder.glob("*.wav")

    return sorted(p for p in wavs if p.is_file())


def run_script_on_file(
    script_path: Path,
    wav_path: Path,
    out_path: Path,
    preset: str | None,
    extra_args: list[str],
) -> int:
    """
    Runs: python <script_path> --cli --in <wav_path> --out <out_path> [--preset <preset>] <extra_args...>
    Returns the exit code.
    """
    cmd = [
        sys.executable,                 # the current python interpreter
        str(script_path),
        "--cli",
        "--in",
        str(wav_path),
        "--out",
        str(out_path),
    ]
    if preset:
        cmd.extend(["--preset", preset])
    cmd.extend(extra_args)

    print(f"\n> Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout:
        print("STDOUT:\n", result.stdout)

    if result.stderr:
        print("STDERR:\n", result.stderr)

    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch process .wav files in a folder.")
    parser.add_argument("--folder", required=True, help="Folder containing .wav files.")
    parser.add_argument("--script", required=True, help="Script to run on each .wav file.")
    parser.add_argument("--recursive", action="store_true", help="Search subfolders too.")
    parser.add_argument("--out_dir", help="Output folder for processed files.")
    parser.add_argument("--preset", help="Preset name to pass to the script.")
    parser.add_argument("--suffix", default="_sfd_v3", help="Suffix appended to output file names.")
    parser.add_argument("--limit", type=int, default=0, help="Optional max files to process (0 = no limit).")
    parser.add_argument(
        "--",
        dest="pass_through",
        nargs=argparse.REMAINDER,
        help="Everything after -- is passed to the called script.",
    )
    args = parser.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    script_path = Path(args.script).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else None
    extra_args = args.pass_through or []

    if not folder.exists() or not folder.is_dir():
        print(f"ERROR: Folder not found: {folder}")
        return 2

    if not script_path.exists() or not script_path.is_file():
        print(f"ERROR: Script not found: {script_path}")
        return 2

    wav_files = find_wavs(folder, recursive=args.recursive)

    if not wav_files:
        print(f"No .wav files found in: {folder}")
        return 0
    if args.limit and args.limit > 0:
        wav_files = wav_files[: args.limit]

    print(f"Found {len(wav_files)} WAV files.")
    failures: list[Path] = []

    for wav_path in wav_files:
        target_dir = out_dir or wav_path.parent
        target_dir.mkdir(parents=True, exist_ok=True)
        suffix = args.suffix or ""
        if suffix.lower().endswith(".wav"):
            out_name = f"{wav_path.stem}{suffix}"
        else:
            out_name = f"{wav_path.stem}{suffix}.wav"
        out_path = target_dir / out_name
        code = run_script_on_file(script_path, wav_path, out_path, args.preset, extra_args)
        if code != 0:
            failures.append(wav_path)

    print("\n=== Batch Summary ===")
    print(f"Total: {len(wav_files)}")
    print(f"Failed: {len(failures)}")

    if failures:
        print("\nFailed files:")
        for f in failures:
            print(f"- {f}")

        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
