from __future__ import annotations

import argparse
from pathlib import Path

from .audio_engine import AudioEngine, StereoFieldConfig
from .gui_handler import GUIHandler
from .system_utils import ConfigManager, GPUManager


class StereoFieldDesigner:
    """Orchestrator connecting engine, analyzer, and GUI."""

    def __init__(self, config_path: str | None = None):
        self.config_manager = ConfigManager(config_path=config_path)
        preset = self.config_manager.get_preset("Wide Dream")

        config = StereoFieldConfig()
        for key, value in preset.items():
            if hasattr(config, key):
                setattr(config, key, value)

        self.engine = AudioEngine(config, gpu=GPUManager(preferred="auto"))
        self.gui = None

    def run(self):
        if self.gui is None:
            self.gui = GUIHandler(self.engine, self.config_manager)
        self.gui.launch()


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stereo Field Designer v3")
    parser.add_argument("--cli", action="store_true", help="Run headless CLI processing.")
    parser.add_argument("--in", dest="inp", help="Input audio path (wav/flac/mp3).")
    parser.add_argument("--out", dest="out", help="Output audio path (wav).")
    parser.add_argument("--preset", default="Wide Dream", help="Preset name.")
    return parser


def main():
    args = build_cli().parse_args()
    designer = StereoFieldDesigner()
    preset = designer.config_manager.get_preset(args.preset)
    for key, value in preset.items():
        if hasattr(designer.engine.config, key):
            setattr(designer.engine.config, key, value)

    if args.cli:
        if not args.inp or not args.out:
            raise SystemExit("--cli requires --in and --out")
        analysis = designer.engine.process_file(args.inp, args.out)
        print(f"Exported: {args.out}")
        print(f"LUFS: {analysis.lufs:.2f} | Corr: {analysis.correlation:.2f} | Width: {analysis.width_index:.2f}")
    else:
        designer.run()


if __name__ == "__main__":
    main()
