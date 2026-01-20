from __future__ import annotations

from pathlib import Path
import os

import numpy as np
import soundfile as sf

from .audio_engine import AudioEngine
from .system_utils import ConfigManager

try:
    import dearpygui.dearpygui as dpg  # type: ignore
except Exception:  # pragma: no cover - GUI optional in headless tests
    dpg = None


class StereoVisualizer:
    """Stereo field energy visualizer (Lissajous + waveform)."""

    def __init__(self):
        self.scope_tag = "sfd_scope_plot"
        self.scope_series = "sfd_scope_series"
        self.wave_tag = "sfd_wave_plot"
        self.wave_left = "sfd_wave_left"
        self.wave_right = "sfd_wave_right"
        self.spec_tag = "sfd_spec_plot"
        self.spec_series = "sfd_spec_series"

    def build(self):
        with dpg.plot(label="Stereo Field Energy", height=250, tag=self.scope_tag):
            dpg.add_plot_legend()
            dpg.add_plot_axis(dpg.mvXAxis, label="Left")
            dpg.add_plot_axis(dpg.mvYAxis, label="Right")
            dpg.add_scatter_series([], [], label="Stereo Orbit", parent=dpg.last_item(), tag=self.scope_series)

        with dpg.plot(label="Waveform", height=200, tag=self.wave_tag):
            dpg.add_plot_legend()
            dpg.add_plot_axis(dpg.mvXAxis, label="Time")
            with dpg.plot_axis(dpg.mvYAxis, label="Amplitude"):
                dpg.add_line_series([], [], label="Left", tag=self.wave_left)
                dpg.add_line_series([], [], label="Right", tag=self.wave_right)

        with dpg.plot(label="Spectrum", height=200, tag=self.spec_tag):
            dpg.add_plot_legend()
            dpg.add_plot_axis(dpg.mvXAxis, label="Hz")
            with dpg.plot_axis(dpg.mvYAxis, label="dB"):
                dpg.add_line_series([], [], label="Avg Spectrum", tag=self.spec_series)

    def update(self, stereo: np.ndarray, sr: int):
        if stereo is None or stereo.size == 0:
            return
        left = stereo[:, 0]
        right = stereo[:, 1]

        points = min(3000, len(left))
        idx = np.linspace(0, len(left) - 1, points).astype(int)
        l = left[idx]
        r = right[idx]

        dpg.set_value(self.scope_series, [l.tolist(), r.tolist()])

        wave_x = np.linspace(0, 1.0, points).tolist()
        dpg.set_value(self.wave_left, [wave_x, l.tolist()])
        dpg.set_value(self.wave_right, [wave_x, r.tolist()])

        mono = 0.5 * (left + right)
        n_fft = int(min(8192, mono.size))
        if n_fft > 16:
            window = np.hanning(n_fft)
            fft_mag = np.abs(np.fft.rfft(mono[:n_fft] * window)) + 1e-9
            fft_db = 20.0 * np.log10(fft_mag / np.max(fft_mag))
            freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
            keep = min(400, freqs.size)
            idx_f = np.linspace(0, freqs.size - 1, keep).astype(int)
            dpg.set_value(self.spec_series, [freqs[idx_f].tolist(), fft_db[idx_f].tolist()])


class GUIHandler:
    """Standalone GUI for Stereo Field Designer v3."""

    def __init__(self, engine: AudioEngine, config_manager: ConfigManager):
        if dpg is None:
            raise RuntimeError("DearPyGui is not installed. pip install dearpygui")
        self.engine = engine
        self.config_manager = config_manager
        self.visualizer = StereoVisualizer()
        self.audio = None
        self.processed = None
        self.analysis = None
        self.input_path = None
        self.output_path = None
        self.sample_rate = engine.config.sample_rate
        self.control_tags: dict[str, str] = {}
        self.accent = (0, 220, 200)
        self.accent_hot = (155, 255, 245)
        self.muted = (140, 150, 165)

    def _apply_theme(self):
        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvAll):
                def add_color(attr_name, color, category=None):
                    if hasattr(dpg, attr_name):
                        dpg.add_theme_color(getattr(dpg, attr_name), color, category=category)

                def add_style(attr_name, *values):
                    if hasattr(dpg, attr_name):
                        dpg.add_theme_style(getattr(dpg, attr_name), *values)

                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (10, 12, 18))
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (16, 18, 26))
                dpg.add_theme_color(dpg.mvThemeCol_PopupBg, (18, 20, 30))
                dpg.add_theme_color(dpg.mvThemeCol_Border, (35, 38, 48))
                dpg.add_theme_color(dpg.mvThemeCol_TitleBg, (16, 18, 28))
                dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, (26, 30, 42))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (22, 26, 36))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (30, 36, 48))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (34, 42, 58))
                dpg.add_theme_color(dpg.mvThemeCol_Button, (30, 75, 110))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (35, 110, 150))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (45, 140, 175))
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, (0, 200, 255))
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, self.accent_hot)
                dpg.add_theme_color(dpg.mvThemeCol_CheckMark, self.accent)
                dpg.add_theme_color(dpg.mvThemeCol_Text, (235, 240, 250))
                dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, (120, 130, 145))
                dpg.add_theme_color(dpg.mvThemeCol_PlotLines, self.accent)
                dpg.add_theme_color(dpg.mvThemeCol_PlotLinesHovered, self.accent_hot)
                dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram, (200, 110, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg, (14, 16, 24))
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab, (35, 40, 55))
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabHovered, (45, 55, 75))
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabActive, (65, 75, 95))
                add_style("mvStyleVar_WindowRounding", 8)
                add_style("mvStyleVar_ChildRounding", 8)
                add_style("mvStyleVar_FrameRounding", 6)
                add_style("mvStyleVar_PopupRounding", 6)
                add_style("mvStyleVar_GrabRounding", 8)
                add_style("mvStyleVar_ScrollbarRounding", 8)
                add_style("mvStyleVar_WindowPadding", 14, 12)
                add_style("mvStyleVar_FramePadding", 10, 6)
                add_style("mvStyleVar_ItemSpacing", 8, 6)
                add_style("mvStyleVar_ItemInnerSpacing", 6, 4)
                add_style("mvStyleVar_ScrollbarSize", 12)
                add_style("mvStyleVar_GrabMinSize", 10)
                add_style("mvStyleVar_PlotPadding", 12, 10)

                add_color("mvPlotCol_PlotBg", (14, 16, 24), dpg.mvThemeCat_Plots)
                add_color("mvPlotCol_PlotBorder", (34, 38, 48), dpg.mvThemeCat_Plots)
                add_color("mvPlotCol_AxisGrid", (30, 32, 40), dpg.mvThemeCat_Plots)
                add_color("mvPlotCol_LegendBg", (16, 18, 26), dpg.mvThemeCat_Plots)
                add_color("mvPlotCol_LegendBorder", (34, 38, 48), dpg.mvThemeCat_Plots)
                add_color("mvPlotCol_LegendText", (210, 218, 230), dpg.mvThemeCat_Plots)
                add_color("mvPlotCol_TitleText", (220, 230, 245), dpg.mvThemeCat_Plots)
        dpg.bind_theme(theme)

    def _update_config(self, sender, app_data, user_data):
        key = user_data
        setattr(self.engine.config, key, app_data)

    def _add_hint(self, item, text: str):
        with dpg.tooltip(item):
            dpg.add_text(text, wrap=280)

    def _apply_preset(self, sender, app_data):
        preset = self.config_manager.get_preset(app_data)
        for key, value in preset.items():
            if hasattr(self.engine.config, key):
                setattr(self.engine.config, key, value)
                if key in self.control_tags:
                    dpg.set_value(self.control_tags[key], value)

    def _load_audio(self, path: str):
        audio, sr = self.engine.load_audio(path)
        self.audio = audio
        self.sample_rate = sr
        self.input_path = path
        self.output_path = str(Path(path).with_suffix(".sfd_v3.wav"))
        self.visualizer.update(self.audio, sr)
        dpg.set_value("sfd_input_path", path)
        dpg.set_value("sfd_output_path", self.output_path)

    def _on_file_dialog(self, sender, app_data):
        path = self._resolve_dialog_path(app_data)
        if path:
            self._load_audio(path)

    def _on_drop(self, sender, app_data):
        if app_data:
            self._load_audio(app_data[0])

    def _resolve_dialog_path(self, app_data) -> str | None:
        if not app_data:
            return None
        selections = app_data.get("selections")
        if selections:
            return next(iter(selections.values()))

        path = app_data.get("file_path_name") or app_data.get("file_name")
        if not path:
            return None
        if path.endswith(".*"):
            base = Path(path[:-2])
            file_name = app_data.get("file_name")
            current_path = app_data.get("current_path")
            if file_name and current_path and file_name != "*.*":
                candidate = Path(current_path) / file_name
                if candidate.exists():
                    return str(candidate)

            candidates = list(base.parent.glob(base.name + ".*"))
            for ext in (".wav", ".flac", ".mp3", ".aiff", ".aif", ".ogg", ".m4a"):
                for candidate in candidates:
                    if candidate.suffix.lower() == ext:
                        return str(candidate)
            if candidates:
                return str(candidates[0])
            return None
        return path

    def _process(self):
        if self.audio is None:
            return
        processed, analysis = self.engine.process(self.audio, sr=self.engine.config.sample_rate)
        self.processed = processed
        self.analysis = analysis
        self.visualizer.update(self.processed, self.sample_rate)
        dpg.set_value("sfd_lufs", f"{analysis.lufs:.2f} LUFS")
        dpg.set_value("sfd_corr", f"{analysis.correlation:.2f}")
        dpg.set_value("sfd_width", f"{analysis.width_index:.2f}")
        dpg.set_value("sfd_emotion", f"{analysis.stereo_emotion_index:.2f}")
        corr_val = float(np.clip((analysis.correlation + 1.0) / 2.0, 0.0, 1.0))
        width_val = float(np.clip(analysis.width_index / 0.6, 0.0, 1.0))
        emotion_val = float(np.clip(analysis.stereo_emotion_index, 0.0, 1.0))
        dpg.set_value("sfd_corr_bar", corr_val)
        dpg.set_value("sfd_width_bar", width_val)
        dpg.set_value("sfd_emotion_bar", emotion_val)
        dpg.configure_item("sfd_corr_bar", overlay=f"Correlation {analysis.correlation:.2f}")
        dpg.configure_item("sfd_width_bar", overlay=f"Width {analysis.width_index:.2f}")
        dpg.configure_item("sfd_emotion_bar", overlay=f"Emotion {analysis.stereo_emotion_index:.2f}")

    def _export(self):
        if self.processed is None:
            return
        output_path = dpg.get_value("sfd_output_path")
        if not output_path:
            return
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), self.processed, self.sample_rate)

    def _process_export(self):
        self._process()
        self._export()

    def _open_output_folder(self):
        output_path = dpg.get_value("sfd_output_path")
        if not output_path:
            return
        folder = Path(output_path).parent
        if folder.exists():
            os.startfile(folder)

    def _build_controls(self):
        presets = self.config_manager.list_presets()
        dpg.add_text("Stereo Field Designer v3", color=self.accent)
        dpg.add_text("Spatial mastering for melodic trap", color=self.muted)
        dpg.add_spacer()
        combo = dpg.add_combo(
            presets,
            default_value=presets[0] if presets else "Wide Dream",
            label="Artistic Profile",
            callback=self._apply_preset,
            tag="sfd_preset_combo",
        )
        self._add_hint(combo, "Preset macro that sets width, motion, air, and mono safety defaults.")
        if hasattr(dpg, "add_spacer"):
            dpg.add_spacer(height=4)
        else:
            dpg.add_spacing(count=1)
        with dpg.group(horizontal=True):
            btn_load = dpg.add_button(label="Load Audio", width=140, callback=lambda: dpg.show_item("sfd_file_dialog"), tag="sfd_btn_load")
            btn_process = dpg.add_button(label="Process", width=140, callback=lambda: self._process(), tag="sfd_btn_process")
        with dpg.group(horizontal=True):
            btn_export = dpg.add_button(label="Export", width=140, callback=lambda: self._export(), tag="sfd_btn_export")
            btn_process_export = dpg.add_button(label="Process + Export", width=140, callback=lambda: self._process_export(), tag="sfd_btn_process_export")
        btn_open = dpg.add_button(label="Open Output Folder", width=286, callback=lambda: self._open_output_folder(), tag="sfd_btn_open")
        input_path = dpg.add_input_text(label="Input", tag="sfd_input_path", width=286, readonly=True)
        output_path = dpg.add_input_text(label="Output", tag="sfd_output_path", width=286)
        self._add_hint(btn_load, "Load a stereo audio file for analysis and processing.")
        self._add_hint(btn_process, "Run the stereo engine and update meters/plots.")
        self._add_hint(btn_export, "Export the last processed audio to Output path.")
        self._add_hint(btn_process_export, "One-click process and export.")
        self._add_hint(btn_open, "Open the output folder in your file explorer.")
        self._add_hint(input_path, "Read-only path for the loaded audio.")
        self._add_hint(output_path, "Target path for exported audio.")
        dpg.add_separator()

        def slider(label, attr, min_v, max_v):
            tag = f"sfd_{attr}"
            self.control_tags[attr] = tag
            dpg.add_slider_float(
                label=label,
                min_value=min_v,
                max_value=max_v,
                default_value=getattr(self.engine.config, attr),
                callback=self._update_config,
                user_data=attr,
                tag=tag,
                width=260,
            )
            return tag

        dpg.add_text("Spatial Engine", color=self.muted)
        bypass = dpg.add_checkbox(
            label="Bypass Processing",
            default_value=self.engine.config.bypass_processing,
            callback=self._update_config,
            user_data="bypass_processing",
            tag="sfd_bypass",
        )
        self._add_hint(bypass, "A/B compare. When enabled, processing is bypassed.")
        tag = slider("Width (dB)", "width_db", -0.5, 3.0)
        self._add_hint(tag, "Base stereo width. Small moves keep mono safe.")
        tag = slider("Motion", "motion_depth", 0.0, 1.0)
        self._add_hint(tag, "Tempo-synced width movement. Higher = more animation.")
        tag = slider("Morph", "morph_strength", 0.0, 1.0)
        self._add_hint(tag, "Neural spectral morph texture. Keep low for subtlety.")
        tag = slider("Air", "air_strength", 0.0, 1.0)
        self._add_hint(tag, "Bright bias for widening decisions and sparkle.")
        tag = slider("Warmth", "warmth_strength", 0.0, 1.0)
        self._add_hint(tag, "Mid focus bias. Higher reduces width for cohesion.")
        dpg.add_separator()
        dpg.add_text("Psychoacoustics", color=self.muted)
        tag = slider("Psycho Width", "psycho_strength", 0.0, 1.0)
        self._add_hint(tag, "Ear-weighted phase widening. Keep below 0.7 for mono safety.")
        tag = slider("Phase (deg)", "phase_degrees", 0.0, 20.0)
        self._add_hint(tag, "Max phase rotation for width. Lower keeps center tighter.")
        tag = slider("HRTF Tint", "hrtf_amount", 0.0, 1.0)
        self._add_hint(tag, "Adds small interaural level differences for depth.")
        tag = slider("Mono Guard Hz", "mono_guard_hz", 40.0, 220.0)
        self._add_hint(tag, "Below this stays mono to keep kick and 808 stable.")
        tag = slider("Mono Guard Slope", "mono_guard_slope_hz", 40.0, 140.0)
        self._add_hint(tag, "Smoother transition into width above mono guard.")
        tag = slider("Correlation Guard", "correlation_guard", -0.2, 0.5)
        self._add_hint(tag, "Reduces side energy if correlation drops too low.")
        dpg.add_separator()
        dpg.add_text("Loudness", color=self.muted)
        tag = slider("Loudness Target (LUFS)", "target_lufs", -20.0, -8.0)
        self._add_hint(tag, "Auto gain to hit target LUFS before peak control.")
        soft_clip = dpg.add_checkbox(
            label="Soft Clip",
            default_value=self.engine.config.enable_soft_clip,
            callback=self._update_config,
            user_data="enable_soft_clip",
            tag="sfd_soft_clip",
        )
        self._add_hint(soft_clip, "Gentle peak shaping to lift loudness without hard limiting.")
        tag = slider("Clip Drive", "soft_clip_drive", 1.0, 4.0)
        self._add_hint(tag, "Strength of soft clip. More = louder, less dynamic.")

        dpg.add_separator()
        dpg.add_text("Meters", color=self.muted)
        lufs = dpg.add_text("LUFS: --", tag="sfd_lufs")
        corr = dpg.add_text("Correlation: --", tag="sfd_corr")
        width = dpg.add_text("Width Index: --", tag="sfd_width")
        emotion = dpg.add_text("Stereo Emotion: --", tag="sfd_emotion")
        dpg.add_progress_bar(default_value=0.5, tag="sfd_corr_bar")
        dpg.add_progress_bar(default_value=0.0, tag="sfd_width_bar")
        dpg.add_progress_bar(default_value=0.0, tag="sfd_emotion_bar")
        self._add_hint(lufs, "Integrated loudness after processing.")
        self._add_hint(corr, "Stereo correlation from -1 to 1. Lower is wider but riskier.")
        self._add_hint(width, "Side-to-mid energy ratio. Higher means wider image.")
        self._add_hint(emotion, "Perceptual index from width, air, and focus.")

    def launch(self):
        dpg.create_context()
        if hasattr(dpg, "set_viewport_drop_callback"):
            dpg.set_viewport_drop_callback(self._on_drop)

        with dpg.window(label="Stereo Field Designer v3", width=1000, height=700):
            with dpg.group(horizontal=True):
                with dpg.child_window(width=320, height=-1):
                    self._build_controls()
                with dpg.child_window(width=-1, height=-1):
                    self.visualizer.build()

        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            callback=self._on_file_dialog,
            tag="sfd_file_dialog",
            width=700,
            height=400,
            file_count=1,
        ):
            dpg.add_file_extension(".*")
            dpg.add_file_extension(".wav", color=(0, 200, 255))
            dpg.add_file_extension(".flac", color=(0, 200, 255))
            dpg.add_file_extension(".mp3", color=(0, 200, 255))

        self._apply_theme()
        dpg.create_viewport(title="Stereo Field Designer v3", width=1100, height=750)
        if hasattr(dpg, "set_viewport_clear_color"):
            dpg.set_viewport_clear_color((8, 10, 14, 255))
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()
