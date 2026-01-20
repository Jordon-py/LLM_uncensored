import numpy as np
import pytest

pytest.importorskip("librosa")

from stereo_field_designer_v3.audio_engine import AudioEngine, StereoFieldConfig
from stereo_field_designer_v3.system_utils import TestSuite
from stereo_field_designer_v3.width_controller import NeuralMorpher


def test_engine_process_stability():
    config = StereoFieldConfig()
    engine = AudioEngine(config)
    audio = TestSuite.generate_example(sr=config.sample_rate, seconds=1.0)
    processed, analysis = engine.process(audio, sr=config.sample_rate)

    assert processed.shape == audio.shape
    assert np.isfinite(processed).all()
    assert analysis.width_index >= 0.0


def test_neural_morph_curve_shape():
    morpher = NeuralMorpher()
    mag = np.abs(np.random.randn(1025, 12)).astype(np.float32)
    curve = morpher.morph_curve(mag, strength=0.5)
    assert curve.shape[0] == mag.shape[0]
    assert np.isfinite(curve).all()
