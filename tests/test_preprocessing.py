"""
Tests unitaires du module de prétraitement audio.

On génère des signaux audio synthétiques (sinusoïdes, silence, bruit)
avec numpy/soundfile plutôt que de dépendre de fichiers audio externes,
pour que les tests soient rapides et reproductibles.
"""

import os
import tempfile

import numpy as np
import pytest
import soundfile as sf

from app.preprocessing.audio_preprocessing import (
    preprocess_audio,
    to_mono,
    resample,
    normalize_amplitude,
    check_not_silent,
    check_duration,
    validate_file_format,
)
from app.exceptions import (
    UnsupportedFormatError,
    EmptyAudioError,
    SilentAudioError,
    AudioTooLongError,
)
from app.config import AUDIO_CONFIG


def _make_wav_file(duration_s=2.0, sample_rate=22050, freq=440.0, silent=False, stereo=False):
    """Crée un fichier .wav temporaire contenant une sinusoïde (ou du silence)."""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    signal = np.zeros_like(t) if silent else 0.5 * np.sin(2 * np.pi * freq * t)

    if stereo:
        signal = np.stack([signal, signal], axis=-1)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, signal, sample_rate)
    return tmp.name


class TestValidateFileFormat:
    def test_wav_is_valid(self):
        validate_file_format("audio.wav")  # ne doit pas lever d'exception

    def test_mp3_is_valid(self):
        validate_file_format("audio.mp3")

    def test_unsupported_format_raises(self):
        with pytest.raises(UnsupportedFormatError):
            validate_file_format("audio.ogg")


class TestLoadAndPreprocess:
    def test_preprocess_valid_audio_returns_float32_array(self):
        path = _make_wav_file(duration_s=1.0, sample_rate=22050)
        try:
            result = preprocess_audio(path)
            assert isinstance(result, np.ndarray)
            assert result.dtype == np.float32
        finally:
            os.remove(path)

    def test_preprocess_resamples_to_16khz(self):
        path = _make_wav_file(duration_s=1.0, sample_rate=44100)
        try:
            result = preprocess_audio(path)
            expected_len = AUDIO_CONFIG.TARGET_SAMPLE_RATE  # ~1 seconde à 16kHz
            assert abs(len(result) - expected_len) < 100  # tolérance d'arrondi
        finally:
            os.remove(path)

    def test_preprocess_converts_stereo_to_mono(self):
        path = _make_wav_file(duration_s=1.0, sample_rate=16000, stereo=True)
        try:
            result = preprocess_audio(path)
            assert result.ndim == 1
        finally:
            os.remove(path)

    def test_empty_file_raises(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        try:
            with pytest.raises(EmptyAudioError):
                preprocess_audio(tmp.name)
        finally:
            os.remove(tmp.name)

    def test_nonexistent_file_raises(self):
        with pytest.raises(EmptyAudioError):
            preprocess_audio("/tmp/ce_fichier_n_existe_pas_12345.wav")

    def test_silent_audio_raises(self):
        path = _make_wav_file(duration_s=1.0, silent=True)
        try:
            with pytest.raises(SilentAudioError):
                preprocess_audio(path)
        finally:
            os.remove(path)

    def test_unsupported_format_raises(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".ogg", delete=False)
        tmp.write(b"fake content")
        tmp.close()
        try:
            with pytest.raises(UnsupportedFormatError):
                preprocess_audio(tmp.name)
        finally:
            os.remove(tmp.name)

    def test_too_long_audio_raises(self):
        # On simule directement via check_duration plutôt que de générer
        # un vrai fichier de 5+ minutes (coûteux en temps de test).
        fake_signal = np.zeros(AUDIO_CONFIG.TARGET_SAMPLE_RATE * (AUDIO_CONFIG.MAX_DURATION_SECONDS + 10))
        with pytest.raises(AudioTooLongError):
            check_duration(fake_signal, AUDIO_CONFIG.TARGET_SAMPLE_RATE)


class TestHelperFunctions:
    def test_to_mono_averages_channels(self):
        stereo = np.array([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]])
        mono = to_mono(stereo)
        assert mono.ndim == 1

    def test_normalize_amplitude_bounds(self):
        signal = np.array([0.1, -5.0, 3.0, -0.2])
        normalized = normalize_amplitude(signal)
        assert np.max(np.abs(normalized)) <= 1.0 + 1e-6

    def test_normalize_zero_signal_stays_zero(self):
        signal = np.zeros(100)
        normalized = normalize_amplitude(signal)
        assert np.all(normalized == 0)

    def test_check_not_silent_passes_for_loud_signal(self):
        t = np.linspace(0, 1, 16000)
        loud_signal = 0.5 * np.sin(2 * np.pi * 440 * t)
        check_not_silent(loud_signal)  # ne doit pas lever d'exception

    def test_resample_changes_length(self):
        signal = np.random.randn(22050)  # 1s à 22050 Hz
        resampled = resample(signal, orig_sr=22050, target_sr=16000)
        assert abs(len(resampled) - 16000) < 50
