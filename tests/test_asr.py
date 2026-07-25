"""
Tests unitaires du module ASR (Wav2Vec2Transcriber).

On mocke Wav2Vec2Processor/Wav2Vec2ForCTC pour ne pas dépendre d'un
téléchargement réseau du modèle (~1.2 Go) ni d'un GPU pendant les tests.
Ces tests valident la LOGIQUE du wrapper (gestion d'erreurs, sample rate,
singleton), pas la qualité de transcription elle-même — celle-ci doit être
vérifiée manuellement avec de vrais fichiers audio (cf. démonstration).
"""

from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import torch

from app.exceptions import TranscriptionError


@pytest.fixture
def mock_wav2vec2(monkeypatch):
    """Mocke Wav2Vec2Processor et Wav2Vec2ForCTC pour éviter le téléchargement réel."""
    mock_processor_instance = MagicMock()
    mock_processor_instance.return_value = MagicMock(
        input_values=torch.zeros((1, 16000))
    )
    mock_processor_instance.batch_decode.return_value = ["bonjour le monde"]

    mock_model_instance = MagicMock()
    mock_model_instance.to.return_value = mock_model_instance
    mock_model_instance.eval.return_value = None
    mock_logits = torch.randn(1, 50, 32)  # (batch, seq_len, vocab_size) factice
    mock_model_instance.return_value = MagicMock(logits=mock_logits)

    with patch("app.asr.transcriber.Wav2Vec2Processor") as mock_proc_cls, \
         patch("app.asr.transcriber.Wav2Vec2ForCTC") as mock_model_cls:
        mock_proc_cls.from_pretrained.return_value = mock_processor_instance
        mock_model_cls.from_pretrained.return_value = mock_model_instance
        yield {
            "processor_cls": mock_proc_cls,
            "model_cls": mock_model_cls,
            "processor_instance": mock_processor_instance,
            "model_instance": mock_model_instance,
        }


class TestWav2Vec2Transcriber:
    def test_model_loads_successfully(self, mock_wav2vec2):
        from app.asr.transcriber import Wav2Vec2Transcriber

        transcriber = Wav2Vec2Transcriber(model_name="fake-model")
        assert transcriber.model_name == "fake-model"
        mock_wav2vec2["processor_cls"].from_pretrained.assert_called_once_with("fake-model")

    def test_transcribe_returns_string(self, mock_wav2vec2):
        from app.asr.transcriber import Wav2Vec2Transcriber

        transcriber = Wav2Vec2Transcriber(model_name="fake-model")
        signal = np.random.randn(16000).astype(np.float32)

        result = transcriber.transcribe(signal, sample_rate=16000)
        assert isinstance(result, str)
        assert result == "bonjour le monde"

    def test_transcribe_wrong_sample_rate_raises(self, mock_wav2vec2):
        from app.asr.transcriber import Wav2Vec2Transcriber

        transcriber = Wav2Vec2Transcriber(model_name="fake-model")
        signal = np.random.randn(8000).astype(np.float32)

        with pytest.raises(TranscriptionError, match="Sample rate"):
            transcriber.transcribe(signal, sample_rate=8000)

    def test_model_load_failure_raises_transcription_error(self, monkeypatch):
        from app.asr.transcriber import Wav2Vec2Transcriber

        with patch("app.asr.transcriber.Wav2Vec2Processor") as mock_proc_cls:
            mock_proc_cls.from_pretrained.side_effect = OSError("network error")
            with pytest.raises(TranscriptionError, match="Échec du chargement"):
                Wav2Vec2Transcriber(model_name="fake-model")

    def test_inference_failure_raises_transcription_error(self, mock_wav2vec2):
        from app.asr.transcriber import Wav2Vec2Transcriber

        transcriber = Wav2Vec2Transcriber(model_name="fake-model")
        mock_wav2vec2["model_instance"].side_effect = RuntimeError("CUDA out of memory")

        signal = np.random.randn(16000).astype(np.float32)
        with pytest.raises(TranscriptionError, match="Échec de la transcription"):
            transcriber.transcribe(signal, sample_rate=16000)


class TestSingleton:
    def test_get_transcriber_returns_same_instance(self, mock_wav2vec2):
        import app.asr.transcriber as transcriber_module

        transcriber_module._transcriber_instance = None  # reset avant le test
        instance1 = transcriber_module.get_transcriber()
        instance2 = transcriber_module.get_transcriber()

        assert instance1 is instance2
        # Le modèle ne doit être chargé qu'UNE seule fois
        mock_wav2vec2["processor_cls"].from_pretrained.assert_called_once()
