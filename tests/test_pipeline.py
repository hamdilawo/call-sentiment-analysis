"""
Tests unitaires du pipeline complet (run_pipeline).

On mocke les 3 sous-modules (preprocessing, asr, sentiment) pour tester
UNIQUEMENT la logique d'orchestration et de gestion d'erreurs du pipeline,
indépendamment du bon fonctionnement de chaque brique (déjà testée dans
test_preprocessing.py, test_asr.py, test_sentiment.py).
"""

from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from app.exceptions import (
    EmptyAudioError,
    SilentAudioError,
    TranscriptionError,
    SentimentAnalysisError,
)


@pytest.fixture
def mock_pipeline_success():
    """Mocke les 3 étapes pour un scénario de succès complet."""
    fake_signal = np.zeros(16000, dtype=np.float32)

    mock_transcriber = MagicMock()
    mock_transcriber.transcribe.return_value = "Je suis très content du service"

    mock_analyzer = MagicMock()
    mock_analyzer.analyze.return_value = {
        "sentiment": "positif",
        "confidence": 0.87,
        "all_scores": {"positif": 0.87, "négatif": 0.05, "neutre": 0.08},
    }

    with patch("app.pipeline.preprocess_audio", return_value=fake_signal) as mock_preprocess, \
         patch("app.pipeline.get_transcriber", return_value=mock_transcriber), \
         patch("app.pipeline.get_sentiment_analyzer", return_value=mock_analyzer):
        yield {
            "preprocess": mock_preprocess,
            "transcriber": mock_transcriber,
            "analyzer": mock_analyzer,
        }


class TestRunPipelineSuccess:
    def test_pipeline_returns_expected_structure(self, mock_pipeline_success):
        from app.pipeline import run_pipeline

        result = run_pipeline("fake_audio.wav")

        assert result["transcription"] == "Je suis très content du service"
        assert result["sentiment"] == "positif"
        assert result["confidence"] == 0.87
        assert "all_scores" in result
        assert "processing_time_seconds" in result
        assert result["processing_time_seconds"] >= 0

    def test_pipeline_calls_each_stage_once(self, mock_pipeline_success):
        from app.pipeline import run_pipeline

        run_pipeline("fake_audio.wav")

        mock_pipeline_success["preprocess"].assert_called_once_with("fake_audio.wav")
        mock_pipeline_success["transcriber"].transcribe.assert_called_once()
        mock_pipeline_success["analyzer"].analyze.assert_called_once_with(
            "Je suis très content du service"
        )


class TestRunPipelinePreprocessingErrors:
    def test_empty_audio_error_propagates(self, mock_pipeline_success):
        from app.pipeline import run_pipeline

        mock_pipeline_success["preprocess"].side_effect = EmptyAudioError("fichier vide")
        with pytest.raises(EmptyAudioError):
            run_pipeline("fake_audio.wav")

    def test_silent_audio_error_propagates(self, mock_pipeline_success):
        from app.pipeline import run_pipeline

        mock_pipeline_success["preprocess"].side_effect = SilentAudioError("silence")
        with pytest.raises(SilentAudioError):
            run_pipeline("fake_audio.wav")

    def test_unexpected_preprocessing_error_wrapped(self, mock_pipeline_success):
        from app.pipeline import run_pipeline
        from app.exceptions import PipelineError

        mock_pipeline_success["preprocess"].side_effect = ValueError("erreur inattendue")
        with pytest.raises(PipelineError):
            run_pipeline("fake_audio.wav")


class TestRunPipelineTranscriptionErrors:
    def test_transcription_error_propagates(self, mock_pipeline_success):
        from app.pipeline import run_pipeline

        mock_pipeline_success["transcriber"].transcribe.side_effect = TranscriptionError(
            "échec ASR"
        )
        with pytest.raises(TranscriptionError):
            run_pipeline("fake_audio.wav")

    def test_empty_transcription_raises(self, mock_pipeline_success):
        from app.pipeline import run_pipeline

        mock_pipeline_success["transcriber"].transcribe.return_value = "   "
        with pytest.raises(TranscriptionError, match="Aucune parole détectée"):
            run_pipeline("fake_audio.wav")

    def test_sentiment_analyzer_not_called_if_transcription_empty(self, mock_pipeline_success):
        from app.pipeline import run_pipeline

        mock_pipeline_success["transcriber"].transcribe.return_value = ""
        with pytest.raises(TranscriptionError):
            run_pipeline("fake_audio.wav")

        mock_pipeline_success["analyzer"].analyze.assert_not_called()


class TestRunPipelineSentimentErrors:
    def test_sentiment_error_propagates(self, mock_pipeline_success):
        from app.pipeline import run_pipeline

        mock_pipeline_success["analyzer"].analyze.side_effect = SentimentAnalysisError(
            "échec sentiment"
        )
        with pytest.raises(SentimentAnalysisError):
            run_pipeline("fake_audio.wav")
