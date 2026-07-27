"""
Tests unitaires de l'API FastAPI.

On mocke run_pipeline() pour tester UNIQUEMENT la couche API (validation
de fichiers, codes HTTP, format des réponses), indépendamment du bon
fonctionnement du pipeline ML lui-même (déjà testé dans test_pipeline.py).
"""

import io
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.exceptions import EmptyAudioError, UnsupportedFormatError, TranscriptionError

client = TestClient(app)


def _fake_wav_bytes():
    """Génère un contenu binaire factice (le vrai contenu n'importe pas car run_pipeline est mocké)."""
    return io.BytesIO(b"RIFF....WAVEfmt fake content for testing")


class TestGeneralEndpoints:
    def test_root_returns_200(self):
        response = client.get("/")
        assert response.status_code == 200

    def test_health_returns_model_info(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "asr_model" in data
        assert "sentiment_model" in data


class TestPredictEndpointSuccess:
    def test_predict_returns_expected_schema(self):
        fake_result = {
            "transcription": "Je suis très content",
            "sentiment": "positif",
            "confidence": 0.87,
            "all_scores": {"positif": 0.87, "négatif": 0.05, "neutre": 0.08},
            "processing_time_seconds": 1.23,
        }
        with patch("app.api.main.run_pipeline", return_value=fake_result):
            response = client.post(
                "/predict",
                files={"file": ("test.wav", _fake_wav_bytes(), "audio/wav")},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["transcription"] == "Je suis très content"
        assert data["sentiment"] == "positif"
        assert data["confidence"] == 0.87
        assert "all_scores" in data
        assert "processing_time_seconds" in data

    def test_predict_accepts_mp3(self):
        fake_result = {
            "transcription": "test",
            "sentiment": "neutre",
            "confidence": 0.5,
            "all_scores": {"positif": 0.25, "négatif": 0.25, "neutre": 0.5},
            "processing_time_seconds": 1.0,
        }
        with patch("app.api.main.run_pipeline", return_value=fake_result):
            response = client.post(
                "/predict",
                files={"file": ("test.mp3", _fake_wav_bytes(), "audio/mpeg")},
            )
        assert response.status_code == 200


class TestPredictEndpointValidation:
    def test_unsupported_format_returns_400(self):
        response = client.post(
            "/predict",
            files={"file": ("test.ogg", _fake_wav_bytes(), "audio/ogg")},
        )
        assert response.status_code == 400

    def test_no_file_returns_422(self):
        response = client.post("/predict")
        assert response.status_code == 422  # FastAPI validation error


class TestPredictEndpointPipelineErrors:
    def test_empty_audio_error_returns_400(self):
        with patch("app.api.main.run_pipeline", side_effect=EmptyAudioError("fichier vide")):
            response = client.post(
                "/predict",
                files={"file": ("test.wav", _fake_wav_bytes(), "audio/wav")},
            )
        assert response.status_code == 400
        assert "message" in response.json() or "detail" in response.json()

    def test_transcription_error_returns_500(self):
        with patch("app.api.main.run_pipeline", side_effect=TranscriptionError("échec ASR")):
            response = client.post(
                "/predict",
                files={"file": ("test.wav", _fake_wav_bytes(), "audio/wav")},
            )
        assert response.status_code == 500

    def test_unexpected_error_returns_500(self):
        with patch("app.api.main.run_pipeline", side_effect=RuntimeError("boom")):
            response = client.post(
                "/predict",
                files={"file": ("test.wav", _fake_wav_bytes(), "audio/wav")},
            )
        assert response.status_code == 500
