"""
Tests unitaires de la fonction process_audio() utilisée par l'interface Gradio.

On mocke run_pipeline() pour tester uniquement la logique d'adaptation
entrée/sortie Gradio (gestion du cas "pas de fichier", formatage des
erreurs en messages utilisateur), pas le pipeline ML lui-même.
"""

from unittest.mock import patch

from app.exceptions import EmptyAudioError, SilentAudioError


def test_process_audio_no_file_returns_warning():
    from app.gradio_app import process_audio

    transcription, scores, temps = process_audio(None)
    assert "⚠️" in transcription
    assert scores == {}


def test_process_audio_success_returns_expected_tuple():
    from app.gradio_app import process_audio

    fake_result = {
        "transcription": "Je suis très content",
        "sentiment": "positif",
        "confidence": 0.87,
        "all_scores": {"positif": 0.87, "négatif": 0.05, "neutre": 0.08},
        "processing_time_seconds": 1.23,
    }
    with patch("app.gradio_app.run_pipeline", return_value=fake_result):
        transcription, scores, temps = process_audio("fake_audio.wav")

    assert transcription == "Je suis très content"
    assert scores == {"positif": 0.87, "négatif": 0.05, "neutre": 0.08}
    assert "1.23" in temps


def test_process_audio_pipeline_error_returns_friendly_message():
    from app.gradio_app import process_audio

    with patch("app.gradio_app.run_pipeline", side_effect=EmptyAudioError("fichier vide")):
        transcription, scores, temps = process_audio("fake_audio.wav")

    assert "❌" in transcription
    assert scores == {}


def test_process_audio_silent_audio_error_returns_friendly_message():
    from app.gradio_app import process_audio

    with patch("app.gradio_app.run_pipeline", side_effect=SilentAudioError("silence")):
        transcription, scores, temps = process_audio("fake_audio.wav")

    assert "❌" in transcription
    assert scores == {}


def test_process_audio_unexpected_error_handled_gracefully():
    from app.gradio_app import process_audio

    with patch("app.gradio_app.run_pipeline", side_effect=RuntimeError("boom")):
        transcription, scores, temps = process_audio("fake_audio.wav")

    assert "❌" in transcription
    assert scores == {}
