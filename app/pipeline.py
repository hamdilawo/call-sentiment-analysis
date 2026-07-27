"""
Pipeline complet : Audio -> Prétraitement -> ASR -> Sentiment.

Ce module assemble les 3 briques développées précédemment
(app.preprocessing, app.asr, app.sentiment) en une seule fonction
d'entrée unique, utilisée à la fois par l'API FastAPI et l'interface Gradio.

Centraliser l'assemblage ici évite de dupliquer la logique métier dans
plusieurs points d'entrée (principe DRY), et permet de gérer les erreurs
de chaque étape de façon uniforme.
"""

import logging
import time

from app.preprocessing.audio_preprocessing import preprocess_audio
from app.asr.transcriber import get_transcriber
from app.sentiment.analyzer import get_sentiment_analyzer
from app.exceptions import (
    PipelineError,
    UnsupportedFormatError,
    EmptyAudioError,
    SilentAudioError,
    AudioTooLongError,
    TranscriptionError,
    SentimentAnalysisError,
)

logger = logging.getLogger(__name__)


def run_pipeline(file_path: str) -> dict:
    """
    Exécute le pipeline complet sur un fichier audio.

    Étapes :
        1. Prétraitement audio (validation, mono, 16kHz, normalisation)
        2. Transcription (ASR, Wav2Vec2)
        3. Analyse de sentiment (CamemBERT)

    Args:
        file_path: chemin vers le fichier audio (.wav ou .mp3).

    Returns:
        dict avec les clés :
            - "transcription": texte transcrit
            - "sentiment": "positif" | "négatif" | "neutre"
            - "confidence": score de confiance (float)
            - "all_scores": scores détaillés pour les 3 classes
            - "processing_time_seconds": durée totale de traitement

    Raises:
        UnsupportedFormatError, EmptyAudioError, SilentAudioError,
        AudioTooLongError, TranscriptionError, SentimentAnalysisError
        (toutes héritent de PipelineError, capturable en une seule fois
        par l'API si le détail n'est pas nécessaire)
    """
    start_time = time.time()
    logger.info(f"Démarrage du pipeline pour : {file_path}")

    # --- Étape 1 : Prétraitement audio ---
    try:
        signal = preprocess_audio(file_path)
    except (UnsupportedFormatError, EmptyAudioError, SilentAudioError, AudioTooLongError):
        # Ces erreurs sont déjà explicites et destinées à l'utilisateur final
        # (ex: "votre fichier est vide"), on les laisse remonter telles quelles.
        raise
    except Exception as e:
        # Filet de sécurité pour toute erreur imprévue de prétraitement
        raise PipelineError(f"Erreur inattendue lors du prétraitement audio: {e}") from e

    # --- Étape 2 : Transcription (ASR) ---
    try:
        transcriber = get_transcriber()
        transcription = transcriber.transcribe(signal)
    except TranscriptionError:
        raise
    except Exception as e:
        raise TranscriptionError(f"Erreur inattendue lors de la transcription: {e}") from e

    if not transcription or not transcription.strip():
        raise TranscriptionError(
            "Aucune parole détectée dans l'audio (transcription vide). "
            "Vérifiez que le fichier contient bien de la voix compréhensible."
        )

    # --- Étape 3 : Analyse de sentiment ---
    try:
        analyzer = get_sentiment_analyzer()
        sentiment_result = analyzer.analyze(transcription)
    except SentimentAnalysisError:
        raise
    except Exception as e:
        raise SentimentAnalysisError(f"Erreur inattendue lors de l'analyse de sentiment: {e}") from e

    processing_time = round(time.time() - start_time, 3)

    result = {
        "transcription": transcription,
        "sentiment": sentiment_result["sentiment"],
        "confidence": sentiment_result["confidence"],
        "all_scores": sentiment_result["all_scores"],
        "processing_time_seconds": processing_time,
    }

    logger.info(
        f"Pipeline terminé en {processing_time}s -> "
        f"sentiment='{result['sentiment']}' (confiance={result['confidence']})"
    )
    return result
