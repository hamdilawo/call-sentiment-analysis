"""
Prétraitement audio pour le pipeline Audio -> Sentiment.

Ce module gère la première étape du pipeline :
  fichier audio brut (.wav/.mp3) -> tableau numpy prêt pour Wav2Vec2

Étapes appliquées :
    1. Validation du format et de la taille du fichier
    2. Chargement de l'audio (librosa gère nativement .wav et .mp3)
    3. Conversion en mono si stéréo
    4. Rééchantillonnage à 16 kHz (requis par Wav2Vec2)
    5. Normalisation de l'amplitude dans [-1, 1]
    6. Détection de silence total (protection contre les fichiers vides)
"""

import os
import logging

import numpy as np
import librosa

from app.config import AUDIO_CONFIG
from app.exceptions import (
    UnsupportedFormatError,
    EmptyAudioError,
    SilentAudioError,
    AudioTooLongError,
)

logger = logging.getLogger(__name__)

# Seuil en dessous duquel on considère l'audio comme silencieux.
# RMS (Root Mean Square) moyen de l'amplitude du signal.
SILENCE_RMS_THRESHOLD = 1e-4


def validate_file_format(file_path: str) -> None:
    """
    Vérifie que l'extension du fichier est supportée.

    Args:
        file_path: chemin vers le fichier audio.

    Raises:
        UnsupportedFormatError: si l'extension n'est pas .wav ou .mp3.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in AUDIO_CONFIG.SUPPORTED_FORMATS:
        raise UnsupportedFormatError(
            f"Format '{ext}' non supporté. "
            f"Formats acceptés : {AUDIO_CONFIG.SUPPORTED_FORMATS}"
        )


def load_audio(file_path: str) -> tuple[np.ndarray, int]:
    """
    Charge un fichier audio depuis le disque.

    Args:
        file_path: chemin vers le fichier .wav ou .mp3.

    Returns:
        Tuple (signal audio en float32, sample rate d'origine).

    Raises:
        UnsupportedFormatError: format de fichier invalide.
        EmptyAudioError: fichier vide, corrompu, ou illisible.
    """
    validate_file_format(file_path)

    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        raise EmptyAudioError(f"Le fichier '{file_path}' est vide ou introuvable.")

    try:
        # sr=None : on garde le sample rate d'origine pour l'instant,
        # le rééchantillonnage est fait explicitement à l'étape suivante.
        signal, sample_rate = librosa.load(file_path, sr=None, mono=False)
    except Exception as e:
        raise EmptyAudioError(
            f"Impossible de lire le fichier audio '{file_path}': {e}"
        ) from e

    if signal is None or signal.size == 0:
        raise EmptyAudioError(f"Le fichier '{file_path}' ne contient aucun échantillon audio.")

    return signal, sample_rate


def to_mono(signal: np.ndarray) -> np.ndarray:
    """Convertit un signal stéréo/multi-canal en mono par moyenne des canaux."""
    if signal.ndim > 1:
        return librosa.to_mono(signal)
    return signal


def resample(signal: np.ndarray, orig_sr: int, target_sr: int = None) -> np.ndarray:
    """Rééchantillonne le signal au sample rate cible (16 kHz par défaut)."""
    target_sr = target_sr or AUDIO_CONFIG.TARGET_SAMPLE_RATE
    if orig_sr == target_sr:
        return signal
    return librosa.resample(signal, orig_sr=orig_sr, target_sr=target_sr)


def normalize_amplitude(signal: np.ndarray) -> np.ndarray:
    """Normalise l'amplitude du signal dans l'intervalle [-1, 1]."""
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        return signal / max_val
    return signal


def check_not_silent(signal: np.ndarray) -> None:
    """
    Vérifie que l'audio n'est pas silencieux (RMS trop faible).

    Raises:
        SilentAudioError: si le signal est considéré comme silencieux.
    """
    rms = np.sqrt(np.mean(signal.astype(np.float64) ** 2))
    if rms < SILENCE_RMS_THRESHOLD:
        raise SilentAudioError(
            "L'audio est silencieux ou quasi-inaudible (RMS trop faible). "
            "Vérifiez le fichier source."
        )


def check_duration(signal: np.ndarray, sample_rate: int) -> None:
    """
    Vérifie que la durée de l'audio ne dépasse pas la limite autorisée.

    Raises:
        AudioTooLongError: si la durée dépasse AUDIO_CONFIG.MAX_DURATION_SECONDS.
    """
    duration_seconds = len(signal) / sample_rate
    if duration_seconds > AUDIO_CONFIG.MAX_DURATION_SECONDS:
        raise AudioTooLongError(
            f"Durée audio ({duration_seconds:.1f}s) supérieure à la limite "
            f"autorisée ({AUDIO_CONFIG.MAX_DURATION_SECONDS}s)."
        )


def preprocess_audio(file_path: str) -> np.ndarray:
    """
    Pipeline complet de prétraitement d'un fichier audio.

    Chargement -> validation -> mono -> resampling 16kHz -> normalisation.

    Args:
        file_path: chemin vers le fichier .wav ou .mp3.

    Returns:
        Signal audio prétraité, prêt à être passé au modèle ASR (Wav2Vec2).

    Raises:
        UnsupportedFormatError, EmptyAudioError, SilentAudioError, AudioTooLongError
    """
    logger.info(f"Prétraitement de l'audio : {file_path}")

    signal, orig_sr = load_audio(file_path)
    signal = to_mono(signal)
    signal = resample(signal, orig_sr, AUDIO_CONFIG.TARGET_SAMPLE_RATE)

    check_duration(signal, AUDIO_CONFIG.TARGET_SAMPLE_RATE)
    check_not_silent(signal)

    signal = normalize_amplitude(signal)

    logger.info(
        f"Audio prétraité : {len(signal) / AUDIO_CONFIG.TARGET_SAMPLE_RATE:.2f}s "
        f"@ {AUDIO_CONFIG.TARGET_SAMPLE_RATE}Hz, mono"
    )
    return signal.astype(np.float32)
