"""
Configuration centrale du pipeline Audio -> Sentiment.

Toutes les constantes du projet (modèles, paramètres audio, limites)
sont définies ici pour éviter de les disperser dans le code.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class AudioConfig:
    """Paramètres de prétraitement audio."""
    TARGET_SAMPLE_RATE: int = 16_000      # requis par Wav2Vec 2.0
    TARGET_CHANNELS: int = 1               # mono
    MAX_DURATION_SECONDS: int = 300        # 5 minutes max (cf. cahier des charges)
    SUPPORTED_FORMATS: tuple = (".wav", ".mp3")


@dataclass(frozen=True)
class ModelConfig:
    """Modèles pré-entraînés utilisés dans le pipeline."""
    # ASR : Wav2Vec 2.0 fine-tuné pour le français
    ASR_MODEL_NAME: str = "jonatasgrosman/wav2vec2-large-xlsr-53-french"

    # Sentiment : CamemBERT fine-tuné pour l'analyse de sentiment (FR)
    # Choisi car il gère nativement les 3 classes positif/négatif/neutre
    # requises par le cahier des charges (contrairement à tblard/tf-allocine
    # qui est binaire positif/négatif).
    SENTIMENT_MODEL_NAME: str = "ac0hik/Sentiment_Analysis_French"
    # Alternative multilingue si le modèle principal pose problème :
    SENTIMENT_MODEL_FALLBACK: str = "nlptown/bert-base-multilingual-uncased-sentiment"


@dataclass(frozen=True)
class APIConfig:
    """Paramètres de l'API FastAPI."""
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    MAX_FILE_SIZE_MB: int = 25


AUDIO_CONFIG = AudioConfig()
MODEL_CONFIG = ModelConfig()
API_CONFIG = APIConfig()
