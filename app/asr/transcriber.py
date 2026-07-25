"""
Module ASR (Automatic Speech Recognition) basé sur Wav2Vec 2.0.

Transcrit un signal audio prétraité (mono, 16kHz, normalisé) en texte français
via le modèle pré-entraîné jonatasgrosman/wav2vec2-large-xlsr-53-french.

Le modèle est chargé une seule fois (lazy loading + singleton) car son
chargement est coûteux (~1.2 Go, plusieurs secondes) : on ne veut pas le
recharger à chaque requête de l'API.
"""

import logging

import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from app.config import MODEL_CONFIG
from app.exceptions import TranscriptionError

logger = logging.getLogger(__name__)


class Wav2Vec2Transcriber:
    """
    Wrapper autour de Wav2Vec 2.0 pour la transcription audio -> texte.

    Le modèle et le processor sont chargés une seule fois à l'initialisation
    (ou au premier appel si on utilise get_transcriber() ci-dessous).
    """

    def __init__(self, model_name: str = None, device: str = None):
        self.model_name = model_name or MODEL_CONFIG.ASR_MODEL_NAME
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Chargement du modèle ASR '{self.model_name}' sur {self.device}...")
        try:
            self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
            self.model = Wav2Vec2ForCTC.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
        except Exception as e:
            raise TranscriptionError(
                f"Échec du chargement du modèle ASR '{self.model_name}': {e}"
            ) from e

        logger.info("Modèle ASR chargé avec succès.")

    def transcribe(self, audio_signal: np.ndarray, sample_rate: int = 16_000) -> str:
        """
        Transcrit un signal audio en texte.

        Args:
            audio_signal: signal audio mono, normalisé, en float32.
            sample_rate: doit être 16000 Hz (requis par le modèle).

        Returns:
            Le texte transcrit (str), éventuellement vide si aucune parole détectée.

        Raises:
            TranscriptionError: si l'inférence échoue.
        """
        if sample_rate != 16_000:
            raise TranscriptionError(
                f"Sample rate {sample_rate}Hz invalide, le modèle attend 16000Hz. "
                "Vérifiez que le prétraitement a bien été appliqué."
            )

        try:
            inputs = self.processor(
                audio_signal,
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True,
            )
            input_values = inputs.input_values.to(self.device)

            with torch.no_grad():
                logits = self.model(input_values).logits

            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]

        except Exception as e:
            raise TranscriptionError(f"Échec de la transcription audio: {e}") from e

        return transcription.strip()


# --- Singleton pour éviter de recharger le modèle à chaque requête API ---

_transcriber_instance: Wav2Vec2Transcriber | None = None


def get_transcriber() -> Wav2Vec2Transcriber:
    """Retourne l'instance unique du transcripteur (chargement paresseux)."""
    global _transcriber_instance
    if _transcriber_instance is None:
        _transcriber_instance = Wav2Vec2Transcriber()
    return _transcriber_instance
