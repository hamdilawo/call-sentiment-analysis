"""
Exceptions personnalisées pour le pipeline audio -> sentiment.

Centraliser les exceptions ici permet à l'API de les intercepter
proprement et de renvoyer des codes HTTP / messages appropriés,
au lieu de laisser remonter des erreurs génériques (ValueError, etc.)
difficiles à distinguer les unes des autres.
"""


class PipelineError(Exception):
    """Exception de base pour toutes les erreurs du pipeline."""
    pass


class UnsupportedFormatError(PipelineError):
    """Levée quand le format du fichier audio n'est pas supporté (.wav/.mp3 uniquement)."""
    pass


class EmptyAudioError(PipelineError):
    """Levée quand le fichier audio est vide ou corrompu (0 échantillon)."""
    pass


class SilentAudioError(PipelineError):
    """Levée quand l'audio ne contient que du silence (amplitude ~ 0 partout)."""
    pass


class AudioTooLongError(PipelineError):
    """Levée quand la durée de l'audio dépasse la limite autorisée (5 min)."""
    pass


class TranscriptionError(PipelineError):
    """Levée quand l'étape ASR (Wav2Vec2) échoue."""
    pass


class SentimentAnalysisError(PipelineError):
    """Levée quand l'étape d'analyse de sentiment (BERT) échoue."""
    pass
