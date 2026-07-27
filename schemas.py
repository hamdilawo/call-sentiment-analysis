"""
Modèles Pydantic pour les requêtes/réponses de l'API.

Définir des schémas explicites permet à FastAPI de générer automatiquement
une documentation interactive (Swagger UI sur /docs) et de valider les
réponses avant de les renvoyer au client.
"""

from pydantic import BaseModel, Field


class SentimentScores(BaseModel):
    """Scores de confiance détaillés pour chacune des 3 classes de sentiment."""
    positif: float = Field(..., ge=0, le=1)
    négatif: float = Field(..., ge=0, le=1)
    neutre: float = Field(..., ge=0, le=1)


class PredictResponse(BaseModel):
    """Réponse renvoyée par l'endpoint POST /predict."""
    transcription: str = Field(..., description="Texte transcrit à partir de l'audio")
    sentiment: str = Field(..., description="Classe de sentiment prédite : positif, négatif ou neutre")
    confidence: float = Field(..., ge=0, le=1, description="Score de confiance de la prédiction")
    all_scores: dict = Field(..., description="Scores détaillés pour les 3 classes")
    processing_time_seconds: float = Field(..., description="Temps de traitement total en secondes")

    model_config = {
        "json_schema_extra": {
            "example": {
                "transcription": "Bonjour, je suis très satisfait du service, merci beaucoup",
                "sentiment": "positif",
                "confidence": 0.8486,
                "all_scores": {"positif": 0.8486, "négatif": 0.0507, "neutre": 0.1007},
                "processing_time_seconds": 2.34,
            }
        }
    }


class ErrorResponse(BaseModel):
    """Réponse renvoyée en cas d'erreur (format uniforme pour tous les codes HTTP)."""
    error: str = Field(..., description="Type d'erreur")
    message: str = Field(..., description="Description détaillée de l'erreur")


class HealthResponse(BaseModel):
    """Réponse de l'endpoint de santé /health."""
    status: str
    asr_model: str
    sentiment_model: str
