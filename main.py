"""
API REST pour le pipeline Audio -> Sentiment.

Endpoint principal : POST /predict
    Accepte un fichier audio (.wav ou .mp3) et retourne :
    transcription, sentiment (positif/négatif/neutre), score de confiance.

Lancement local :
    uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000

Documentation interactive auto-générée :
    http://localhost:8000/docs
"""

import logging
import os
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from app.pipeline import run_pipeline
from app.config import API_CONFIG, AUDIO_CONFIG, MODEL_CONFIG
from app.api.schemas import PredictResponse, ErrorResponse, HealthResponse
from app.exceptions import (
    UnsupportedFormatError,
    EmptyAudioError,
    SilentAudioError,
    AudioTooLongError,
    TranscriptionError,
    SentimentAnalysisError,
    PipelineError,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="API de Détection de Sentiment dans les Appels Vocaux",
    description=(
        "Pipeline ASR (Wav2Vec2) + Analyse de sentiment (CamemBERT) "
        "pour détecter automatiquement si un client est satisfait, "
        "mécontent ou neutre à partir d'un enregistrement vocal."
    ),
    version="1.0.0",
)

# Mapping des exceptions métier vers les codes HTTP appropriés.
# 400 : le fichier envoyé par le client est en cause (mauvaise requête)
# 500 : erreur interne du pipeline (modèle, calcul, etc.)
_ERROR_STATUS_MAP = {
    UnsupportedFormatError: 400,
    EmptyAudioError: 400,
    SilentAudioError: 400,
    AudioTooLongError: 400,
    TranscriptionError: 500,
    SentimentAnalysisError: 500,
}


@app.get("/", tags=["Général"])
def root():
    """Point d'entrée simple confirmant que l'API est en ligne."""
    return {
        "message": "API de détection de sentiment vocal - voir /docs pour la documentation",
    }


@app.get("/health", response_model=HealthResponse, tags=["Général"])
def health_check():
    """Vérifie que l'API est opérationnelle et indique les modèles utilisés."""
    return HealthResponse(
        status="ok",
        asr_model=MODEL_CONFIG.ASR_MODEL_NAME,
        sentiment_model=MODEL_CONFIG.SENTIMENT_MODEL_NAME,
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["Prédiction"],
)
async def predict(file: UploadFile = File(..., description="Fichier audio .wav ou .mp3, 5 min max")):
    """
    Analyse un fichier audio et retourne la transcription + le sentiment détecté.

    - **file** : fichier audio au format .wav ou .mp3 (durée max 5 minutes)

    Retourne la transcription, le sentiment (positif/négatif/neutre),
    le score de confiance, et les scores détaillés pour chaque classe.
    """
    # --- Validation préliminaire de la taille du fichier ---
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > API_CONFIG.MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"Fichier trop volumineux ({size_mb:.1f} Mo). "
                   f"Limite : {API_CONFIG.MAX_FILE_SIZE_MB} Mo.",
        )

    # --- Sauvegarde temporaire du fichier pour traitement ---
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in AUDIO_CONFIG.SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Format '{ext}' non supporté. Formats acceptés : {AUDIO_CONFIG.SUPPORTED_FORMATS}",
        )

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        logger.info(f"Traitement de la requête pour le fichier : {file.filename}")
        result = run_pipeline(tmp_path)
        return PredictResponse(**result)

    except PipelineError as e:
        status_code = _ERROR_STATUS_MAP.get(type(e), 500)
        logger.warning(f"Erreur pipeline ({type(e).__name__}): {e}")
        raise HTTPException(status_code=status_code, detail=str(e))

    except Exception as e:
        logger.error(f"Erreur interne inattendue: {e}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur.")

    finally:
        # Nettoyage du fichier temporaire, même en cas d'erreur
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Formate toutes les erreurs HTTP dans un format JSON uniforme."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": type(exc).__name__, "message": exc.detail},
    )
