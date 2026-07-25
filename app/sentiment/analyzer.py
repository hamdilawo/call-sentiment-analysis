"""
Module d'analyse de sentiment basé sur CamemBERT (3 classes : positif/négatif/neutre).

Transcrit -> texte français -> classification de sentiment avec score de confiance.

Le modèle utilisé (ac0hik/Sentiment_Analysis_French) est un CamemBERT fine-tuné
sur le dataset tweet_sentiment_multilingual (portion française), nativement
3 classes. On lit le mapping label -> texte directement depuis la config du
modèle (model.config.id2label) plutôt que de le coder en dur, pour éviter
les erreurs si le modèle utilise un ordre de labels différent de celui attendu.

IMPORTANT : la normalisation des labels bruts du modèle (souvent au format
"LABEL_0", "LABEL_1", "LABEL_2" ou "positive"/"negative"/"neutral") vers nos
3 catégories françaises (positif/négatif/neutre) doit être VÉRIFIÉE
manuellement au premier lancement réel (cf. fonction verify_label_mapping()).
"""

import logging

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from app.config import MODEL_CONFIG
from app.exceptions import SentimentAnalysisError

logger = logging.getLogger(__name__)

# Classes finales exposées par notre pipeline (cahier des charges : 3 classes)
SENTIMENT_CLASSES = ("positif", "négatif", "neutre")

# Mapping de secours si le modèle expose des labels génériques (LABEL_0/1/2)
# et qu'aucune correspondance textuelle n'est trouvée dans sa config.
# Basé sur la convention la plus courante pour ce type de modèle 3-classes :
# LABEL_0 = négatif, LABEL_1 = neutre, LABEL_2 = positif.
# -> À VÉRIFIER avec verify_label_mapping() avant de faire confiance à ce mapping.
_FALLBACK_GENERIC_MAPPING = {
    "LABEL_0": "négatif",
    "LABEL_1": "neutre",
    "LABEL_2": "positif",
}

# Normalisation des variantes textuelles possibles (anglais, majuscules, etc.)
# vers nos 3 classes françaises.
_TEXTUAL_LABEL_NORMALIZATION = {
    "positive": "positif",
    "positif": "positif",
    "négatif": "négatif",
    "negatif": "négatif",
    "negative": "négatif",
    "neutral": "neutre",
    "neutre": "neutre",
}


def _normalize_label(raw_label: str) -> str:
    """Convertit un label brut du modèle vers l'une de nos 3 classes françaises."""
    key = raw_label.strip().lower()

    if key in _TEXTUAL_LABEL_NORMALIZATION:
        return _TEXTUAL_LABEL_NORMALIZATION[key]

    if raw_label in _FALLBACK_GENERIC_MAPPING:
        logger.warning(
            f"Label générique '{raw_label}' rencontré : mapping de secours utilisé "
            f"({_FALLBACK_GENERIC_MAPPING[raw_label]}). "
            "Vérifiez ce mapping avec verify_label_mapping() avant de faire confiance "
            "aux résultats en production."
        )
        return _FALLBACK_GENERIC_MAPPING[raw_label]

    raise SentimentAnalysisError(
        f"Label inconnu retourné par le modèle : '{raw_label}'. "
        f"Labels attendus : {SENTIMENT_CLASSES} ou {list(_FALLBACK_GENERIC_MAPPING.keys())}."
    )


class CamembertSentimentAnalyzer:
    """
    Wrapper autour d'un CamemBERT fine-tuné pour l'analyse de sentiment (FR, 3 classes).

    Comme pour le transcripteur ASR, le modèle est chargé une seule fois
    (singleton via get_sentiment_analyzer()) pour éviter le rechargement
    à chaque requête API.
    """

    def __init__(self, model_name: str = None, device: str = None):
        self.model_name = model_name or MODEL_CONFIG.SENTIMENT_MODEL_NAME
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Chargement du modèle de sentiment '{self.model_name}' sur {self.device}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(
                self.device
            )
            self.model.eval()
        except Exception as e:
            raise SentimentAnalysisError(
                f"Échec du chargement du modèle de sentiment '{self.model_name}': {e}"
            ) from e

        # id2label est défini dans la config du modèle Hugging Face,
        # ex: {0: "négatif", 1: "neutre", 2: "positif"} ou {0: "LABEL_0", ...}
        self.id2label = self.model.config.id2label
        logger.info(f"Mapping de labels du modèle : {self.id2label}")
        logger.info("Modèle de sentiment chargé avec succès.")

    def analyze(self, text: str) -> dict:
        """
        Analyse le sentiment d'un texte français.

        Args:
            text: texte transcrit (sortie du module ASR).

        Returns:
            dict avec les clés :
                - "sentiment": une des 3 classes ("positif", "négatif", "neutre")
                - "confidence": score de confiance (float, entre 0 et 1)
                - "all_scores": dict {classe: score} pour les 3 classes

        Raises:
            SentimentAnalysisError: si le texte est vide ou si l'inférence échoue.
        """
        if not text or not text.strip():
            raise SentimentAnalysisError(
                "Impossible d'analyser le sentiment d'un texte vide "
                "(la transcription ASR n'a peut-être détecté aucune parole)."
            )

        try:
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, padding=True, max_length=512
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits

            probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]

        except Exception as e:
            raise SentimentAnalysisError(f"Échec de l'analyse de sentiment: {e}") from e

        all_scores = {}
        for idx, prob in enumerate(probabilities.tolist()):
            raw_label = self.id2label[idx]
            normalized_label = _normalize_label(raw_label)
            all_scores[normalized_label] = prob

        predicted_class = max(all_scores, key=all_scores.get)
        confidence = all_scores[predicted_class]

        return {
            "sentiment": predicted_class,
            "confidence": round(confidence, 4),
            "all_scores": {k: round(v, 4) for k, v in all_scores.items()},
        }


def verify_label_mapping():
    """
    Fonction utilitaire À LANCER MANUELLEMENT UNE FOIS pour vérifier que le
    mapping label -> sentiment est correct pour le modèle réellement chargé.

    Teste 3 phrases sans ambiguïté et affiche le résultat pour validation humaine.
    Usage : python -c "from app.sentiment.analyzer import verify_label_mapping; verify_label_mapping()"
    """
    analyzer = CamembertSentimentAnalyzer()

    test_cases = [
        ("Je suis vraiment très content, c'est excellent, merci beaucoup !", "positif"),
        ("C'est vraiment nul, je suis très déçu et en colère.", "négatif"),
        ("Le magasin ouvre à neuf heures et ferme à dix-huit heures.", "neutre"),
    ]

    print(f"\nMapping brut du modèle (id2label) : {analyzer.id2label}\n")
    print("Vérification sur 3 phrases sans ambiguïté :")
    print("-" * 60)

    for text, expected in test_cases:
        result = analyzer.analyze(text)
        status = "✅" if result["sentiment"] == expected else "❌ À VÉRIFIER"
        print(f"{status} Attendu: {expected:10s} | Obtenu: {result['sentiment']:10s} "
              f"(confiance: {result['confidence']:.2f})")
        print(f"   Texte: \"{text}\"")
        print(f"   Tous les scores: {result['all_scores']}")
        print("-" * 60)


# --- Singleton pour éviter de recharger le modèle à chaque requête API ---

_analyzer_instance: CamembertSentimentAnalyzer | None = None


def get_sentiment_analyzer() -> CamembertSentimentAnalyzer:
    """Retourne l'instance unique de l'analyseur de sentiment (chargement paresseux)."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = CamembertSentimentAnalyzer()
    return _analyzer_instance
