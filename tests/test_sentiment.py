"""
Tests unitaires du module d'analyse de sentiment (CamembertSentimentAnalyzer).

Comme pour l'ASR, on mocke AutoTokenizer/AutoModelForSequenceClassification
pour ne pas dépendre d'un téléchargement réseau du modèle. Ces tests valident
la LOGIQUE du wrapper (normalisation des labels, gestion d'erreurs, singleton),
pas la qualité de classification elle-même — celle-ci doit être vérifiée
manuellement avec verify_label_mapping() et de vrais textes.
"""

from unittest.mock import patch, MagicMock

import pytest
import torch

from app.exceptions import SentimentAnalysisError
from app.sentiment.analyzer import _normalize_label, SENTIMENT_CLASSES


def _make_mock_model(id2label, logits_values):
    """Crée un mock de modèle renvoyant des logits fixes pour un id2label donné."""
    mock_config = MagicMock()
    mock_config.id2label = id2label

    mock_model_instance = MagicMock()
    mock_model_instance.to.return_value = mock_model_instance
    mock_model_instance.eval.return_value = None
    mock_model_instance.config = mock_config
    mock_model_instance.return_value = MagicMock(
        logits=torch.tensor([logits_values])
    )
    return mock_model_instance


@pytest.fixture
def mock_camembert_textual_labels():
    """Mock avec labels textuels déjà explicites (positif/négatif/neutre)."""
    id2label = {0: "négatif", 1: "neutre", 2: "positif"}
    # logits élevés sur l'indice 2 ("positif")
    mock_model_instance = _make_mock_model(id2label, [0.1, 0.2, 5.0])

    mock_tokenizer_instance = MagicMock()
    mock_tokenizer_instance.return_value = MagicMock(to=lambda device: mock_tokenizer_instance.return_value)

    with patch("app.sentiment.analyzer.AutoTokenizer") as mock_tok_cls, \
         patch("app.sentiment.analyzer.AutoModelForSequenceClassification") as mock_model_cls:
        mock_tok_cls.from_pretrained.return_value = mock_tokenizer_instance
        mock_model_cls.from_pretrained.return_value = mock_model_instance
        yield {
            "tokenizer_cls": mock_tok_cls,
            "model_cls": mock_model_cls,
            "model_instance": mock_model_instance,
        }


@pytest.fixture
def mock_camembert_generic_labels():
    """Mock avec labels génériques (LABEL_0/1/2), nécessitant le mapping de secours."""
    id2label = {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}
    # logits élevés sur l'indice 0 ("LABEL_0" -> "négatif" via fallback)
    mock_model_instance = _make_mock_model(id2label, [5.0, 0.1, 0.1])

    mock_tokenizer_instance = MagicMock()
    mock_tokenizer_instance.return_value = MagicMock(to=lambda device: mock_tokenizer_instance.return_value)

    with patch("app.sentiment.analyzer.AutoTokenizer") as mock_tok_cls, \
         patch("app.sentiment.analyzer.AutoModelForSequenceClassification") as mock_model_cls:
        mock_tok_cls.from_pretrained.return_value = mock_tokenizer_instance
        mock_model_cls.from_pretrained.return_value = mock_model_instance
        yield {"model_instance": mock_model_instance}


class TestNormalizeLabel:
    def test_textual_french_labels(self):
        assert _normalize_label("positif") == "positif"
        assert _normalize_label("négatif") == "négatif"
        assert _normalize_label("neutre") == "neutre"

    def test_textual_english_labels(self):
        assert _normalize_label("positive") == "positif"
        assert _normalize_label("negative") == "négatif"
        assert _normalize_label("neutral") == "neutre"

    def test_case_insensitive(self):
        assert _normalize_label("POSITIVE") == "positif"

    def test_generic_labels_use_fallback(self):
        assert _normalize_label("LABEL_0") == "négatif"
        assert _normalize_label("LABEL_1") == "neutre"
        assert _normalize_label("LABEL_2") == "positif"

    def test_unknown_label_raises(self):
        with pytest.raises(SentimentAnalysisError):
            _normalize_label("SOMETHING_UNEXPECTED")


class TestCamembertSentimentAnalyzer:
    def test_model_loads_successfully(self, mock_camembert_textual_labels):
        from app.sentiment.analyzer import CamembertSentimentAnalyzer

        analyzer = CamembertSentimentAnalyzer(model_name="fake-model")
        assert analyzer.model_name == "fake-model"

    def test_analyze_returns_expected_structure(self, mock_camembert_textual_labels):
        from app.sentiment.analyzer import CamembertSentimentAnalyzer

        analyzer = CamembertSentimentAnalyzer(model_name="fake-model")
        result = analyzer.analyze("Je suis très content !")

        assert "sentiment" in result
        assert "confidence" in result
        assert "all_scores" in result
        assert result["sentiment"] in SENTIMENT_CLASSES
        assert 0 <= result["confidence"] <= 1
        assert set(result["all_scores"].keys()) == set(SENTIMENT_CLASSES)

    def test_analyze_picks_highest_probability_class(self, mock_camembert_textual_labels):
        from app.sentiment.analyzer import CamembertSentimentAnalyzer

        analyzer = CamembertSentimentAnalyzer(model_name="fake-model")
        result = analyzer.analyze("Je suis très content !")

        # Le mock a des logits élevés sur l'indice 2 ("positif")
        assert result["sentiment"] == "positif"

    def test_analyze_with_generic_labels_uses_fallback_mapping(self, mock_camembert_generic_labels):
        from app.sentiment.analyzer import CamembertSentimentAnalyzer

        analyzer = CamembertSentimentAnalyzer(model_name="fake-model")
        result = analyzer.analyze("Un texte quelconque")

        # Le mock a des logits élevés sur l'indice 0 ("LABEL_0" -> "négatif")
        assert result["sentiment"] == "négatif"

    def test_empty_text_raises(self, mock_camembert_textual_labels):
        from app.sentiment.analyzer import CamembertSentimentAnalyzer

        analyzer = CamembertSentimentAnalyzer(model_name="fake-model")
        with pytest.raises(SentimentAnalysisError, match="texte vide"):
            analyzer.analyze("")

    def test_whitespace_only_text_raises(self, mock_camembert_textual_labels):
        from app.sentiment.analyzer import CamembertSentimentAnalyzer

        analyzer = CamembertSentimentAnalyzer(model_name="fake-model")
        with pytest.raises(SentimentAnalysisError, match="texte vide"):
            analyzer.analyze("   ")

    def test_model_load_failure_raises(self):
        from app.sentiment.analyzer import CamembertSentimentAnalyzer

        with patch("app.sentiment.analyzer.AutoTokenizer") as mock_tok_cls:
            mock_tok_cls.from_pretrained.side_effect = OSError("network error")
            with pytest.raises(SentimentAnalysisError, match="Échec du chargement"):
                CamembertSentimentAnalyzer(model_name="fake-model")


class TestSingleton:
    def test_get_sentiment_analyzer_returns_same_instance(self, mock_camembert_textual_labels):
        import app.sentiment.analyzer as analyzer_module

        analyzer_module._analyzer_instance = None  # reset avant le test
        instance1 = analyzer_module.get_sentiment_analyzer()
        instance2 = analyzer_module.get_sentiment_analyzer()

        assert instance1 is instance2
