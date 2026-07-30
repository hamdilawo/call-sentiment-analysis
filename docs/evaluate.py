"""
Script d'évaluation quantitative du pipeline.

Mesure :
    - WER (Word Error Rate) pour l'ASR : compare la transcription du modèle
      à une transcription de référence (ce qui a réellement été dit).
    - Accuracy / F1-score pour la classification de sentiment : compare
      le sentiment prédit au sentiment réel attendu.

Usage :
    python docs/evaluate.py
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import jiwer
from sklearn.metrics import accuracy_score, f1_score, classification_report

from app.asr.transcriber import get_transcriber
from app.sentiment.analyzer import get_sentiment_analyzer
from app.preprocessing.audio_preprocessing import preprocess_audio


# --- Jeu de données d'évaluation ---
# À AJUSTER : reference_text doit correspondre exactement à ce qui est dit
# dans le fichier audio correspondant.
TEST_CASES = [
    {
        "audio_path": "audio_samples/positif.wav",
        "reference_text": "bonjour je suis très satisfaite du service merci beaucoup c'est excellent",
        "reference_sentiment": "positif",
    },
    {
        "audio_path": "audio_samples/negatif.wav",
        "reference_text": "je suis très mécontente le service est vraiment décevant je ne suis pas content du tout",
        "reference_sentiment": "négatif",
    },
    {
        "audio_path": "audio_samples/neutre.wav",
        "reference_text": "bonjour je voudrais des informations sur les horaires d'ouverture s'il vous plaît",
        "reference_sentiment": "neutre",
    },
]


def normalize_for_wer(text: str) -> str:
    """Normalisation minimale avant calcul du WER (minuscules, espaces)."""
    return " ".join(text.lower().split())


def evaluate() -> dict:
    """Exécute le pipeline sur chaque cas de test et calcule les métriques."""
    transcriber = get_transcriber()
    analyzer = get_sentiment_analyzer()

    references_text = []
    hypotheses_text = []
    true_sentiments = []
    pred_sentiments = []
    per_file_results = []

    for case in TEST_CASES:
        signal = preprocess_audio(case["audio_path"])
        hypothesis = transcriber.transcribe(signal)
        sentiment_result = analyzer.analyze(hypothesis)

        ref_norm = normalize_for_wer(case["reference_text"])
        hyp_norm = normalize_for_wer(hypothesis)
        file_wer = jiwer.wer(ref_norm, hyp_norm)

        references_text.append(ref_norm)
        hypotheses_text.append(hyp_norm)
        true_sentiments.append(case["reference_sentiment"])
        pred_sentiments.append(sentiment_result["sentiment"])

        per_file_results.append({
            "audio_path": case["audio_path"],
            "reference_text": case["reference_text"],
            "hypothesis_text": hypothesis,
            "wer": round(file_wer, 4),
            "reference_sentiment": case["reference_sentiment"],
            "predicted_sentiment": sentiment_result["sentiment"],
            "confidence": sentiment_result["confidence"],
            "sentiment_correct": case["reference_sentiment"] == sentiment_result["sentiment"],
        })

    global_wer = jiwer.wer(references_text, hypotheses_text)
    accuracy = accuracy_score(true_sentiments, pred_sentiments)
    f1 = f1_score(true_sentiments, pred_sentiments, average="macro", zero_division=0)
    report = classification_report(
        true_sentiments, pred_sentiments, zero_division=0
    )

    return {
        "per_file_results": per_file_results,
        "global_wer": round(global_wer, 4),
        "accuracy": round(accuracy, 4),
        "f1_macro": round(f1, 4),
        "classification_report": report,
    }


def print_results(results: dict):
    print("\n" + "=" * 70)
    print("RÉSULTATS PAR FICHIER")
    print("=" * 70)
    for r in results["per_file_results"]:
        status = "✅" if r["sentiment_correct"] else "❌"
        print(f"\n{r['audio_path']}")
        print(f"  Référence   : {r['reference_text']}")
        print(f"  Transcrit   : {r['hypothesis_text']}")
        print(f"  WER         : {r['wer']:.2%}")
        print(f"  Sentiment   : attendu={r['reference_sentiment']} | "
              f"obtenu={r['predicted_sentiment']} ({r['confidence']:.2f}) {status}")

    print("\n" + "=" * 70)
    print("MÉTRIQUES GLOBALES")
    print("=" * 70)
    print(f"WER global (ASR)        : {results['global_wer']:.2%}")
    print(f"Accuracy (sentiment)    : {results['accuracy']:.2%}")
    print(f"F1-score macro (sentiment) : {results['f1_macro']:.4f}")
    print(f"\nRapport de classification détaillé :\n{results['classification_report']}")


def write_markdown_report(results: dict, output_path: str = "docs/evaluation_results.md"):
    """Génère un rapport Markdown lisible, à inclure dans les livrables."""
    lines = [
        "# Résultats d'évaluation quantitative\n",
        "Évaluation du pipeline ASR (Wav2Vec2) + Sentiment (CamemBERT) "
        "sur les 3 fichiers audio de démonstration.\n",
        "## Résultats par fichier\n",
        "| Fichier | WER | Sentiment attendu | Sentiment obtenu | Confiance | Correct |",
        "|---|---|---|---|---|---|",
    ]
    for r in results["per_file_results"]:
        status = "✅" if r["sentiment_correct"] else "❌"
        lines.append(
            f"| {r['audio_path']} | {r['wer']:.2%} | {r['reference_sentiment']} | "
            f"{r['predicted_sentiment']} | {r['confidence']:.2f} | {status} |"
        )

    lines += [
        "\n## Métriques globales\n",
        f"- **WER global (ASR)** : {results['global_wer']:.2%}",
        f"- **Accuracy (sentiment)** : {results['accuracy']:.2%}",
        f"- **F1-score macro (sentiment)** : {results['f1_macro']:.4f}",
        "\n## Rapport de classification détaillé\n",
        "```",
        results["classification_report"],
        "```",
        "\n## Notes",
        "- Le WER est calculé après normalisation basique (minuscules, espaces).",
        "- Évaluation réalisée sur un échantillon volontairement restreint (3 fichiers, "
        "1 par classe) ; les résultats donnent une indication mais ne remplacent pas "
        "une évaluation sur un jeu de données plus large.",
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n📄 Rapport Markdown généré : {output_path}")


if __name__ == "__main__":
    results = evaluate()
    print_results(results)
    write_markdown_report(results)