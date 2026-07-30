# Résultats d'évaluation quantitative

Évaluation du pipeline ASR (Wav2Vec2) + Sentiment (CamemBERT) sur les 3 fichiers audio de démonstration.

## Résultats par fichier

| Fichier | WER | Sentiment attendu | Sentiment obtenu | Confiance | Correct |
|---|---|---|---|---|---|
| audio_samples/positif.wav | 9.09% | positif | positif | 0.85 | ✅ |
| audio_samples/negatif.wav | 12.50% | négatif | négatif | 0.93 | ✅ |
| audio_samples/neutre.wav | 50.00% | neutre | neutre | 0.61 | ✅ |

## Métriques globales

- **WER global (ASR)** : 23.08%
- **Accuracy (sentiment)** : 100.00%
- **F1-score macro (sentiment)** : 1.0000

## Rapport de classification détaillé

```
              precision    recall  f1-score   support

      neutre       1.00      1.00      1.00         1
     négatif       1.00      1.00      1.00         1
     positif       1.00      1.00      1.00         1

    accuracy                           1.00         3
   macro avg       1.00      1.00      1.00         3
weighted avg       1.00      1.00      1.00         3

```

## Notes
- Le WER est calculé après normalisation basique (minuscules, espaces).
- Évaluation réalisée sur un échantillon volontairement restreint (3 fichiers, 1 par classe) ; les résultats donnent une indication mais ne remplacent pas une évaluation sur un jeu de données plus large.