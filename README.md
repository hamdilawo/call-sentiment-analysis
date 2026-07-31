# Détection Automatique de Sentiment dans des Appels Vocaux

Pipeline complet **Audio → Transcription → Sentiment** combinant **Wav2Vec 2.0** (ASR) et **CamemBERT** (analyse de sentiment) pour détecter automatiquement si un client est **satisfait**, **mécontent** ou **neutre** à partir d'un enregistrement vocal en français.

> Projet réalisé dans le cadre du module Deep Learning 2 — Dakar Institute of Technology (DIT), Master 2 IA — 2026.

---

## Table des matières

- [Architecture](#architecture)
- [Modèles utilisés](#modèles-utilisés)
- [Installation](#installation)
- [Utilisation](#utilisation)
  - [API REST](#api-rest)
  - [Interface Gradio](#interface-gradio)
  - [Docker](#docker)
- [Structure du projet](#structure-du-projet)
- [Tests](#tests)
- [Évaluation quantitative](#évaluation-quantitative-bonus)
- [Cas d'usage](#cas-dusage)
- [Limites connues](#limites-connues)
- [Fichiers de démonstration](#fichiers-de-démonstration)

---

## Architecture

```
Audio (.wav / .mp3)
        │
        ▼
┌───────────────────┐
│   Prétraitement    │  mono, 16 kHz, normalisation, validation
└───────────────────┘
        │
        ▼
┌───────────────────┐
│   ASR (Wav2Vec2)   │  audio → texte français
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Sentiment (CamemBERT) │  texte → positif / négatif / neutre
└───────────────────┘
        │
        ▼
   JSON de sortie
```

Le pipeline est exposé de deux façons :
- **API REST** (FastAPI) — endpoint `POST /predict`, pour intégration dans d'autres systèmes.
- **Interface Gradio** — pour tester le pipeline de façon interactive (upload ou micro).

Les deux interfaces réutilisent la même fonction centrale `app/pipeline.py::run_pipeline()`, garantissant un comportement identique.

---

## Modèles utilisés

| Étape | Modèle | Lien Hugging Face | Justification |
|---|---|---|---|
| ASR | `jonatasgrosman/wav2vec2-large-xlsr-53-french` | [huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-french](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-french) | Wav2Vec2 XLSR-53 fine-tuné spécifiquement pour le français, largement utilisé et documenté, bonnes performances sur audio conversationnel. |
| Sentiment | `ac0hik/Sentiment_Analysis_French` | [huggingface.co/ac0hik/Sentiment_Analysis_French](https://huggingface.co/ac0hik/Sentiment_Analysis_French) | CamemBERT fine-tuné nativement sur 3 classes (positif/négatif/neutre) — requis par le cahier des charges. Contrairement à des alternatives comme `tblard/tf-allocine` (binaire positif/négatif uniquement), ce modèle gère directement la classe neutre sans heuristique additionnelle. |

Les deux modèles sont chargés une seule fois au démarrage (singleton), évitant un rechargement coûteux à chaque requête.

---

## Installation

### Prérequis
- Python ≥ 3.9 (testé avec 3.11 et 3.14)
- ~3 Go d'espace disque pour les modèles (téléchargés automatiquement au premier lancement)
- (Optionnel) Docker, pour l'exécution conteneurisée

### Étapes

```bash
git clone https://github.com/hamdilawo/call-sentiment-analysis.git
cd call-sentiment-analysis

python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

**Recommandé** : configurez un token Hugging Face pour accélérer les téléchargements et éviter les limites de débit anonymes :
```bash
export HF_TOKEN="votre_token"           # Linux/Mac
setx HF_TOKEN "votre_token"             # Windows (rouvrir le terminal après)
```
(Créez un token gratuit sur [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens))

---

## Utilisation

### API REST

Lancer le serveur :
```bash
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
```

Documentation interactive (Swagger) : **http://localhost:8000/docs**

**Exemple d'appel avec curl :**
```bash
curl -X POST "http://localhost:8000/predict" -F "file=@audio_samples/positif.wav"
```

**Exemple de réponse :**
```json
{
  "transcription": "bonjour je suis très satisfaite du service merci beaucoup cest excellent",
  "sentiment": "positif",
  "confidence": 0.8486,
  "all_scores": {"négatif": 0.0507, "neutre": 0.1007, "positif": 0.8486},
  "processing_time_seconds": 2.34
}
```

**Exemple d'appel en Python :**
```python
import requests

with open("audio_samples/positif.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": f},
    )

print(response.json())
```

**Endpoints disponibles :**
| Méthode | Route | Description |
|---|---|---|
| GET | `/` | Message de bienvenue |
| GET | `/health` | Vérifie que l'API est opérationnelle |
| POST | `/predict` | Analyse un fichier audio (transcription + sentiment) |

### Interface Gradio

```bash
python -m app.gradio_app
```

Ouvre automatiquement **http://localhost:7860** — interface permettant d'uploader un fichier ou d'enregistrer directement au micro, avec affichage de la transcription et d'un graphique des scores de sentiment.

### Docker

```bash
docker build -t call-sentiment-analysis .
docker run -p 8000:8000 call-sentiment-analysis
```

L'API est alors accessible sur `http://localhost:8000` comme en exécution locale.

> Note technique : le `Dockerfile` installe PyTorch en version **CPU-only** explicitement (`--index-url https://download.pytorch.org/whl/cpu`), car l'installation par défaut de `torch`/`torchaudio` embarque des dépendances CUDA volumineuses et inutiles en l'absence de GPU dans le conteneur.

---

## Structure du projet

```
call-sentiment-analysis/
├── app/
│   ├── config.py              # Configuration centralisée (modèles, paramètres audio)
│   ├── exceptions.py          # Exceptions personnalisées du pipeline
│   ├── pipeline.py            # Assemblage complet : prétraitement -> ASR -> sentiment
│   ├── gradio_app.py          # Interface Gradio
│   ├── preprocessing/
│   │   └── audio_preprocessing.py
│   ├── asr/
│   │   └── transcriber.py     # Wrapper Wav2Vec2
│   ├── sentiment/
│   │   └── analyzer.py        # Wrapper CamemBERT
│   └── api/
│       ├── main.py            # Application FastAPI
│       └── schemas.py         # Modèles Pydantic (requêtes/réponses)
├── tests/                     # Tests unitaires (pytest)
├── audio_samples/             # Fichiers audio de démonstration (1 par classe)
├── docs/                      # Documentation additionnelle, résultats d'évaluation
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Tests

```bash
python -m pytest tests/ -v
```

La suite de tests couvre chaque module indépendamment (prétraitement, ASR, sentiment, pipeline, API, Gradio), avec **mocking** des modèles Hugging Face pour des tests rapides et reproductibles sans téléchargement réseau. Le comportement réel des modèles est validé séparément via les scripts de démonstration et d'évaluation (voir sections suivantes).

---

## Évaluation quantitative

Un script d'évaluation (`docs/evaluate.py`) mesure :
- **WER** (Word Error Rate) pour l'ASR, en comparant les transcriptions du modèle à des transcriptions de référence rédigées manuellement.
- **Accuracy / F1-score** pour la classification de sentiment, sur un petit jeu de phrases annotées.

```bash
python docs/evaluate.py
```

Les résultats détaillés sont disponibles dans `docs/evaluation_results.md`.

---

## Cas d'usage

- **Centres d'appels** : analyse automatique de la satisfaction client sur des enregistrements d'appels, sans intervention humaine.
- **Support client** : priorisation des tickets/appels selon le niveau de mécontentement détecté.
- **Suivi qualité** : détection d'appels à forte insatisfaction pour audit ou formation des agents.
- **Sondages vocaux** : analyse de réponses vocales libres dans des enquêtes de satisfaction.

---

## Limites connues

- **Qualité de transcription** : le modèle Wav2Vec2 peut faire des erreurs sur des mots rares, des noms propres, ou dans des conditions audio bruitées (ex: "horaires" transcrit "olaires" dans certains tests). Ces erreurs mineures affectent rarement la classification de sentiment globale, le contexte restant généralement suffisant.
- **Classe "neutre"** : par nature plus ambiguë que les classes positif/négatif, elle affiche souvent une confiance plus faible.
- **Langue** : le pipeline est calibré pour le français ; il n'est pas conçu pour du contenu multilingue ou du code-switching.
- **Durée maximale** : 5 minutes par fichier (limite du cahier des charges) ; les fichiers plus longs sont rejetés plutôt que tronqués.
- **Absence de diarisation** : le pipeline ne distingue pas les locuteurs (agent vs client) ; il analyse le sentiment global du fichier audio fourni.
- **Performance CPU** : sans GPU, le traitement d'un fichier de plusieurs minutes peut prendre quelques secondes supplémentaires par rapport à une exécution GPU.

---

## Fichiers de démonstration

Le dossier `audio_samples/` contient 3 fichiers audio de test, un par classe de sentiment :

| Fichier | Sentiment attendu | Transcription obtenue |
|---|---|---|
| `positif.wav` | positif | "bonjour je suis très satisfaite du service merci beaucoup cest excellent" |
| `negatif.wav` | négatif | "je suis très moins content le service est vraiment décevant je ne suis pas content du tout" |
| `neutre.wav` | neutre | "un bonjour je voudrais des informatior sur les olaires douverture sil vous plait" |

---

## Auteur

Hamady Ngansou SABALY DIT Master 2 IA
