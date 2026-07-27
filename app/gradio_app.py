"""
Interface Gradio pour le pipeline Audio -> Sentiment.

Permet de tester le pipeline complet (ASR + sentiment) de façon interactive :
upload ou enregistrement audio -> transcription affichée -> sentiment détecté
avec scores de confiance visualisés.

Lancement local :
    python app/gradio_app.py

Ouvre automatiquement une interface web sur http://localhost:7860
"""

import logging

import gradio as gr

from app.pipeline import run_pipeline
from app.exceptions import PipelineError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_audio(audio_file_path: str):
    """
    Fonction appelée par Gradio à chaque soumission d'un fichier audio.

    Args:
        audio_file_path: chemin fourni par Gradio vers le fichier audio
                          (upload ou enregistrement micro).

    Returns:
        Tuple (transcription: str, scores: dict, message_temps: str)
        adapté aux composants de sortie Gradio (Textbox, Label, Textbox).
    """
    if audio_file_path is None:
        return "⚠️ Aucun fichier audio fourni.", {}, ""

    try:
        result = run_pipeline(audio_file_path)
    except PipelineError as e:
        # Erreur "métier" attendue (fichier vide, silence, format...) :
        # on affiche un message clair à l'utilisateur plutôt qu'un stacktrace.
        logger.warning(f"Erreur pipeline: {e}")
        return f"❌ Erreur : {e}", {}, ""
    except Exception as e:
        logger.error(f"Erreur inattendue: {e}")
        return f"❌ Erreur inattendue : {e}", {}, ""

    transcription = result["transcription"]
    # gr.Label attend un dict {classe: score} pour afficher un graphique à barres
    scores = result["all_scores"]
    temps_info = f"⏱️ Temps de traitement : {result['processing_time_seconds']}s"

    return transcription, scores, temps_info


# --- Construction de l'interface ---

with gr.Blocks(title="Détection de Sentiment Vocal") as demo:
    gr.Markdown(
        """
        # 🎙️ Détection Automatique de Sentiment dans les Appels Vocaux

        Pipeline **Wav2Vec 2.0** (transcription) + **CamemBERT** (analyse de sentiment).

        Uploadez un fichier audio (.wav ou .mp3, 5 min max) ou enregistrez directement
        avec votre micro pour détecter si le client est **satisfait**, **mécontent**
        ou **neutre**.
        """
    )

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Fichier audio (.wav / .mp3)",
            )
            submit_btn = gr.Button("Analyser", variant="primary")

        with gr.Column():
            transcription_output = gr.Textbox(
                label="📝 Transcription",
                lines=3,
                interactive=False,
            )
            sentiment_output = gr.Label(
                label="📊 Sentiment détecté",
                num_top_classes=3,
            )
            time_output = gr.Textbox(
                label="",
                interactive=False,
                show_label=False,
            )

    submit_btn.click(
        fn=process_audio,
        inputs=[audio_input],
        outputs=[transcription_output, sentiment_output, time_output],
    )

    # Permet aussi de lancer l'analyse automatiquement après un enregistrement micro
    audio_input.change(
        fn=process_audio,
        inputs=[audio_input],
        outputs=[transcription_output, sentiment_output, time_output],
    )

    gr.Markdown(
        """
        ---
        **Limites connues** : la qualité de la transcription peut varier sur des mots
        rares ou peu fréquents ; le sentiment "neutre" est parfois moins confiant
        que les classes positif/négatif par nature.
        """
    )


if __name__ == "__main__":
    demo.launch()
