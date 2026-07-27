"""
Script de test manuel pour vérifier le pipeline ASR sur de vrais fichiers audio.

Usage :
    python test_manual.py

Place tes fichiers audio de test dans audio_samples/ avant de lancer ce script.
"""

from app.asr.transcriber import get_transcriber
from app.preprocessing.audio_preprocessing import preprocess_audio

# Modifie cette liste selon les fichiers que tu as dans audio_samples/
fichiers_test = [
    "audio_samples/positif.wav",
    "audio_samples/negatif.wav",
    "audio_samples/neutre.wav",
]

transcriber = get_transcriber()

for fichier in fichiers_test:
    print(f"\n--- {fichier} ---")
    try:
        signal = preprocess_audio(fichier)
        texte = transcriber.transcribe(signal)
        print(f"Transcription : {texte}")
    except Exception as e:
        print(f"Erreur : {e}")
