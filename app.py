"""
Point d'entrée pour le déploiement Hugging Face Spaces.

Hugging Face Spaces (SDK Gradio) exécute automatiquement un fichier nommé
`app.py` à la racine du dépôt. Ce fichier réutilise l'interface définie
dans app/gradio_app.py sans dupliquer de logique.
"""

from app.gradio_app import demo

if __name__ == "__main__":
    demo.launch()