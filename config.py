"""Configuration pour le système RAG."""
from pathlib import Path

# Chemins des dossiers
BASE_DIR = Path(__file__).parent
PDF_FOLDER = BASE_DIR / "pdfs"  # Dossier par défaut pour les PDFs
TEMP_DIR = BASE_DIR / "temp_images"

# Créer les dossiers s'ils n'existent pas
PDF_FOLDER.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# URLs des services
VLLM_URL = "http://localhost:8085/v1/chat/completions"

# Paramètres du modèle
MODEL_PARAMS = {
    "temperature": 0.7,
    "max_tokens": 1000
}

# Paramètres de recherche
SEARCH_PARAMS = {
    "max_results": 8,
    "max_images_per_page": 4
}

# Modèle d'embedding
EMBEDDING_MODEL = "bert-base-multilingual-cased"
