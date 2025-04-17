"""Configuration pour le système RAG."""
from pathlib import Path

# Chemins des dossiers
BASE_DIR = Path(__file__).parent
PDF_FOLDER = BASE_DIR / "pdfs"  # Dossier par défaut pour les PDFs
TEMP_DIR = BASE_DIR / "temp_images"

# Créer les dossiers s'ils n'existent pas
PDF_FOLDER.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# URLs et modèles
VLLM_URL = "http://localhost:8085/v1/chat/completions"
PIXTRAL_PATH = "/home/llama/models/base_models/Pixtral-12B-2409"  # Chemin vers le modèle Pixtral
MISTRAL_PATH = "Mistral-Large-Instruct-2407-AWQ"  # Modèle pour le texte

# Paramètres du modèle
MODEL_PARAMS = {
    "temperature": 0.7,
    "max_tokens": 1000
}

# Paramètres de recherche
SEARCH_PARAMS = {
    "max_results": 3,  # Nombre de résultats à retourner
    "max_images_per_page": 2,  # Nombre maximum d'images à traiter par page
    "max_total_images": 4  # Nombre maximum d'images total pour Pixtral
}

# Modèle d'embedding
EMBEDDING_MODEL = "bert-base-multilingual-cased"
