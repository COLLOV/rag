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
PIXTRAL_URL = "http://localhost:8085/v1/chat/completions"  # Port pour Pixtral
MISTRAL_URL = "http://localhost:5263/v1/chat/completions"  # Port pour Mistral
PIXTRAL_PATH = "/home/llama/models/base_models/Pixtral-12B-2409"  # Modèle Pixtral
MISTRAL_PATH = "Mistral-Large-Instruct-2407-AWQ"  # Modèle Mistral

# Paramètres du modèle
MODEL_PARAMS = {
    "temperature": 0.7,
    "max_tokens": 1000
}

# Paramètres de recherche
SEARCH_PARAMS = {
    "max_results": 3,  # Nombre de résultats à retourner
    "max_images_per_page": 2,  # Nombre maximum d'images à traiter par page
    "max_total_images": 4,  # Nombre maximum d'images total pour Pixtral
    "chunk_size": 512,  # Taille des chunks de texte
    "chunk_overlap": 50,  # Chevauchement entre les chunks
    "min_chunk_length": 50  # Longueur minimale d'un chunk
}

# Modèle d'embedding
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"  # Meilleur modèle multilingue pour Pixtral
