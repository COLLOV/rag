import os
import base64
import requests
import numpy as np
import faiss
from typing import List, Dict, Tuple
from pathlib import Path
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from PIL import Image
from transformers import AutoTokenizer, AutoModel
import torch
from dataclasses import dataclass
import uuid

@dataclass
class PDFContent:
    source: str
    pages: List[dict]  # Liste des pages avec leur texte et images
    total_text: str  # Texte complet du PDF

from config import (VLLM_URL, MODEL_PARAMS, SEARCH_PARAMS, 
                  TEMP_DIR, EMBEDDING_MODEL, PDF_FOLDER, MODEL_PATH)

class PDFProcessor:
    def __init__(self, pdf_folder: str, vllm_url: str = VLLM_URL):
        self.pdf_folder = Path(pdf_folder)
        self.vllm_url = vllm_url
        self.temp_dir = Path(TEMP_DIR)
        self.temp_dir.mkdir(exist_ok=True)
        # Charger le tokenizer et le modèle
        self.tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
        self.model = AutoModel.from_pretrained(EMBEDDING_MODEL)
        self.index = None
        self.pdfs: List[PDFContent] = []
        
    def extract_from_pdf(self, pdf_path: str) -> PDFContent:
        """Extrait le texte et les images du PDF complet."""
        reader = PdfReader(pdf_path)
        pdf_images = convert_from_path(pdf_path)
        filename = Path(pdf_path).name
        
        pages = []
        full_text = []
        
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text = page.extract_text()
            full_text.append(text)
            
            pages.append({
                'number': page_num + 1,
                'text': text,
                'images': [pdf_images[page_num]]
            })
        
        return PDFContent(
            source=filename,
            pages=pages,
            total_text='\n'.join(full_text)
        )

    def encode_image_base64(self, image: Image.Image) -> dict:
        """Encode une image en base64 pour l'API VLLM."""
        # Sauvegarder temporairement l'image
        temp_path = self.temp_dir / f"temp_{uuid.uuid4()}.jpg"
        image.save(temp_path, "JPEG")
        
        # Lire et encoder en base64
        with open(temp_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        
        # Nettoyer le fichier temporaire
        temp_path.unlink()
        
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encoded}"
            }
        }

    def analyze_image(self, image: Image.Image) -> dict:
        """Prépare une image pour l'API VLLM."""
        # Encoder directement l'image en base64 avec le bon format
        return self.encode_image_base64(image)

    def process_pdf_directory(self) -> None:
        """Traite tous les PDF dans le dossier et crée l'index de recherche."""
        print(f"Début du traitement des PDF dans {self.pdf_folder}")
        self.pdfs = []
        all_texts = []
        
        for filename in os.listdir(self.pdf_folder):
            if filename.endswith('.pdf'):
                pdf_path = os.path.join(self.pdf_folder, filename)
                print(f"Traitement de {filename}...")
                
                # Extraire tout le contenu du PDF
                pdf_content = self.extract_from_pdf(pdf_path)
                
                # Analyser les images de chaque page
                image_descriptions_by_page = []
                
                for page in pdf_content.pages:
                    max_images = SEARCH_PARAMS["max_images_per_page"]
                    images_to_process = page['images'][:max_images]
                    
                    page_descriptions = []
                    for img in images_to_process:
                        try:
                            desc = self.analyze_image(img)
                            page_descriptions.append(str(desc))  # Convertir en string
                        except Exception as e:
                            print(f"Erreur lors de l'analyse d'une image : {e}")
                    
                    image_descriptions_by_page.append(page_descriptions)
                
                # Combiner le texte et les descriptions d'images
                combined_text = pdf_content.total_text
                for page_num, descriptions in enumerate(image_descriptions_by_page, 1):
                    if descriptions:
                        combined_text += "\nImages de la page " + str(page_num) + ":\n"
                        combined_text += "\n".join(descriptions) + "\n"
                
                all_texts.append(combined_text)
                self.pdfs.append(pdf_content)
        
        # Créer les embeddings avec le modèle
        embeddings = []
        for text in all_texts:
            inputs = self.tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
        
        # Convertir en array numpy
        embeddings = np.vstack(embeddings)
        
        # Créer l'index FAISS
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype(np.float32))

    def search(self, query: str, k: int = SEARCH_PARAMS["max_results"]) -> List[Tuple[PDFContent, float]]:
        """Recherche les pages les plus pertinentes."""

        """Recherche les pages les plus pertinentes pour la question."""
        # Encoder la requête
        inputs = self.tokenizer(query, padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        query_embedding = outputs.last_hidden_state.mean(dim=1).numpy()
        
        # Rechercher les k*2 plus proches voisins
        D, I = self.index.search(query_embedding.astype(np.float32), k*2)
        
        # Retourner les PDFs les plus pertinents
        final_results = []
        for idx, score in zip(I[0], D[0]):
            if idx < len(self.pdfs):  # Vérifier que l'index est valide
                final_results.append((self.pdfs[idx], score))
        
        # Trier par score (distance)
        final_results.sort(key=lambda x: x[1])
        return final_results[:k]

    def generate_response(self, query: str, k: int = 3) -> str:
        """Génère une réponse à la question en utilisant les documents pertinents."""
        # Récupérer les PDFs pertinents
        relevant_pdfs = self.search(query, k=k)
        
        # Préparer le contexte pour le LLM
        context = ""
        for pdf, score in relevant_pdfs:
            context += f"\nDocument: {pdf.source}\n"
            context += f"Contenu:\n{pdf.total_text}\n"
            
            # Ajouter les descriptions des images pour chaque page
            for page in pdf.pages:
                images = page['images'][:SEARCH_PARAMS['max_images_per_page']]
                if images:
                    context += f"\nImages de la page {page['number']}:\n"
                    for img in images:
                        try:
                            desc = self.analyze_image(img)
                            context += f"- Image : {desc}\n"
                        except Exception as e:
                            print(f"Erreur lors de l'analyse d'une image : {e}")
        
        # Préparer le message utilisateur avec le texte et les images
        user_content = [
            {
                "type": "text",
                "text": f"Voici des extraits de documents. Ta tâche est de répondre précisément à la question suivante en utilisant ces informations : {query}\n\nContexte:\n{context}"
            }
        ]
        
        # Ajouter les images au message
        for pdf in relevant_pdfs:
            for page in pdf.pages:
                images = page['images'][:SEARCH_PARAMS['max_images_per_page']]
                for img in images:
                    try:
                        image_content = self.analyze_image(img)
                        user_content.append(image_content)
                    except Exception as e:
                        print(f"Erreur lors de l'analyse d'une image : {e}")
        
        # Appeler l'API VLLM
        response = requests.post(
            self.vllm_url,
            headers={"Content-Type": "application/json"},
            json={
                "model": MODEL_PATH,
                "messages": [
                    {
                        "role": "user",
                        "content": user_content
                    }
                ],
                **MODEL_PARAMS
            }
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            print(f"Erreur : {response.status_code}")
            print(f"Détails : {response.text}")
            return "Désolé, je n'ai pas pu générer de réponse."

def main():
    # Initialiser le processeur avec le dossier PDF par défaut
    processor = PDFProcessor(pdf_folder=PDF_FOLDER)
    
    # Traiter les PDF et créer l'index
    print("Traitement des PDF et création de l'index...")
    processor.process_pdf_directory()
    
    # Exemple de requête
    query = "Comment changer de mot de passe ?"
    print(f"\nQuestion : {query}")
    
    # Générer une réponse
    response = processor.generate_response(query)
    print("\nRéponse :")
    print(response)

if __name__ == "__main__":
    main()
