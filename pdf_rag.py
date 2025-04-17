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
                  TEMP_DIR, EMBEDDING_MODEL, PDF_FOLDER, PIXTRAL_PATH,
                  MISTRAL_PATH)

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
                
                # Stocker le contenu du PDF
                self.pdfs.append(pdf_content)
                all_texts.append(pdf_content.total_text)
        
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

    def analyze_images_with_pixtral(self, query: str, images: list) -> str:
        """Analyse un lot d'images avec Pixtral."""
        user_content = [
            {
                "type": "text",
                "text": f"Voici des images extraites d'un document. Décris ce que tu vois dans ces images en lien avec la question : {query}"
            }
        ]
        
        # Ajouter les images au message
        for img in images:
            try:
                image_content = self.analyze_image(img)
                user_content.append(image_content)
            except Exception as e:
                print(f"Erreur lors de l'analyse d'une image : {e}")
        
        # Appeler Pixtral
        response = requests.post(
            self.vllm_url,
            headers={"Content-Type": "application/json"},
            json={
                "model": PIXTRAL_PATH,
                "messages": [
                    {
                        "role": "system",
                        "content": "Tu es un assistant expert en analyse d'images. Ta tâche est de décrire précisément le contenu des images en lien avec la question posée. Décris les éléments visuels importants, le texte visible, et tout ce qui pourrait aider à répondre à la question."
                    },
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
            print(f"Erreur Pixtral : {response.status_code}")
            print(f"Détails : {response.text}")
            return ""
    
    def generate_response(self, query: str, k: int = 3) -> str:
        """Génère une réponse à la question en utilisant les documents pertinents."""
        # Récupérer les PDFs pertinents
        relevant_pdfs = self.search(query, k=k)
        
        # Préparer le contexte textuel
        context = ""
        all_images = []
        
        for pdf_content, score in relevant_pdfs:
            context += f"\nDocument: {pdf_content.source}\n"
            context += f"Contenu:\n{pdf_content.total_text}\n"
            
            # Collecter toutes les images
            for page in pdf_content.pages:
                images = page['images'][:SEARCH_PARAMS['max_images_per_page']]
                all_images.extend(images)
        
        # Analyser les images par lots de 4
        image_descriptions = []
        for i in range(0, len(all_images), SEARCH_PARAMS['max_total_images']):
            batch = all_images[i:i + SEARCH_PARAMS['max_total_images']]
            description = self.analyze_images_with_pixtral(query, batch)
            if description:
                image_descriptions.append(description)
        
        # Combiner toutes les informations
        full_context = f"Contexte textuel:\n{context}\n\nAnalyse des images:\n"
        full_context += "\n".join(image_descriptions)
        
        # Générer la réponse finale avec Mistral
        response = requests.post(
            self.vllm_url,
            headers={"Content-Type": "application/json"},
            json={
                "model": MISTRAL_PATH,
                "messages": [
                    {
                        "role": "system",
                        "content": "Tu es un assistant expert qui répond aux questions en utilisant uniquement les informations fournies. Si tu ne trouves pas l'information dans le contexte, dis-le clairement."
                    },
                    {
                        "role": "user",
                        "content": f"Question : {query}\n\nInformations disponibles :\n{full_context}"
                    }
                ],
                **MODEL_PARAMS
            }
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            print(f"Erreur Mistral : {response.status_code}")
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
