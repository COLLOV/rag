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

@dataclass
class PDFContent:
    source: str
    pages: List[dict]  # Liste des pages avec leur texte et images
    total_text: str  # Texte complet du PDF

from config import (VLLM_URL, MODEL_PARAMS, SEARCH_PARAMS, 
                  TEMP_DIR, EMBEDDING_MODEL, PDF_FOLDER)

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
        self.pages: List[PageContent] = []
        
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
        temp_path = self.temp_dir / "temp.jpg"
        image.save(temp_path, "JPEG")
        
        with open(temp_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encoded}"
            }
        }

    def analyze_image(self, image: Image.Image) -> str:
        """Analyse une image avec Pixtral via VLLM et retourne une description."""
        image_data = self.encode_image_base64(image)
        
        messages = [
            {"role": "user", "content": [
                image_data,
                "Décris cette image en détail."
            ]}
        ]
        
        response = requests.post(
            self.vllm_url,
            json={
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 500
            }
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return "Erreur lors de l'analyse de l'image"

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
                            page_descriptions.append(desc)
                        except Exception as e:
                            print(f"Erreur lors de l'analyse d'une image : {e}")
                    
                    image_descriptions_by_page.append(page_descriptions)
                
                # Combiner tout le contenu du PDF
                combined_text = f"""Document: {pdf_content.source}\n"""
                combined_text += f"Contenu textuel:\n{pdf_content.total_text}\n"
                combined_text += "\nDescriptions des images:\n"
                
                for page_num, descriptions in enumerate(image_descriptions_by_page, 1):
                    if descriptions:
                        combined_text += f"Page {page_num}:\n"
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

    def search(self, query: str, k: int = SEARCH_PARAMS["max_results"]) -> List[Tuple[PageContent, float]]:
        """Recherche les pages les plus pertinentes."""

        """Recherche les pages les plus pertinentes pour la question."""
        # Encoder la requête
        inputs = self.tokenizer(query, padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        query_embedding = outputs.last_hidden_state.mean(dim=1).numpy()
        
        # Rechercher les k*2 plus proches voisins
        D, I = self.index.search(query_embedding.astype(np.float32), k*2)
        
        # Organiser les résultats par document
        docs_pages = {}
        for idx, dist in zip(I[0], D[0]):
            page = self.pages[idx]
            key = page.source
            if key not in docs_pages:
                docs_pages[key] = []
            docs_pages[key].append((page, dist))
        
        # Trier les pages par numéro pour chaque document
        for key in docs_pages:
            docs_pages[key].sort(key=lambda x: x[0].page_number)
        
        # Sélectionner les séquences de pages consécutives
        final_results = []
        for doc_pages in docs_pages.values():
            sequences = []
            current_seq = [doc_pages[0]]
            
            for i in range(1, len(doc_pages)):
                if doc_pages[i][0].page_number == doc_pages[i-1][0].page_number + 1:
                    current_seq.append(doc_pages[i])
                else:
                    sequences.append(current_seq)
                    current_seq = [doc_pages[i]]
            sequences.append(current_seq)
            
            if sequences:
                longest_seq = max(sequences, key=len)
                final_results.extend(longest_seq)
        
        # Trier par score de similarité et retourner les k meilleurs résultats
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
                            context += f"- {desc}\n"
                        except Exception as e:
                            print(f"Erreur lors de l'analyse d'une image : {e}")
        
        # Construire le prompt pour le LLM
        messages = [
            {"role": "system", "content": "Tu es un assistant serviable qui répond aux questions en utilisant uniquement les informations fournies dans le contexte. Si tu ne trouves pas l'information dans le contexte, dis-le clairement."}, 
            {"role": "user", "content": f"Contexte:\n{context}\n\nQuestion: {query}\n\nRéponds à la question en utilisant uniquement les informations du contexte ci-dessus. Sois précis et cite les numéros de page et les documents sources."}
        ]
        
        # Appeler l'API VLLM pour générer la réponse
        response = requests.post(
            self.vllm_url,
            json={
                "messages": messages,
                **MODEL_PARAMS
            }
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
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
