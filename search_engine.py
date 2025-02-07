import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import faiss
import numpy as np
from typing import List, Union, Dict
import os

class MultiModalSearch:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        # Initialize CLIP model and processor
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Initialize storage
        self.index = None
        self.image_paths = []
        
    def _encode_image(self, image: Union[str, Image.Image]) -> np.ndarray:
        """Encode image to embedding vector"""
        if isinstance(image, str):
            image = Image.open(image)
        inputs = self.processor(images=image, return_tensors="pt")
        image_features = self.model.get_image_features(**inputs)
        return image_features.detach().numpy()
    
    def _encode_text(self, text: str) -> np.ndarray:
        """Encode text to embedding vector"""
        inputs = self.processor(text=text, return_tensors="pt", padding=True)
        text_features = self.model.get_text_features(**inputs)
        return text_features.detach().numpy()
    
    def index_images(self, image_paths: List[str]):
        """Index a list of images for search"""
        embeddings = []
        for path in image_paths:
            embedding = self._encode_image(path)
            embeddings.append(embedding)
            self.image_paths.append(path)
            
        embeddings = np.vstack(embeddings)
        
        # Initialize FAISS index if not exists
        if self.index is None:
            self.index = faiss.IndexFlatIP(embeddings.shape[1])
        
        # Add embeddings to index
        self.index.add(embeddings)
    
    def search_by_text(self, query_text: str, k: int = 5) -> List[Dict]:
        """Search for similar images using a text query"""
        if not self.index:
            return []
            
        query_embedding = self._encode_text(query_text)
        D, I = self.index.search(query_embedding, min(k, len(self.image_paths)))
        
        return [
            {
                "path": self.image_paths[i],
                "score": float(d),
                "filename": os.path.basename(self.image_paths[i])
            } 
            for d, i in zip(D[0], I[0])
        ]