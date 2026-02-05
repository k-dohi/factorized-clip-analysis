"""
CLIP Anomaly Scorer

Implements the anomaly scoring function from the paper:
    s_CLIP = exp(ℓ^A/T) / (exp(ℓ^N/T) + exp(ℓ^A/T))

where:
    - ℓ^N = cosine similarity to normal text embedding
    - ℓ^A = cosine similarity to anomaly text embedding
    - T = 0.07 (temperature parameter)
"""

import torch
from PIL import Image
import open_clip
from typing import Tuple, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TEMPERATURE


class CLIPScorer:
    """CLIP-based anomaly scorer using softmax with temperature."""
    
    def __init__(self, 
                 model_name: str = 'ViT-H-14',
                 pretrained: str = 'laion2b_s32b_b79k',
                 device: str = 'auto',
                 temperature: float = TEMPERATURE):
        """
        Initialize CLIP scorer.
        
        Args:
            model_name: CLIP model architecture
            pretrained: Pretrained weights identifier
            device: Device to use ('auto', 'cuda', or 'cpu')
            temperature: Softmax temperature (default: 0.07)
        """
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.temperature = temperature
        self.model_name = model_name
        self.pretrained = pretrained
        
        print(f"Loading CLIP: {model_name} ({pretrained}) on {self.device}")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        
        # Cache for text embeddings
        self._text_cache = {}
    
    def _get_text_features(self, object_class: str, 
                           normal_template: str = "a photo of a {obj}",
                           anomaly_template: str = "a photo of a damaged {obj}") -> torch.Tensor:
        """
        Get or compute cached text features.
        
        Returns:
            Text features tensor of shape (2, D) where D is embedding dimension
        """
        cache_key = (object_class, normal_template, anomaly_template)
        if cache_key not in self._text_cache:
            normal_text = normal_template.format(obj=object_class)
            anomaly_text = anomaly_template.format(obj=object_class)
            
            tokens = self.tokenizer([normal_text, anomaly_text]).to(self.device)
            with torch.no_grad():
                text_features = self.model.encode_text(tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            self._text_cache[cache_key] = text_features
        
        return self._text_cache[cache_key]
    
    def compute_score(self, 
                      image: Image.Image, 
                      object_class: str,
                      normal_template: str = "a photo of a {obj}",
                      anomaly_template: str = "a photo of a damaged {obj}"
                      ) -> Tuple[float, float, float]:
        """
        Compute anomaly score using softmax with temperature.
        
        Args:
            image: PIL Image to score
            object_class: Object class name (e.g., 'bottle')
            normal_template: Template for normal text prompt
            anomaly_template: Template for anomaly text prompt
        
        Returns:
            Tuple of (anomaly_score, sim_normal, sim_anomaly)
            - anomaly_score: Softmax probability of anomaly [0, 1]
            - sim_normal: Cosine similarity to normal text
            - sim_anomaly: Cosine similarity to anomaly text
        """
        text_features = self._get_text_features(object_class, normal_template, anomaly_template)
        
        # Encode image
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Compute similarities
        similarities = (image_features @ text_features.T).squeeze()
        sim_normal = similarities[0].item()
        sim_anomaly = similarities[1].item()
        
        # Softmax with temperature (eq. clip_score in paper)
        logits = similarities / self.temperature
        probs = torch.softmax(logits, dim=0)
        anomaly_score = probs[1].item()  # P(anomaly)
        
        return anomaly_score, sim_normal, sim_anomaly
    
    def compute_scores_batch(self, 
                             images: list,
                             object_class: str,
                             normal_template: str = "a photo of a {obj}",
                             anomaly_template: str = "a photo of a damaged {obj}"
                             ) -> Tuple[list, list, list]:
        """
        Compute anomaly scores for a batch of images.
        
        Args:
            images: List of PIL Images
            object_class: Object class name
            normal_template: Template for normal text prompt
            anomaly_template: Template for anomaly text prompt
        
        Returns:
            Tuple of (anomaly_scores, sim_normals, sim_anomalies)
        """
        text_features = self._get_text_features(object_class, normal_template, anomaly_template)
        
        # Batch encode images
        tensors = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(tensors)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Compute similarities
        similarities = image_features @ text_features.T  # (N, 2)
        
        # Softmax with temperature
        logits = similarities / self.temperature
        probs = torch.softmax(logits, dim=1)
        
        anomaly_scores = probs[:, 1].cpu().numpy().tolist()
        sim_normals = similarities[:, 0].cpu().numpy().tolist()
        sim_anomalies = similarities[:, 1].cpu().numpy().tolist()
        
        return anomaly_scores, sim_normals, sim_anomalies
