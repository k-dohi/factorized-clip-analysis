#!/usr/bin/env python3
"""
Experiment 2: Variation-Specific Prompt Evaluation

Corresponds to Section 5.2 "Variation-Specific Prompt Evaluation" in the paper.

Objective:
- Evaluate whether variation-specific prompts can detect each variation type
- Evaluate all 17 variation types (5 categories)
- Three evaluation settings: Neg=N, Neg=N+A, Neg=A

Protocol:
- Normal prompt: "a photo of a {class}"
- Variation prompt: "a photo of a {class} [suffix]" (variation-specific)
- Positive: pseudo-images with variation type v
- Negative: 
  - Neg=N: real normal test images
  - Neg=N+A: real normal + real anomaly test images
  - Neg=A: real anomaly test images only

Expected Results (Table VI from paper):
- Object category: 86.2% average (Neg=N)
- Lighting category: 59.3% average (Neg=N)
- Medium category: 80.9% average (Neg=N)
- Foreign category: 91.6% average (Neg=N)
- Camera category: 76.4% average (Neg=N)
- Overall: 80.0% average (Neg=N)
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# Project root
SCRIPT_DIR = Path(__file__).parent
PUBLIC_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = PUBLIC_DIR.parent.parent

sys.path.insert(0, str(PUBLIC_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import open_clip
except ImportError:
    print("Error: open_clip not installed. Run: pip install open_clip_torch")
    sys.exit(1)

from sklearn.metrics import roc_auc_score

from config import (
    OBJECT_CLASSES, N_SAMPLES, RANDOM_SEED, TEMPERATURE,
    DEFAULT_MODEL, DEFAULT_PRETRAINED
)
from utils.data_loader import MVTecDataLoader
from utils.metrics import compute_auroc
from generate_pseudo_images import PseudoImageLoader


# Variation-Specific Prompts (Table III from paper)
VARIATION_PROMPTS = {
    # Category 1: Object Anomalies
    'object': {
        'cutpaste': 'with damage',
        'scratchmix': 'with scratches',
        'texture': 'with unusual texture patterns',
        'sd_inpaint': 'with damage',
    },
    # Category 2: Lighting
    'lighting': {
        'colorshift': 'with abnormal color shift',
        'contrast_change': 'with abnormal contrast',
        'brightness': 'with abnormal brightness',
    },
    # Category 3: Medium (Atmospheric)
    'medium': {
        'fog': 'in foggy conditions',
        'smoke': 'obscured by smoke',
        'heathaze': 'distorted by heat haze',
    },
    # Category 4: Foreign Objects
    'foreign': {
        'simple_shape_foreign': 'with simple geometric foreign objects',
        'signs_foreign': 'with warning signs or labels',
        'texture_foreign': 'with textured foreign patches',
        'sd_foreign': 'with intrusive foreign objects',
    },
    # Category 5: Camera Artifacts
    'camera': {
        'motionblur': 'with motion blur',
        'noise': 'with image noise',
        'defocusblur': 'with defocus blur',
    },
}


class VariationSpecificPromptEvaluator:
    """
    Experiment 2: Variation-Specific Prompt Evaluation
    
    Evaluate whether variation-specific prompts can detect each variation type.
    Uses pre-generated pseudo images from PseudoImageLoader.
    """
    
    def __init__(
        self,
        model_name: str = "ViT-H-14",
        pretrained: str = "laion2b_s32b_b79k",
        device: str = None,
        pseudo_loader: PseudoImageLoader = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.pretrained = pretrained
        
        # Load CLIP model
        print(f"Loading CLIP model: {model_name} ({pretrained})")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # PseudoImageLoader for pre-generated images
        self.pseudo_loader = pseudo_loader or PseudoImageLoader()
        
    def _encode_text(self, text: str) -> torch.Tensor:
        """Encode text"""
        with torch.no_grad():
            tokens = self.tokenizer([text]).to(self.device)
            features = self.model.encode_text(tokens)
            return F.normalize(features, dim=-1)
    
    def _encode_image(self, image: Image.Image) -> torch.Tensor:
        """Encode image"""
        with torch.no_grad():
            img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            features = self.model.encode_image(img_tensor)
            return F.normalize(features, dim=-1)
    
    def compute_score(
        self,
        image: Image.Image,
        object_class: str,
        variation_suffix: str,
    ) -> float:
        """
        Compute score with variation-specific prompt
        
        Args:
            image: Input image
            object_class: Object class name
            variation_suffix: variation-specific suffix
            
        Returns:
            anomaly_score: softmax-normalized anomaly score
        """
        # Prompts
        normal_prompt = f"a photo of a {object_class}"
        anomaly_prompt = f"a photo of a {object_class} {variation_suffix}"
        
        # Encode
        normal_features = self._encode_text(normal_prompt)
        anomaly_features = self._encode_text(anomaly_prompt)
        image_features = self._encode_image(image)
        
        # Compute similarities
        sim_normal = (image_features @ normal_features.T).item()
        sim_anomaly = (image_features @ anomaly_features.T).item()
        
        # Softmax with temperature
        exp_normal = np.exp(sim_normal / TEMPERATURE)
        exp_anomaly = np.exp(sim_anomaly / TEMPERATURE)
        score = exp_anomaly / (exp_normal + exp_anomaly)
        
        return score
    
    def evaluate_variation_type(
        self,
        data_loader: MVTecDataLoader,
        object_class: str,
        category: str,
        variation_type: str,
        variation_suffix: str,
    ) -> Dict:
        """
        Evaluate a single variation type
        
        Args:
            data_loader: MVTec data loader
            object_class: Object class name
            category: Category name
            variation_type: Variation type name
            variation_suffix: variation-specific suffix
            
        Returns:
            results dict with AUROC for Neg=N, Neg=N+A, Neg=A
        """
        # Load pre-generated normal images
        try:
            normal_pil = self.pseudo_loader.get_normal_images(object_class)
        except FileNotFoundError:
            print(f"  Warning: Normal images not found for {object_class}")
            return None
        
        # Load pre-generated pseudo images for this variation type
        try:
            pseudo_pil = self.pseudo_loader.get_pseudo_images(object_class, variation_type)
        except FileNotFoundError:
            print(f"  Warning: {variation_type} images not found for {object_class}")
            return None
        
        # Get real anomaly images
        real_anomaly_data = data_loader.get_test_anomaly_images(object_class)
        
        # Positive: pre-generated pseudo-images with this variation type
        positive_scores = []
        for _, pil_img in pseudo_pil:
            score = self.compute_score(pil_img, object_class, variation_suffix)
            positive_scores.append(score)
        
        if len(positive_scores) == 0:
            return None
        
        # Negative=N: pre-generated normal images
        neg_n_scores = []
        for _, pil_img in normal_pil:
            score = self.compute_score(pil_img, object_class, variation_suffix)
            neg_n_scores.append(score)
        
        # Negative=A: real anomaly test images
        neg_a_scores = []
        for _, img, _ in real_anomaly_data:
            pil_img = Image.fromarray(img)
            score = self.compute_score(pil_img, object_class, variation_suffix)
            neg_a_scores.append(score)
        
        # Compute AUROCs
        results = {}
        
        # Neg=N
        if len(neg_n_scores) > 0:
            auroc_n = compute_auroc(neg_n_scores, positive_scores)
            results['neg_n'] = auroc_n * 100
        
        # Neg=N+A
        if len(neg_n_scores) > 0 and len(neg_a_scores) > 0:
            neg_na_scores = neg_n_scores + neg_a_scores
            auroc_na = compute_auroc(neg_na_scores, positive_scores)
            results['neg_na'] = auroc_na * 100
        
        # Neg=A
        if len(neg_a_scores) > 0:
            auroc_a = compute_auroc(neg_a_scores, positive_scores)
            results['neg_a'] = auroc_a * 100
        
        return results
    
    def run(
        self,
        data_dir: str,
        object_classes: List[str] = None,
    ) -> Dict:
        """
        Run evaluation for all variation types
        
        Args:
            data_dir: MVTecAD data directory
            object_classes: Objects to evaluate (None for all objects)
            
        Returns:
            results dict
        """
        print("\n" + "="*70)
        print("Experiment 2: Variation-Specific Prompt Evaluation")
        print("="*70)
        
        if object_classes is None:
            object_classes = OBJECT_CLASSES
        
        data_loader = MVTecDataLoader(data_dir)
        
        all_results = {}
        
        for category, variations in VARIATION_PROMPTS.items():
            print(f"\n--- Category: {category.upper()} ---")
            category_results = {}
            
            for variation_type, suffix in variations.items():
                print(f"\n  Variation: {variation_type}")
                variation_results = {'neg_n': [], 'neg_na': [], 'neg_a': []}
                
                for obj_class in tqdm(object_classes, desc=f"[Exp6] {category}/{variation_type}"):
                    result = self.evaluate_variation_type(
                        data_loader, obj_class, category, variation_type, suffix
                    )
                    
                    if result:
                        for key in ['neg_n', 'neg_na', 'neg_a']:
                            if key in result:
                                variation_results[key].append(result[key])
                
                # Average across objects
                avg_results = {}
                for key in ['neg_n', 'neg_na', 'neg_a']:
                    if variation_results[key]:
                        avg_results[key] = np.mean(variation_results[key])
                
                category_results[variation_type] = avg_results
                
                if 'neg_n' in avg_results:
                    print(f"    Neg=N: {avg_results['neg_n']:.1f}%")
            
            all_results[category] = category_results
        
        return all_results


def print_results_table(results: Dict):
    """Display results in paper Table VI style"""
    print("\n" + "="*70)
    print("Variation-Specific Prompt Detection AUROC (%)")
    print("="*70)
    print(f"{'Category':<12} {'Variation Type':<20} {'Neg=N':<10} {'Neg=N+A':<10} {'Neg=A':<10}")
    print("-"*70)
    
    overall_neg_n = []
    overall_neg_na = []
    overall_neg_a = []
    
    for category, variations in results.items():
        cat_neg_n = []
        cat_neg_na = []
        cat_neg_a = []
        
        for var_type, scores in variations.items():
            neg_n = scores.get('neg_n', float('nan'))
            neg_na = scores.get('neg_na', float('nan'))
            neg_a = scores.get('neg_a', float('nan'))
            
            print(f"{category:<12} {var_type:<20} {neg_n:>8.1f}%  {neg_na:>8.1f}%  {neg_a:>8.1f}%")
            
            if not np.isnan(neg_n):
                cat_neg_n.append(neg_n)
                overall_neg_n.append(neg_n)
            if not np.isnan(neg_na):
                cat_neg_na.append(neg_na)
                overall_neg_na.append(neg_na)
            if not np.isnan(neg_a):
                cat_neg_a.append(neg_a)
                overall_neg_a.append(neg_a)
        
        # Category average
        if cat_neg_n:
            avg_n = np.mean(cat_neg_n)
            avg_na = np.mean(cat_neg_na) if cat_neg_na else float('nan')
            avg_a = np.mean(cat_neg_a) if cat_neg_a else float('nan')
            print(f"{'':12} {'Category Avg':<20} {avg_n:>8.1f}%  {avg_na:>8.1f}%  {avg_a:>8.1f}%")
        print("-"*70)
    
    # Overall average
    if overall_neg_n:
        print(f"{'OVERALL':<12} {'Average':<20} {np.mean(overall_neg_n):>8.1f}%  {np.mean(overall_neg_na):>8.1f}%  {np.mean(overall_neg_a):>8.1f}%")
    print("="*70)


def main():
    """Main execution function"""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Experiment 2: Variation-Specific Prompt Evaluation")
    parser.add_argument("--data_dir", type=str, required=True, help="MVTecAD data directory")
    parser.add_argument("--model_name", type=str, default="ViT-H-14", help="CLIP model name")
    parser.add_argument("--pretrained", type=str, default="laion2b_s32b_b79k", help="Pretrained weights")
    parser.add_argument("--output_file", type=str, default=None, help="Output JSON file")
    args = parser.parse_args()
    
    # Run experiment
    evaluator = VariationSpecificPromptEvaluator(
        model_name=args.model_name,
        pretrained=args.pretrained,
    )
    
    results = evaluator.run(data_dir=args.data_dir)
    
    # Display results
    print_results_table(results)
    
    # Save results
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    
    return results


def run_variation_specific_experiment(scorer, data_loader, pseudo_loader, output_dir):
    """
    Wrapper function to run variation-specific prompt evaluation.
    
    This allows integration with the unified run_all_experiments.py script.
    
    Args:
        scorer: CLIPScorer instance (not used directly - evaluator creates its own)
        data_loader: MVTecDataLoader instance
        pseudo_loader: PseudoImageLoader instance for pre-generated images
        output_dir: Path to save results
        
    Returns:
        Results dictionary
    """
    import json
    
    # Verify pseudo images exist
    print("Verifying pre-generated pseudo images...")
    for obj in OBJECT_CLASSES:
        if not pseudo_loader.has_variation(obj, "colorshift"):
            raise RuntimeError(
                f"Pseudo images not found for {obj}. "
                "Run `python scripts/public/generate_pseudo_images.py` first."
            )
    print("  Pseudo images verified.\n")
    
    # Use scorer's model info
    evaluator = VariationSpecificPromptEvaluator(
        model_name=scorer.model_name,
        pretrained=scorer.pretrained,
        pseudo_loader=pseudo_loader,
    )
    
    # Run evaluation
    results = evaluator.run(data_dir=str(data_loader.data_dir))
    
    # Print summary
    print_results_table(results)
    
    # Save results
    output_path = output_dir / "variation_specific_results.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    main()
