"""
Experiment 2: Background Masking Analysis

Evaluates whether background masking can mitigate CLIP's sensitivity
to foreign object intrusions.

Protocol:
- Load pre-generated pseudo images from PseudoImageLoader
- Use AutoBBoxSAMSegmenter (same as pseudo image generation) for segmentation
- Compare detection with black/white background masking
- Evaluate on both Real vs Real and Foreign vs Real scenarios

NOTE: Pseudo images must be generated BEFORE running this experiment.
Run `python scripts/public/generate_pseudo_images.py` first.
"""

import argparse
import json
import sys
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import yaml
from tqdm import tqdm
import torch

# Add parent directories to path
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    OBJECT_CLASSES, N_SAMPLES, RANDOM_SEED,
    DEFAULT_MODEL, DEFAULT_PRETRAINED, get_output_dir
)
from utils.clip_scorer import CLIPScorer
from utils.data_loader import MVTecDataLoader
from utils.metrics import compute_auroc
from generate_pseudo_images import PseudoImageLoader

# Import AutoBBoxSAMSegmenter from project source
from src.datasets.segmenters.auto_bbox_sam_segmenter import AutoBBoxSAMSegmenter

# Path to segmentation config
SEGMENTATION_CONFIG = PROJECT_ROOT / "config" / "segmentation" / "mvtec_objects.yaml"


def apply_mask(image: np.ndarray, mask: np.ndarray, 
               bg_color: str = 'black') -> np.ndarray:
    """
    Apply background mask to image.
    
    Args:
        image: RGB image
        mask: Binary mask (255 = foreground, or 1 = foreground)
        bg_color: 'black' or 'white'
    
    Returns:
        Masked image
    """
    if bg_color == 'black':
        bg = np.zeros_like(image)
    else:  # white
        bg = np.ones_like(image) * 255
    
    # Normalize mask to 0-1 range
    mask_normalized = mask.astype(np.float32)
    if mask_normalized.max() > 1:
        mask_normalized = mask_normalized / 255.0
    
    mask_3ch = np.stack([mask_normalized] * 3, axis=-1)
    return (image * mask_3ch + bg * (1 - mask_3ch)).astype(np.uint8)


def init_segmenter():
    """Initialize AutoBBoxSAMSegmenter with same config as pseudo image generation."""
    print("Loading segmentation config...")
    with open(SEGMENTATION_CONFIG, 'r') as f:
        seg_configs = yaml.safe_load(f)
    
    print("Initializing AutoBBoxSAMSegmenter with SAM 3...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    segmenter = AutoBBoxSAMSegmenter(
        model_id="facebook/sam3",
        device=device,
        object_type_configs=seg_configs
    )
    print(f"  ✓ Segmenter initialized on {device}")
    return segmenter


def segment_image(segmenter, image: np.ndarray, object_class: str) -> np.ndarray:
    """
    Segment image using AutoBBoxSAMSegmenter.
    
    Args:
        segmenter: AutoBBoxSAMSegmenter instance
        image: RGB image (np.ndarray)
        object_class: Object class name
    
    Returns:
        Binary mask (255 = foreground, 0 = background)
    """
    # Convert RGB to BGR for SAM
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    segmenter.set_object_type(object_class)
    config = segmenter.get_config_for_object(object_class)
    sam_mode = config.get("sam_mode", "sam_combined")
    
    seg_results = segmenter.segment_all_modes(img_bgr, object_class)
    
    if sam_mode not in seg_results:
        raise ValueError(f"SAM mode '{sam_mode}' not available. Available: {list(seg_results.keys())}")
    
    mask = seg_results[sam_mode]
    if mask is None:
        raise ValueError(f"SAM mode '{sam_mode}' returned None")
    
    # Ensure mask is 0-255 range
    mask = mask.astype(np.uint8)
    if mask.max() <= 1:
        mask = mask * 255
    
    return mask


def run_masking_experiment(scorer: CLIPScorer,
                           data_loader: MVTecDataLoader,
                           pseudo_loader: PseudoImageLoader,
                           segmenter,
                           output_dir: Path) -> dict:
    """
    Run background masking experiment.
    
    Compares:
    - No masking (baseline)
    - Black background masking
    - White background masking
    
    Uses pre-generated pseudo images from PseudoImageLoader.
    Uses AutoBBoxSAMSegmenter for consistent segmentation.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Foreign object variations to test
    foreign_variation_types = [
        'simple_shape_foreign',
        'signs_foreign',
        'texture_foreign',
        'sd_foreign',
    ]
    
    results = {
        'real_vs_real': {'no_mask': {}, 'black': {}, 'white': {}},
        'foreign_vs_real': {'no_mask': {}, 'black': {}, 'white': {}}
    }
    
    for i, object_class in enumerate(tqdm(OBJECT_CLASSES, desc="[Exp2] Masking")):
        print(f"\n  [{i+1}/{len(OBJECT_CLASSES)}] {object_class}")
        
        # Load pre-generated normal images
        normal_pil_images = pseudo_loader.get_normal_images(object_class)
        normal_images = [(idx, np.array(img)) for idx, img in normal_pil_images]
        
        # Get anomaly data from MVTec
        anomaly_data = data_loader.get_test_anomaly_images(object_class)
        
        # Pre-compute masks for normal images
        print(f"    Computing masks for normal images...")
        normal_masks = {}
        for idx, img in normal_images:
            normal_masks[idx] = segment_image(segmenter, img, object_class)
        
        # Pre-compute masks for anomaly images
        print(f"    Computing masks for anomaly images...")
        anomaly_masks = {}
        for j, (_, img, _) in enumerate(anomaly_data):
            anomaly_masks[j] = segment_image(segmenter, img, object_class)
        
        # === Real vs Real ===
        print(f"    Evaluating Real vs Real...")
        for mask_type in ['no_mask', 'black', 'white']:
            normal_scores = []
            for idx, img in normal_images:
                img_copy = img.copy()
                if mask_type != 'no_mask':
                    img_copy = apply_mask(img_copy, normal_masks[idx], mask_type)
                
                pil_img = Image.fromarray(img_copy)
                score, _, _ = scorer.compute_score(pil_img, object_class)
                normal_scores.append(score)
            
            anomaly_scores = []
            for j, (_, img, _) in enumerate(anomaly_data):
                img_copy = img.copy()
                if mask_type != 'no_mask':
                    img_copy = apply_mask(img_copy, anomaly_masks[j], mask_type)
                
                pil_img = Image.fromarray(img_copy)
                score, _, _ = scorer.compute_score(pil_img, object_class)
                anomaly_scores.append(score)
            
            auroc = compute_auroc(normal_scores, anomaly_scores) * 100
            results['real_vs_real'][mask_type][object_class] = auroc
        
        # === Foreign vs Real ===
        print(f"    Evaluating Foreign vs Real...")
        
        # Load pre-generated foreign object images for each variation type
        pre_generated_foreign = {}  # {var_name: [(idx, img), ...]}
        for var_type in foreign_variation_types:
            try:
                pre_generated_foreign[var_type] = pseudo_loader.get_pseudo_images_np(
                    object_class, var_type
                )
            except FileNotFoundError as e:
                print(f"      Warning: {var_type} not found: {e}")
                pre_generated_foreign[var_type] = []
        
        # Pre-compute masks for foreign images
        foreign_masks = {}  # {var_type: {idx: mask}}
        for var_type in foreign_variation_types:
            foreign_masks[var_type] = {}
            for idx, img in pre_generated_foreign.get(var_type, []):
                foreign_masks[var_type][idx] = segment_image(segmenter, img, object_class)
        
        # Now test each mask type using the pre-generated foreign images
        for mask_type in ['no_mask', 'black', 'white']:
            # Collect scores for all foreign variations
            all_foreign_aurocs = []
            
            for var_type in foreign_variation_types:
                foreign_images = pre_generated_foreign[var_type]
                if not foreign_images:
                    continue
                
                foreign_scores = []
                for idx, foreign_img in foreign_images:
                    img_to_score = foreign_img.copy()
                    if mask_type != 'no_mask':
                        img_to_score = apply_mask(img_to_score, foreign_masks[var_type][idx], mask_type)
                    
                    pil_img = Image.fromarray(img_to_score)
                    score, _, _ = scorer.compute_score(pil_img, object_class)
                    foreign_scores.append(score)
                
                if len(foreign_scores) > 0:
                    # Anomaly scores are the same for all variations (already computed above)
                    anomaly_scores_masked = []
                    for j, (_, img, _) in enumerate(anomaly_data):
                        img_copy = img.copy()
                        if mask_type != 'no_mask':
                            img_copy = apply_mask(img_copy, anomaly_masks[j], mask_type)
                        
                        pil_img = Image.fromarray(img_copy)
                        score, _, _ = scorer.compute_score(pil_img, object_class)
                        anomaly_scores_masked.append(score)
                    
                    auroc = compute_auroc(foreign_scores, anomaly_scores_masked) * 100
                    all_foreign_aurocs.append(auroc)
            
            # Average over all foreign variations
            if all_foreign_aurocs:
                avg_auroc = np.mean(all_foreign_aurocs)
                results['foreign_vs_real'][mask_type][object_class] = avg_auroc
    
    # Compute averages
    for scenario in ['real_vs_real', 'foreign_vs_real']:
        for mask_type in ['no_mask', 'black', 'white']:
            vals = list(results[scenario][mask_type].values())
            results[scenario][mask_type]['average'] = np.mean(vals)
    
    # Save results
    output_path = output_dir / "masking_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    print_masking_summary(results)
    
    return results


def print_masking_summary(results: dict):
    """Print formatted masking results summary."""
    print("\n" + "="*70)
    print("MASKING EXPERIMENT SUMMARY")
    print("="*70)
    
    print("\n--- Real vs Real ---")
    print(f"{'Object':<15} {'No Mask':<10} {'Black':<10} {'White':<10}")
    for obj in OBJECT_CLASSES:
        row = f"{obj:<15}"
        for mask in ['no_mask', 'black', 'white']:
            row += f" {results['real_vs_real'][mask][obj]:<10.1f}"
        print(row)
    
    row = f"{'Average':<15}"
    for mask in ['no_mask', 'black', 'white']:
        row += f" {results['real_vs_real'][mask]['average']:<10.1f}"
    print(row)
    
    print("\n--- Foreign vs Real ---")
    print(f"{'Object':<15} {'No Mask':<10} {'Black':<10} {'White':<10}")
    for obj in OBJECT_CLASSES:
        row = f"{obj:<15}"
        for mask in ['no_mask', 'black', 'white']:
            row += f" {results['foreign_vs_real'][mask][obj]:<10.1f}"
        print(row)
    
    row = f"{'Average':<15}"
    for mask in ['no_mask', 'black', 'white']:
        row += f" {results['foreign_vs_real'][mask]['average']:<10.1f}"
    print(row)


def main():
    parser = argparse.ArgumentParser(description='Experiment 2: Background Masking')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL)
    parser.add_argument('--pretrained', type=str, default=DEFAULT_PRETRAINED)
    args = parser.parse_args()
    
    output_dir = get_output_dir("experiment2_masking")
    
    # Initialize segmenter first (same as pseudo image generation)
    segmenter = init_segmenter()
    
    scorer = CLIPScorer(args.model, args.pretrained)
    data_loader = MVTecDataLoader()
    pseudo_loader = PseudoImageLoader()
    
    # Verify pseudo images exist
    print("Verifying pre-generated pseudo images...")
    for obj in OBJECT_CLASSES:
        if not pseudo_loader.has_variation(obj, "simple_shape_foreign"):
            raise RuntimeError(
                f"Pseudo images not found for {obj}. "
                "Run `python scripts/public/generate_pseudo_images.py` first."
            )
    print("  Pseudo images verified.\n")
    
    run_masking_experiment(scorer, data_loader, pseudo_loader, segmenter, output_dir)


if __name__ == '__main__':
    main()
