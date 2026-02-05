"""
Experiment 1: Zero-Shot Detection under Contextual Variations

Main experiment evaluating CLIP's detection performance across:
- Scenario 1: Real normal vs Real anomaly
- Scenario 2: Pseudo-normal (with variation) vs Real anomaly  
- Scenario 3: Real normal vs Object pseudo-anomaly

Protocol:
- Use pre-generated pseudo images from PseudoImageLoader
- Compute AUROC using softmax score with T=0.07

NOTE: Pseudo images must be generated BEFORE running this experiment.
Run `python scripts/public/generate_pseudo_images.py` first.
"""

import argparse
import json
import sys
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    OBJECT_CLASSES, VARIATION_TAXONOMY, N_SAMPLES, RANDOM_SEED,
    DEFAULT_MODEL, DEFAULT_PRETRAINED, get_output_dir
)
from utils.clip_scorer import CLIPScorer
from utils.data_loader import MVTecDataLoader
from utils.metrics import compute_auroc
from generate_pseudo_images import PseudoImageLoader


def run_scenario1(scorer: CLIPScorer, 
                  data_loader: MVTecDataLoader,
                  object_class: str) -> float:
    """
    Scenario 1: Real normal vs Real anomaly
    
    Baseline detection performance without any variations.
    """
    # Sample test normal images
    normal_images = data_loader.sample_test_normal_images(object_class)
    anomaly_data = data_loader.get_test_anomaly_images(object_class)
    
    # Score normal images
    normal_scores = []
    for _, img in normal_images:
        pil_img = Image.fromarray(img)
        score, _, _ = scorer.compute_score(pil_img, object_class)
        normal_scores.append(score)
    
    # Score anomaly images
    anomaly_scores = []
    for _, img, _ in anomaly_data:
        pil_img = Image.fromarray(img)
        score, _, _ = scorer.compute_score(pil_img, object_class)
        anomaly_scores.append(score)
    
    return compute_auroc(normal_scores, anomaly_scores) * 100


def run_scenario2(scorer: CLIPScorer,
                  data_loader: MVTecDataLoader,
                  pseudo_loader: PseudoImageLoader,
                  object_class: str,
                  variation_type: str) -> float:
    """
    Scenario 2: Pseudo-normal (test normal + variation) vs Real anomaly
    
    Evaluates CLIP's robustness to contextual variations.
    Uses pre-generated pseudo images.
    """
    # Load pre-generated pseudo images
    pseudo_images = pseudo_loader.get_pseudo_images(object_class, variation_type)
    anomaly_data = data_loader.get_test_anomaly_images(object_class)
    
    # Score pseudo-normal images
    pseudo_normal_scores = []
    for _, img in pseudo_images:
        score, _, _ = scorer.compute_score(img, object_class)
        pseudo_normal_scores.append(score)
    
    # Score anomaly images (no variation)
    anomaly_scores = []
    for _, img, _ in anomaly_data:
        pil_img = Image.fromarray(img)
        score, _, _ = scorer.compute_score(pil_img, object_class)
        anomaly_scores.append(score)
    
    return compute_auroc(pseudo_normal_scores, anomaly_scores) * 100


def run_scenario3(scorer: CLIPScorer,
                  data_loader: MVTecDataLoader,
                  pseudo_loader: PseudoImageLoader,
                  object_class: str,
                  object_variation_type: str) -> float:
    """
    Scenario 3: Real normal vs Object pseudo-anomaly
    
    Evaluates CLIP's ability to detect synthesized object anomalies.
    Uses pre-generated pseudo images.
    """
    # Load pre-generated normal images (from pseudo_loader for consistency)
    normal_images = pseudo_loader.get_normal_images(object_class)
    
    # Load pre-generated pseudo-anomaly images
    pseudo_anomaly_images = pseudo_loader.get_pseudo_images(object_class, object_variation_type)
    
    # Score real normal images
    normal_scores = []
    for _, img in normal_images:
        score, _, _ = scorer.compute_score(img, object_class)
        normal_scores.append(score)
    
    # Score pseudo-anomaly images
    pseudo_anomaly_scores = []
    for _, img in pseudo_anomaly_images:
        score, _, _ = scorer.compute_score(img, object_class)
        pseudo_anomaly_scores.append(score)
    
    return compute_auroc(normal_scores, pseudo_anomaly_scores) * 100


def run_full_experiment(model_name: str = DEFAULT_MODEL,
                        pretrained: str = DEFAULT_PRETRAINED,
                        output_dir: Path = None) -> dict:
    """
    Run full Experiment 1 across all object classes and variation types.
    """
    if output_dir is None:
        output_dir = get_output_dir("experiment1_main")
    
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    scorer = CLIPScorer(model_name, pretrained)
    data_loader = MVTecDataLoader()
    pseudo_loader = PseudoImageLoader()
    
    # Verify pseudo images exist
    print("Verifying pre-generated pseudo images...")
    for obj in OBJECT_CLASSES:
        if not pseudo_loader.has_variation(obj, "colorshift"):
            raise RuntimeError(
                f"Pseudo images not found for {obj}. "
                "Run `python scripts/public/generate_pseudo_images.py` first."
            )
    print("  Pseudo images verified.\n")
    
    results = {
        'model': f"{model_name}/{pretrained}",
        'n_samples': N_SAMPLES,
        'scenario1': {},  # Real vs Real
        'scenario2': {},  # Per-category averages
        'scenario3': {},  # Object pseudo-anomaly
        'detailed': {}    # Per-variation results
    }
    
    # Contextual variation types (for Scenario 2)
    contextual_categories = ['Lighting', 'Medium', 'Foreign', 'Camera']
    
    # Object variation types (for Scenario 3)  
    object_variations = VARIATION_TAXONOMY['Object']
    
    for i, object_class in enumerate(tqdm(OBJECT_CLASSES, desc="[Exp1] Zero-Shot")):
        print(f"\n{'='*50}")
        print(f"[Exp1] Processing {i+1}/{len(OBJECT_CLASSES)}: {object_class}")
        print(f"{'='*50}")
        
        results['detailed'][object_class] = {}
        
        # Scenario 1: Real vs Real
        s1_auroc = run_scenario1(scorer, data_loader, object_class)
        results['scenario1'][object_class] = s1_auroc
        print(f"  Scenario 1 (Real vs Real): {s1_auroc:.1f}%")
        
        # Scenario 2: Per-category
        for category in contextual_categories:
            variation_types = VARIATION_TAXONOMY[category]
            category_aurocs = []
            
            for var_type in variation_types:
                try:
                    auroc = run_scenario2(scorer, data_loader, pseudo_loader,
                                         object_class, var_type)
                    category_aurocs.append(auroc)
                    results['detailed'][object_class][var_type] = auroc
                except Exception as e:
                    print(f"    Warning: {var_type} failed: {e}")
            
            if category_aurocs:
                avg_auroc = np.mean(category_aurocs)
                if category not in results['scenario2']:
                    results['scenario2'][category] = {}
                results['scenario2'][category][object_class] = avg_auroc
                print(f"  Scenario 2 ({category}): {avg_auroc:.1f}%")
        
        # Scenario 3: Object pseudo-anomaly
        object_aurocs = []
        for var_type in object_variations:
            try:
                auroc = run_scenario3(scorer, data_loader, pseudo_loader,
                                     object_class, var_type)
                object_aurocs.append(auroc)
                results['detailed'][object_class][f"object_{var_type}"] = auroc
            except Exception as e:
                print(f"    Warning: object_{var_type} failed: {e}")
        
        if object_aurocs:
            avg_auroc = np.mean(object_aurocs)
            results['scenario3'][object_class] = avg_auroc
            print(f"  Scenario 3 (Object): {avg_auroc:.1f}%")
    
    # Compute overall averages
    results['averages'] = {
        'scenario1': np.mean(list(results['scenario1'].values())),
        'scenario2': {cat: np.mean(list(results['scenario2'][cat].values())) 
                     for cat in contextual_categories if cat in results['scenario2']},
        'scenario3': np.mean(list(results['scenario3'].values()))
    }
    
    # Save results
    output_path = output_dir / "experiment1_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    
    # Print summary
    print_summary(results)
    
    return results


def print_summary(results: dict):
    """Print formatted summary of results."""
    print("\n" + "="*70)
    print("EXPERIMENT 1 SUMMARY")
    print("="*70)
    
    print(f"\nModel: {results['model']}")
    print(f"N samples: {results['n_samples']}")
    
    print("\n--- Scenario 1: Real Normal vs Real Anomaly ---")
    print(f"{'Object':<15} {'AUROC (%)':<10}")
    for obj, auroc in results['scenario1'].items():
        print(f"{obj:<15} {auroc:<10.1f}")
    print(f"{'Average':<15} {results['averages']['scenario1']:<10.1f}")
    
    print("\n--- Scenario 2: Pseudo-Normal vs Real Anomaly (Category Avg) ---")
    print(f"{'Object':<15} {'Lighting':<10} {'Medium':<10} {'Foreign':<10} {'Camera':<10}")
    for obj in OBJECT_CLASSES:
        row = f"{obj:<15}"
        for cat in ['Lighting', 'Medium', 'Foreign', 'Camera']:
            if cat in results['scenario2'] and obj in results['scenario2'][cat]:
                row += f" {results['scenario2'][cat][obj]:<10.1f}"
            else:
                row += f" {'N/A':<10}"
        print(row)
    
    # Category averages
    row = f"{'Average':<15}"
    for cat in ['Lighting', 'Medium', 'Foreign', 'Camera']:
        if cat in results['averages']['scenario2']:
            row += f" {results['averages']['scenario2'][cat]:<10.1f}"
        else:
            row += f" {'N/A':<10}"
    print(row)
    
    print("\n--- Scenario 3: Real Normal vs Object Pseudo-Anomaly ---")
    print(f"{'Object':<15} {'AUROC (%)':<10}")
    for obj, auroc in results['scenario3'].items():
        print(f"{obj:<15} {auroc:<10.1f}")
    print(f"{'Average':<15} {results['averages']['scenario3']:<10.1f}")


def main():
    parser = argparse.ArgumentParser(description='Experiment 1: Zero-Shot Detection')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                       help=f'CLIP model name (default: {DEFAULT_MODEL})')
    parser.add_argument('--pretrained', type=str, default=DEFAULT_PRETRAINED,
                       help=f'Pretrained weights (default: {DEFAULT_PRETRAINED})')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    run_full_experiment(args.model, args.pretrained, output_dir)


if __name__ == '__main__':
    main()
