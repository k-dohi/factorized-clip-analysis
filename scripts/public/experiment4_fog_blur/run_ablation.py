"""
Experiment 4: Fog and Defocus Blur Ablation Study

Parameter ablation experiments for:
- Fog density: 0.0 to 1.0
- Defocus Blur radius: 0 to 21

Generates:
- Parameter tables (LaTeX format)
- Frequency distribution plots showing score distributions
"""

import argparse
import json
import sys
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    OBJECT_CLASSES, N_SAMPLES, RANDOM_SEED, TEMPERATURE,
    DEFAULT_MODEL, DEFAULT_PRETRAINED, get_output_dir, LATEX_FIG_DIR,
    get_model_id
)
from utils.clip_scorer import CLIPScorer
from utils.data_loader import MVTecDataLoader
from utils.variations import VariationGenerator
from utils.metrics import compute_auroc
from generate_pseudo_images import PseudoImageLoader


def run_fog_ablation(scorer: CLIPScorer,
                     data_loader: MVTecDataLoader,
                     pseudo_loader: PseudoImageLoader,
                     variation_gen: VariationGenerator,
                     output_dir: Path) -> dict:
    """
    Run Fog density ablation experiment.
    
    Fog densities: 0.0 (baseline) to 1.0
    Uses pre-sampled normal images from PseudoImageLoader for consistency.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    densities = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    results_by_density = {d: {'auroc': [], 'sim_normal': [], 'sim_anomaly': []} 
                          for d in densities}
    object_results = {}
    
    for i, object_class in enumerate(tqdm(OBJECT_CLASSES, desc="[Exp4] Fog Ablation")):
        print(f"\n  [{i+1}/{len(OBJECT_CLASSES)}] {object_class}")
        
        # Load pre-sampled normal images from PseudoImageLoader
        normal_pil = pseudo_loader.get_normal_images(object_class)
        normal_images = [(idx, np.array(img)) for idx, img in normal_pil]
        anomaly_data = data_loader.get_test_anomaly_images(object_class)
        
        # Score anomaly images once (no variation)
        anomaly_scores = []
        for _, img, _ in anomaly_data:
            pil_img = Image.fromarray(img)
            score, _, _ = scorer.compute_score(pil_img, object_class)
            anomaly_scores.append(score)
        
        baseline_auroc = None
        best_auroc = 0
        best_density = None
        
        for density in densities:
            pseudo_scores = []
            sim_normals = []
            sim_anomalies = []
            
            for _, img in normal_images:
                fogged = variation_gen.apply_fog(img, density=density)
                pil_img = Image.fromarray(fogged)
                score, sim_n, sim_a = scorer.compute_score(pil_img, object_class)
                pseudo_scores.append(score)
                sim_normals.append(sim_n)
                sim_anomalies.append(sim_a)
            
            auroc = compute_auroc(pseudo_scores, anomaly_scores) * 100
            
            results_by_density[density]['auroc'].append(auroc)
            results_by_density[density]['sim_normal'].append(np.mean(sim_normals))
            results_by_density[density]['sim_anomaly'].append(np.mean(sim_anomalies))
            
            if density == 0.0:
                baseline_auroc = auroc
            if auroc > best_auroc:
                best_auroc = auroc
                best_density = density
        
        object_results[object_class] = {
            'baseline': baseline_auroc,
            'best_density': best_density,
            'best_auroc': best_auroc,
            'delta': best_auroc - baseline_auroc
        }
    
    # Compute averages
    results = []
    for density in densities:
        results.append({
            'density': density,
            'auroc': np.mean(results_by_density[density]['auroc']),
            'sim_normal': np.mean(results_by_density[density]['sim_normal']),
            'sim_anomaly': np.mean(results_by_density[density]['sim_anomaly']),
        })
    
    # Save results
    output = {
        'parameter': 'fog_density',
        'results': results,
        'object_results': object_results
    }
    
    output_path = output_dir / "fog_ablation_results.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    # Generate LaTeX table
    generate_latex_table(results, 'Density', output_dir / 'fog_parameter_table.tex')
    
    print(f"\nFog ablation results saved to: {output_path}")
    
    return output


def run_defocusblur_ablation(scorer: CLIPScorer,
                              data_loader: MVTecDataLoader,
                              pseudo_loader: PseudoImageLoader,
                              variation_gen: VariationGenerator,
                              output_dir: Path) -> dict:
    """
    Run Defocus Blur radius ablation experiment.
    
    Radii: 0 (baseline) to 21
    Uses pre-sampled normal images from PseudoImageLoader for consistency.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    radii = [0, 1, 3, 5, 7, 9, 11, 15, 21]
    
    results_by_radius = {r: {'auroc': [], 'sim_normal': [], 'sim_anomaly': []} 
                         for r in radii}
    object_results = {}
    
    for i, object_class in enumerate(tqdm(OBJECT_CLASSES, desc="[Exp4] Defocus Blur Ablation")):
        print(f"\n  [{i+1}/{len(OBJECT_CLASSES)}] {object_class}")
        
        # Load pre-sampled normal images from PseudoImageLoader
        normal_pil = pseudo_loader.get_normal_images(object_class)
        normal_images = [(idx, np.array(img)) for idx, img in normal_pil]
        anomaly_data = data_loader.get_test_anomaly_images(object_class)
        
        # Score anomaly images once
        anomaly_scores = []
        for _, img, _ in anomaly_data:
            pil_img = Image.fromarray(img)
            score, _, _ = scorer.compute_score(pil_img, object_class)
            anomaly_scores.append(score)
        
        baseline_auroc = None
        best_auroc = 0
        best_radius = None
        
        for radius in radii:
            pseudo_scores = []
            sim_normals = []
            sim_anomalies = []
            
            for _, img in normal_images:
                blurred = variation_gen.apply_defocusblur(img, radius=radius)
                pil_img = Image.fromarray(blurred)
                score, sim_n, sim_a = scorer.compute_score(pil_img, object_class)
                pseudo_scores.append(score)
                sim_normals.append(sim_n)
                sim_anomalies.append(sim_a)
            
            auroc = compute_auroc(pseudo_scores, anomaly_scores) * 100
            
            results_by_radius[radius]['auroc'].append(auroc)
            results_by_radius[radius]['sim_normal'].append(np.mean(sim_normals))
            results_by_radius[radius]['sim_anomaly'].append(np.mean(sim_anomalies))
            
            if radius == 0:
                baseline_auroc = auroc
            if auroc > best_auroc:
                best_auroc = auroc
                best_radius = radius
        
        object_results[object_class] = {
            'baseline': baseline_auroc,
            'best_radius': best_radius,
            'best_auroc': best_auroc,
            'delta': best_auroc - baseline_auroc
        }
    
    # Compute averages
    results = []
    for radius in radii:
        results.append({
            'radius': radius,
            'auroc': np.mean(results_by_radius[radius]['auroc']),
            'sim_normal': np.mean(results_by_radius[radius]['sim_normal']),
            'sim_anomaly': np.mean(results_by_radius[radius]['sim_anomaly']),
        })
    
    # Save results
    output = {
        'parameter': 'defocusblur_radius',
        'results': results,
        'object_results': object_results
    }
    
    output_path = output_dir / "defocusblur_ablation_results.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    # Generate LaTeX table
    generate_latex_table(results, 'Radius', output_dir / 'defocusblur_parameter_table.tex')
    
    print(f"\nDefocus blur ablation results saved to: {output_path}")
    
    return output


def generate_latex_table(results: list, param_name: str, output_path: Path):
    """Generate LaTeX table from results."""
    param_key = param_name.lower()
    
    with open(output_path, 'w') as f:
        f.write(f"% {param_name} ablation table\n")
        f.write(f"% Protocol: test normal (N={N_SAMPLES}) + variation vs test anomaly\n")
        f.write(f"% Score: softmax with T={TEMPERATURE}\n\n")
        f.write("\\begin{tabular}{c|c|cc}\n")
        f.write("\\hline\n")
        f.write(f"{param_name} & AUROC (\\%) & Sim to Normal & Sim to Anomaly \\\\\n")
        f.write("\\hline\n")
        
        for r in results:
            param_val = r.get(param_key, r.get('density', r.get('radius', '')))
            f.write(f"{param_val} & {r['auroc']:.1f} & {r['sim_normal']:.4f} & {r['sim_anomaly']:.4f} \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
    
    print(f"LaTeX table saved to: {output_path}")


def generate_distribution_plot(scorer: CLIPScorer,
                               data_loader: MVTecDataLoader,
                               pseudo_loader: PseudoImageLoader,
                               object_class: str,
                               variation_type: str,  # 'fog' or 'defocusblur'
                               output_path: Path,
                               expected_baseline_auroc: float = None,
                               expected_variation_auroc: float = None):
    """
    Generate frequency distribution plot for a specific object class and variation.
    
    Uses the same pre-generated pseudo images as experiment1.
    """
    
    # Load pre-sampled normal images from PseudoImageLoader (same as experiment1)
    normal_pil = pseudo_loader.get_normal_images(object_class)
    anomaly_data = data_loader.get_test_anomaly_images(object_class)
    
    # Load pre-generated pseudo images with variation (same as experiment1)
    pseudo_images = pseudo_loader.get_pseudo_images(object_class, variation_type)
    
    # Baseline scores (no variation) - test normal without any modification
    baseline_scores = []
    for _, img in normal_pil:
        score, _, _ = scorer.compute_score(img, object_class)
        baseline_scores.append(score)
    
    # With variation scores - use pre-generated pseudo images (same as experiment1)
    variation_scores = []
    for _, img in pseudo_images:
        score, _, _ = scorer.compute_score(img, object_class)
        variation_scores.append(score)
    
    # Anomaly scores (no variation)
    anomaly_scores = []
    for _, img, _ in anomaly_data:
        pil_img = Image.fromarray(img)
        score, _, _ = scorer.compute_score(pil_img, object_class)
        anomaly_scores.append(score)
    
    # Compute AUROC from actual scores
    baseline_auroc = compute_auroc(baseline_scores, anomaly_scores) * 100
    variation_auroc = compute_auroc(variation_scores, anomaly_scores) * 100

    # Warn if JSON-provided values disagree with computed values
    if expected_baseline_auroc is not None and abs(expected_baseline_auroc - baseline_auroc) > 0.5:
        print(f"  [Warn] Baseline AUROC mismatch for {object_class}/{variation_type}: "
              f"expected {expected_baseline_auroc:.2f}, computed {baseline_auroc:.2f}")
    if expected_variation_auroc is not None and abs(expected_variation_auroc - variation_auroc) > 0.5:
        print(f"  [Warn] Variation AUROC mismatch for {object_class}/{variation_type}: "
              f"expected {expected_variation_auroc:.2f}, computed {variation_auroc:.2f}")

    # Create frequency distribution plot
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Define colors and labels
    var_name = 'Fog' if variation_type == 'fog' else 'Defocus Blur'
    colors = {
        'baseline': '#2ecc71',      # Green - normal baseline
        'variation': '#3498db',     # Blue - normal with variation
        'anomaly': '#e74c3c'        # Red - anomaly
    }
    
    # Plot histograms
    bins = np.linspace(0.25, 0.75, 51)  # 50 bins from 0.25 to 0.75
    
    # Baseline (no variation) - test normal
    ax.hist(baseline_scores, bins=bins, alpha=0.5, color=colors['baseline'],
            density=True, label='Test Normal (no variation)')
    
    # With variation - test normal with fog/blur
    ax.hist(variation_scores, bins=bins, alpha=0.5, color=colors['variation'],
            density=True, label=f'Test Normal (with {var_name})')
    
    # Anomaly (no variation)
    ax.hist(anomaly_scores, bins=bins, alpha=0.5, color=colors['anomaly'],
            density=True, label='Test Anomaly (no variation)')
    
    # Add vertical lines for means
    ax.axvline(np.mean(baseline_scores), color=colors['baseline'], 
               linestyle='--', linewidth=1.5, alpha=0.8)
    ax.axvline(np.mean(variation_scores), color=colors['variation'],
               linestyle='--', linewidth=1.5, alpha=0.8)
    ax.axvline(np.mean(anomaly_scores), color=colors['anomaly'],
               linestyle='--', linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel('Anomaly Score', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_xlim(0.25, 0.75)
    
    delta = variation_auroc - baseline_auroc
    sign = "+" if delta >= 0 else ""
    ax.set_title(f'{object_class} — {var_name}\n'
                f'AUROC: Baseline={baseline_auroc:.1f}% → With {var_name}={variation_auroc:.1f}% '
                f'(Δ={sign}{delta:.1f}pt)', fontsize=13)
    
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Distribution plot saved to: {output_path}")
    
    # Save scores to CSV for debugging/analysis
    csv_path = output_path.with_suffix('.csv')
    import csv
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sample_type', 'index', 'anomaly_score'])
        for i, score in enumerate(baseline_scores):
            writer.writerow(['baseline', i, score])
        for i, score in enumerate(variation_scores):
            writer.writerow(['variation', i, score])
        for i, score in enumerate(anomaly_scores):
            writer.writerow(['anomaly', i, score])
    print(f"Scores CSV saved to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='Experiment 4: Fog/Defocus Blur Ablation')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL)
    parser.add_argument('--pretrained', type=str, default=DEFAULT_PRETRAINED)
    parser.add_argument('--variation', type=str, default='both',
                       choices=['fog', 'defocusblur', 'both'])
    parser.add_argument('--distribution-only', action='store_true',
                       help='Only generate distribution plots from experiment1 results (skip ablation)')
    parser.add_argument('--exp1-results', type=str, default=None,
                       help='Path to experiment1_results.json (overrides default public_experiments path)')
    parser.add_argument('--fig-dir', type=str, default=None,
                       help='Override figure output directory (default: latex/figures/fog_defocusblur)')
    args = parser.parse_args()
    
    output_dir = get_output_dir("experiment4_fog_blur")
    fig_dir = Path(args.fig_dir) if args.fig_dir else (LATEX_FIG_DIR / "fog_defocusblur")
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize
    scorer = CLIPScorer(args.model, args.pretrained)
    data_loader = MVTecDataLoader()
    pseudo_loader = PseudoImageLoader()
    variation_gen = VariationGenerator(RANDOM_SEED)
    
    # If --distribution-only, load from experiment1 results and generate plots
    if getattr(args, 'distribution_only', False):
        print("Loading results from experiment1 JSON...")
        
        # Load experiment1 results to get AUROC for each variation
        # Default path: outputs/public_experiments/experiment1_zero_shot/{model_id}/experiment1_results.json
        model_id = get_model_id(args.model, args.pretrained)
        default_exp1_json = get_output_dir("experiment1_zero_shot") / model_id / "experiment1_results.json"
        exp1_json = Path(args.exp1_results) if args.exp1_results else default_exp1_json
        
        if not exp1_json.exists():
            raise RuntimeError(
                f"Experiment1 results not found: {exp1_json}. "
                "Run experiment1 first or pass --exp1-results."
            )
        
        with open(exp1_json, 'r') as f:
            exp1_results = json.load(f)
        print(f"  Loaded: {exp1_json}")
        
        for var_type in ['fog', 'defocusblur']:
            if args.variation not in [var_type, 'both']:
                continue
            
            # Collect AUROC delta for each object class
            class_deltas = []
            for object_class in OBJECT_CLASSES:
                try:
                    # Get baseline AUROC (scenario1)
                    baseline_auroc = exp1_results['scenario1'][object_class]
                    # Get variation AUROC (detailed -> object_class -> variation)
                    variation_auroc = exp1_results['detailed'][object_class][var_type]
                except KeyError as e:
                    raise KeyError(
                        f"Missing key in experiment1_results.json for {object_class}/{var_type}: {e}"
                    ) from e
                delta = variation_auroc - baseline_auroc
                class_deltas.append((object_class, baseline_auroc, variation_auroc, delta))
            
            # Sort by delta
            sorted_by_delta = sorted(class_deltas, key=lambda x: x[3], reverse=True)
            
            # Best class (most improvement)
            best_obj, best_baseline, best_variation, best_delta = sorted_by_delta[0]
            print(f"\n  {var_type} best: {best_obj} (Δ={best_delta:+.1f}pt)")
            generate_distribution_plot(
                scorer, data_loader, pseudo_loader,
                best_obj, var_type,
                fig_dir / f"{var_type}_{best_obj}_best_distribution.png",
                expected_baseline_auroc=best_baseline,
                expected_variation_auroc=best_variation
            )
            
            # Worst class (most degradation)
            worst_obj, worst_baseline, worst_variation, worst_delta = sorted_by_delta[-1]
            print(f"  {var_type} worst: {worst_obj} (Δ={worst_delta:+.1f}pt)")
            generate_distribution_plot(
                scorer, data_loader, pseudo_loader,
                worst_obj, var_type,
                fig_dir / f"{var_type}_{worst_obj}_worst_distribution.png",
                expected_baseline_auroc=worst_baseline,
                expected_variation_auroc=worst_variation
            )
        
        print(f"\nDistribution plots saved to: {fig_dir}")
        return
    
    # Verify pseudo images exist
    print("Verifying pre-generated pseudo images...")
    for obj in OBJECT_CLASSES:
        if not pseudo_loader.has_variation(obj, "fog"):
            raise RuntimeError(
                f"Pseudo images not found for {obj}. "
                "Run `python scripts/public/generate_pseudo_images.py` first."
            )
    print("  Pseudo images verified.\n")
    
    results = {}
    
    # Run ablations
    if args.variation in ['fog', 'both']:
        results['fog'] = run_fog_ablation(scorer, data_loader, pseudo_loader, 
                                          variation_gen, output_dir)
    
    if args.variation in ['defocusblur', 'both']:
        results['defocusblur'] = run_defocusblur_ablation(scorer, data_loader, 
                                                          pseudo_loader, variation_gen, output_dir)


if __name__ == '__main__':
    main()
