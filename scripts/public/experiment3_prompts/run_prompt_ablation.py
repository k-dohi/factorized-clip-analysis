"""
Experiment 3: Prompt Ablation Study (Table V)

Evaluates the effect of different anomaly prompt formulations on detection performance
across ALL THREE SCENARIOS as reported in the paper.

Protocol (matches paper Table V):
- Scenario 1 (Real): Real normal vs Real anomaly
- Scenario 2 (Lighting, Medium, Foreign, Camera): Pseudo-normal (with variation) vs Real anomaly
- Scenario 3 (Object): Real normal vs Object pseudo-anomaly

Anomaly Prompt Variations:
1. "a photo of a damaged {obj}" (baseline)
2. "a photo of a {obj} with damage on its surface"
3. "a photo of a {obj} with a defect"
4. "a photo of a flawed {obj}"
5. "a photo of a defective {obj}"
6. "a photo of an anomalous {obj}"
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
    DEFAULT_MODEL, DEFAULT_PRETRAINED, get_output_dir,
    ANOMALY_PROMPTS, NORMAL_PROMPT
)
from utils.clip_scorer import CLIPScorer
from utils.data_loader import MVTecDataLoader
from utils.metrics import compute_auroc
from generate_pseudo_images import PseudoImageLoader


# Prompt templates for ablation (anomaly prompt variations)
# Normal prompt is fixed: "a photo of a {obj}"
# Order matches Table V in paper
PROMPT_TEMPLATES = {
    'damaged': {
        'normal': NORMAL_PROMPT,
        'anomaly': ANOMALY_PROMPTS['damaged']
    },
    'surface_damage': {
        'normal': NORMAL_PROMPT,
        'anomaly': ANOMALY_PROMPTS['surface_damage']
    },
    'with_defect': {
        'normal': NORMAL_PROMPT,
        'anomaly': ANOMALY_PROMPTS['with_defect']
    },
    'flawed': {
        'normal': NORMAL_PROMPT,
        'anomaly': ANOMALY_PROMPTS['flawed']
    },
    'defective': {
        'normal': NORMAL_PROMPT,
        'anomaly': ANOMALY_PROMPTS['defective']
    },
    'anomalous': {
        'normal': NORMAL_PROMPT,
        'anomaly': ANOMALY_PROMPTS['anomalous']
    },
}


def run_prompt_ablation(scorer: CLIPScorer,
                        data_loader: MVTecDataLoader,
                        pseudo_loader: PseudoImageLoader,
                        output_dir: Path) -> dict:
    """
    Run prompt template ablation experiment across all 3 scenarios.
    
    Matches Table V in paper:
    - Scenario 1 (Real): Real normal vs Real anomaly
    - Scenario 2: 4 contextual categories (Lighting, Medium, Foreign, Camera)
    - Scenario 3 (Object): Real normal vs Object pseudo-anomaly
    
    Uses pre-generated pseudo images from PseudoImageLoader.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Contextual variation categories (for Scenario 2)
    contextual_categories = {
        'Lighting': VARIATION_TAXONOMY['Lighting'],
        'Medium': VARIATION_TAXONOMY['Medium'],
        'Foreign': VARIATION_TAXONOMY['Foreign'],
        'Camera': VARIATION_TAXONOMY['Camera']
    }
    
    # Object variation types (for Scenario 3)
    object_variations = VARIATION_TAXONOMY['Object']
    
    # Results structure matching Table V
    results = {}
    for template_name in PROMPT_TEMPLATES:
        results[template_name] = {
            'scenario1': {},  # Real vs Real (per object)
            'scenario2': {    # Category averages (per object)
                'Lighting': {},
                'Medium': {},
                'Foreign': {},
                'Camera': {}
            },
            'scenario3': {}   # Object pseudo-anomaly (per object)
        }
    
    for i, object_class in enumerate(tqdm(OBJECT_CLASSES, desc="[Exp3] Prompt Ablation")):
        print(f"\n  [{i+1}/{len(OBJECT_CLASSES)}] {object_class}")
        
        # Load pre-generated normal images
        normal_pil_list = pseudo_loader.get_normal_images(object_class)
        normal_pil = [img for _, img in normal_pil_list]
        
        # Get anomaly data from MVTec
        anomaly_data = data_loader.get_test_anomaly_images(object_class)
        anomaly_pil = [Image.fromarray(img) for _, img, _ in anomaly_data]
        
        # ======================================================================
        # Load pre-generated pseudo images from PseudoImageLoader
        # This ensures the SAME pseudo images are used across all prompt templates
        # ======================================================================
        
        # Load Scenario 2 pseudo-normal images (contextual variations)
        # Structure: {category: {var_type: [PIL images]}}
        print(f"    Loading pre-generated pseudo-normal images...")
        scenario2_pseudo_images = {}
        for category, variation_types in contextual_categories.items():
            scenario2_pseudo_images[category] = {}
            for var_type in variation_types:
                try:
                    pseudo_list = pseudo_loader.get_pseudo_images(object_class, var_type)
                    pseudo_images = [img for _, img in pseudo_list]
                    scenario2_pseudo_images[category][var_type] = pseudo_images
                except FileNotFoundError as e:
                    print(f"    Warning: {var_type} not found: {e}")
        
        # Load Scenario 3 pseudo-anomaly images (object variations)
        # Structure: {var_type: [PIL images]}
        print(f"    Loading pre-generated pseudo-anomaly images...")
        scenario3_pseudo_images = {}
        for var_type in object_variations:
            try:
                pseudo_list = pseudo_loader.get_pseudo_images(object_class, var_type)
                pseudo_images = [img for _, img in pseudo_list]
                scenario3_pseudo_images[var_type] = pseudo_images
            except FileNotFoundError as e:
                print(f"    Warning: object_{var_type} not found: {e}")
        
        # ======================================================================
        # Now test each prompt template using the SAME pre-generated images
        # ======================================================================
        for template_name, templates in PROMPT_TEMPLATES.items():
            normal_template = templates['normal']
            anomaly_template = templates['anomaly']
            
            # === Scenario 1: Real normal vs Real anomaly ===
            normal_scores = []
            for pil_img in normal_pil:
                score, _, _ = scorer.compute_score(
                    pil_img, object_class, normal_template, anomaly_template
                )
                normal_scores.append(score)
            
            anomaly_scores = []
            for pil_img in anomaly_pil:
                score, _, _ = scorer.compute_score(
                    pil_img, object_class, normal_template, anomaly_template
                )
                anomaly_scores.append(score)
            
            auroc_s1 = compute_auroc(normal_scores, anomaly_scores) * 100
            results[template_name]['scenario1'][object_class] = auroc_s1
            
            # === Scenario 2: Pseudo-normal (with variation) vs Real anomaly ===
            # Uses pre-generated pseudo images (same across all prompts)
            for category, variation_types in contextual_categories.items():
                category_aurocs = []
                
                for var_type in variation_types:
                    if var_type not in scenario2_pseudo_images.get(category, {}):
                        continue
                    
                    pseudo_images = scenario2_pseudo_images[category][var_type]
                    pseudo_scores = []
                    for pil_img in pseudo_images:
                        score, _, _ = scorer.compute_score(
                            pil_img, object_class, normal_template, anomaly_template
                        )
                        pseudo_scores.append(score)
                    
                    auroc = compute_auroc(pseudo_scores, anomaly_scores) * 100
                    category_aurocs.append(auroc)
                
                if category_aurocs:
                    avg_auroc = np.mean(category_aurocs)
                    results[template_name]['scenario2'][category][object_class] = avg_auroc
            
            # === Scenario 3: Real normal vs Object pseudo-anomaly ===
            # Uses pre-generated pseudo images (same across all prompts)
            object_aurocs = []
            for var_type in object_variations:
                if var_type not in scenario3_pseudo_images:
                    continue
                
                pseudo_images = scenario3_pseudo_images[var_type]
                pseudo_anomaly_scores = []
                for pil_img in pseudo_images:
                    score, _, _ = scorer.compute_score(
                        pil_img, object_class, normal_template, anomaly_template
                    )
                    pseudo_anomaly_scores.append(score)
                
                auroc = compute_auroc(normal_scores, pseudo_anomaly_scores) * 100
                object_aurocs.append(auroc)
            
            if object_aurocs:
                avg_auroc = np.mean(object_aurocs)
                results[template_name]['scenario3'][object_class] = avg_auroc
    
    # Compute averages for each template
    for template_name in PROMPT_TEMPLATES:
        # Scenario 1 average
        s1_vals = list(results[template_name]['scenario1'].values())
        results[template_name]['scenario1']['average'] = np.mean(s1_vals)
        
        # Scenario 2 category averages
        for category in contextual_categories:
            cat_vals = list(results[template_name]['scenario2'][category].values())
            if cat_vals:
                results[template_name]['scenario2'][category]['average'] = np.mean(cat_vals)
        
        # Scenario 3 average
        s3_vals = list(results[template_name]['scenario3'].values())
        if s3_vals:
            results[template_name]['scenario3']['average'] = np.mean(s3_vals)
    
    # Save results
    output_path = output_dir / "prompt_ablation_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate LaTeX table matching Table V format
    generate_latex_table(results, output_dir / "prompt_ablation_table.tex")
    
    print(f"\nResults saved to: {output_path}")
    print_summary(results)
    
    return results


def generate_latex_table(results: dict, output_path: Path):
    """Generate LaTeX table matching paper Table V format."""
    templates = list(PROMPT_TEMPLATES.keys())
    
    # Template display names matching paper
    template_labels = {
        'damaged': r'"a photo of a damaged \{class\}"',
        'surface_damage': r'"a photo of a \{class\} with damage on its surface"',
        'with_defect': r'"a photo of a \{class\} with a defect"',
        'flawed': r'"a photo of a flawed \{class\}"',
        'defective': r'"a photo of a defective \{class\}"',
        'anomalous': r'"a photo of an anomalous \{class\}"',
    }
    
    with open(output_path, 'w') as f:
        f.write("% Prompt ablation results (Table V format)\n")
        f.write("\\begin{tabular}{|l|c|cccc|c|}\n")
        f.write("\\hline\n")
        f.write(" & \\textbf{Scenario 1} & \\multicolumn{4}{c|}{\\textbf{Scenario 2}} & \\textbf{Scenario 3} \\\\\n")
        f.write("\\textbf{Prompt Template} & Real & Lighting & Medium & Foreign & Camera & Object \\\\\n")
        f.write("\\hline\n")
        
        for template in templates:
            label = template_labels.get(template, template)
            s1 = results[template]['scenario1'].get('average', float('nan'))
            lighting = results[template]['scenario2']['Lighting'].get('average', float('nan'))
            medium = results[template]['scenario2']['Medium'].get('average', float('nan'))
            foreign = results[template]['scenario2']['Foreign'].get('average', float('nan'))
            camera = results[template]['scenario2']['Camera'].get('average', float('nan'))
            s3 = results[template]['scenario3'].get('average', float('nan'))
            
            f.write(f"{label} & {s1:.1f} & {lighting:.1f} & {medium:.1f} & {foreign:.1f} & {camera:.1f} & {s3:.1f} \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
    
    print(f"LaTeX table saved to: {output_path}")


def print_summary(results: dict):
    """Print formatted summary matching Table V format."""
    templates = list(PROMPT_TEMPLATES.keys())
    
    print("\n" + "="*100)
    print("PROMPT ABLATION SUMMARY (Table V Format)")
    print("="*100)
    
    # Header
    print(f"{'Prompt Template':<20} {'Scenario1':<10} {'Lighting':<10} {'Medium':<10} {'Foreign':<10} {'Camera':<10} {'Object':<10}")
    print("-" * 100)
    
    for template in templates:
        s1 = results[template]['scenario1'].get('average', float('nan'))
        lighting = results[template]['scenario2']['Lighting'].get('average', float('nan'))
        medium = results[template]['scenario2']['Medium'].get('average', float('nan'))
        foreign = results[template]['scenario2']['Foreign'].get('average', float('nan'))
        camera = results[template]['scenario2']['Camera'].get('average', float('nan'))
        s3 = results[template]['scenario3'].get('average', float('nan'))
        
        print(f"{template:<20} {s1:<10.1f} {lighting:<10.1f} {medium:<10.1f} {foreign:<10.1f} {camera:<10.1f} {s3:<10.1f}")
    
    print("="*100)
    
    # Find best template for Scenario 1
    best = max(templates, key=lambda t: results[t]['scenario1'].get('average', 0))
    print(f"\nBest template for Scenario 1: {best} ({results[best]['scenario1']['average']:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Experiment 3: Prompt Ablation (Table V)')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL)
    parser.add_argument('--pretrained', type=str, default=DEFAULT_PRETRAINED)
    args = parser.parse_args()
    
    output_dir = get_output_dir("experiment3_prompts")
    
    scorer = CLIPScorer(args.model, args.pretrained)
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
    
    run_prompt_ablation(scorer, data_loader, pseudo_loader, output_dir)


if __name__ == '__main__':
    main()
