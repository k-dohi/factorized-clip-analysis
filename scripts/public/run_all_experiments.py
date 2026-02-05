"""
Run All Paper Experiments - Complete Implementation

This script executes all experiments required for the paper.

Paper Tables and corresponding experiments:
  - Table I-III (Exp 1): Zero-shot detection with ALL 7 CLIP models
  - Table IV (Exp 2): Background Masking Analysis (ViT-H-14 only)
  - Table V (Exp 3): Prompt Ablation Study (ViT-H-14 only)
  - Table VI (Exp 4): Fog/Blur Ablation (ViT-H-14 only)
  - Table VII (Exp 5): Spatial Sensitivity (ViT-H-14 only)
  - Table VIII (Exp 6): Variation-Specific Prompts (ViT-H-14 only)

Usage:
  # Run all experiments (full paper reproduction)
  python run_all_experiments.py
  
  # Run only Experiment 1 with all 7 models
  python run_all_experiments.py --only 1
  
  # Run experiments 2-6 (ViT-H-14 only)
  python run_all_experiments.py --skip 1
"""

import argparse
import json
import sys
from datetime import datetime 
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    CLIP_MODELS, DEFAULT_MODEL, DEFAULT_PRETRAINED,
    OBJECT_CLASSES, RANDOM_SEED, get_output_dir, get_model_id
)


def run_experiment1_all_models(output_base: Path) -> dict:
    """
    Experiment 1: Zero-Shot Detection under Contextual Variations
    
    Paper Tables I-III: Run with ALL 7 CLIP models.
    """
    from experiment1_zero_shot.run_experiment1 import run_full_experiment
    
    all_results = {}
    
    for i, model_config in enumerate(CLIP_MODELS):
        model_name = model_config['name']
        pretrained = model_config['pretrained']
        model_id = get_model_id(model_name, pretrained)
        
        print(f"\n{'='*70}")
        print(f"Model {i+1}/{len(CLIP_MODELS)}: {model_name} ({pretrained})")
        print(f"{'='*70}")
        
        try:
            output_dir = output_base / model_id
            output_dir.mkdir(parents=True, exist_ok=True)
            
            results = run_full_experiment(model_name, pretrained, output_dir)
            all_results[model_id] = results
            
            # Print summary
            if 'averages' in results:
                print(f"  Scenario 1 avg: {results['averages'].get('scenario1', 'N/A'):.1f}%")
                print(f"  Scenario 3 avg: {results['averages'].get('scenario3', 'N/A'):.1f}%")
        
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_results[model_id] = {'error': str(e)}
    
    # Save aggregated results
    summary_path = output_base / "all_models_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nExperiment 1 summary saved to: {summary_path}")
    return all_results


def run_experiments_vith14(skip_experiments: list, output_base: Path) -> dict:
    """
    Experiments 2-6: Run with ViT-H-14 only.
    
    Paper Tables IV-VIII.
    Uses pre-generated pseudo images from PseudoImageLoader.
    """
    from utils.clip_scorer import CLIPScorer
    from utils.data_loader import MVTecDataLoader
    from utils.variations import VariationGenerator
    from generate_pseudo_images import PseudoImageLoader
    
    results = {}
    
    # Initialize components with ViT-H-14
    print(f"\nLoading CLIP: {DEFAULT_MODEL} ({DEFAULT_PRETRAINED})")
    scorer = CLIPScorer(DEFAULT_MODEL, DEFAULT_PRETRAINED)
    data_loader = MVTecDataLoader()
    variation_gen = VariationGenerator(RANDOM_SEED)
    pseudo_loader = PseudoImageLoader()
    
    # Verify pseudo images exist
    print("\nVerifying pre-generated pseudo images...")
    for obj in OBJECT_CLASSES:
        if not pseudo_loader.has_variation(obj, "colorshift"):
            raise RuntimeError(
                f"Pseudo images not found for {obj}. "
                "Run `python scripts/public/generate_pseudo_images.py` first."
            )
    print("  Pseudo images verified.\n")
    
    # Experiment 2: Background Masking Analysis (Table IV)
    if 2 not in skip_experiments:
        print("\n" + "="*70)
        print("EXPERIMENT 2: Background Masking Analysis (Table IV)")
        print("="*70)
        try:
            from experiment2_masking.run_masking import run_masking_experiment, init_segmenter
            segmenter = init_segmenter()
            output_dir = output_base / "experiment2_masking"
            exp2_results = run_masking_experiment(scorer, data_loader, pseudo_loader, segmenter, output_dir)
            results['experiment2'] = exp2_results
            print("Experiment 2 completed successfully")
        except Exception as e:
            print(f"Experiment 2 failed: {e}")
            import traceback
            traceback.print_exc()
            results['experiment2'] = {'error': str(e)}
    
    # Experiment 3: Prompt Ablation Study (Table V)
    if 3 not in skip_experiments:
        print("\n" + "="*70)
        print("EXPERIMENT 3: Prompt Ablation Study (Table V)")
        print("="*70)
        try:
            from experiment3_prompts.run_prompt_ablation import run_prompt_ablation
            output_dir = output_base / "experiment3_prompts"
            exp3_results = run_prompt_ablation(scorer, data_loader, pseudo_loader, output_dir)
            results['experiment3'] = exp3_results
            print("Experiment 3 completed successfully")
        except Exception as e:
            print(f"Experiment 3 failed: {e}")
            import traceback
            traceback.print_exc()
            results['experiment3'] = {'error': str(e)}
    
    # Experiment 4: Fog and Defocus Blur Ablation (Table VI)
    if 4 not in skip_experiments:
        print("\n" + "="*70)
        print("EXPERIMENT 4: Fog and Defocus Blur Ablation (Table VI)")
        print("="*70)
        try:
            from experiment4_fog_blur.run_ablation import run_fog_ablation, run_defocusblur_ablation
            output_dir = output_base / "experiment4_fog_blur"
            fog_results = run_fog_ablation(scorer, data_loader, pseudo_loader, variation_gen, output_dir)
            blur_results = run_defocusblur_ablation(scorer, data_loader, pseudo_loader, variation_gen, output_dir)
            results['experiment4'] = {
                'fog': fog_results,
                'defocus_blur': blur_results
            }
            print("Experiment 4 completed successfully")
        except Exception as e:
            print(f"Experiment 4 failed: {e}")
            import traceback
            traceback.print_exc()
            results['experiment4'] = {'error': str(e)}
    
    # Experiment 5: Spatial Sensitivity Analysis (Table VII)
    if 5 not in skip_experiments:
        print("\n" + "="*70)
        print("EXPERIMENT 5: Spatial Sensitivity Analysis (Table VII)")
        print("="*70)
        try:
            from experiment5_spatial.sliding_window import run_sliding_window_analysis
            output_dir = output_base / "experiment5_spatial"
            exp5_results = run_sliding_window_analysis(scorer, pseudo_loader, output_dir)
            results['experiment5'] = exp5_results
            print("Experiment 5 completed successfully")
        except Exception as e:
            print(f"Experiment 5 failed: {e}")
            import traceback
            traceback.print_exc()
            results['experiment5'] = {'error': str(e)}
    
    # Experiment 6: Variation-Specific Prompt Evaluation (Table VIII)
    if 6 not in skip_experiments:
        print("\n" + "="*70)
        print("EXPERIMENT 6: Variation-Specific Prompt Evaluation (Table VIII)")
        print("="*70)
        try:
            from experiment6_variation_specific.run_variation_specific import run_variation_specific_experiment
            output_dir = output_base / "experiment6_variation_specific"
            exp6_results = run_variation_specific_experiment(scorer, data_loader, pseudo_loader, output_dir)
            results['experiment6'] = exp6_results
            print("Experiment 6 completed successfully")
        except Exception as e:
            print(f"Experiment 6 failed: {e}")
            import traceback
            traceback.print_exc()
            results['experiment6'] = {'error': str(e)}
    
    return results


def run_all_experiments(skip_experiments: list = None):
    """
    Run all paper experiments.
    
    Args:
        skip_experiments: List of experiment numbers to skip (1-6)
    """
    skip = skip_experiments or []
    start_time = datetime.now()
    output_base = get_output_dir("")
    
    all_results = {
        'timestamp': datetime.now().isoformat(),
    }
    
    print("#" * 70)
    print("# PAPER EXPERIMENTS - COMPLETE RUN")
    print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# Experiments to skip: {skip if skip else 'None'}")
    print("#" * 70)
    
    # Experiment 1: All 7 CLIP models
    if 1 not in skip:
        print("\n" + "=" * 70)
        print("EXPERIMENT 1: Zero-Shot Detection (ALL 7 MODELS)")
        print("Paper Tables I-III")
        print("=" * 70)
        
        exp1_output = output_base / "experiment1_zero_shot"
        exp1_output.mkdir(parents=True, exist_ok=True)
        all_results['experiment1'] = run_experiment1_all_models(exp1_output)

    # Experiments 2-6: ViT-H-14 only
    exp2_6_results = run_experiments_vith14(skip, output_base)
    all_results.update(exp2_6_results)
    
    # Summary
    elapsed = (datetime.now() - start_time).total_seconds()
    all_results['elapsed_seconds'] = elapsed
    
    print("\n" + "#" * 70)
    print("# ALL EXPERIMENTS COMPLETED")
    print(f"# Total time: {elapsed:.1f} seconds ({elapsed/3600:.2f} hours)")
    print("#" * 70)
    
    # Save final summary
    summary_path = output_base / "final_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nFinal summary saved to: {summary_path}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description='Run All Paper Experiments (Tables I-VIII)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full paper reproduction (all 7 models for Exp1, ViT-H-14 for Exp2-6)
  python run_all_experiments.py
  
  # Run only Experiment 1 (all 7 models)
  python run_all_experiments.py --only 1
  
  # Skip Experiment 1, run Experiments 2-6 only
  python run_all_experiments.py --skip 1
        """
    )
    parser.add_argument('--skip', type=int, nargs='+', default=[],
                       help='Experiment numbers to skip (1-6)')
    parser.add_argument('--only', type=int, nargs='+', default=[],
                       help='Only run these experiment numbers (1-6)')
    args = parser.parse_args()
    
    if args.only:
        skip = [i for i in range(1, 7) if i not in args.only]
    else:
        skip = args.skip
    
    run_all_experiments(skip_experiments=skip)


if __name__ == '__main__':
    main()
