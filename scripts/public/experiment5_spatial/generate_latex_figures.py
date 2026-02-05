#!/usr/bin/env python3
"""
Generate LaTeX figures from experiment5 heatmap data.

Reads heatmap data from final_summary.json and generates:
- {class}_original.png: Original test normal image
- {class}_heatmap.png: Heatmap overlay with colorbar

Uses the same test normal images as the experiment (via PseudoImageLoader).
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OBJECT_CLASSES, LATEX_FIG_DIR, get_output_dir
from generate_pseudo_images import PseudoImageLoader


def load_heatmap_data(summary_path: Path) -> dict:
    """Load experiment5 heatmap data from final_summary.json."""
    with open(summary_path, 'r') as f:
        data = json.load(f)
    
    if 'experiment5' not in data:
        raise ValueError("experiment5 not found in final_summary.json")
    
    return data['experiment5']


def generate_original_image(image: np.ndarray, output_path: Path):
    """Save original image only."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(image)
    ax.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def generate_heatmap_image(image: np.ndarray, 
                           heatmap: np.ndarray,
                           output_path: Path,
                           fixed_vmax: float = None):
    """
    Save heatmap overlay with colorbar.
    
    Args:
        image: Original image (RGB)
        heatmap: 16x16 heatmap of Δs values
        output_path: Output file path
        fixed_vmax: Fixed vmax for colorbar (None for auto)
    """
    h, w = image.shape[:2]
    
    # Compute vmax
    if fixed_vmax is not None:
        vmax = fixed_vmax
    else:
        vmax = max(abs(heatmap.min()), abs(heatmap.max()), 0.01)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    
    # Show original image
    ax.imshow(image)
    
    # Overlay heatmap (upscale to image size)
    hm = ax.imshow(heatmap, cmap='RdBu_r', alpha=0.6,
                   vmin=-vmax, vmax=vmax,
                   extent=[0, w, h, 0],
                   interpolation='nearest')
    ax.axis('off')
    
    # Add colorbar
    cbar = plt.colorbar(hm, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label('$\\Delta s$', fontsize=20)
    cbar.ax.tick_params(labelsize=18)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Generate LaTeX figures from experiment5 heatmap data'
    )
    parser.add_argument('--summary', type=str, default=None,
                        help='Path to final_summary.json')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for figures')
    parser.add_argument('--fixed_vmax', type=float, default=None,
                        help='Fixed vmax for colorbar')
    parser.add_argument('--classes', nargs='+', default=None,
                        help='Specific classes to generate (default: all)')
    args = parser.parse_args()
    
    # Set paths
    if args.summary is None:
        summary_path = get_output_dir("") / "final_summary.json"
    else:
        summary_path = Path(args.summary)
    
    if args.output_dir is None:
        output_dir = LATEX_FIG_DIR / "sliding_window"
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading heatmap data from: {summary_path}")
    exp5_data = load_heatmap_data(summary_path)
    
    # Load pseudo loader for test normal images
    print("Loading PseudoImageLoader for test normal images...")
    pseudo_loader = PseudoImageLoader()
    
    # Determine classes to process
    classes = args.classes if args.classes else OBJECT_CLASSES
    
    print(f"\nGenerating figures for {len(classes)} classes...")
    print(f"Output directory: {output_dir}\n")
    
    for object_class in classes:
        if object_class not in exp5_data:
            print(f"  [{object_class}] Skipped - no heatmap data")
            continue
        
        # Get heatmap
        heatmap = np.array(exp5_data[object_class]['avg_heatmap'])
        
        # Get original image (first test normal image)
        normal_images = pseudo_loader.get_normal_images(object_class)
        if len(normal_images) == 0:
            print(f"  [{object_class}] Skipped - no normal images")
            continue
        
        # Use first image (same as experiment) - returns (index, PIL.Image) tuple
        _, original_pil = normal_images[0]
        original_np = np.array(original_pil)
        
        # Generate original image
        orig_path = output_dir / f"{object_class}_original.png"
        generate_original_image(original_np, orig_path)
        
        # Generate heatmap image
        hm_path = output_dir / f"{object_class}_heatmap.png"
        generate_heatmap_image(original_np, heatmap, hm_path, args.fixed_vmax)
        
        print(f"  [{object_class}] Generated: {orig_path.name}, {hm_path.name}")
    
    print(f"\nDone! Figures saved to: {output_dir}")


if __name__ == '__main__':
    main()
