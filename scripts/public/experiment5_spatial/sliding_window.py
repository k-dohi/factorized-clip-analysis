"""
Experiment 5: Spatial Sensitivity Analysis (Sliding Window Heatmap)

Analyzes where CLIP perceives "damage" when texture patches are placed
at different spatial locations.

Protocol:
- Sample test normal images (with replacement)
- For each image, place texture patches at grid positions
- Use TextureForeignObjectAug for consistent texture generation
- Compute Δs = s_perturbed - s_original at each position
- Generate heatmaps showing spatial sensitivity
"""

import argparse
import json
import sys
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cv2
from tqdm import tqdm
from scipy.fft import fft2, ifft2, fftshift, ifftshift

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    OBJECT_CLASSES, N_SAMPLES, RANDOM_SEED,
    DEFAULT_MODEL, DEFAULT_PRETRAINED, get_output_dir, LATEX_FIG_DIR
)
from utils.clip_scorer import CLIPScorer
from utils.data_loader import MVTecDataLoader
from generate_pseudo_images import PseudoImageLoader


def generate_texture_patch(patch: np.ndarray, seed: int = None) -> np.ndarray:
    """
    Generate texture patch using the same method as TextureForeignObjectAug.
    
    Spectral mixing: mix amplitude of original patch with noise, keep phase.
    
    Args:
        patch: Input patch (H, W, 3)
        seed: Random seed for reproducibility
    
    Returns:
        Texture patch (H, W, 3)
    """
    if seed is not None:
        np.random.seed(seed)
        import random
        random.seed(seed)
    
    import random
    
    ph, pw = patch.shape[:2]
    
    # ---- Noise generation (mid-frequency) ----
    noise = np.random.normal(128, 30, patch.shape).astype(np.float32)
    k = random.choice([7, 11, 15])
    noise = cv2.GaussianBlur(noise, (k, k), 0)
    noise = cv2.normalize(noise, None, 0, 255, cv2.NORM_MINMAX)
    
    # ---- Spectral mixing ----
    mix_alpha = random.uniform(0.4, 0.8)  # 0 = original, 1 = full noise
    out_patch = np.zeros_like(patch, np.float32)
    
    for c in range(3):
        F_o = fftshift(fft2(patch[:, :, c].astype(np.float32)))
        F_n = fftshift(fft2(noise[:, :, c]))
        Amp_o, Ph_o = np.abs(F_o), np.angle(F_o)
        Amp_n = np.abs(F_n)
        
        Amp_mix = (1 - mix_alpha) * Amp_o + mix_alpha * Amp_n
        F_mix = Amp_mix * np.exp(1j * Ph_o)
        out_c = np.real(ifft2(ifftshift(F_mix)))
        out_patch[:, :, c] = out_c
    
    out_patch = np.clip(out_patch, 0, 255).astype(np.uint8)
    return out_patch


def compute_sliding_window_heatmap(scorer: CLIPScorer,
                                   image: np.ndarray,
                                   object_class: str,
                                   grid_size: int = 16,
                                   num_textures: int = 5,
                                   base_seed: int = 42) -> np.ndarray:
    """
    Compute spatial sensitivity heatmap using sliding window.
    
    Uses TextureForeignObjectAug-style spectral mixing for texture generation.
    
    Args:
        scorer: CLIP scorer
        image: Input image (RGB)
        object_class: Object class name
        grid_size: Number of grid cells per dimension
        num_textures: Number of random textures to average over
        base_seed: Base random seed
    
    Returns:
        Heatmap of Δs values (grid_size x grid_size)
    """
    h, w = image.shape[:2]
    L = min(h, w) // grid_size  # Patch size
    
    # Compute base score
    pil_img = Image.fromarray(image)
    base_score, _, _ = scorer.compute_score(pil_img, object_class)
    
    # Compute delta at each grid position
    heatmap = np.zeros((grid_size, grid_size))
    
    for gy in range(grid_size):
        for gx in range(grid_size):
            y1 = gy * L
            x1 = gx * L
            y2 = min(y1 + L, h)
            x2 = min(x1 + L, w)
            
            # Extract original patch
            original_patch = image[y1:y2, x1:x2].copy()
            
            deltas = []
            for t in range(num_textures):
                # Generate texture using spectral mixing (same as TextureForeignObjectAug)
                seed = base_seed + gy * grid_size * num_textures + gx * num_textures + t
                texture_patch = generate_texture_patch(original_patch, seed=seed)
                
                # Create perturbed image
                perturbed = image.copy()
                perturbed[y1:y2, x1:x2] = texture_patch
                
                # Compute score
                pil_perturbed = Image.fromarray(perturbed)
                perturbed_score, _, _ = scorer.compute_score(pil_perturbed, object_class)
                
                deltas.append(perturbed_score - base_score)
            
            heatmap[gy, gx] = np.mean(deltas)
    
    return heatmap


def run_sliding_window_analysis(scorer: CLIPScorer,
                                 pseudo_loader: PseudoImageLoader,
                                 output_dir: Path,
                                 num_images: int = 5,
                                 grid_size: int = 16,
                                 num_textures: int = 5) -> dict:
    """
    Run sliding window analysis for all object classes.
    Uses pre-sampled normal images from PseudoImageLoader for consistency.
    Uses TextureForeignObjectAug-style spectral mixing for texture generation.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for i, object_class in enumerate(tqdm(OBJECT_CLASSES, desc="[Exp5] Spatial Sensitivity")):
        print(f"\n  [{i+1}/{len(OBJECT_CLASSES)}] {object_class}")
        
        # Load pre-sampled normal images from PseudoImageLoader
        normal_pil = pseudo_loader.get_normal_images(object_class)
        # Only use first num_images
        normal_images = [(idx, np.array(img)) for idx, img in normal_pil[:num_images]]
        
        class_heatmaps = []
        
        for img_idx, (idx, img) in enumerate(normal_images):
            # Different base seed for each image
            base_seed = RANDOM_SEED + img_idx * 10000
            heatmap = compute_sliding_window_heatmap(
                scorer, img, object_class, grid_size, num_textures, base_seed
            )
            class_heatmaps.append(heatmap)
            
            # Save individual heatmap
            save_heatmap(heatmap, img, 
                        output_dir / f"{object_class}_image{img_idx:02d}_heatmap.png",
                        object_class)
        
        # Average heatmap for this class
        avg_heatmap = np.mean(class_heatmaps, axis=0)
        results[object_class] = {
            'avg_heatmap': avg_heatmap.tolist(),
            'num_images': len(class_heatmaps)
        }
        
        # Save class average heatmap
        save_heatmap(avg_heatmap, normal_images[0][1],
                    output_dir / f"{object_class}_avg_heatmap.png",
                    object_class)
    
    # Save results
    output_path = output_dir / "sliding_window_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def save_heatmap(heatmap: np.ndarray, 
                 original_image: np.ndarray,
                 output_path: Path,
                 object_class: str):
    """Save heatmap visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title(f'{object_class} - Original', fontsize=12)
    axes[0].axis('off')
    
    # Heatmap
    vmax = max(abs(heatmap.min()), abs(heatmap.max()))
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'diverging', ['blue', 'white', 'red']
    )
    im = axes[1].imshow(heatmap, cmap=cmap, vmin=-vmax, vmax=vmax)
    axes[1].set_title(f'Δs Heatmap (texture patch sensitivity)', fontsize=12)
    axes[1].axis('off')
    
    plt.colorbar(im, ax=axes[1], label='Δs (anomaly score change)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Experiment 3: Spatial Sensitivity')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL)
    parser.add_argument('--pretrained', type=str, default=DEFAULT_PRETRAINED)
    parser.add_argument('--num_images', type=int, default=5,
                       help='Number of images per class')
    parser.add_argument('--grid_size', type=int, default=16,
                       help='Grid size for sliding window')
    args = parser.parse_args()
    
    output_dir = get_output_dir("experiment3_spatial")
    
    scorer = CLIPScorer(args.model, args.pretrained)
    pseudo_loader = PseudoImageLoader()
    
    # Verify pseudo images exist
    print("Verifying pre-generated pseudo images...")
    for obj in OBJECT_CLASSES:
        normal_images = pseudo_loader.get_normal_images(obj)
        if len(normal_images) == 0:
            raise RuntimeError(
                f"Pseudo images not found for {obj}. "
                "Run `python scripts/public/generate_pseudo_images.py` first."
            )
    print("  Pseudo images verified.\n")
    
    run_sliding_window_analysis(scorer, pseudo_loader, output_dir,
                                args.num_images, args.grid_size)


if __name__ == '__main__':
    main()
