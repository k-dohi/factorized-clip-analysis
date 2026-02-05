"""
Pseudo Image Pre-Generation Script

Generates and saves all pseudo images ONCE before running experiments.
This avoids redundant generation across experiments 1-6.

Structure:
  outputs/public_experiments/pseudo_images/
    ├── {object_class}/
    │   ├── normal/           # Original test normal images (for reference)
    │   │   ├── 0000.png
    │   │   └── ...
    │   ├── {variation_type}/  # Pseudo images for each variation
    │   │   ├── 0000.png
    │   │   └── ...
    │   └── ...
    └── ...
"""

import argparse
import json
import os
import sys
import cv2
import yaml
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import time

# Add parent directory to path
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

# Add project root to path for segmenter access
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    OBJECT_CLASSES, VARIATION_TAXONOMY, N_SAMPLES, RANDOM_SEED,
    get_output_dir
)
from utils.data_loader import MVTecDataLoader
from utils.variations import VariationGenerator, VARIATION_TO_AUGMENTER

# Import AutoBBoxSAMSegmenter from project source
from src.datasets.segmenters.auto_bbox_sam_segmenter import AutoBBoxSAMSegmenter

# Path to segmentation config
SEGMENTATION_CONFIG = PROJECT_ROOT / "config" / "segmentation" / "mvtec_objects.yaml"


def get_all_variation_types() -> list:
    """Get flat list of all variation types from taxonomy."""
    all_types = []
    for category, types in VARIATION_TAXONOMY.items():
        all_types.extend(types)
    return all_types


# Variation types that are slow (SD-based) - run last
SLOW_VARIATIONS = ['sd_inpaint', 'sd_foreign']

# Variation types that are fast (non-SD) - run first
FAST_VARIATIONS = [v for v in get_all_variation_types() if v not in SLOW_VARIATIONS]


def generate_pseudo_images(
    output_base_dir: Path,
    object_classes: list = None,
    variation_types: list = None,
    force_regenerate: bool = False,
):
    """
    Generate and save all pseudo images.
    
    Args:
        output_base_dir: Base directory for saving pseudo images
        object_classes: List of object classes to process (default: all)
        variation_types: List of variation types to generate (default: all)
        force_regenerate: If True, regenerate even if images exist
    """
    import random
    
    output_base_dir = Path(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Set global random seed for reproducibility
    # This sets the initial state, but each augmentation call will advance the state
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # Initialize
    data_loader = MVTecDataLoader()
    variation_gen = VariationGenerator(RANDOM_SEED)
    
    if object_classes is None:
        object_classes = OBJECT_CLASSES
    
    if variation_types is None:
        variation_types = get_all_variation_types()
    
    # Load segmentation config
    print("Loading segmentation config...")
    with open(SEGMENTATION_CONFIG, 'r') as f:
        seg_configs = yaml.safe_load(f)
    
    # Initialize SAM 3 segmenter (SAM 3 is required for sam_combined mode)
    print("Initializing AutoBBoxSAMSegmenter with SAM 3...")
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    segmenter = AutoBBoxSAMSegmenter(
        model_id="facebook/sam3",
        device=device,
        object_type_configs=seg_configs
    )
    print(f"  ✓ Segmenter initialized on {device}")
    
    # Determine which variations need masks
    MASK_REQUIRED_VARIATIONS = []
    for var_type in variation_types:
        if var_type in VARIATION_TO_AUGMENTER:
            aug_cls = VARIATION_TO_AUGMENTER[var_type]
            aug = aug_cls() if var_type not in SLOW_VARIATIONS else None
            if aug is not None:
                mask_aware = getattr(aug, 'mask_aware', True)
                region = getattr(aug, 'region', 'foreground')
                if mask_aware and region in ['foreground', 'background']:
                    MASK_REQUIRED_VARIATIONS.append(var_type)
    # Also add SD-based variations that need masks
    MASK_REQUIRED_VARIATIONS.extend(['sd_inpaint', 'sd_foreign', 'cutpaste', 'scratchmix', 
                                      'texture', 'simple_shape_foreign', 'signs_foreign', 
                                      'texture_foreign'])
    MASK_REQUIRED_VARIATIONS = list(set(MASK_REQUIRED_VARIATIONS))
    print(f"Variations requiring mask: {MASK_REQUIRED_VARIATIONS}")
    
    # Metadata to track what was generated
    metadata = {
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'random_seed': RANDOM_SEED,
        'n_samples': N_SAMPLES,
        'object_classes': object_classes,
        'variation_types': variation_types,
        'status': {}
    }
    
    print("=" * 70)
    print("PSEUDO IMAGE PRE-GENERATION")
    print("=" * 70)
    print(f"Output directory: {output_base_dir}")
    print(f"Object classes: {len(object_classes)}")
    print(f"Variation types: {len(variation_types)}")
    print(f"Samples per class: {N_SAMPLES}")
    print("=" * 70)
    
    total_start = time.time()
    
    # Calculate total work for progress tracking
    total_variations = len(object_classes) * len(variation_types)
    completed_variations = 0
    
    # Main progress bar for overall progress
    pbar_main = tqdm(total=total_variations, desc="Overall Progress", position=0)
    
    for obj_idx, object_class in enumerate(object_classes):
        tqdm.write(f"\n{'='*60}")
        tqdm.write(f"[{obj_idx+1}/{len(object_classes)}] Processing {object_class}")
        tqdm.write(f"{'='*60}")
        
        obj_dir = output_base_dir / object_class
        obj_dir.mkdir(parents=True, exist_ok=True)
        
        metadata['status'][object_class] = {}
        
        # Sample test normal images
        normal_images = data_loader.sample_test_normal_images(object_class)
        
        # Save original normal images (for reference)
        normal_dir = obj_dir / "normal"
        normal_dir.mkdir(exist_ok=True)
        
        for i, (img_path, img) in enumerate(normal_images):
            save_path = normal_dir / f"{i:04d}.png"
            if not save_path.exists() or force_regenerate:
                Image.fromarray(img).save(save_path)
        
        metadata['status'][object_class]['normal'] = len(normal_images)
        tqdm.write(f"  Normal images: {len(normal_images)}")
        
        # Pre-compute masks for all images if any variation needs them
        needs_masks = any(v in MASK_REQUIRED_VARIATIONS for v in variation_types)
        masks_cache = {}
        
        if needs_masks:
            tqdm.write(f"  Pre-computing masks using SAM for {object_class}...")
            segmenter.set_object_type(object_class)
            config = segmenter.get_config_for_object(object_class)
            sam_mode = config.get("sam_mode", "sam_combined")
            tqdm.write(f"    Config: {config}")
            tqdm.write(f"    SAM mode: {sam_mode}")
            
            success_masks = 0
            for i, (img_path, img) in enumerate(tqdm(normal_images, desc="  Computing masks", position=1, leave=False)):
                # Convert RGB to BGR for SAM
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                try:
                    seg_results = segmenter.segment_all_modes(img_bgr, object_class)
                    if i == 0:
                        tqdm.write(f"    Available modes: {list(seg_results.keys())}")
                    
                    # Use configured sam_mode directly (SAM 3 required)
                    if sam_mode not in seg_results:
                        raise ValueError(f"SAM mode '{sam_mode}' not available. Available: {list(seg_results.keys())}. SAM 3 is required.")
                    
                    mask = seg_results[sam_mode]
                    if mask is None:
                        raise ValueError(f"SAM mode '{sam_mode}' returned None for image {i}")
                    
                    masks_cache[i] = mask.astype(np.uint8)
                    success_masks += 1
                    if i == 0:
                        tqdm.write(f"    Using mode: {sam_mode}")
                        tqdm.write(f"    First mask shape: {masks_cache[i].shape}, sum: {masks_cache[i].sum()}")
                except Exception as e:
                    raise RuntimeError(f"ERROR computing mask for image {i}: {e}")
            tqdm.write(f"  ✓ Computed {success_masks}/{len(masks_cache)} masks successfully")
            
            # Save masks to disk for later use by other experiments
            mask_dir = obj_dir / "masks"
            mask_dir.mkdir(exist_ok=True)
            for i, mask in masks_cache.items():
                mask_path = mask_dir / f"{i:04d}.png"
                # Save as single-channel grayscale (0 or 255)
                mask_img = (mask * 255).astype(np.uint8) if mask.max() <= 1 else mask
                Image.fromarray(mask_img).save(mask_path)
            tqdm.write(f"  ✓ Saved {len(masks_cache)} masks to {mask_dir}")
        
        # Generate pseudo images for each variation type
        for var_idx, var_type in enumerate(variation_types):
            var_dir = obj_dir / var_type
            var_dir.mkdir(exist_ok=True)
            
            # Check if already generated
            existing = list(var_dir.glob("*.png"))
            if len(existing) >= len(normal_images) and not force_regenerate:
                metadata['status'][object_class][var_type] = len(existing)
                pbar_main.update(1)
                completed_variations += 1
                tqdm.write(f"  [{var_idx+1}/{len(variation_types)}] {var_type}: SKIPPED (already {len(existing)} images)")
                continue
            
            # Get variation function
            try:
                var_func = variation_gen.get_variation_function(var_type)
            except Exception as e:
                tqdm.write(f"  [{var_idx+1}/{len(variation_types)}] {var_type}: ERROR - {e}")
                metadata['status'][object_class][var_type] = f"error: {e}"
                pbar_main.update(1)
                completed_variations += 1
                continue
            
            # Check if this variation needs mask
            var_needs_mask = var_type in MASK_REQUIRED_VARIATIONS
            
            # Generate and save pseudo images with progress bar
            success_count = 0
            is_slow = var_type in SLOW_VARIATIONS
            desc = f"  {var_type} ({'SD' if is_slow else 'fast'})"
            
            pbar_samples = tqdm(
                enumerate(normal_images), 
                total=len(normal_images),
                desc=desc,
                position=1,
                leave=False
            )
            
            for i, (img_path, img) in pbar_samples:
                save_path = var_dir / f"{i:04d}.png"
                
                if save_path.exists() and not force_regenerate:
                    success_count += 1
                    continue
                
                try:
                    # Get mask if needed
                    mask = masks_cache.get(i) if var_needs_mask else None
                    
                    # Apply variation with mask
                    # Use per-sample seed for reproducibility: base_seed + sample_index
                    sample_seed = RANDOM_SEED + i
                    pseudo_img = var_func(img, mask=mask, obj_name=object_class, seed=sample_seed)
                    Image.fromarray(pseudo_img).save(save_path)
                    success_count += 1
                except Exception as e:
                    # Save error marker
                    error_path = var_dir / f"{i:04d}.error"
                    error_path.write_text(str(e))
            
            pbar_samples.close()
            metadata['status'][object_class][var_type] = success_count
            pbar_main.update(1)
            completed_variations += 1
            tqdm.write(f"  [{var_idx+1}/{len(variation_types)}] {var_type}: {success_count}/{len(normal_images)} images")
        
        # Save metadata after each object (for resume capability)
        metadata_path = output_base_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    pbar_main.close()
    
    total_time = time.time() - total_start
    print("\n" + "=" * 70)
    print(f"GENERATION COMPLETE")
    print(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"Metadata saved to: {metadata_path}")
    print("=" * 70)
    
    return metadata


class PseudoImageLoader:
    """
    Loader for pre-generated pseudo images.
    Use this in experiments instead of generating on-the-fly.
    """
    
    def __init__(self, base_dir: Path = None):
        if base_dir is None:
            base_dir = get_output_dir("pseudo_images")
        self.base_dir = Path(base_dir)
        
        # Load metadata if exists
        metadata_path = self.base_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                self.metadata = json.load(f)
        else:
            self.metadata = None
    
    def _load_and_verify_image(self, img_path: Path) -> Image.Image:
        """
        Load and verify an image file.
        Raises an error if the image is corrupted.
        """
        try:
            img = Image.open(img_path)
            # Verify the image is not corrupted
            img.verify()
            # Re-open because verify() closes the file
            img = Image.open(img_path)
            return img
        except Exception as e:
            raise RuntimeError(
                f"Corrupted image file: {img_path}\n"
                f"Error: {e}\n"
                f"Please regenerate with: python generate_pseudo_images.py "
                f"--objects {img_path.parent.parent.name} "
                f"--variations {img_path.parent.name} --force"
            )
    
    def get_normal_images(self, object_class: str) -> list:
        """
        Get original normal images for an object class.
        
        Returns:
            List of (index, PIL.Image) tuples
        
        Raises:
            FileNotFoundError: If normal images directory doesn't exist
            RuntimeError: If any image file is corrupted
        """
        normal_dir = self.base_dir / object_class / "normal"
        if not normal_dir.exists():
            raise FileNotFoundError(f"Normal images not found for {object_class}")
        
        images = []
        for img_path in sorted(normal_dir.glob("*.png")):
            idx = int(img_path.stem)
            img = self._load_and_verify_image(img_path)
            images.append((idx, img))
        
        return images
    
    def get_pseudo_images(self, object_class: str, variation_type: str) -> list:
        """
        Get pre-generated pseudo images for an object class and variation type.
        
        Returns:
            List of (index, PIL.Image) tuples
        
        Raises:
            FileNotFoundError: If pseudo images directory doesn't exist
            RuntimeError: If any image file is corrupted
        """
        var_dir = self.base_dir / object_class / variation_type
        if not var_dir.exists():
            raise FileNotFoundError(
                f"Pseudo images not found for {object_class}/{variation_type}"
            )
        
        images = []
        for img_path in sorted(var_dir.glob("*.png")):
            idx = int(img_path.stem)
            img = self._load_and_verify_image(img_path)
            images.append((idx, img))
        
        return images
    
    def get_pseudo_images_np(self, object_class: str, variation_type: str) -> list:
        """
        Get pre-generated pseudo images as numpy arrays.
        
        Returns:
            List of (index, np.ndarray) tuples
        """
        pil_images = self.get_pseudo_images(object_class, variation_type)
        return [(idx, np.array(img)) for idx, img in pil_images]
    
    def has_variation(self, object_class: str, variation_type: str) -> bool:
        """Check if pseudo images exist for given object/variation."""
        var_dir = self.base_dir / object_class / variation_type
        if not var_dir.exists():
            return False
        return len(list(var_dir.glob("*.png"))) > 0
    
    def get_masks(self, object_class: str) -> list:
        """
        Get pre-computed SAM masks for an object class.
        
        Returns:
            List of (index, np.ndarray) tuples where mask is binary (0 or 255)
        
        Raises:
            FileNotFoundError: If masks directory doesn't exist
        """
        mask_dir = self.base_dir / object_class / "masks"
        if not mask_dir.exists():
            raise FileNotFoundError(
                f"Masks not found for {object_class}. "
                f"Regenerate pseudo images with: python scripts/public/generate_pseudo_images.py --objects {object_class} --force"
            )
        
        masks = []
        for mask_path in sorted(mask_dir.glob("*.png")):
            idx = int(mask_path.stem)
            mask = np.array(Image.open(mask_path).convert('L'))
            masks.append((idx, mask))
        
        return masks
    
    def has_masks(self, object_class: str) -> bool:
        """Check if pre-computed masks exist for an object class."""
        mask_dir = self.base_dir / object_class / "masks"
        if not mask_dir.exists():
            return False
        return len(list(mask_dir.glob("*.png"))) > 0
    
    def get_available_variations(self, object_class: str) -> list:
        """Get list of available variation types for an object class."""
        obj_dir = self.base_dir / object_class
        if not obj_dir.exists():
            return []
        
        variations = []
        for var_dir in obj_dir.iterdir():
            if var_dir.is_dir() and var_dir.name != "normal":
                if len(list(var_dir.glob("*.png"))) > 0:
                    variations.append(var_dir.name)
        
        return sorted(variations)


def main():
    parser = argparse.ArgumentParser(
        description='Pre-generate pseudo images for all experiments'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Output directory (default: outputs/public_experiments/pseudo_images)'
    )
    parser.add_argument(
        '--objects', type=str, nargs='+', default=None,
        help='Specific object classes to process (default: all)'
    )
    parser.add_argument(
        '--variations', type=str, nargs='+', default=None,
        help='Specific variation types to generate (default: all)'
    )
    parser.add_argument(
        '--force', action='store_true',
        help='Force regeneration even if images exist'
    )
    parser.add_argument(
        '--fast-only', action='store_true',
        help='Only generate fast (non-SD) variations'
    )
    parser.add_argument(
        '--slow-only', action='store_true',
        help='Only generate slow (SD-based) variations'
    )
    args = parser.parse_args()
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = get_output_dir("pseudo_images")
    
    # Determine which variations to generate
    if args.variations:
        variations = args.variations
    elif args.fast_only:
        variations = FAST_VARIATIONS
    elif args.slow_only:
        variations = SLOW_VARIATIONS
    else:
        # Run fast first, then slow
        variations = FAST_VARIATIONS + SLOW_VARIATIONS
    
    generate_pseudo_images(
        output_base_dir=output_dir,
        object_classes=args.objects,
        variation_types=variations,
        force_regenerate=args.force,
    )


if __name__ == '__main__':
    main()
