"""
Variation Generator - Self-contained implementation

This module provides a simple interface to the augmenters that are
now included in scripts/public/augmenters/

NO external dependencies - uses only local augmenters.
"""

import numpy as np
from pathlib import Path
import sys

# Add parent directory to path to import local augmenters
SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

# Import local augmenters (self-contained in public directory)
from augmenters import (
    # Category 1: Object Defect
    CutPasteAug,
    ScratchMixAug,
    FourierTextureAug,
    SDInpaintAug,
    
    # Category 2: Lighting
    ColorShiftAug,
    ContrastAnomalyAug,
    BrightnessAnomalyAug,
    
    # Category 3: Atmospheric
    FogAug,
    SmokeAug,
    HeatHazeAug,
    
    # Category 4: Foreign Object
    SimpleShapeForeignObjectAug,
    ProceduralForeignObjectAug,
    TextureForeignObjectAug,
    SDForeignObjectAug,
    
    # Category 5: Camera
    MotionBlurAug,
    NoiseAug,
    DefocusBlurAug,
    
    # Registry
    get_augmenter,
)

# Mapping from variation name to augmenter class
VARIATION_TO_AUGMENTER = {
    # Lighting
    'colorshift': ColorShiftAug,
    'contrast_change': ContrastAnomalyAug,
    'brightness': BrightnessAnomalyAug,
    
    # Medium/Atmospheric
    'fog': FogAug,
    'smoke': SmokeAug,
    'heathaze': HeatHazeAug,
    
    # Foreign
    'simple_shape_foreign': SimpleShapeForeignObjectAug,
    'signs_foreign': ProceduralForeignObjectAug,
    'texture_foreign': TextureForeignObjectAug,
    'sd_foreign': SDForeignObjectAug,
    
    # Camera
    'motionblur': MotionBlurAug,
    'noise': NoiseAug,
    'defocusblur': DefocusBlurAug,
    
    # Object
    'cutpaste': CutPasteAug,
    'scratchmix': ScratchMixAug,
    'texture': FourierTextureAug,
    'sd_inpaint': SDInpaintAug,
}


class VariationGenerator:
    """
    Wrapper that provides a simple interface to existing augmenters.
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize variation generator.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        self._augmenter_cache = {}
    
    def _get_augmenter(self, variation_type: str):
        """Get or create cached augmenter instance."""
        if variation_type not in self._augmenter_cache:
            if variation_type not in VARIATION_TO_AUGMENTER:
                raise ValueError(f"Unknown variation type: {variation_type}. "
                               f"Available: {list(VARIATION_TO_AUGMENTER.keys())}")
            self._augmenter_cache[variation_type] = VARIATION_TO_AUGMENTER[variation_type]()
        return self._augmenter_cache[variation_type]
    
    def apply(self, img, variation_type: str, mask=None, seed=None, **kwargs):
        """
        Apply a variation to an image.
        
        Args:
            img: Input image (BGR, uint8)
            variation_type: Name of the variation
            mask: Optional foreground mask (if None, creates dummy mask for foreign object augs)
            seed: Optional random seed
            **kwargs: Additional arguments for the augmenter
        
        Returns:
            Tuple of (augmented_image, metadata)
        """
        import numpy as np
        
        augmenter = self._get_augmenter(variation_type)
        if seed is None:
            seed = self.random_seed
        
        # Foreign object augmenters require a mask (background = 0, foreground = 1)
        # If no mask provided, create a simple center-based mask
        if mask is None and variation_type in ['simple_shape_foreign', 'signs_foreign', 
                                                 'texture_foreign', 'sd_foreign']:
            h, w = img.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            # Create circular foreground in center (objects typically in center)
            cy, cx = h // 2, w // 2
            radius = min(h, w) // 3
            y, x = np.ogrid[:h, :w]
            circle_mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
            mask[circle_mask] = 1
        
        return augmenter(img, mask=mask, seed=seed, **kwargs)
    
    def get_variation_function(self, variation_type: str):
        """
        Get a callable that applies the specified variation.
        
        Args:
            variation_type: Name of the variation
        
        Returns:
            Callable that takes an image and returns the augmented image
        """
        import numpy as np
        
        augmenter = self._get_augmenter(variation_type)
        
        # Check augmenter properties
        mask_aware = getattr(augmenter, 'mask_aware', True)
        region = getattr(augmenter, 'region', 'foreground')
        needs_mask = mask_aware and region in ['foreground', 'background']
        use_sd = variation_type == 'sd_foreign'  # SDForeignObjectAug uses SD generation
        
        def apply_variation(img, mask=None, **kwargs):
            # If mask is required but not provided, raise error
            # (mask should be provided by generate_pseudo_images.py via SAM)
            if mask is None and needs_mask:
                raise ValueError(
                    f"Variation '{variation_type}' requires a foreground mask but none was provided. "
                    f"Please ensure SAM segmentation is working correctly."
                )
            
            if use_sd:
                kwargs['use_sd_generator'] = True
            # Use per-sample seed for reproducibility: caller should pass seed=base_seed+index
            # If seed is not provided, use None (random)
            seed = kwargs.pop('seed', None)
            result, _ = augmenter(img, mask=mask, seed=seed, **kwargs)
            return result
        
        return apply_variation
    
    @staticmethod
    def get_all_variation_types():
        """Get list of all available variation types."""
        return list(VARIATION_TO_AUGMENTER.keys())
    
    @staticmethod
    def get_augmenter_class(variation_type: str):
        """Get the augmenter class for a variation type."""
        return VARIATION_TO_AUGMENTER.get(variation_type)
    
    # =========================================================================
    # Convenience methods for specific augmentations (used by experiment2/4)
    # =========================================================================
    
    def apply_fog(self, img, density=None, **kwargs):
        """Apply fog augmentation."""
        augmenter = self._get_augmenter('fog')
        if density is not None:
            kwargs['density'] = density
        result, _ = augmenter(img, mask=None, seed=None, **kwargs)
        return result
    
    def apply_defocusblur(self, img, radius=None, **kwargs):
        """Apply defocus blur augmentation."""
        augmenter = self._get_augmenter('defocusblur')
        if radius is not None:
            kwargs['radius'] = radius
        result, _ = augmenter(img, mask=None, seed=None, **kwargs)
        return result
    
    def apply_simple_shape_foreign(self, img, mask=None, **kwargs):
        """Apply simple shape foreign object augmentation."""
        import numpy as np
        
        if mask is None:
            h, w = img.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            cy, cx = h // 2, w // 2
            radius = min(h, w) // 3
            y, x = np.ogrid[:h, :w]
            circle_mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
            mask[circle_mask] = 1
        
        augmenter = self._get_augmenter('simple_shape_foreign')
        result, _ = augmenter(img, mask=mask, seed=None, **kwargs)
        return result
    
    def apply_noise(self, img, **kwargs):
        """Apply noise augmentation."""
        augmenter = self._get_augmenter('noise')
        result, _ = augmenter(img, mask=None, seed=None, **kwargs)
        return result
    
    def apply_motionblur(self, img, **kwargs):
        """Apply motion blur augmentation."""
        augmenter = self._get_augmenter('motionblur')
        result, _ = augmenter(img, mask=None, seed=None, **kwargs)
        return result
    
    def apply_colorshift(self, img, **kwargs):
        """Apply color shift augmentation."""
        augmenter = self._get_augmenter('colorshift')
        result, _ = augmenter(img, mask=None, seed=None, **kwargs)
        return result
    
    def apply_contrast_change(self, img, **kwargs):
        """Apply contrast change augmentation."""
        augmenter = self._get_augmenter('contrast_change')
        result, _ = augmenter(img, mask=None, seed=None, **kwargs)
        return result
    
    def apply_brightness(self, img, **kwargs):
        """Apply brightness augmentation."""
        augmenter = self._get_augmenter('brightness')
        result, _ = augmenter(img, mask=None, seed=None, **kwargs)
        return result
    
    def apply_smoke(self, img, **kwargs):
        """Apply smoke augmentation."""
        augmenter = self._get_augmenter('smoke')
        result, _ = augmenter(img, mask=None, seed=None, **kwargs)
        return result
    
    def apply_heathaze(self, img, **kwargs):
        """Apply heat haze augmentation."""
        augmenter = self._get_augmenter('heathaze')
        result, _ = augmenter(img, mask=None, seed=None, **kwargs)
        return result
    
    def apply_signs_foreign(self, img, mask=None, **kwargs):
        """Apply signs/procedural foreign object augmentation."""
        import numpy as np
        if mask is None:
            h, w = img.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            cy, cx = h // 2, w // 2
            radius = min(h, w) // 3
            y, x = np.ogrid[:h, :w]
            circle_mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
            mask[circle_mask] = 1
        augmenter = self._get_augmenter('signs_foreign')
        result, _ = augmenter(img, mask=mask, seed=None, **kwargs)
        return result
    
    def apply_texture_foreign(self, img, mask=None, **kwargs):
        """Apply texture foreign object augmentation."""
        import numpy as np
        if mask is None:
            h, w = img.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            cy, cx = h // 2, w // 2
            radius = min(h, w) // 3
            y, x = np.ogrid[:h, :w]
            circle_mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
            mask[circle_mask] = 1
        augmenter = self._get_augmenter('texture_foreign')
        result, _ = augmenter(img, mask=mask, seed=None, **kwargs)
        return result
    
    def apply_cutpaste(self, img, mask=None, **kwargs):
        """Apply cutpaste augmentation."""
        augmenter = self._get_augmenter('cutpaste')
        result, _ = augmenter(img, mask=mask, seed=None, **kwargs)
        return result
    
    def apply_scratchmix(self, img, mask=None, **kwargs):
        """Apply scratchmix augmentation."""
        augmenter = self._get_augmenter('scratchmix')
        result, _ = augmenter(img, mask=mask, seed=None, **kwargs)
        return result
    
    def apply_texture(self, img, mask=None, **kwargs):
        """Apply Fourier texture augmentation."""
        augmenter = self._get_augmenter('texture')
        result, _ = augmenter(img, mask=mask, seed=None, **kwargs)
        return result
    
    def apply_sd_inpaint(self, img, mask=None, **kwargs):
        """Apply SD Inpaint augmentation."""
        augmenter = self._get_augmenter('sd_inpaint')
        result, _ = augmenter(img, mask=mask, seed=None, **kwargs)
        return result
    
    def apply_sd_foreign(self, img, mask=None, **kwargs):
        """Apply SD Foreign object augmentation."""
        import numpy as np
        if mask is None:
            h, w = img.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            cy, cx = h // 2, w // 2
            radius = min(h, w) // 3
            y, x = np.ogrid[:h, :w]
            circle_mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
            mask[circle_mask] = 1
        augmenter = self._get_augmenter('sd_foreign')
        # Use SD generation (not loading from directory)
        result, _ = augmenter(img, mask=mask, seed=None, use_sd_generator=True, **kwargs)
        return result
