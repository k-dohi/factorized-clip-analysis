"""
Augmenters package - 5-category pseudo anomaly generators

Category classification:
- Category1: Object Defect
- Category2: Lighting Variation
- Category3: Atmospheric Distortion
- Category4: Foreign Object
- Category5: Camera Anomaly
"""

from .object_defect import (
    CutPasteAug,
    ScratchMixAug,
    FourierTextureAug,
    SDInpaintAug,
)

from .lighting import (
    ColorShiftAug,
    ContrastAnomalyAug,
    BrightnessAnomalyAug,
)

from .atmospheric import (
    FogAug,
    SmokeAug,
    HeatHazeAug,
)

from .foreign_object import (
    SimpleShapeForeignObjectAug,
    ProceduralForeignObjectAug,
    TextureForeignObjectAug,
    SDForeignObjectAug,
)

from .camera import (
    MotionBlurAug,
    NoiseAug,
    DefocusBlurAug,
)

from .base import BaseAugmenter

# Export all classes
__all__ = [
    # Base
    "BaseAugmenter",
    
    # Category1: Object Defect
    "CutPasteAug",
    "ScratchMixAug",
    "FourierTextureAug",
    "SDInpaintAug",
    
    # Category 2: Lighting
    "ColorShiftAug",
    "ContrastAnomalyAug",
    "BrightnessAnomalyAug",
    
    # Category 3: Atmospheric
    "FogAug",
    "SmokeAug",
    "HeatHazeAug",
    
    # Category 4: Foreign Object
    "SimpleShapeForeignObjectAug",
    "ProceduralForeignObjectAug",
    "TextureForeignObjectAug",
    "SDForeignObjectAug",
    
    # Category 5: Camera
    "MotionBlurAug",
    "NoiseAug",
    "DefocusBlurAug",
]


# ==========================================================
# Registry & Helper Functions
# ==========================================================
_REGISTRY = {
    # Category 1: Object Defect (foreground)
    "cutpaste":      CutPasteAug,
    "scratchmix":    ScratchMixAug,
    "texture":       FourierTextureAug,
    "sd_inpaint":    SDInpaintAug,
    
    # Category 2: Lighting (global)
    "colorshift":    ColorShiftAug,
    "contrast":      ContrastAnomalyAug,
    "contrast_change": ContrastAnomalyAug,
    "brightness":    BrightnessAnomalyAug,
    "brightness_change": BrightnessAnomalyAug,
    
    # Category 3: Atmospheric (global)
    "fog":           FogAug,
    "smoke":         SmokeAug,
    "heathaze":      HeatHazeAug,
    
    # Category 4: Foreign Object (background)
    "simple_shape_foreign": SimpleShapeForeignObjectAug,
    "signs_foreign": ProceduralForeignObjectAug,
    "texture_foreign":    TextureForeignObjectAug,
    "sd_foreign":         SDForeignObjectAug,
    
    # Category 5: Camera (global)
    "motionblur":    MotionBlurAug,
    "noise":         NoiseAug,
    "defocusblur":   DefocusBlurAug,
}


def get_augmenter(name: str) -> BaseAugmenter:
    """Get an Augmenter instance by name"""
    name = name.lower()
    if name not in _REGISTRY:
        raise ValueError(f"{name} not in registry. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name]()


def get_all_augmenter_names():
    """Get all augmenter names"""
    return list(_REGISTRY.keys())
