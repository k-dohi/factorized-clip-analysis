"""
Category 2: Lighting Variation

Augmentation methods that simulate appearance changes due to lighting/illumination variations.
Lighting affects the entire image regardless of foreground/background.

Handles the following 3 lighting changes:
1. ColorShiftAug: Color temperature changes
2. ContrastAnomalyAug: Contrast changes
3. BrightnessAnomalyAug: Brightness changes
"""

import cv2
import random
import numpy as np
from typing import Dict, Optional, Tuple

from .base import BaseAugmenter


class ColorShiftAug(BaseAugmenter):
    """Color shift (lighting color temperature change)"""
    region = "global"
    mask_aware = False

    def _random_aug(self, img, mask, **kw):
        h, w = img.shape[:2]
        h_shift = int(random.uniform(-25, 25))
        s_shift = int(random.uniform(-40, 40))
        v_shift = int(random.uniform(-30, 30))
        gray_p = random.uniform(0, 0.4)
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = np.clip(hsv[:, :, 0] + h_shift/2, 0, 179)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] + s_shift, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] + v_shift, 0, 255)
        
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        if gray_p > 0:
            gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
            gray_mask = np.random.random((h, w)) < gray_p
            out[gray_mask] = np.stack([gray[gray_mask]] * 3, axis=1)

        return out, {
            "hue": h_shift,
            "sat": s_shift,
            "val": v_shift,
            "gray_p": round(gray_p, 2)
        }


class ContrastAnomalyAug(BaseAugmenter):
    """Contrast anomaly (lighting contrast change)"""
    region = "global"
    mask_aware = False
    
    def _random_aug(self, img, mask, **kw):
        h, w = img.shape[:2]
        factor = random.uniform(0.3, 2.0)  # Contrast modification
        
        # Lighting changes affect the entire image
        mean = np.mean(img)
        out = np.clip((img - mean) * factor + mean, 0, 255).astype(np.uint8)
            
        return out, {"contrast_factor": factor}


class BrightnessAnomalyAug(BaseAugmenter):
    """Brightness anomaly (lighting brightness change)"""
    region = "global"
    mask_aware = False
    
    def _random_aug(self, img, mask, **kw):
        h, w = img.shape[:2]
        factor = random.uniform(-0.5, 0.8)  # Darken to brighten
        
        # Lighting changes affect the entire image
        out = np.clip(img.astype(np.float32) * (1 + factor), 0, 255).astype(np.uint8)
            
        return out, {"brightness_factor": factor}
