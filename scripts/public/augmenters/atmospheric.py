"""
Category 3: Atmospheric Distortion

Augmentation methods that simulate appearance changes due to optical property changes in air.

Handles the following 3 representative atmospheric changes:
1. FogAug: Fog/mist (scattering by fine water droplets)
2. SmokeAug: Smoke/smog/dust (scattering + absorption by particles)
3. HeatHazeAug: Heat shimmer/mirage (refractive index fluctuation)
"""

import cv2
import random
import numpy as np
from typing import Dict, Optional, Tuple

from .base import BaseAugmenter


class FogAug(BaseAugmenter):
    """
    Fog/Mist (scattering by fine water droplets)
    
    Medium change: Water droplets float in large quantities in air, light scatters in all directions
    
    Effect on photo:
    - Reduced contrast, colors become pale
    - Bluish-white transparent haze
    - Overall bright and soft impression
    """
    region = "global"
    mask_aware = False
    
    def _random_aug(self, img, mask, **kw):
        h, w = img.shape[:2]
        
        # Fog density: can be specified externally, otherwise random
        fog_density = kw.get('fog_density', kw.get('density', None))
        if fog_density is None:
            fog_density = random.uniform(0.3, 0.6)
        
        # Strong blur to emphasize softness
        blur_img = cv2.GaussianBlur(img, (11, 11), 2.0)
        
        # Bluish-white fog (bright)
        fog_color = np.array([250, 250, 245], dtype=np.float32)  # Almost white with slight blue
        fog_overlay = np.full((h, w, 3), fog_color, dtype=np.float32)
        
        # Uniformly blend fog across entire image (transparent overlay)
        out = blur_img.astype(np.float32) * (1 - fog_density) + fog_overlay * fog_density
        out = np.clip(out, 0, 255).astype(np.uint8)
        
        return out, {
            "fog_density": round(fog_density, 3)
        }


class SmokeAug(BaseAugmenter):
    """
    Smoke/Smog/Dust (scattering + absorption by particles)
    
    Medium change: Solid particles like carbon, sulfur, dust scatter and absorb light
    
    Effect on photo:
    - Dense and opaque with particle texture
    - Warm to neutral colors with heavy atmosphere
    - Significantly reduced contrast
    """
    region = "global"
    mask_aware = False
    
    def _random_aug(self, img, mask, **kw):
        h, w = img.shape[:2]
        
        # Smoke density (dense)
        smoke_density = random.uniform(0.4, 0.7)
        
        # Light blur (not as soft as fog)
        blur_img = cv2.GaussianBlur(img, (5, 5), 0.8)
        
        # Dark warm-toned overlay (grayish brown)
        smoke_color = np.array([140, 150, 160], dtype=np.float32)  # Dark gray to brown
        smoke_overlay = np.full((h, w, 3), smoke_color, dtype=np.float32)
        
        # Add random particle noise (for particle texture effect)
        noise = np.random.randint(-15, 15, (h, w, 3), dtype=np.int16)
        smoke_overlay = np.clip(smoke_overlay + noise, 0, 255).astype(np.float32)
        
        # Blend smoke overlay (opaque feel)
        out = blur_img.astype(np.float32) * (1 - smoke_density) + smoke_overlay * smoke_density
        out = np.clip(out, 0, 255).astype(np.uint8)
        
        return out, {
            "smoke_density": round(smoke_density, 3)
        }


class HeatHazeAug(BaseAugmenter):
    """
    Heat shimmer/mirage (refractive index fluctuation)
    
    Medium change: Temperature differences cause local changes in air refractive index, light path fluctuates
    
    Effect on photo:
    - Distant objects or objects near ground appear to waver
    - Blurred edges, reduced detail resolution
    - Especially prominent in telephoto shots (summer roads, factory exhaust, deserts, etc.)
    """
    region = "global"
    mask_aware = False
    
    def _random_aug(self, img, mask, **kw):
        h, w = img.shape[:2]
        
        # Shimmer intensity and wave characteristics (very weak)
        distortion_strength = random.uniform(0.5, 1.5)  # Weak distortion
        wave_frequency = random.uniform(0.02, 0.05)
        
        # Generate distortion map (wavy displacement)
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Sinusoidal fluctuation (applied uniformly across entire image)
        dx = distortion_strength * np.sin(y * wave_frequency * 2 * np.pi)
        dy = distortion_strength * np.sin(x * wave_frequency * 2 * np.pi)
        
        # Add random noise for irregularity (very weak)
        noise_x = np.random.randn(h, w) * distortion_strength * 0.2
        noise_y = np.random.randn(h, w) * distortion_strength * 0.2
        
        dx = dx + noise_x
        dy = dy + noise_y
        
        # Remapping
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)
        
        # Boundary handling
        map_x = np.clip(map_x, 0, w - 1)
        map_y = np.clip(map_y, 0, h - 1)
        
        # Apply distortion
        distorted = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        # Slight blur (subtle resolution reduction)
        out = cv2.GaussianBlur(distorted, (3, 3), 0.7)
        
        return out, {
            "distortion_strength": round(distortion_strength, 2),
            "wave_frequency": round(wave_frequency, 4)
        }
