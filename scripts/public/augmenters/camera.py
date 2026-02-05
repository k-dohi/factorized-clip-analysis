"""
Category 5: Camera Anomaly

Augmentation methods that simulate image degradation due to camera/imaging device issues.

The three most common "imaging system variation factors" in real-world photography:
1. Motion Blur (Camera Shake / Motion Blur) - Hand shake, shutter shock
2. Image Noise - High ISO, sensor noise in low-light conditions
3. Defocus Blur - Focus miss, out of depth of field
"""

import cv2
import random
import numpy as np
from typing import Dict, Optional, Tuple

from .base import BaseAugmenter


class MotionBlurAug(BaseAugmenter):
    """
    Motion Blur (Camera shake / Hand shake)
    
    Blur caused by camera movement during exposure:
    - Hand shake
    - Shutter shock
    - Tripod vibration
    
    Appearance: Entire image flows/stretches in a line/loses sharpness
    Likely conditions: Low light, telephoto, slow shutter, handheld
    
    Physical explanation: Image is convolved in the "motion vector direction"
    """
    region = "global"
    mask_aware = False
    
    def _random_aug(self, img, mask, **kw):
        h, w = img.shape[:2]
        k = random.randint(7, 35) | 1
        angle = random.uniform(0, 360)
        
        kernel = self._create_motion_kernel(k, angle)
        out = cv2.filter2D(img, -1, kernel)
                
        return out, {"kernel": k, "angle": angle}
    
    @staticmethod
    def _create_motion_kernel(size, angle):
        """Generate motion blur kernel"""
        kernel = np.zeros((size, size))
        center = size // 2
        
        angle_rad = np.deg2rad(angle)
        
        for i in range(size):
            offset = i - center
            x = int(center + offset * np.cos(angle_rad))
            y = int(center + offset * np.sin(angle_rad))
            
            if 0 <= x < size and 0 <= y < size:
                kernel[y, x] = 1
        
        if kernel.sum() > 0:
            kernel /= kernel.sum()
        else:
            kernel[center, center] = 1
            
        return kernel


class NoiseAug(BaseAugmenter):
    """
    Sensor Noise (Image Noise)
    
    Noise from photon randomness + electronic circuit noise:
    - Shot noise (Poisson distribution)
    - Read noise (Gaussian distribution)
    
    Appearance: Graininess and color unevenness in dark areas and uniform surfaces (especially at high ISO)
    Likely conditions: Low light, high ISO, long exposure, small sensor cameras
    
    Physical explanation: I = S + N_shot + N_read
    """
    region = "global"
    mask_aware = False
    
    def _random_aug(self, img, mask, **kw):
        h, w = img.shape[:2]
        var = random.uniform(10, 60)
        
        noise = np.random.normal(0, np.sqrt(var), img.shape)
        out = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
                
        return out, {"var": var}


class DefocusBlurAug(BaseAugmenter):
    """
    Defocus Blur (Out of focus)
    
    Simulates out-of-focus state:
    - AF malfunction
    - Out of depth of field
    - Bokeh at wide aperture
    
    Physical explanation: Point image spreads as "PSF (Point Spread Function)"
    """
    region = "global"
    mask_aware = False
    
    def _random_aug(self, img, mask, **kw):
        h, w = img.shape[:2]
        
        # If radius is specified externally, use disk blur
        radius = kw.get('radius', None)
        if radius is not None:
            if radius == 0:
                # If radius=0, no blur (baseline)
                return img.copy(), {"blur_type": "none", "radius": 0}
            kernel = self._create_disk_kernel(radius)
            out = cv2.filter2D(img, -1, kernel)
            return out, {"blur_type": "disk", "radius": radius}
        
        # Select blur type
        blur_type = random.choice(["gaussian", "disk", "motion"])
        
        if blur_type == "gaussian":
            # Gaussian blur (general out-of-focus)
            kernel_size = random.randrange(5, 25, 2)
            sigma = random.uniform(1.0, 5.0)
            out = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
            return out, {
                "blur_type": "gaussian",
                "kernel_size": kernel_size,
                "sigma": sigma
            }
            
        elif blur_type == "disk":
            # Disk blur (circular bokeh, wide aperture bokeh)
            radius = random.randint(3, 12)
            kernel = self._create_disk_kernel(radius)
            out = cv2.filter2D(img, -1, kernel)
            return out, {
                "blur_type": "disk",
                "radius": radius
            }
            
        else:  # motion
            # Light motion blur (combination of camera shake and defocus)
            kernel_size = random.randint(5, 15) | 1
            angle = random.uniform(0, 360)
            kernel = MotionBlurAug._create_motion_kernel(kernel_size, angle)
            out = cv2.filter2D(img, -1, kernel)
            
            # Additional Gaussian blur
            out = cv2.GaussianBlur(out, (5, 5), 1.0)
            
            return out, {
                "blur_type": "motion_defocus",
                "kernel_size": kernel_size,
                "angle": angle
            }
    
    @staticmethod
    def _create_disk_kernel(radius):
        """
        Generate circular kernel (disk PSF).
        Simulates circular bokeh at wide aperture.
        """
        size = radius * 2 + 1
        kernel = np.zeros((size, size), dtype=np.float32)
        center = radius
        
        y, x = np.ogrid[-center:center+1, -center:center+1]
        mask = x*x + y*y <= radius*radius
        kernel[mask] = 1.0
        
        if kernel.sum() > 0:
            kernel /= kernel.sum()
        
        return kernel
