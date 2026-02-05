"""
Base Augmenter class

This file defines the base class (BaseAugmenter) only.
Actual augmentation classes are in separate files.
"""

import cv2
import random
import json
import math
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Tuple


# ==========================================================
# Base class (mask-aware, every call random)
# ==========================================================
class BaseAugmenter(ABC):
    region: str = "foreground"         # "foreground", "background", or "global"
    mask_aware: bool = True            # mask-aware or not

    # -------- public API --------
    def __call__(self,
                 image: np.ndarray,
                 mask: Optional[np.ndarray] = None,
                 seed: Optional[int] = None,
                 **kw) -> Tuple[np.ndarray, Dict]:
        """
        image : BGR uint8 (H,W,3)
        mask  : binary (H,W) 1 = foreground (None means entire image)
        Returns
        -------
        aug_img : uint8 BGR
        meta    : dict containing actually sampled continuous parameters
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        aug_img, params = self._random_aug(image.copy(), mask, **kw)

        if mask is not None and self.mask_aware:
            m3 = np.stack([mask]*3, 2)
            if self.region == "foreground":
                aug_img = aug_img*m3 + image*(1-m3)
            else:                           # background
                aug_img = aug_img*(1-m3) + image*m3

        return aug_img.astype(np.uint8), params

    # -------- subclass must implement --------
    @abstractmethod
    def _random_aug(self,
                    img: np.ndarray,
                    mask: Optional[np.ndarray],
                    **kw) -> Tuple[np.ndarray, Dict]:
        ...
