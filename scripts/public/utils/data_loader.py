"""
MVTec AD Data Loader

Provides utilities for loading MVTec AD images with proper sampling protocol:
- Test normal: Random sampling with replacement (N=100)
- Test anomaly: All images from defect subdirectories
"""

import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Tuple, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DATASETS_DIR, OBJECT_CLASSES, N_SAMPLES, RANDOM_SEED, EXCLUDED_DEFECT_TYPES
)


class MVTecDataLoader:
    """Data loader for MVTec AD dataset with proper sampling protocol."""
    
    def __init__(self, 
                 dataset_dir: Path = DATASETS_DIR,
                 n_samples: int = N_SAMPLES,
                 random_seed: int = RANDOM_SEED):
        """
        Initialize data loader.
        
        Args:
            dataset_dir: Path to MVTec AD dataset
            n_samples: Number of test normal images to sample
            random_seed: Random seed for reproducibility
        """
        self.dataset_dir = Path(dataset_dir)
        self.data_dir = self.dataset_dir  # Alias for compatibility
        self.n_samples = n_samples
        self.rng = np.random.default_rng(random_seed)
    
    def get_test_normal_images(self, object_class: str) -> List[Tuple[Path, np.ndarray]]:
        """
        Load all test normal images for an object class.
        
        Args:
            object_class: Object class name (e.g., 'bottle')
        
        Returns:
            List of (path, image_array) tuples
        """
        test_good_dir = self.dataset_dir / object_class / "test" / "good"
        if not test_good_dir.exists():
            raise FileNotFoundError(f"Test good directory not found: {test_good_dir}")
        
        images = []
        for img_path in sorted(test_good_dir.glob("*.png")):
            img = np.array(Image.open(img_path).convert('RGB'))
            images.append((img_path, img))
        
        return images
    
    def sample_test_normal_images(self, 
                                  object_class: str,
                                  n_samples: Optional[int] = None
                                  ) -> List[Tuple[Path, np.ndarray]]:
        """
        Sample test normal images with replacement.
        
        This is the protocol used in the paper: randomly sample N images
        from test/good with replacement to create pseudo-normal set.
        
        Args:
            object_class: Object class name
            n_samples: Number of samples (default: self.n_samples)
        
        Returns:
            List of (path, image_array) tuples
        """
        if n_samples is None:
            n_samples = self.n_samples
        
        all_images = self.get_test_normal_images(object_class)
        if not all_images:
            raise ValueError(f"No test normal images found for {object_class}")
        
        # Sample with replacement
        indices = self.rng.choice(len(all_images), size=n_samples, replace=True)
        return [all_images[i] for i in indices]
    
    def get_test_anomaly_images(self, object_class: str) -> List[Tuple[Path, np.ndarray, str]]:
        """
        Load all test anomaly images for an object class.
        
        Args:
            object_class: Object class name
        
        Returns:
            List of (path, image_array, defect_type) tuples
        """
        test_dir = self.dataset_dir / object_class / "test"
        if not test_dir.exists():
            raise FileNotFoundError(f"Test directory not found: {test_dir}")
        
        # Excluded defect types for this object class
        excluded = EXCLUDED_DEFECT_TYPES.get(object_class, [])
        
        images = []
        for subdir in sorted(test_dir.iterdir()):
            if not subdir.is_dir():
                continue
            if subdir.name == "good":
                continue
            if subdir.name in excluded:
                continue
            
            defect_type = subdir.name
            for img_path in sorted(subdir.glob("*.png")):
                img = np.array(Image.open(img_path).convert('RGB'))
                images.append((img_path, img, defect_type))
        
        return images
    
    def get_train_normal_images(self, object_class: str) -> List[Tuple[Path, np.ndarray]]:
        """
        Load all training normal images for an object class.
        
        NOTE: This is only for reference/comparison. The paper experiments
        use test normal images, not train normal.
        
        Args:
            object_class: Object class name
        
        Returns:
            List of (path, image_array) tuples
        """
        train_good_dir = self.dataset_dir / object_class / "train" / "good"
        if not train_good_dir.exists():
            raise FileNotFoundError(f"Train good directory not found: {train_good_dir}")
        
        images = []
        for img_path in sorted(train_good_dir.glob("*.png")):
            img = np.array(Image.open(img_path).convert('RGB'))
            images.append((img_path, img))
        
        return images
    
    @staticmethod
    def get_object_classes() -> List[str]:
        """Get list of object classes."""
        return OBJECT_CLASSES.copy()
