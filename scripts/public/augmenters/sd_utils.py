"""
Stable Diffusion Utility Classes for Augmentation

Provides common functionality for SD-based augmentation.
"""

import random
import csv
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List, Any
from PIL import Image


# Stable Diffusion model presets
SD_MODEL_PRESETS = {
    "sd15": {
        "key": "sd15",
        "label": "Stable Diffusion v1.5",
        "inpaint_model_id": "runwayml/stable-diffusion-inpainting",
        "img2img_model_id": "runwayml/stable-diffusion-v1-5",
        "torch_dtype": "float16",
        "default_guidance": 3.0,
        "default_steps": 30,
        "default_img2img_strength": 0.65,
    },
    "sd2": {
        "key": "sd2",
        "label": "Stable Diffusion v2.0",
        "inpaint_model_id": "stabilityai/stable-diffusion-2-inpainting",
        "img2img_model_id": "stabilityai/stable-diffusion-2",
        "torch_dtype": "float16",
        "default_guidance": 5.0,
        "default_steps": 40,
        "default_img2img_strength": 0.6,
    },
    "sdxl": {
        "key": "sdxl",
        "label": "Stable Diffusion XL",
        "inpaint_model_id": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        "img2img_model_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "torch_dtype": "float16",
        "default_guidance": 5.5,
        "default_steps": 35,
        "default_img2img_strength": 0.55,
        "load_kwargs": {"variant": "fp16"},
    }
}


class SDPipelineManager:
    """
    Stable Diffusion Pipeline Management Class
    
    Provides common functionality for SD-based augmentation including
    model presets, device/dtype management, and prompt management.
    """
    
    def __init__(self,
                 model_preset: str = "sd15",
                 model_id: Optional[str] = None,
                 pipeline_type: str = "inpaint",
                 device: str = "auto",
                 fp16: bool = True,
                 prompt_csv: Optional[str] = None):
        """
        Args:
            model_preset: Model preset ("sd15", "sd2", "sdxl")
            model_id: Custom model ID (overrides preset)
            pipeline_type: Pipeline type ("inpaint" or "img2img")
            device: Device ("auto", "cuda", "cpu")
            fp16: FP16 usage flag
            prompt_csv: Prompt CSV path
        """
        # Load model preset
        if model_preset in SD_MODEL_PRESETS:
            preset = SD_MODEL_PRESETS[model_preset]
            if pipeline_type == "inpaint":
                self.model_id = model_id or preset["inpaint_model_id"]
            elif pipeline_type == "img2img":
                self.model_id = model_id or preset["img2img_model_id"]
            else:
                raise ValueError(f"Invalid pipeline_type: {pipeline_type}")
            
            self.model_label = preset["label"]
            self.default_guidance = preset["default_guidance"]
            self.default_steps = preset["default_steps"]
            self.default_strength = preset.get("default_img2img_strength", 0.65)
            self.load_kwargs = preset.get("load_kwargs", {})
        else:
            raise ValueError(f"Invalid model_preset: {model_preset}. Choose from {list(SD_MODEL_PRESETS.keys())}")
        
        self.pipeline_type = pipeline_type
        
        # Auto-detect device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # dtype configuration
        if self.device == "cuda" and fp16:
            self.torch_dtype = torch.float16
        else:
            self.torch_dtype = torch.float32
        
        # Prompt management
        self.prompts = self._load_prompts(prompt_csv)
        
        # Pipeline is lazily initialized
        self.pipeline = None
        
        print(f"[SDPipelineManager] Configured {self.model_label} ({pipeline_type}) on {self.device}")
        print(f"[SDPipelineManager] Loaded {len(self.prompts)} prompts")
    
    def _load_prompts(self, prompt_csv: Optional[str]) -> List[str]:
        """
        Load prompts from CSV
        
        CSV format:
        - prompt,category,description
        - "prompt text",defect,"description"
        """
        # Default prompts (vary by pipeline_type)
        if self.pipeline_type == "inpaint":
            default_prompts = [
                "a small deep scratch with rusty edges, photorealistic, high detail",
                "cracked surface with damaged texture, realistic crack pattern",
                "corroded metal with rust and oxidation, weathered surface",
                "burnt and charred surface with scorch marks, fire damage",
            ]
        else:  # img2img (object generation)
            default_prompts = [
                "a small screw on white background, product photo, high detail",
                "a small bolt on white background, studio lighting",
                "a paper clip on white background, office supply",
                "a small coin on white background, metallic shiny",
            ]
        
        if prompt_csv is None:
            return default_prompts
        
        csv_path = Path(prompt_csv)
        if not csv_path.exists():
            print(f"[SDPipelineManager] Warning: prompt CSV not found: {prompt_csv}")
            return default_prompts
        
        prompts = []
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'prompt' in row and row['prompt'].strip():
                        prompts.append(row['prompt'].strip())
            
            if prompts:
                print(f"[SDPipelineManager] Loaded {len(prompts)} prompts from {csv_path}")
            else:
                print(f"[SDPipelineManager] Warning: No valid prompts in CSV, using defaults")
                return default_prompts
                
        except Exception as e:
            print(f"[SDPipelineManager] Error loading prompts from CSV: {e}")
            return default_prompts
        
        return prompts
    
    def get_pipeline(self):
        """Get pipeline (lazy initialization)"""
        if self.pipeline is not None:
            return self.pipeline
        
        print(f"[SDPipelineManager] Loading pipeline: {self.model_label} ({self.pipeline_type})...")
        
        pipeline_kwargs = {
            "torch_dtype": self.torch_dtype,
            "safety_checker": None,
            "requires_safety_checker": False,
        }
        pipeline_kwargs.update(self.load_kwargs)
        
        if self.pipeline_type == "inpaint":
            from diffusers import AutoPipelineForInpainting
            self.pipeline = AutoPipelineForInpainting.from_pretrained(
                self.model_id,
                **pipeline_kwargs
            )
        elif self.pipeline_type == "img2img":
            from diffusers import StableDiffusionPipeline
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                **pipeline_kwargs
            )
        else:
            raise ValueError(f"Invalid pipeline_type: {self.pipeline_type}")
        
        # Memory optimization
        if self.device == "cuda":
            try:
                # Try to use model offloading (requires accelerate)
                self.pipeline.enable_model_cpu_offload()
                print("[SDPipelineManager] Enabled model CPU offload")
            except Exception as e:
                print(f"[SDPipelineManager] Could not enable CPU offload: {e}. Moving to CUDA directly.")
                self.pipeline = self.pipeline.to(self.device)
        else:
            self.pipeline = self.pipeline.to(self.device)

        if hasattr(self.pipeline, 'enable_attention_slicing'):
            self.pipeline.enable_attention_slicing()
        
        if hasattr(self.pipeline, 'enable_vae_slicing'):
            self.pipeline.enable_vae_slicing()
            
        if hasattr(self.pipeline, 'enable_vae_tiling'):
            self.pipeline.enable_vae_tiling()
        
        print(f"[SDPipelineManager] Pipeline loaded successfully")
        return self.pipeline
    
    def get_random_prompt(self) -> str:
        """Get a random prompt"""
        return random.choice(self.prompts)
    
    def create_generator(self, seed: Optional[int] = None) -> torch.Generator:
        """Create a generator"""
        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(seed)
        else:
            generator.manual_seed(random.randint(0, 2**32-1))
        return generator
    
    def prepare_image_and_mask(self, 
                               image: np.ndarray, 
                               mask: np.ndarray,
                               target_size: int = 512) -> tuple:
        """
        Preprocess image and mask for SD
        
        Returns:
            (pil_image, pil_mask, original_size)
        """
        import cv2
        
        original_size = image.shape[:2]  # (h, w)
        
        # Convert to uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)
        
        # Resize (multiple of 8)
        img_resized = cv2.resize(image, (target_size, target_size))
        mask_resized = cv2.resize(mask, (target_size, target_size))
        
        # PIL conversion
        pil_image = Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
        pil_mask = Image.fromarray(mask_resized).convert("L")
        
        return pil_image, pil_mask, original_size
    
    def postprocess_result(self, 
                          result_pil: Image.Image,
                          original_size: tuple,
                          original_image: np.ndarray,
                          mask: Optional[np.ndarray] = None,
                          mask_feather: int = 0) -> np.ndarray:
        """
        Post-process SD generation result
        
        Args:
            result_pil: SD generation result (PIL Image)
            original_size: Original image size (h, w)
            original_image: Original image (numpy array)
            mask: Mask (optional, used for blending)
            mask_feather: Mask blur amount
            
        Returns:
            Processed image (numpy array, BGR)
        """
        import cv2
        
        # Resize to original size
        h, w = original_size
        result_pil = result_pil.resize((w, h), Image.LANCZOS)
        result_np = np.array(result_pil)
        result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
        
        # Mask blend
        if mask is not None and mask_feather > 0:
            mask_float = mask.astype(np.float32)
            if mask_float.max() > 1.0:
                mask_float = mask_float / 255.0
            
            kernel_size = mask_feather * 2 + 1
            kernel_size = max(1, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
            mask_float = cv2.GaussianBlur(mask_float, (kernel_size, kernel_size), 0)
            mask_float = np.clip(mask_float, 0.0, 1.0)
            
            if mask_float.ndim == 2:
                mask_float = np.repeat(mask_float[..., None], 3, axis=2)
            
            img_float = original_image.astype(np.float32)
            result_float = result_bgr.astype(np.float32)
            blended = result_float * mask_float + img_float * (1.0 - mask_float)
            result_bgr = np.clip(np.round(blended), 0, 255).astype(np.uint8)
        
        return result_bgr


def apply_tone_matching(obj: np.ndarray, roi: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """
    Tone matching in L*a*b* color space.
    Match the color tone of the object to the surrounding ROI.
    
    Args:
        obj: Object image (BGR, float32)
        roi: Background ROI image (BGR, float32)
        alpha: Alpha mask (float32, 0-1)
        
    Returns:
        Tone-matched image (BGR, float32)
    """
    import cv2
    
    # L*a*b* conversion
    obj_float = np.clip(obj, 0, 255).astype(np.uint8)
    roi_float = np.clip(roi, 0, 255).astype(np.uint8)
    
    lab_obj = cv2.cvtColor(obj_float, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_roi = cv2.cvtColor(roi_float, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    # Define mask region and ring region
    mask_binary = (alpha > 0.05).astype(np.uint8)
    if mask_binary.sum() == 0:
        return obj
    
    # Ring region (surrounding 5 pixels)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    dilated = cv2.dilate(mask_binary, kernel)
    eroded = cv2.erode(mask_binary, kernel)
    ring = np.clip(dilated - eroded, 0, 1)
    ring = np.where(mask_binary == 1, 0, ring).astype(bool)
    
    if ring.sum() == 0:
        ring = (1 - mask_binary).astype(bool)
    
    # Statistical matching for each channel
    for ch in range(3):
        obj_vals = lab_obj[:, :, ch][mask_binary.astype(bool)]
        roi_vals = lab_roi[:, :, ch][ring]
        
        if obj_vals.size == 0 or roi_vals.size == 0:
            continue
        
        obj_mean = float(obj_vals.mean())
        roi_mean = float(roi_vals.mean())
        obj_std = float(obj_vals.std() + 1e-4)
        roi_std = float(roi_vals.std() + 1e-4)
        
        # Statistics matching
        scale = roi_std / obj_std
        lab_obj[:, :, ch] = (lab_obj[:, :, ch] - obj_mean) * scale + roi_mean
    
    # BGR conversion
    matched = cv2.cvtColor(np.clip(lab_obj, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    return matched.astype(np.float32)


def inject_texture_noise(obj: np.ndarray, roi: np.ndarray, noise_scale: float = 0.25) -> np.ndarray:
    """
    Inject noise based on surrounding texture.
    
    Args:
        obj: Object image (BGR, float32)
        roi: Background ROI image (BGR, float32)
        noise_scale: Noise scale coefficient
        
    Returns:
        Noise-injected image (BGR, float32)
    """
    import cv2
    
    # Determine noise level from grayscale variance of ROI
    roi_gray = cv2.cvtColor(np.clip(roi, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    sigma = float(roi_gray.std()) * noise_scale
    
    if sigma < 1e-3:
        return obj
    
    # Generate Gaussian noise
    noise = np.random.normal(0.0, sigma, size=obj.shape).astype(np.float32)
    return np.clip(obj + noise, 0, 255)


def feather_alpha_advanced(alpha: np.ndarray, 
                           feather_distance: float = 8.0, 
                           gradient_power: float = 1.5) -> np.ndarray:
    """
    Advanced feathering technique.
    
    Create natural boundaries by combining distance transform and gradient mask.
    
    Args:
        alpha: Alpha mask (float32, 0-1)
        feather_distance: Feathering distance (pixels)
        gradient_power: Gradient strength
        
    Returns:
        Feathered alpha mask (float32, 0-1)
    """
    import cv2
    
    alpha_clean = np.clip(alpha, 0.0, 1.0)
    
    # Create binary mask
    binary_mask = (alpha_clean > 0.1).astype(np.uint8)
    if binary_mask.sum() == 0:
        return alpha_clean
    
    # Distance transform (inner)
    dist_inner = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    
    # Distance transform (outer)
    inverted = 1 - binary_mask
    dist_outer = cv2.distanceTransform(inverted, cv2.DIST_L2, 5)
    
    # Inner gradient (from center to boundary)
    inner_grad = np.clip(dist_inner / feather_distance, 0.0, 1.0)
    inner_grad = np.power(inner_grad, 1.0 / gradient_power)
    
    # Outer gradient (from boundary to outside)
    outer_grad = np.clip(1.0 - (dist_outer / feather_distance), 0.0, 1.0)
    outer_grad = np.power(outer_grad, gradient_power)
    
    # Combine
    combined = np.where(binary_mask > 0, inner_grad, outer_grad)
    feathered = alpha_clean * combined
    
    # Finish with light Gaussian blur
    ksize = max(3, int(feather_distance / 4) | 1)
    if ksize % 2 == 0:
        ksize += 1
    feathered = cv2.GaussianBlur(feathered, (ksize, ksize), 0)
    
    return feathered


def blend_object_advanced(img: np.ndarray, 
                          obj_bgr: np.ndarray, 
                          alpha_mask: np.ndarray, 
                          x: int, 
                          y: int, 
                          blend_mode: str = "poisson") -> np.ndarray:
    """
    Composite object image with advanced blending techniques.
    
    Uses techniques from defect_patch_compositor.py:
    - Tone matching (L*a*b* space)
    - Noise injection
    - Advanced feathering
    - Poisson blending
    
    Args:
        img: Base image (BGR, uint8)
        obj_bgr: Object image (BGR, uint8 or float32)
        alpha_mask: Alpha mask (float32, 0-1)
        x, y: Placement position
        blend_mode: "poisson" or "alpha"
        
    Returns:
        Composited image (BGR, uint8)
    """
    import cv2
    
    h, w = obj_bgr.shape[:2]
    roi = img[y:y+h, x:x+w].copy()
    
    # Convert to float32 (if necessary)
    if obj_bgr.dtype != np.float32:
        obj_bgr = obj_bgr.astype(np.float32)
    if roi.dtype != np.float32:
        roi = roi.astype(np.float32)
    
    # 1. Tone matching (L*a*b* space)
    obj_toned = apply_tone_matching(obj_bgr, roi, alpha_mask)
    
    # 2. Noise injection (blend with surrounding texture)
    obj_noised = inject_texture_noise(obj_toned, roi)
    
    # 3. Advanced feathering (natural boundaries)
    alpha_feathered = feather_alpha_advanced(alpha_mask)
    
    # 4. Blending
    img_result = img.copy()
    
    if blend_mode == "poisson":
        # Poisson blending (most natural)
        mask_uint8 = (alpha_feathered * 255).astype(np.uint8)
        try:
            obj_uint8 = np.clip(obj_noised, 0, 255).astype(np.uint8)
            blended_region = cv2.seamlessClone(obj_uint8, roi.astype(np.uint8), mask_uint8, 
                                               (w // 2, h // 2), cv2.MIXED_CLONE)
            img_result[y:y+h, x:x+w] = blended_region
        except Exception as e:
            # Fallback to alpha compositing if Poisson fails
            print(f"[blend_object_advanced] Poisson blend failed, using alpha: {e}")
            alpha_expanded = alpha_feathered[:, :, None]
            blended = roi * (1.0 - alpha_expanded) + obj_noised * alpha_expanded
            img_result[y:y+h, x:x+w] = np.clip(blended, 0, 255).astype(np.uint8)
    else:
        # Alpha compositing
        alpha_expanded = alpha_feathered[:, :, None]
        blended = roi * (1.0 - alpha_expanded) + obj_noised * alpha_expanded
        img_result[y:y+h, x:x+w] = np.clip(blended, 0, 255).astype(np.uint8)
    
    return img_result
