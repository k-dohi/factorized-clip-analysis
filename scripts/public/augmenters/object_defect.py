"""
Category 1: Object Defect

Augmentation methods that simulate physical defects and damage on objects.
"""

import cv2
import random
import math
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from PIL import Image

from .base import BaseAugmenter
from .sd_utils import SDPipelineManager


class CutPasteAug(BaseAugmenter):
    """Cut a part of the object and paste it to another location"""
    region = "foreground"

    def _random_aug(self, img, mask, **kw):
        h, w = img.shape[:2]
        
        # Modified to work only within mask region
        if mask is not None:
            fg_y, fg_x = np.where(mask == 1)
            if len(fg_x) == 0:
                return img, {"note": "no_foreground_mask"}
        
        area_ratio = random.uniform(0.005, 0.25)          # 0.5% to 25%
        side = int(math.sqrt(area_ratio * h * w))
        
        # Patch extraction position (optimized: random selection)
        if mask is not None and len(fg_x) > 0:
            # Offset from randomly selected foreground pixel
            idx = random.randint(0, len(fg_x) - 1)
            cx, cy = fg_x[idx], fg_y[idx]
            x1 = np.clip(cx - side//2, 0, w - side)
            y1 = np.clip(cy - side//2, 0, h - side)
        else:
            x1 = random.randint(0, max(0, w - side - 1))
            y1 = random.randint(0, max(0, h - side - 1))
            
        patch = img[y1:y1+side, x1:x1+side].copy()

        rot = random.uniform(0, 359)
        M   = cv2.getRotationMatrix2D((side/2, side/2), rot, 1)
        patch = cv2.warpAffine(patch, M, (side, side),
                               borderMode=cv2.BORDER_REFLECT_101)

        # Paste position (optimized: random selection)
        if mask is not None and len(fg_x) > 0:
            idx = random.randint(0, len(fg_x) - 1)
            cx, cy = fg_x[idx], fg_y[idx]
            x2 = np.clip(cx - side//2, 0, w - side)
            y2 = np.clip(cy - side//2, 0, h - side)
        else:
            x2 = random.randint(0, max(0, w - side - 1))
            y2 = random.randint(0, max(0, h - side - 1))
            
        feather = random.uniform(0., 0.4)
        roi = img[y2:y2+side, x2:x2+side]
        img[y2:y2+side, x2:x2+side] = cv2.addWeighted(
            patch, 1-feather, roi, feather, 0)

        return img, {"area_ratio": area_ratio,
                     "rotation": rot,
                     "feather": feather,
                     "patch_pos": (x1, y1),
                     "paste_pos": (x2, y2)}


class ScratchMixAug(BaseAugmenter):
    """Simulation of scratch marks"""
    region = "foreground"

    def _random_aug(self, img, mask, **kw):
        h, w = img.shape[:2]
        
        # Modified to draw lines only within mask region
        if mask is not None:
            fg_y, fg_x = np.where(mask == 1)
            if len(fg_x) == 0:
                return img, {"note": "no_foreground_mask"}
        
        n_line = random.randint(8, 60)
        max_len = int(0.3 * max(h, w))
        width   = random.randint(1, 4)
        drawn_lines = []
        
        # Draw all lines at once (optimized)
        for _ in range(n_line):
            if mask is not None and len(fg_x) > 0:
                # Randomly select foreground pixel
                idx = random.randint(0, len(fg_x) - 1)
                x1, y1 = fg_x[idx], fg_y[idx]
            else:
                x1, y1 = random.randint(0, w-1), random.randint(0, h-1)
            
            length = random.randint(20, max_len)
            angle  = random.uniform(0, 2*math.pi)
            x2 = int(np.clip(x1 + length*math.cos(angle), 0, w-1))
            y2 = int(np.clip(y1 + length*math.sin(angle), 0, h-1))
            
            color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
            cv2.line(img, (x1,y1), (x2,y2), color, width, cv2.LINE_AA)
            drawn_lines.append({"start": (x1,y1), "end": (x2,y2), "color": color})
        
        return img, {"n_line": n_line, "width": width, "lines": drawn_lines}


class FourierTextureAug(BaseAugmenter):
    """
    Fourier space texture synthesis.
    Mixes the amplitude spectrum of a foreground patch with random noise
    to create pseudo-anomalies that only change the texture. No additional materials needed.
    """
    region = "foreground"

    def _random_aug(self, img, mask, **kw):
        h, w = img.shape[:2]

        # ---- 1. Determine patch range within foreground mask ----
        if mask is not None:
            ys, xs = np.where(mask == 1)
            if len(xs) == 0:
                return img, {"note": "no_fg"}
        else:
            ys, xs = np.arange(h), np.arange(w)
            
        # Patch size is 10-25% of diagonal
        size_ratio = random.uniform(0.1, 0.25)
        pw = int(w * size_ratio)
        ph = int(h * size_ratio)
        
        # Patch center point (optimized: direct selection)
        if len(xs) > 0:
            idx = random.randint(0, len(xs) - 1)
            cx, cy = xs[idx], ys[idx]
        else:
            cx, cy = w//2, h//2
            
        x1 = np.clip(cx - pw // 2, 0, w - pw)
        y1 = np.clip(cy - ph // 2, 0, h - ph)

        patch = img[y1:y1 + ph, x1:x1 + pw].copy()

        # ---- 2. Noise generation (mid-frequency) ----
        noise = np.random.normal(128, 30, patch.shape).astype(np.float32)
        k = random.choice([7, 11, 15])
        noise = cv2.GaussianBlur(noise, (k, k), 0)
        noise = cv2.normalize(noise, None, 0, 255, cv2.NORM_MINMAX)

        # ---- 3. Spectral mixing ----
        # FFT (each channel independently)
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

        # ---- 4. PoissonClone for boundary blending ----
        center = (x1 + pw // 2, y1 + ph // 2)
        mask_roi = 255 * np.ones((ph, pw), np.uint8)
        img_clone = cv2.seamlessClone(out_patch, img, mask_roi,
                                      center, cv2.NORMAL_CLONE)

        return img_clone, {
            "size_ratio": round(size_ratio, 3),
            "alpha_amp": round(mix_alpha, 2),
            "blur_k": k
        }


class SDInpaintAug(BaseAugmenter):
    """
    Stable Diffusion Inpaint.
    Generate anomalies by inpainting foreground patches with Stable Diffusion.
    
    Common implementation using sd_utils.SDPipelineManager.
    """
    region = "foreground"

    def __init__(self,
                 model_preset="sd15",
                 model_id=None,
                 device="auto",
                 fp16=True,
                 prompt_csv=None,
                 debug_save_dir=None):
        
        # SD Pipeline Manager initialization
        # Keep prompt_csv for backward compatibility, but basically load from directory
        self.sd_manager = SDPipelineManager(
            model_preset=model_preset,
            model_id=model_id,
            pipeline_type="inpaint",
            device=device,
            fp16=fp16,
            prompt_csv=prompt_csv
        )
        
        # Load object-specific prompts
        self.prompts_by_category = self._load_category_prompts()
        
        if debug_save_dir:
            self.debug_save_dir = Path(debug_save_dir)
            self.debug_save_dir.mkdir(parents=True, exist_ok=True)
            print(f"[SDInpaintAug] Initialized with debug_save_dir={debug_save_dir}")
        else:
            self.debug_save_dir = None

    def _load_category_prompts(self) -> Dict[str, List[str]]:
        """Load prompts from CSV files under config/prompts/inpaint/"""
        prompts = {}
        
        # Path resolution from project root
        # scripts/public/augmenters/ -> project root
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        prompt_dir = project_root / "config" / "prompts" / "inpaint"
        
        if not prompt_dir.exists():
            print(f"[SDInpaintAug] Warning: Prompt directory not found: {prompt_dir}")
            return {}
            
        for csv_file in prompt_dir.glob("*.csv"):
            category = csv_file.stem  # metal, wood, etc.
            loaded = self.sd_manager._load_prompts(str(csv_file))
            if loaded:
                prompts[category] = loaded
                print(f"[SDInpaintAug] Loaded {len(loaded)} prompts for '{category}'")
        
        return prompts

    def _get_prompt_for_object(self, obj_name: str) -> str:
        """Select appropriate prompt based on object name"""
        
        # Use corresponding prompt if available for the object name
        if obj_name in self.prompts_by_category:
            return random.choice(self.prompts_by_category[obj_name])
        
        # If not found, use default (SDPipelineManager's default)
        # This is normal behavior, so no warning is issued
        return self.sd_manager.get_random_prompt()

    def _generate_irregular_mask(self, h, w, cx, cy, rad):
        """Generate mask with irregular shape"""
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Create irregular shape using random walk or polygon
        n_pts = random.randint(5, 12)
        pts = []
        for i in range(n_pts):
            theta = 2 * np.pi * i / n_pts
            # Randomly vary the radius
            r = rad * random.uniform(0.6, 1.4)
            x = int(cx + r * np.cos(theta))
            y = int(cy + r * np.sin(theta))
            pts.append([x, y])
        
        pts = np.array([pts], dtype=np.int32)
        cv2.fillPoly(mask, pts, 255)
        
        # Add noise to roughen the boundary
        noise = np.random.randint(0, 2, (h, w), dtype=np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_dilated = cv2.dilate(mask, kernel, iterations=2)
        mask_eroded = cv2.erode(mask, kernel, iterations=2)
        
        # Apply noise to edge region
        edge = cv2.bitwise_xor(mask_dilated, mask_eroded)
        noisy_edge = cv2.bitwise_and(edge, noise)
        
        final_mask = cv2.bitwise_or(mask_eroded, noisy_edge)
        
        # Apply closing to prevent holes
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
        
        return final_mask

    def _random_aug(self, img, mask, **kw):
        h, w = img.shape[:2]
        obj_name = kw.get("obj_name", "unknown")

        # ------- 1. Determine defect position within foreground mask --------
        if mask is not None:
            fg_y, fg_x = np.where(mask == 1)
            if len(fg_x) == 0:
                return img, {"note": "no_foreground_mask"}
        else:
            fg_y, fg_x = np.arange(h), np.arange(w)

        rad = random.randint(int(0.10*min(h,w)), int(0.18*min(h,w)))
        
        # Place center point within foreground mask
        if mask is not None and len(fg_x) > 0:
            max_attempts = 50
            valid_center = False
            for _ in range(max_attempts):
                idx = random.randint(0, len(fg_x) - 1)
                cx, cy = fg_x[idx], fg_y[idx]
                
                if (cx - rad >= 0 and cx + rad < w and 
                    cy - rad >= 0 and cy + rad < h):
                    # Simple check: is more than 70% of circular area in foreground
                    circle_mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.circle(circle_mask, (cx, cy), rad, 255, -1)
                    circle_fg = np.logical_and(circle_mask == 255, mask == 1)
                    if np.sum(circle_fg) > 0.7 * np.sum(circle_mask == 255):
                        valid_center = True
                        break
            
            if not valid_center:
                idx = random.randint(0, len(fg_x) - 1)
                cx, cy = fg_x[idx], fg_y[idx]
                rad = min(rad, cx, w-cx-1, cy, h-cy-1)
                rad = max(rad, 5)
        else:
            cx, cy = random.randint(rad, w-rad-1), random.randint(rad, h-rad-1)

        # ------- 2. Create irregular defect mask --------
        hole = self._generate_irregular_mask(h, w, cx, cy, rad)

        # ------- 3. Crop processing (for resolution preservation) --------
        # Crop rectangle containing the defect area (with margin)
        crop_margin = int(rad * 1.5)
        x1 = max(0, cx - rad - crop_margin)
        y1 = max(0, cy - rad - crop_margin)
        x2 = min(w, cx + rad + crop_margin)
        y2 = min(h, cy + rad + crop_margin)
        
        crop_w = x2 - x1
        crop_h = y2 - y1
        
        # Skip if crop area is too small
        if crop_w < 64 or crop_h < 64:
             return img, {"note": "crop_too_small"}

        img_crop = img[y1:y2, x1:x2].copy()
        hole_crop = hole[y1:y2, x1:x2].copy()

        # ------- 4. SD preprocessing (using sd_utils) --------
        # Resize cropped image to 512x512 for Inpaint
        pil_img, pil_mask, original_crop_size = self.sd_manager.prepare_image_and_mask(
            img_crop, hole_crop, target_size=512
        )

        # ------- 5. Prompt selection and parameter retrieval --------
        prompt = self._get_prompt_for_object(obj_name)
        guidance_scale = kw.get("guidance_scale", self.sd_manager.default_guidance)
        num_inference_steps = kw.get("num_inference_steps", self.sd_manager.default_steps)
        seed = kw.get("seed")
        generator = self.sd_manager.create_generator(seed)

        # ------- 6. Execute SD Inpaint --------
        pipe = self.sd_manager.get_pipeline()
        with torch.autocast(self.sd_manager.device, enabled=self.sd_manager.device=="cuda"):
            result = pipe(
                prompt=prompt,
                negative_prompt="blurry, low quality, distorted, deformed, text, watermark",
                image=pil_img,
                mask_image=pil_mask,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                strength=1.0,
                generator=generator
            ).images[0]

        # ------- 7. Post-processing (using sd_utils) --------
        # Write back to crop area (with feathering enabled)
        out_crop = self.sd_manager.postprocess_result(
            result, original_crop_size, img_crop, mask=hole_crop, mask_feather=5
        )
        
        # Restore to full image
        out_full = img.copy()
        out_full[y1:y2, x1:x2] = out_crop

        # ------- 8. Debug save --------
        if self.debug_save_dir:
            # Mask visualization (red overlay)
            vis_img = img.copy()
            # Paint mask region in red (BGR: 0, 0, 255)
            vis_img[hole == 255] = (vis_img[hole == 255] * 0.3 + np.array([0, 0, 255]) * 0.7).astype(np.uint8)
            
            # Draw crop area rectangle (green)
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            debug_file_name = kw.get("debug_file_name")
            if debug_file_name:
                save_path = self.debug_save_dir / debug_file_name
            else:
                import time
                timestamp = str(int(time.time() * 1000))
                save_path = self.debug_save_dir / f"inpaint_{obj_name}_{timestamp}.png"
                
            cv2.imwrite(str(save_path), vis_img)
            # print(f"[SDInpaintAug] Saved debug image to {save_path}")

        return out_full, {
            "radius": rad,
            "center": (cx, cy),
            "prompt": prompt,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "seed": generator.initial_seed(),
            "model": self.sd_manager.model_label,
            "crop_rect": (x1, y1, x2, y2)
        }

