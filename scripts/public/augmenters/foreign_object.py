"""
Category 4: Foreign Object

Augmentation methods that simulate foreign objects intruding or adhering to the object's surroundings.

Classified by generation method (MECE):
1. ProceduralForeignObjectAug: Programmatic generation
   - Stains (oil spots, water drops, dust)
   - Simple shapes (circles, rectangles, lines, polygons)
   - UI elements (popups, labels, barcodes)
   
2. SDForeignObjectAug: Stable Diffusion generation
   - Physical objects (screws, bolts, coins, clips, etc.)
   - Textures (stains, spots, etc.)
"""

import cv2
import random
import math
import numpy as np
import string
import os
from pathlib import Path
from typing import Dict, Optional, Tuple
from scipy.fft import fft2, ifft2, fftshift, ifftshift

from .base import BaseAugmenter
from .sd_utils import SDPipelineManager, blend_object_advanced


def _add_margin_to_mask(mask: np.ndarray, h: int, w: int, margin_ratio: float = 0.05) -> np.ndarray:
    """
    Add margin (dilation) to mask.
    Returns original mask if background region disappears.
    """
    margin_size = int(min(h, w) * margin_ratio)
    if margin_size <= 0:
        return mask

    kernel = np.ones((margin_size, margin_size), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    
    if np.any(dilated_mask == 0):
        return dilated_mask
    
    return mask


def _clip_object_to_image(
    x: int, y: int, sz: int, w: int, h: int, 
    obj_bgr: np.ndarray, alpha_2d: np.ndarray
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int, int]:
    """
    Clip object that extends beyond image boundaries.
    
    Returns:
        (obj_bgr_cropped, alpha_2d_cropped, x_start, y_start)
        Cropped object image, alpha mask, and paste start position.
        Returns (None, None, 0, 0) if no paste region exists.
    """
    # Clipping when placement position extends beyond image
    x_start = max(0, x)
    y_start = max(0, y)
    x_end = min(w, x + sz)
    y_end = min(h, y + sz)
    
    # Paste region size
    paste_w = x_end - x_start
    paste_h = y_end - y_start
    
    if paste_w <= 0 or paste_h <= 0:
        return None, None, 0, 0
        
    # Crop object image and mask
    obj_x_start = x_start - x
    obj_y_start = y_start - y
    obj_x_end = obj_x_start + paste_w
    obj_y_end = obj_y_start + paste_h
    
    obj_bgr_cropped = obj_bgr[obj_y_start:obj_y_end, obj_x_start:obj_x_end]
    alpha_2d_cropped = alpha_2d[obj_y_start:obj_y_end, obj_x_start:obj_x_end]
    
    return obj_bgr_cropped, alpha_2d_cropped, x_start, y_start


def _get_placement_position(mask, w, h, sz, max_attempts=20):
    """
    Determine object placement position (common processing).
    Randomly select center point from effective_mask (mask),
    then check if overlap between foreign object and effective mask is 50% or more.
    Retry if not satisfied.
    
    Args:
        sz: int (square size) or tuple (width, height)
    
    Returns:
        (x, y): Placement position. Returns None if placement is not possible.
    """
    # mask == 0 is background (placeable region)
    bg_y, bg_x = np.where(mask == 0)
    
    if len(bg_x) == 0:
        return None
        
    if isinstance(sz, (tuple, list)):
        obj_w, obj_h = sz
    else:
        obj_w, obj_h = sz, sz
        
    for _ in range(max_attempts):
        idx = random.randint(0, len(bg_x) - 1)
        cx, cy = bg_x[idx], bg_y[idx]
        x = int(cx - obj_w // 2)
        y = int(cy - obj_h // 2)
        
        # Rectangle within image bounds
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w, x + obj_w)
        y2 = min(h, y + obj_h)
        
        if x2 <= x1 or y2 <= y1:
            continue
            
        # Get mask of placement region
        # mask: 0 is background, 1 is foreground (forbidden region)
        # Count number of 0s in placement region
        region_mask = mask[y1:y2, x1:x2]
        bg_pixels = np.sum(region_mask == 0)
        
        # Check ratio of background (valid region) to object area
        # overlap >= 50%
        if bg_pixels / (obj_w * obj_h) >= 0.5:
                return x, y
                
    return None


class SimpleShapeForeignObjectAug(BaseAugmenter):
    """
    Simple shapes (circles, rectangles, lines, polygons) placement
    """
    region = "background"
    
    def __init__(self, margin_ratio: float = 0.05, **kwargs):
        super().__init__(**kwargs)
        self.margin_ratio = margin_ratio

    def _random_aug(self, img, mask, **kw):
        h, w = img.shape[:2]
        effective_mask = _add_margin_to_mask(mask, h, w, margin_ratio=self.margin_ratio)
        
        # Create copy for drawing
        aug_img = img.copy()
        
        aug_img, metadata = self._draw_simple_shape(aug_img, effective_mask)
        
        # Mask processing: restore original image where effective_mask is 1 (object + margin)
        if effective_mask.ndim == 2:
            mask_expanded = effective_mask[:, :, None]
        else:
            mask_expanded = effective_mask
            
        final_img = np.where(mask_expanded == 1, img, aug_img)
        
        return final_img, metadata

    def _draw_simple_shape(self, img, mask):
        """Simple shapes (circles, rectangles, lines, polygons)"""
        h, w = img.shape[:2]
        
        shape_type = random.choice(["circle", "rectangle", "line", "polygon", "ellipse"])
        n_shapes = random.randint(1, 5)
        placed_shapes = []
        
        for _ in range(n_shapes):
            # Random color
            color = tuple([random.randint(0, 255) for _ in range(3)])
            thickness = random.choice([-1, 2, 3, 5])  # -1 is filled
            
            bg_y, bg_x = np.where(mask == 0)
            idx = random.randint(0, len(bg_x) - 1)
            cx, cy = bg_x[idx], bg_y[idx]
            
            if shape_type == "circle":
                radius = random.randint(int(0.03*min(h,w)), int(0.08*min(h,w)))
                cv2.circle(img, (cx, cy), radius, color, thickness)
                placed_shapes.append({"type": "circle", "center": (cx, cy), "radius": radius})
                
            elif shape_type == "rectangle":
                w_rect = random.randint(int(0.05*w), int(0.15*w))
                h_rect = random.randint(int(0.05*h), int(0.15*h))
                x1 = max(0, cx - w_rect // 2)
                y1 = max(0, cy - h_rect // 2)
                x2 = min(w, cx + w_rect // 2)
                y2 = min(h, cy + h_rect // 2)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
                placed_shapes.append({"type": "rectangle", "bbox": (x1, y1, x2, y2)})
                
            elif shape_type == "line":
                length = random.randint(int(0.05*w), int(0.15*w))
                angle = random.uniform(0, 2*np.pi)
                x1 = int(cx - length/2 * np.cos(angle))
                y1 = int(cy - length/2 * np.sin(angle))
                x2 = int(cx + length/2 * np.cos(angle))
                y2 = int(cy + length/2 * np.sin(angle))
                cv2.line(img, (x1, y1), (x2, y2), color, max(2, abs(thickness)))
                placed_shapes.append({"type": "line", "start": (x1, y1), "end": (x2, y2)})
                
            elif shape_type == "polygon":
                n_sides = random.randint(3, 8)
                radius = random.randint(int(0.03*min(h,w)), int(0.08*min(h,w)))
                pts = []
                for i in range(n_sides):
                    theta = 2*np.pi*i/n_sides + random.uniform(-0.2, 0.2)
                    r = radius * random.uniform(0.7, 1.3)
                    pts.append([int(cx+r*np.cos(theta)), int(cy+r*np.sin(theta))])
                pts = np.array([pts], dtype=np.int32)
                cv2.fillPoly(img, pts, color) if thickness == -1 else cv2.polylines(img, pts, True, color, max(2, thickness))
                placed_shapes.append({"type": "polygon", "center": (cx, cy), "n_sides": n_sides})
                
            elif shape_type == "ellipse":
                axes = (random.randint(int(0.03*w), int(0.08*w)), 
                       random.randint(int(0.02*h), int(0.06*h)))
                angle = random.randint(0, 180)
                cv2.ellipse(img, (cx, cy), axes, angle, 0, 360, color, thickness)
                placed_shapes.append({"type": "ellipse", "center": (cx, cy), "axes": axes})
        
        return img, {
            "foreign_type": "shape",
            "shape_type": shape_type,
            "n_shapes": n_shapes,
            "shapes": placed_shapes
        }


class ProceduralForeignObjectAug(BaseAugmenter):
    """
    Procedurally generated foreign object placement
    
    Place programmatically generated elements in background region:
    - Stains (oil spots, water drops, dust)
    - Simple shapes (circles, rectangles, lines, polygons)
    - UI elements (popups, labels, barcodes)
    """
    region = "background"
    
    def __init__(self, margin_ratio: float = 0.05, **kwargs):
        super().__init__(**kwargs)
        self.margin_ratio = margin_ratio

    def _random_aug(self, img, mask, **kw):
        h, w = img.shape[:2]
        
        # Draw only in background region
        effective_mask = _add_margin_to_mask(mask, h, w, margin_ratio=self.margin_ratio)
        
        # Create copy for drawing
        aug_img = img.copy()
        
        # Randomly select foreign object type
        foreign_type = random.choice([
            "ui_popup",   # UI popup
            "ui_label",   # UI label
            "ui_barcode"  # Barcode
        ])
        
        if foreign_type == "ui_popup":
            aug_img, metadata = self._draw_popup(aug_img, effective_mask)
        elif foreign_type == "ui_label":
            aug_img, metadata = self._draw_label(aug_img, effective_mask)
        elif foreign_type == "ui_barcode":
            aug_img, metadata = self._draw_barcode(aug_img, effective_mask)
        else:
            return img, {"note": "unknown_type"}
            
        # Mask processing: restore original image where effective_mask is 1 (object + margin)
        if effective_mask.ndim == 2:
            mask_expanded = effective_mask[:, :, None]
        else:
            mask_expanded = effective_mask
            
        final_img = np.where(mask_expanded == 1, img, aug_img)
        
        return final_img, metadata
    
    def _draw_stain(self, img, mask):
        """Stains (oil spots, water drops, dust, etc.)"""
        h, w = img.shape[:2]
        
        # Randomly select stain type
        stain_type = random.choice(["oil", "water", "dust"])
        n_blob = random.randint(3, 15)
        placed_blobs = []
        
        for _ in range(n_blob):
            pts = []
            rad = random.randint(int(0.03*w), int(0.08*w))
            
            bg_y, bg_x = np.where(mask == 0)
            max_attempts = 20
            for _ in range(max_attempts):
                idx = random.randint(0, len(bg_x) - 1)
                cx, cy = bg_x[idx], bg_y[idx]
                if (cx - rad >= 0 and cx + rad < w and 
                    cy - rad >= 0 and cy + rad < h):
                    break
            else:
                cx, cy = random.randint(rad, w-rad), random.randint(rad, h-rad)
            
            n_vert = random.randint(5, 10)
            for i in range(n_vert):
                theta = 2*np.pi*i/n_vert + random.uniform(-0.1, 0.1)
                r = rad * random.uniform(0.6, 1.2)
                pts.append([int(cx+r*np.cos(theta)), int(cy+r*np.sin(theta))])
            pts = np.array([pts], dtype=np.int32)
            
            # Change color based on stain type
            if stain_type == "oil":
                color = (random.randint(30,80), random.randint(40,90), random.randint(80,160))  # Brownish
            elif stain_type == "water":
                color = (random.randint(100,150), random.randint(100,150), random.randint(90,130))  # Grayish
            else:  # dust
                color = (random.randint(80,120), random.randint(80,120), random.randint(70,110))  # Gray to brownish
            
            cv2.fillPoly(img, pts, color)
            placed_blobs.append({"center": (cx, cy), "radius": rad, "vertices": n_vert})
        
        # Blur
        k = random.randrange(3, 11, 2)
        img = cv2.GaussianBlur(img, (k,k), 0)
        
        return img, {
            "foreign_type": "stain",
            "stain_type": stain_type,
            "n_blob": n_blob,
            "blur_k": k,
            "blobs": placed_blobs
        }
    
    def _draw_popup(self, img, mask):
        """UI Popup"""
        h, w = img.shape[:2]
        
        # Resize loop
        for _ in range(5):
            # Random size (width: 10-20%, height: 8-15%)
            pw = int(w * random.uniform(0.10, 0.20))
            ph = int(h * random.uniform(0.08, 0.15))
            
            # Placement position
            pos = _get_placement_position(mask, w, h, (pw, ph))
            if pos is not None:
                px, py = pos
                break
        else:
            return img, {"note": "failed_to_place_popup"}
        
        out = img.copy()
        
        # Shadow (black)
        cv2.rectangle(out, (px + 3, py + 3),
                      (px + pw + 3, py + ph + 3), (0, 0, 0), -1)
        # White background
        cv2.rectangle(out, (px, py),
                      (px + pw, py + ph), (255, 255, 255), -1)
        # Gray border
        cv2.rectangle(out, (px, py),
                      (px + pw, py + ph), (180, 180, 180), 2)
        
        # Random text
        msg = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
        cv2.putText(out, msg,
                    (px + 10, py + int(ph * 0.6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return out, {
            "foreign_type": "ui_popup",
            "msg": msg,
            "pos": (px, py),
            "size": (pw, ph)
        }
    
    def _draw_label(self, img, mask):
        """Warning label / sticker"""
        h, w = img.shape[:2]
        
        # Resize loop
        for _ in range(5):
            # Random size (width: 8-18%, height: 5-10%)
            lw = int(w * random.uniform(0.08, 0.18))
            lh = int(h * random.uniform(0.05, 0.10))
            
            # Placement position
            pos = _get_placement_position(mask, w, h, (lw, lh))
            if pos is not None:
                lx, ly = pos
                break
        else:
            return img, {"note": "failed_to_place_label"}
        
        out = img.copy()
        
        # Yellow warning label
        cv2.rectangle(out, (lx, ly),
                      (lx + lw, ly + lh), (0, 220, 255), -1)
        cv2.rectangle(out, (lx, ly),
                      (lx + lw, ly + lh), (0, 0, 0), 2)
        
        # Warning text
        label_text = random.choice(["WARNING", "CAUTION", "DANGER", "NOTICE"])
        cv2.putText(out, label_text,
                    (lx + 5, ly + int(lh * 0.7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return out, {
            "foreign_type": "ui_label",
            "text": label_text,
            "pos": (lx, ly),
            "size": (lw, lh)
        }
    
    def _draw_barcode(self, img, mask):
        """Barcode / QR code style"""
        h, w = img.shape[:2]
        
        # Resize loop
        for _ in range(5):
            # Random size (width: 8-15%, height: 4-8%)
            bw = int(w * random.uniform(0.08, 0.15))
            bh = int(h * random.uniform(0.04, 0.08))
            
            # Placement position
            pos = _get_placement_position(mask, w, h, (bw, bh))
            if pos is not None:
                bx, by = pos
                break
        else:
            return img, {"note": "failed_to_place_barcode"}
    
        out = img.copy()
        
        # White background
        cv2.rectangle(out, (bx, by),
                      (bx + bw, by + bh), (255, 255, 255), -1)
        
        # Barcode-style vertical lines
        n_bars = random.randint(15, 25)
        for i in range(n_bars):
            bar_x = bx + int(i * bw / n_bars)
            bar_width = max(1, int(bw / n_bars * random.uniform(0.3, 0.7)))
            if random.random() > 0.3:  # 70% chance of black line
                cv2.rectangle(out, (bar_x, by),
                              (bar_x + bar_width, by + bh), (0, 0, 0), -1)
        
        return out, {
            "foreign_type": "ui_barcode",
            "pos": (bx, by),
            "size": (bw, bh),
            "n_bars": n_bars
        }


class TextureForeignObjectAug(BaseAugmenter):
    """
    Fourier space texture synthesis for background region.
    Generates texture anomalies (stains, discoloration, texture changes, etc.) in part of the background.
    """
    region = "background"

    def __init__(self, margin_ratio: float = 0.05, **kwargs):
        super().__init__(**kwargs)
        self.margin_ratio = margin_ratio

    def _random_aug(self, img, mask, **kw):
        h, w = img.shape[:2]
        
        # Draw only in background region
        effective_mask = _add_margin_to_mask(mask, h, w, margin_ratio=self.margin_ratio)
        
        # Select patch center from background region (effective_mask == 0)
        bg_y, bg_x = np.where(effective_mask == 0)
        if len(bg_x) == 0:
            # Do nothing if no background
            return img, {"note": "no_background"}
            
        # Patch size is 10-20% of diagonal
        size_ratio = random.uniform(0.1, 0.20)
        pw = int(w * size_ratio)
        ph = int(h * size_ratio)
        
        # Patch center point
        idx = random.randint(0, len(bg_x) - 1)
        cx, cy = bg_x[idx], bg_y[idx]
            
        x1 = np.clip(cx - pw // 2, 0, w - pw)
        y1 = np.clip(cy - ph // 2, 0, h - ph)

        patch = img[y1:y1 + ph, x1:x1 + pw].copy()

        # ---- Noise generation (mid-frequency) ----
        noise = np.random.normal(128, 30, patch.shape).astype(np.float32)
        k = random.choice([7, 11, 15])
        noise = cv2.GaussianBlur(noise, (k, k), 0)
        noise = cv2.normalize(noise, None, 0, 255, cv2.NORM_MINMAX)

        # ---- Spectral mixing ----
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

        # ---- Simple paste (No Blending) ----
        aug_img = img.copy()
        aug_img[y1:y1+ph, x1:x1+pw] = out_patch

        # Mask processing: restore original image where effective_mask is 1 (object + margin)
        if effective_mask.ndim == 2:
            mask_expanded = effective_mask[:, :, None]
        else:
            mask_expanded = effective_mask
            
        final_img = np.where(mask_expanded == 1, img, aug_img)

        return final_img, {
            "foreign_type": "texture",
            "size_ratio": round(size_ratio, 3),
            "alpha_amp": round(mix_alpha, 2),
            "blur_k": k,
            "pos": (x1, y1)
        }


class SDForeignObjectAug(BaseAugmenter):
    """
    Stable Diffusion generated foreign object placement
    
    Place Stable Diffusion generated images in background region:
    - Physical objects (screws, bolts, coins, clips, etc.)
    - Textures (stains, spots, etc.)
    
    Usage:
    1. Load from directory: Place PNG/JPG images in OBJ_DIR
    2. On-the-fly SD generation: Run with use_sd_generator=True
    """
    region = "background"
    OBJ_DIR = "./objects"  # Directory for SD generated images
    
    def __init__(self, 
                 model_preset="sd15",
                 model_id=None,
                 device="auto",
                 fp16=True,
                 prompt_csv=None,
                 margin_ratio=0.05,
                 min_objects=1,
                 max_objects=3,
                 debug_save_dir=None):
        """
        Args:
            model_preset: SD model preset ("sd15", "sd2", "sdxl")
            model_id: Custom model ID
            device: Device ("auto", "cuda", "cpu")
            fp16: FP16 usage flag
            prompt_csv: Prompt CSV path (default: config/prompts_object.csv)
            margin_ratio: Mask margin ratio
            min_objects: Minimum number of objects to generate
            max_objects: Maximum number of objects to generate
            debug_save_dir: Debug image save directory
        """
        # Use object generation prompt CSV by default
        if prompt_csv is None:
            # Convert to absolute path from project root
            project_root = Path(__file__).resolve().parent.parent.parent.parent
            prompt_csv = str(project_root / "config" / "prompts_object.csv")
        
        # SD Pipeline Manager is lazily initialized (only when use_sd_generator=True)
        self.sd_manager = None
        self.model_preset = model_preset
        self.model_id = model_id
        self.device = device
        self.fp16 = fp16
        self.prompt_csv = prompt_csv
        self.margin_ratio = margin_ratio
        self.min_objects = min_objects
        self.max_objects = max_objects
        
        if debug_save_dir:
            self.debug_save_dir = Path(debug_save_dir)
            self.debug_save_dir.mkdir(parents=True, exist_ok=True)
            print(f"[SDForeignObjectAug] Initialized with debug_save_dir={debug_save_dir}")
        else:
            self.debug_save_dir = None
    
    def _get_sd_manager(self):
        """Lazy initialization of SD Pipeline Manager"""
        if self.sd_manager is None:
            self.sd_manager = SDPipelineManager(
                model_preset=self.model_preset,
                model_id=self.model_id,
                pipeline_type="img2img",  # Use img2img for object generation
                device=self.device,
                fp16=self.fp16,
                prompt_csv=self.prompt_csv
            )
        return self.sd_manager
    
    def _random_aug(self, img, mask, **kw):
        h, w = img.shape[:2]
        
        # Place objects only in background region
        effective_mask = _add_margin_to_mask(mask, h, w, margin_ratio=self.margin_ratio)
        
        n_obj = random.randint(self.min_objects, self.max_objects)
        
        # Select SD generation method
        use_sd_generator = kw.get("use_sd_generator", False)
        
        if use_sd_generator:
            # On-the-fly SD generation
            aug_img, metadata = self._generate_with_sd(img, effective_mask, n_obj, **kw)
        else:
            # Load from directory
            aug_img, metadata = self._load_from_directory(img, effective_mask, n_obj)
            
        # Mask processing: restore original image where effective_mask is 1 (object + margin)
        if effective_mask.ndim == 2:
            mask_expanded = effective_mask[:, :, None]
        else:
            mask_expanded = effective_mask
            
        final_img = np.where(mask_expanded == 1, img, aug_img)
        
        # --- Debug save ---
        if self.debug_save_dir:
            vis_img = final_img.copy()
            
            # Draw rectangles at placed object positions
            if "objects" in metadata:
                for obj in metadata["objects"]:
                    if "pos" in obj and "size" in obj:
                        x, y = obj["pos"]
                        sz = obj["size"]
                        # Handle both tuple and integer sizes
                        if isinstance(sz, (list, tuple)):
                            w_obj, h_obj = sz
                        else:
                            w_obj, h_obj = sz, sz
                            
                        cv2.rectangle(vis_img, (x, y), (x + w_obj, y + h_obj), (0, 0, 255), 2)
            
            debug_file_name = kw.get("debug_file_name")
            if debug_file_name:
                save_path = self.debug_save_dir / debug_file_name
            else:
                import time
                timestamp = str(int(time.time() * 1000))
                save_path = self.debug_save_dir / f"foreign_{timestamp}.png"
                
            cv2.imwrite(str(save_path), vis_img)
            # print(f"[SDForeignObjectAug] Saved debug image to {save_path}")

        return final_img, metadata
    
    def _load_from_directory(self, img, mask, n_obj):
        """Load PNG/JPG images from directory and place them"""
        h, w = img.shape[:2]
        out = img.copy()
        placed_objects = []
        
        # Search for SD generated images
        pngs = []
        if os.path.isdir(self.OBJ_DIR):
            pngs = [p for p in os.listdir(self.OBJ_DIR)
                    if p.lower().endswith((".png", ".jpg", ".jpeg"))]
        
        if not pngs:
            print(f"[ForeignObjectAug] Warning: No images found in {self.OBJ_DIR}")
            return img, {"method": "load_from_directory", "n_obj": 0, "objects": []}
        
        for _ in range(n_obj):
            # Retry loop with different sizes
            valid_placement = False
            x, y, sz = 0, 0, 0
            
            for _ in range(5): # Try 5 times with different sizes
                sz = int(random.uniform(0.08, 0.18) * min(h, w))
                
                # Determine placement position
                pos = _get_placement_position(mask, w, h, sz)
                if pos is not None:
                    x, y = pos
                    valid_placement = True
                    break
            
            if not valid_placement:
                continue

            # Load SD image
            obj_path = os.path.join(self.OBJ_DIR, random.choice(pngs))
            obj = cv2.imread(obj_path, cv2.IMREAD_UNCHANGED)
            if obj is None:
                continue
            
            obj = cv2.resize(obj, (sz, sz))
            
            # Alpha channel processing
            if obj.shape[2] == 4:
                alpha_mask = obj[:, :, 3:] / 255.0
                obj_bgr = obj[:, :, :3]
            else:
                alpha_mask = np.ones((sz, sz, 1), dtype=np.float32)
                obj_bgr = obj
            
            # Blend (using sd_utils common function, Poisson blending)
            alpha_2d = alpha_mask.squeeze() if alpha_mask.ndim == 3 else alpha_mask
            
            # Clipping process
            obj_bgr_cropped, alpha_2d_cropped, x_start, y_start = _clip_object_to_image(
                x, y, sz, w, h, obj_bgr, alpha_2d
            )
            
            if obj_bgr_cropped is None:
                continue
            
            out = blend_object_advanced(out, obj_bgr_cropped, alpha_2d_cropped, x_start, y_start, blend_mode="poisson")
            
            placed_objects.append({
                "pos": (x, y),
                "size": sz,
                "type": "loaded_object",
                "file": os.path.basename(obj_path),
                "blend_mode": "poisson"
            })
        
        return out, {
            "method": "load_from_directory",
            "n_obj": len(placed_objects),
            "objects": placed_objects
        }
    
    def _generate_with_sd(self, img, mask, n_obj, **kw):
        """
        On-the-fly generation of object images using Stable Diffusion and paste onto background.
        Uses sd_utils.SDPipelineManager.
        """
        from PIL import Image
        import torch
        
        # Get SD Pipeline Manager (lazy initialization)
        sd_mgr = self._get_sd_manager()
        pipe = sd_mgr.get_pipeline()
        
        h, w = img.shape[:2]
        out = img.copy()
        placed_objects = []
        
        # Get prompt
        sd_prompt = kw.get("sd_prompt") or sd_mgr.get_random_prompt()
        
        for i in range(n_obj):
            # Retry loop with different sizes
            valid_placement = False
            x, y, sz = 0, 0, 0
            
            for _ in range(5): # Try 5 times with different sizes
                sz = int(random.uniform(0.08, 0.18) * min(h, w))
                
                # Determine placement position
                pos = _get_placement_position(mask, w, h, sz)
                if pos is not None:
                    x, y = pos
                    valid_placement = True
                    break
            
            if not valid_placement:
                continue

            # Create generator
            seed = kw.get("seed")
            generator = sd_mgr.create_generator(seed)
            
            # SD generation (small object image with white background)
            with torch.autocast(sd_mgr.device, enabled=sd_mgr.device=="cuda"):
                gen_img = pipe(
                    prompt=sd_prompt,
                    num_inference_steps=kw.get("num_inference_steps", 25),
                    guidance_scale=kw.get("guidance_scale", 7.0),
                    generator=generator
                ).images[0]
            
            # Resize
            gen_img = gen_img.resize((sz, sz), Image.LANCZOS)
            gen_np = np.array(gen_img)
            
            # RGB -> BGR conversion
            if gen_np.shape[2] == 4:
                alpha_mask = gen_np[:, :, 3:] / 255.0
                obj_bgr = cv2.cvtColor(gen_np[:, :, :3], cv2.COLOR_RGB2BGR)
            else:
                alpha_mask = np.ones((sz, sz, 1), dtype=np.float32)
                obj_bgr = cv2.cvtColor(gen_np, cv2.COLOR_RGB2BGR)
            
            # Blend (using sd_utils common function, Poisson blending)
            alpha_2d = alpha_mask.squeeze() if alpha_mask.ndim == 3 else alpha_mask
            
            # Clipping process
            obj_bgr_cropped, alpha_2d_cropped, x_start, y_start = _clip_object_to_image(
                x, y, sz, w, h, obj_bgr, alpha_2d
            )
            
            if obj_bgr_cropped is None:
                continue
            
            out = blend_object_advanced(out, obj_bgr_cropped, alpha_2d_cropped, x_start, y_start, blend_mode="poisson")
            
            placed_objects.append({
                "pos": (x, y),
                "size": sz,
                "type": "sd_generated",
                "prompt": sd_prompt,
                "seed": generator.initial_seed(),
                "blend_mode": "poisson"
            })
        
        return out, {
            "method": "sd_on_the_fly",
            "n_obj": len(placed_objects),
            "objects": placed_objects,
            "model": sd_mgr.model_label
        }
