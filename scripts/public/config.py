"""
Common configuration for paper experiments.

All experiments share:
- TEMPERATURE = 0.07 (softmax temperature for anomaly score)
- N_SAMPLES = 100 (number of test normal images to sample)
- RANDOM_SEED = 42 (reproducibility)

Paper model configuration:
- Experiment 1 (Table I-III): All 7 CLIP models
- Experiment 2+ (Table IV-VI): ViT-H-14 only
"""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets" / "mvtec"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "public_experiments"
LATEX_FIG_DIR = PROJECT_ROOT / "latex" / "figures"

# Experiment parameters
TEMPERATURE = 0.07  # Softmax temperature for s_CLIP (eq. clip_score in paper)
N_SAMPLES = 100     # Number of test normal images to sample (with replacement)
RANDOM_SEED = 42    # For reproducibility

# MVTec AD object categories (excluding texture categories)
OBJECT_CLASSES = [
    'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut',
    'pill', 'screw', 'toothbrush', 'transistor', 'zipper'
]

# ============================================================
# CLIP Model Configurations (Paper Table I-III)
# ============================================================

# All 7 models used in paper (for Experiment 1)
CLIP_MODELS = [
    # OpenAI CLIP (original)
    {'name': 'ViT-B-32', 'pretrained': 'openai'},
    {'name': 'ViT-B-16', 'pretrained': 'openai'},
    {'name': 'ViT-L-14', 'pretrained': 'openai'},
    # OpenCLIP LAION-400M
    {'name': 'ViT-B-16', 'pretrained': 'laion400m_e32'},
    # OpenCLIP LAION-2B
    {'name': 'ViT-B-32', 'pretrained': 'laion2b_s34b_b79k'},
    {'name': 'ViT-L-14', 'pretrained': 'laion2b_s32b_b82k'},
    {'name': 'ViT-H-14', 'pretrained': 'laion2b_s32b_b79k'},  # Best model
]

# Default model (ViT-H-14 for Experiment 2+)
DEFAULT_MODEL = 'ViT-H-14'
DEFAULT_PRETRAINED = 'laion2b_s32b_b79k'

# ============================================================
# Prompt Templates (Paper Table V - Prompt Ablation)
# ============================================================

# Anomaly prompt variations for ablation study
ANOMALY_PROMPTS = {
    'damaged': "a photo of a damaged {obj}",
    'surface_damage': "a photo of a {obj} with damage on its surface",
    'defective': "a photo of a defective {obj}",
    'with_defect': "a photo of a {obj} with a defect",
    'anomalous': "a photo of an anomalous {obj}",
    'flawed': "a photo of a flawed {obj}",
}

# Normal prompt (fixed across all ablations)
NORMAL_PROMPT = "a photo of a {obj}"

# ============================================================
# Variation Taxonomy (5 Categories, 17 Types)
# ============================================================

VARIATION_TAXONOMY = {
    'Lighting': ['colorshift', 'contrast_change', 'brightness'],
    'Medium': ['fog', 'smoke', 'heathaze'],
    'Foreign': ['simple_shape_foreign', 'signs_foreign', 'texture_foreign', 'sd_foreign'],
    'Camera': ['motionblur', 'noise', 'defocusblur'],
    'Object': ['cutpaste', 'scratchmix', 'texture', 'sd_inpaint']
}

# Variation-specific prompts (Paper Table VI)
VARIATION_SPECIFIC_PROMPTS = {
    # Category 1: Object Anomalies
    'object': {
        'cutpaste': 'with damage',
        'scratchmix': 'with scratches',
        'texture': 'with unusual texture patterns',
        'sd_inpaint': 'with damage',
    },
    # Category 2: Lighting
    'lighting': {
        'colorshift': 'with abnormal color shift',
        'contrast_change': 'with abnormal contrast',
        'brightness': 'with abnormal brightness',
    },
    # Category 3: Medium (Atmospheric)
    'medium': {
        'fog': 'in foggy conditions',
        'smoke': 'obscured by smoke',
        'heathaze': 'distorted by heat haze',
    },
    # Category 4: Foreign Objects
    'foreign': {
        'simple_shape_foreign': 'with simple geometric foreign objects',
        'signs_foreign': 'with warning signs or labels',
        'texture_foreign': 'with textured foreign patches',
        'sd_foreign': 'with intrusive foreign objects',
    },
    # Category 5: Camera Artifacts
    'camera': {
        'motionblur': 'with motion blur',
        'noise': 'with image noise',
        'defocusblur': 'with defocus blur',
    },
}

# Excluded defect types for certain object classes
EXCLUDED_DEFECT_TYPES = {
    'zipper': ['combined', 'fabric_border', 'fabric_interior']
}


def get_output_dir(experiment_name: str) -> Path:
    """Get output directory for a specific experiment."""
    output_dir = OUTPUT_DIR / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_model_id(model_name: str, pretrained: str) -> str:
    """Generate unique model identifier for results."""
    return f"{model_name}_{pretrained}".replace('/', '-')
