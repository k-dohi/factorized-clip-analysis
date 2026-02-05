# Public Code for Paper Reproduction

This directory contains self-contained code to reproduce all experiments in the paper.

## Paper Table to Experiment Mapping

| Paper Table | Experiment | CLIP Models | Description |
|-------------|------------|-------------|-------------|
| Table I-III | Experiment 1 | **All 7 models** | Zero-Shot Detection under Contextual Variations |
| Table IV | Experiment 2 | ViT-H-14 only | Background Masking Analysis |
| Table V | Experiment 3 | ViT-H-14 only | Prompt Ablation Study |
| Table VI | Experiment 4 | ViT-H-14 only | Fog and Defocus Blur Parameter Ablation |
| Table VII | Experiment 5 | ViT-H-14 only | Spatial Sensitivity Analysis |
| Table VIII | Experiment 6 | ViT-H-14 only | Variation-Specific Prompt Evaluation |

## CLIP Models Used (7 variants)

| Model | Pretrained | Source |
|-------|------------|--------|
| ViT-B-32 | openai | OpenAI CLIP |
| ViT-B-16 | openai | OpenAI CLIP |
| ViT-L-14 | openai | OpenAI CLIP |
| ViT-B-16 | laion400m_e32 | OpenCLIP LAION-400M |
| ViT-B-32 | laion2b_s34b_b79k | OpenCLIP LAION-2B |
| ViT-L-14 | laion2b_s32b_b79k | OpenCLIP LAION-2B |
| ViT-H-14 | laion2b_s32b_b79k | OpenCLIP LAION-2B |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Full paper reproduction (all experiments, all models)
# Estimated time: 24-48 hours
python run_all_experiments.py

# Quick test (single model, all experiments)
# Estimated time: 2-4 hours
python run_all_experiments.py --quick_test

# Run only Experiment 1 (all 7 models)
python run_all_experiments.py --only 1

# Skip Experiment 1, run Experiments 2-6 (ViT-H-14 only)
python run_all_experiments.py --skip 1
```

## Directory Structure

```
scripts/public/
├── config.py                    # Common configuration (models, parameters)
├── run_all_experiments.py       # Master script for all experiments
├── requirements.txt             # Dependencies
├── README.md                    # This file
│
├── utils/                       # Shared utilities
│   ├── clip_scorer.py          # CLIP anomaly scoring (softmax with T=0.07)
│   ├── data_loader.py          # MVTec AD data loading
│   ├── variations.py           # Variation generation (17 types)
│   └── metrics.py              # AUROC computation
│
├── augmenters/                  # Pseudo anomaly generation (17 augmenters)
│   ├── base.py                 # Base augmenter class
│   ├── object_defect.py        # Category 1: Object defects
│   ├── lighting.py             # Category 2: Lighting variations
│   ├── atmospheric.py          # Category 3: Atmospheric effects
│   ├── foreign_object.py       # Category 4: Foreign objects
│   └── camera.py               # Category 5: Camera artifacts
│
├── experiment1_zero_shot/       # Experiment 1: Zero-shot detection
├── experiment2_masking/         # Experiment 2: Background masking
├── experiment3_prompts/         # Experiment 3: Prompt ablation
├── experiment4_fog_blur/        # Experiment 4: Fog/Blur ablation
├── experiment5_spatial/         # Experiment 5: Spatial sensitivity
└── experiment6_variation_specific/  # Experiment 6: Variation-specific prompts
```

## Output Structure

Results are saved to `outputs/public_experiments/`:

```
outputs/public_experiments/
├── final_summary.json                # Aggregated results
│
├── experiment1_zero_shot/
│   ├── ViT-B-32_openai/              # Results for each model
│   ├── ViT-B-16_openai/
│   ├── ViT-L-14_openai/
│   ├── ViT-B-16_laion400m_e32/
│   ├── ViT-B-32_laion2b_s34b_b79k/
│   ├── ViT-L-14_laion2b_s32b_b79k/
│   ├── ViT-H-14_laion2b_s32b_b79k/
│   └── all_models_summary.json       # Aggregated results for Table I-III
│
├── experiment2_masking/              # Table IV results
├── experiment3_prompts/              # Table V results
├── experiment4_fog_blur/             # Table VI results
├── experiment5_spatial/              # Table VII results
└── experiment6_variation_specific/   # Table VIII results
```

## Experiment Details

### Experiment 1: Zero-Shot Detection (Tables I-III)

Evaluates CLIP's anomaly detection across 3 scenarios:
- **Scenario 1**: Real normal vs Real anomaly (baseline)
- **Scenario 2**: Pseudo-normal (with variation) vs Real anomaly
- **Scenario 3**: Real normal vs Object pseudo-anomaly

Key finding: CLIP misclassifies contextual variations (lighting, fog, etc.) as anomalies.

### Experiment 2: Background Masking (Table IV)

Tests whether background masking can mitigate false positives:
- No masking (baseline)
- Black background masking
- White background masking

### Experiment 3: Prompt Ablation (Table V)

Compares different text prompt templates:
- Generic prompts ("an object")
- Simple prompts ("{obj}")
- Photo prompts ("a photo of a {obj}")
- Industrial prompts ("a {obj} in industrial inspection")
- Quality prompts ("a normal {obj} without defects")

### Experiment 4: Fog/Blur Ablation (Table VI)

Parameter sweep for fog density and blur radius:
- Fog density: 0.0 - 1.0 (11 levels)
- Defocus blur radius: 0 - 21 (9 levels)

### Experiment 5: Spatial Sensitivity (Table VII)

Sliding window analysis showing where CLIP perceives "damage":
- 16x16 grid analysis
- Fourier texture patches
- Δs heatmap generation

### Experiment 6: Variation-Specific Prompts (Table VIII)

Tests variation-specific prompts (e.g., "with fog", "with motion blur"):
- 17 variation types × 10 object classes
- Three negative settings: Neg=N, Neg=N+A, Neg=A

## Citation

If you use this code, please cite our paper:
```bibtex
@article{...,
  title={...},
  author={...},
  journal={IEEE Access},
  year={2025}
}
```
