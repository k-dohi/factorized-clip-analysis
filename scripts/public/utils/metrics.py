"""
Evaluation Metrics

AUROC calculation for anomaly detection.
"""

import numpy as np
from sklearn.metrics import roc_auc_score
from typing import List, Union


def compute_auroc(normal_scores: Union[List[float], np.ndarray],
                  anomaly_scores: Union[List[float], np.ndarray]) -> float:
    """
    Compute AUROC for anomaly detection.
    
    Args:
        normal_scores: Anomaly scores for normal samples
        anomaly_scores: Anomaly scores for anomaly samples
    
    Returns:
        AUROC value [0, 1]
    """
    labels = [0] * len(normal_scores) + [1] * len(anomaly_scores)
    scores = list(normal_scores) + list(anomaly_scores)
    
    if len(set(labels)) < 2:
        raise ValueError("Need both normal and anomaly samples to compute AUROC")
    
    return roc_auc_score(labels, scores)


def compute_auroc_with_ci(normal_scores: Union[List[float], np.ndarray],
                          anomaly_scores: Union[List[float], np.ndarray],
                          n_bootstrap: int = 1000,
                          confidence: float = 0.95,
                          random_seed: int = 42) -> tuple:
    """
    Compute AUROC with bootstrap confidence interval.
    
    Args:
        normal_scores: Anomaly scores for normal samples
        anomaly_scores: Anomaly scores for anomaly samples
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (e.g., 0.95 for 95% CI)
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (auroc, ci_lower, ci_upper)
    """
    rng = np.random.default_rng(random_seed)
    
    normal_scores = np.array(normal_scores)
    anomaly_scores = np.array(anomaly_scores)
    
    auroc_samples = []
    for _ in range(n_bootstrap):
        # Bootstrap sample
        normal_idx = rng.choice(len(normal_scores), size=len(normal_scores), replace=True)
        anomaly_idx = rng.choice(len(anomaly_scores), size=len(anomaly_scores), replace=True)
        
        normal_boot = normal_scores[normal_idx]
        anomaly_boot = anomaly_scores[anomaly_idx]
        
        auroc = compute_auroc(normal_boot, anomaly_boot)
        auroc_samples.append(auroc)
    
    auroc_samples = np.array(auroc_samples)
    auroc_mean = np.mean(auroc_samples)
    
    alpha = 1 - confidence
    ci_lower = np.percentile(auroc_samples, alpha / 2 * 100)
    ci_upper = np.percentile(auroc_samples, (1 - alpha / 2) * 100)
    
    return auroc_mean, ci_lower, ci_upper
