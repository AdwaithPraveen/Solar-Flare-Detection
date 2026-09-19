import os
import random
import logging
import numpy as np
import torch
import config

def setup_logger(name: str = "SolarFlareLogger") -> logging.Logger:
    """
    Sets up a formatted logger to output execution logs to console.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        try:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        except Exception as e:
            print(f"Failed to set up logger handlers: {e}")
            
    return logger

def set_seed(seed: int = 42) -> None:
    """
    Ensures deterministic behavior across random, numpy, PyTorch, and CUDA.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def verify_hardware_acceleration(logger: logging.Logger) -> None:
    """
    Checks and logs the GPU status to confirm NVIDIA CUDA is working.
    """
    if torch.cuda.is_available():
        logger.info("NVIDIA is being used.")
    else:
        logger.warning("CUDA is NOT available. Falling back to CPU mode.")

def calculate_tss(y_true, y_pred, num_classes: int = None) -> dict:
    """
    Computes True Skill Statistic (TSS = TPR - FPR) per class.
    For binary (0=Quiet, 1=Flare): TSS = Flare_TPR - Flare_FPR
    """
    from sklearn.metrics import confusion_matrix
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_pred = np.asarray(y_pred, dtype=int).ravel()
    
    if num_classes is None:
        num_classes = config.NUM_CLASSES
        
    labels = list(range(num_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    tss_scores = {}
    for i in range(num_classes):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - (tp + fn + fp)
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tss_scores[i] = tpr - fpr
        
    return tss_scores

def calculate_all_metrics(y_true, y_pred, y_probs=None, num_classes: int = None) -> dict:
    """
    Computes comprehensive operational metrics for imbalanced space weather forecasting:
    - Raw Accuracy
    - Balanced Accuracy (Macro Recall)
    - Per-class Precision, Recall (Sensitivity / TPR), Specificity (TNR)
    - F1-Score (Macro and Flare / Class 1)
    - True Skill Statistic (TSS)
    - Heidke Skill Score (HSS)
    - Confusion Matrix (TP, FP, FN, TN)
    - ROC-AUC and PR-AUC (if probabilities provided)
    """
    from sklearn.metrics import (
        confusion_matrix,
        accuracy_score,
        balanced_accuracy_score,
        precision_recall_fscore_support,
        roc_auc_score,
        average_precision_score,
    )
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_pred = np.asarray(y_pred, dtype=int).ravel()
    
    if num_classes is None:
        num_classes = config.NUM_CLASSES
        
    labels = list(range(num_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    raw_acc = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    
    tss_scores = {}
    spec_scores = {}
    for i in range(num_classes):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - (tp + fn + fp)
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tnr = tn / (fp + tn) if (fp + tn) > 0 else 0.0
        tss_scores[i] = tpr - fpr
        spec_scores[i] = tnr

    # Heidke Skill Score (HSS2)
    n = np.sum(cm)
    expected_correct = sum(np.sum(cm[i, :]) * np.sum(cm[:, i]) for i in range(num_classes)) / (n if n > 0 else 1)
    actual_correct = np.trace(cm)
    denom = (n - expected_correct)
    hss = (actual_correct - expected_correct) / denom if denom != 0 else 0.0

    metrics = {
        "raw_accuracy": raw_acc,
        "balanced_accuracy": balanced_acc,
        "tss": tss_scores,
        "specificity": spec_scores,
        "precision": {i: prec[i] for i in range(num_classes)},
        "recall": {i: rec[i] for i in range(num_classes)},
        "f1": {i: f1[i] for i in range(num_classes)},
        "support": {i: int(support[i]) for i in range(num_classes)},
        "hss": hss,
        "confusion_matrix": cm,
    }

    if y_probs is not None:
        try:
            if num_classes == 2:
                prob_col = y_probs[:, 1] if (y_probs.ndim == 2 and y_probs.shape[1] > 1) else y_probs.ravel()
                metrics["roc_auc"] = roc_auc_score(y_true, prob_col)
                metrics["pr_auc"] = average_precision_score(y_true, prob_col)
            else:
                metrics["roc_auc"] = roc_auc_score(y_true, y_probs, multi_class="ovr")
        except Exception:
            pass

    return metrics

if __name__ == "__main__":
    test_logger = setup_logger()
    set_seed(42)
    verify_hardware_acceleration(test_logger)