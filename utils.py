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
    logger.info("=== HARDWARE DIAGNOSTICS ===")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        vram_gb = vram_bytes / (1024 ** 3)
        logger.info(f"Target Compute Device : CUDA (GPU Detected)")
        logger.info(f"GPU Model              : {gpu_name}")
        logger.info(f"Total VRAM Available   : {vram_gb:.2f} GB")
        logger.info(f"PyTorch CUDA Version   : {torch.version.cuda}")
        logger.info(f"AMP (Mixed Precision)  : {'Enabled' if config.USE_AMP else 'Disabled'}")
    else:
        logger.warning("CUDA is NOT available. Falling back to CPU mode.")
    logger.info("============================")

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

if __name__ == "__main__":
    test_logger = setup_logger()
    set_seed(42)
    verify_hardware_acceleration(test_logger)