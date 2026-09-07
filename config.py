import os
import torch
from typing import Dict, Any

# ==========================================
# 1. DIRECTORY & DATA CONFIGURATION
# ==========================================
DATA_DIR = "./Cleaned SWANSF Dataset/Cleaned SWANSF Dataset" # Path discovered from load_data.ipynb
MODEL_SAVE_DIR = "./saved_models"

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

SEQ_LEN = 60
NUM_FEATURES = 24
NUM_CLASSES = 2                  # (0: Quiet, 1: Flare [M/X-Class] in SWANSF WithoutC dataset)

# ==========================================
# 2. HARDWARE CONFIGURATION
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = True                   # Automatic Mixed Precision for RTX Tensor Cores

# ==========================================
# 3. HIGH-CAPACITY DEEP LEARNING (PYTORCH)
# ==========================================
DL_CONFIG: Dict[str, Any] = {
    "batch_size": 64,            # Batch size
    "epochs": 60,                # Training window
    "learning_rate": 3e-4,       # Optimal starting rate for AdamW
    "weight_decay": 1e-3,        # L2 Regularization to stop overfitting
    
    # Scaled Neural Architecture
    "cnn_out_channels": 128,     # Spatial conv filters
    "cnn_kernel_size": 5,        # Broader time window kernel
    "lstm_hidden_size": 256,     # Recurrent capacity
    "lstm_layers": 2,            # 2-layer deep stack with dropout
    "lstm_dropout": 0.3,
    "fc_hidden": 128
}

# ==========================================
# 4. HIGH-CAPACITY MACHINE LEARNING (XGBOOST)
# ==========================================
ML_CONFIG: Dict[str, Any] = {
    "n_estimators": 500,         # Decision trees
    "learning_rate": 0.03,       # Fine-grained boosting steps
    "max_depth": 6,              # Tree depth
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic" if NUM_CLASSES == 2 else "multi:softprob",
    "eval_metric": "logloss" if NUM_CLASSES == 2 else "mlogloss",
    "tree_method": "hist",
    "device": "cuda",
    "random_state": 42
}
if NUM_CLASSES > 2:
    ML_CONFIG["num_class"] = NUM_CLASSES