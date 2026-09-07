import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import config
from utils import setup_logger

logger = setup_logger("DataLoader")

# ==========================================
# 1. PYTORCH DATASET CLASS
# ==========================================
class SWANSFDataset(Dataset):
    """
    Custom PyTorch Dataset for SWANSF Time-Series Data.
    Converts (N, 60, 24) -> (N, 24, 60) for PyTorch 1D-CNN expectations.
    """
    def __init__(self, X, y):
        # Convert to float32 tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        # PyTorch Conv1d expects shape: (Batch, Features, Sequence_Length)
        if X_tensor.ndim == 3 and X_tensor.shape[1] == config.SEQ_LEN:
            X_tensor = X_tensor.permute(0, 2, 1)
            
        self.X = X_tensor
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ==========================================
# 2. FEATURE EXTRACTION FOR MACHINE LEARNING
# ==========================================
def extract_ml_features(X_3d):
    """
    Transforms 3D time-series data (N, 60, 24) into a 2D feature matrix
    by calculating summary statistics across all 60 timesteps.
    
    Output Shape: (N, 24 * 5) = (N, 120)
    """
    logger.info(f"Extracting 2D statistical features from 3D array of shape {X_3d.shape}...")
    
    means = np.mean(X_3d, axis=1)
    stds = np.std(X_3d, axis=1)
    maxs = np.max(X_3d, axis=1)
    mins = np.min(X_3d, axis=1)
    lasts = X_3d[:, -1, :]  # Most recent magnetic parameter state
    
    # Concatenate horizontal summary features
    X_2d = np.hstack([means, stds, maxs, mins, lasts])
    logger.info(f"Extracted 2D feature shape: {X_2d.shape}")
    return X_2d

# ==========================================
# 3. PICKLE DATA LOADERS & DATASET PREPARATION
# ==========================================
def load_pickle_file(file_path):
    """Utility to load a pickled data file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Partition file not found at: {file_path}")
    
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data

def prepare_dataloaders(X_train, y_train, X_test, y_test, batch_size=config.DL_CONFIG["batch_size"]):
    """
    Creates PyTorch DataLoaders optimized for GPU execution.
    """
    train_dataset = SWANSFDataset(X_train, y_train)
    test_dataset = SWANSFDataset(X_test, y_test)
    
    # pin_memory=True speeds up transfer from host RAM to GPU VRAM
    pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=0
    )
    
    logger.info(f"PyTorch DataLoaders created successfully. Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    return train_loader, test_loader

if __name__ == "__main__":
    # Diagnostic dry-run with synthetic SWANSF shapes
    dummy_X = np.random.randn(100, 60, 24).astype(np.float32)
    dummy_y = np.random.randint(0, 4, size=(100,))
    
    # Test ML Feature Extraction
    dummy_2d = extract_ml_features(dummy_X)
    
    # Test DL DataLoader creation
    tr_loader, te_loader = prepare_dataloaders(dummy_X, dummy_y, dummy_X, dummy_y)
    for sample_X, sample_y in tr_loader:
        logger.info(f"Sample Batch Tensor Shape: X={sample_X.shape}, y={sample_y.shape}")
        break