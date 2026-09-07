import os
import pickle
import glob
import re
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

def _partition_number(file_path):
    match = re.search(r"Partition(\d+)", os.path.basename(file_path))
    if not match:
        raise ValueError(f"Could not identify partition number in: {file_path}")
    return int(match.group(1))

def _unwrap_partition(data):
    if isinstance(data, dict):
        if len(data) != 1:
            raise ValueError("Partition dictionaries must contain exactly one array.")
        return next(iter(data.values()))
    return data

def _load_split_partitions(split_name, data_dir=config.DATA_DIR):
    """Load and concatenate every feature/label partition for one split."""
    split_dir = os.path.join(data_dir, split_name)
    files = glob.glob(os.path.join(split_dir, "*.pkl"))
    feature_files = { _partition_number(path): path for path in files if "Labels" not in os.path.basename(path) }
    label_files = { _partition_number(path): path for path in files if "Labels" in os.path.basename(path) }

    partition_ids = sorted(feature_files)
    if not partition_ids or set(partition_ids) != set(label_files):
        raise FileNotFoundError(f"Feature/label partition pairs are incomplete in: {split_dir}")

    features, labels = [], []
    for partition_id in partition_ids:
        X = np.asarray(_unwrap_partition(load_pickle_file(feature_files[partition_id])), dtype=np.float32)
        y = np.asarray(_unwrap_partition(load_pickle_file(label_files[partition_id])), dtype=np.int64).squeeze()
        if X.ndim != 3 or X.shape[1:] != (config.SEQ_LEN, config.NUM_FEATURES):
            raise ValueError(f"Unexpected feature shape in partition {partition_id}: {X.shape}")
        if len(X) != len(y):
            raise ValueError(f"Feature/label count mismatch in partition {partition_id}.")
        features.append(X)
        labels.append(y)

    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)

def load_train_test_partitions(data_dir=config.DATA_DIR):
    """Return all training partitions and all untouched test partitions."""
    X_train, y_train = _load_split_partitions("train", data_dir)
    X_test, y_test = _load_split_partitions("test", data_dir)
    return X_train, y_train, X_test, y_test

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
