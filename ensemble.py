import os
import glob
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

import config
from utils import setup_logger, set_seed, calculate_tss
from models.cnn_lstm_model import SolarFlareCNNLSTM
from data_loader import extract_ml_features, load_train_test_partitions

logger = setup_logger("EnsemblePipeline")


def load_data():
    """Load all training partitions and their separate untouched test partitions."""
    return load_train_test_partitions()

def predict_dl_model(model, X, batch_size=64):
    """Generates probability predictions using PyTorch model on GPU."""
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    probs_list = []
    softmax = torch.nn.Softmax(dim=1)
    
    with torch.no_grad():
        for (inputs,) in loader:
            inputs = inputs.to(config.DEVICE)
            with torch.amp.autocast('cuda', enabled=config.USE_AMP):
                outputs = model(inputs)
                probs = softmax(outputs)
            probs_list.append(probs.cpu())
            
    return torch.cat(probs_list, dim=0).numpy()

def run_ensemble_pipeline():
    set_seed(42)
    logger.info("⚡ Starting GPU-Accelerated PyTorch + XGBoost Ensemble")
    
    X_train, y_train, X_test, y_test = load_data()
    
    # Extract 2D statistical features for XGBoost: (N, 120 features)
    X_train_xgb = extract_ml_features(X_train)
    X_test_xgb = extract_ml_features(X_test)

    # 1. Load Trained PyTorch Checkpoint
    checkpoint_path = os.path.join(config.MODEL_SAVE_DIR, "best_solar_flare_model.pt")
    dl_model = SolarFlareCNNLSTM().to(config.DEVICE)
    
    if os.path.exists(checkpoint_path):
        dl_model.load_state_dict(torch.load(checkpoint_path, map_location=config.DEVICE))
        logger.info(f"Loaded PyTorch checkpoint: {checkpoint_path}")
    else:
        raise FileNotFoundError(
            f"DL checkpoint is required for ensemble evaluation: {checkpoint_path}"
        )

    logger.info("Computing Deep Learning probabilities...")
    dl_probs = predict_dl_model(dl_model, X_test)

    # 2. Train XGBoost Model on RTX GPU
    logger.info("Training XGBoost Classifier on GPU...")
    xgb_clf = xgb.XGBClassifier(**config.ML_CONFIG)
    xgb_clf.fit(X_train_xgb, y_train)
    # X_test_xgb is a NumPy array on CPU.  Predict on CPU to avoid XGBoost's
    # CUDA DMatrix fallback warning, then restore the configured device.
    prediction_device = xgb_clf.get_params().get("device", "cpu")
    try:
        xgb_clf.set_params(device="cpu")
        xgb_probs = xgb_clf.predict_proba(X_test_xgb)
    finally:
        xgb_clf.set_params(device=prediction_device)

    # 3. Fixed Equal-Weight Blending
    # LEAKAGE FIX: The previous grid-search over w used y_test to pick the best weight,
    # which is look-ahead bias — the blending hyperparameter was tuned on the test set.
    # Fix: use a fixed equal-weight blend. If weight optimisation is needed in future,
    # it must be done on a separate held-out VALIDATION set, NOT the test set.
    best_weight = 0.5
    best_ensemble_probs = (best_weight * dl_probs) + ((1.0 - best_weight) * xgb_probs)

    # 4. Comparative Evaluation
    dl_preds = np.argmax(dl_probs, axis=1)
    xgb_preds = np.argmax(xgb_probs, axis=1)
    ensemble_preds = np.argmax(best_ensemble_probs, axis=1)

    dl_tss = calculate_tss(y_test, dl_preds)
    xgb_tss = calculate_tss(y_test, xgb_preds)
    ens_tss = calculate_tss(y_test, ensemble_preds)

    logger.info("\n" + "="*80)
    logger.info("                     ENSEMBLE BENCHMARK COMPARISON                      ")
    logger.info("="*80)
    logger.info(f"Optimal Blending Weight : {best_weight:.2f} DL + {1.0 - best_weight:.2f} XGB")
    logger.info("-" * 80)
    if config.NUM_CLASSES == 2:
        logger.info(f"PyTorch DL Mean TSS     : {np.mean(list(dl_tss.values())):.4f} | (Flare: {dl_tss.get(1, 0.0):.4f})")
        logger.info(f"XGBoost GPU Mean TSS    : {np.mean(list(xgb_tss.values())):.4f} | (Flare: {xgb_tss.get(1, 0.0):.4f})")
        logger.info(f"Ensemble Mean TSS       : {np.mean(list(ens_tss.values())):.4f} | (Flare: {ens_tss.get(1, 0.0):.4f})")
    elif config.NUM_CLASSES == 3:
        logger.info(f"PyTorch DL Mean TSS     : {np.mean(list(dl_tss.values())):.4f} | (M: {dl_tss[1]:.4f}, X: {dl_tss[2]:.4f})")
        logger.info(f"XGBoost GPU Mean TSS    : {np.mean(list(xgb_tss.values())):.4f} | (M: {xgb_tss[1]:.4f}, X: {xgb_tss[2]:.4f})")
        logger.info(f"🏆 ENSEMBLE MEAN TSS    : {np.mean(list(ens_tss.values())):.4f} | (M: {ens_tss[1]:.4f}, X: {ens_tss[2]:.4f})")
    else:
        logger.info(f"PyTorch DL Mean TSS     : {np.mean(list(dl_tss.values())):.4f} | (M: {dl_tss[2]:.4f}, X: {dl_tss[3]:.4f})")
        logger.info(f"XGBoost GPU Mean TSS    : {np.mean(list(xgb_tss.values())):.4f} | (M: {xgb_tss[2]:.4f}, X: {xgb_tss[3]:.4f})")
        logger.info(f"🏆 ENSEMBLE MEAN TSS    : {np.mean(list(ens_tss.values())):.4f} | (M: {ens_tss[2]:.4f}, X: {ens_tss[3]:.4f})")
    logger.info("="*80)

    return best_ensemble_probs

if __name__ == "__main__":
    run_ensemble_pipeline()
