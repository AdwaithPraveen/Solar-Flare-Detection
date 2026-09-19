import os
import glob
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

import config
from utils import setup_logger, set_seed, calculate_tss, calculate_all_metrics
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
    logger.info("Starting GPU-Accelerated PyTorch + XGBoost Ensemble...")
    
    X_train, y_train, X_test, y_test = load_data()
    
    X_train_xgb = extract_ml_features(X_train)
    X_test_xgb = extract_ml_features(X_test)

    checkpoint_path = os.path.join(config.MODEL_SAVE_DIR, "cnn_lstm_final.pt")
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(config.MODEL_SAVE_DIR, "cnn_lstm_partition_1.pt")
        
    dl_model = SolarFlareCNNLSTM(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    if os.path.exists(checkpoint_path):
        dl_model.load_state_dict(torch.load(checkpoint_path, map_location=config.DEVICE))
        logger.info(f"Loaded PyTorch checkpoint: {checkpoint_path}")
    else:
        raise FileNotFoundError(
            f"DL checkpoint is required for ensemble evaluation. Checked: {checkpoint_path}"
        )

    logger.info("Computing Deep Learning probabilities...")
    dl_probs = predict_dl_model(dl_model, X_test)

    logger.info("Training XGBoost Classifier on GPU...")
    xgb_clf = xgb.XGBClassifier(**config.ML_CONFIG)
    xgb_clf.fit(X_train_xgb, y_train)
    prediction_device = xgb_clf.get_params().get("device", "cpu")
    try:
        xgb_clf.set_params(device="cpu")
        xgb_probs = xgb_clf.predict_proba(X_test_xgb)
    finally:
        xgb_clf.set_params(device=prediction_device)

    # Fixed equal-weight blending (prevent test-set lookahead tuning)
    best_weight = 0.5
    ensemble_probs = (best_weight * dl_probs) + ((1.0 - best_weight) * xgb_probs)

    threshold = getattr(config, "OPERATIONAL_THRESHOLD", 0.5)
    if config.NUM_CLASSES == 2:
        dl_preds = (dl_probs[:, 1] >= threshold).astype(int)
        xgb_preds = (xgb_probs[:, 1] >= threshold).astype(int)
        ens_preds = (ensemble_probs[:, 1] >= threshold).astype(int)
    else:
        dl_preds = np.argmax(dl_probs, axis=1)
        xgb_preds = np.argmax(xgb_probs, axis=1)
        ens_preds = np.argmax(ensemble_probs, axis=1)

    dl_m = calculate_all_metrics(y_test, dl_preds, y_probs=dl_probs)
    xgb_m = calculate_all_metrics(y_test, xgb_preds, y_probs=xgb_probs)
    ens_m = calculate_all_metrics(y_test, ens_preds, y_probs=ensemble_probs)

    logger.info("\n" + "="*82)
    logger.info("             ENSEMBLE BENCHMARK COMPARISON (FULL OPERATIONAL SUITE)              ")
    logger.info("="*82)
    logger.info(f"Metric                    | XGBoost        | PyTorch CNN-LSTM | Soft Ensemble")
    logger.info("-" * 82)
    logger.info(f"Raw Accuracy              | {xgb_m['raw_accuracy']*100:6.2f}%        | {dl_m['raw_accuracy']*100:6.2f}%          | {ens_m['raw_accuracy']*100:6.2f}%")
    logger.info(f"Balanced Accuracy         | {xgb_m['balanced_accuracy']*100:6.2f}%        | {dl_m['balanced_accuracy']*100:6.2f}%          | {ens_m['balanced_accuracy']*100:6.2f}%")
    logger.info(f"Flare TSS (TPR - FPR)     | {xgb_m['tss'].get(1, 0.0):6.4f}         | {dl_m['tss'].get(1, 0.0):6.4f}           | {ens_m['tss'].get(1, 0.0):6.4f}")
    logger.info(f"Heidke Skill Score (HSS)  | {xgb_m['hss']:6.4f}         | {dl_m['hss']:6.4f}           | {ens_m['hss']:6.4f}")
    logger.info(f"Flare Recall (TPR)        | {xgb_m['recall'].get(1, 0.0)*100:6.2f}%        | {dl_m['recall'].get(1, 0.0)*100:6.2f}%          | {ens_m['recall'].get(1, 0.0)*100:6.2f}%")
    logger.info(f"Quiet Specificity (TNR)   | {xgb_m['specificity'].get(0, 0.0)*100:6.2f}%        | {dl_m['specificity'].get(0, 0.0)*100:6.2f}%          | {ens_m['specificity'].get(0, 0.0)*100:6.2f}%")
    logger.info(f"Flare Precision           | {xgb_m['precision'].get(1, 0.0)*100:6.2f}%        | {dl_m['precision'].get(1, 0.0)*100:6.2f}%          | {ens_m['precision'].get(1, 0.0)*100:6.2f}%")
    logger.info(f"Flare F1-Score            | {xgb_m['f1'].get(1, 0.0):6.4f}         | {dl_m['f1'].get(1, 0.0):6.4f}           | {ens_m['f1'].get(1, 0.0):6.4f}")
    logger.info("="*82)

    return ensemble_probs


if __name__ == "__main__":
    run_ensemble_pipeline()
