import os
import glob
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import gc

import config
from utils import setup_logger, set_seed, calculate_tss
from models.cnn_lstm_model import SolarFlareCNNLSTM, CNNLSTMTrainer
from presentation import SolarPresentationEngine
from data_loader import load_train_test_partitions

logger = setup_logger("TrainDL_KFold")

def optimize_gpu_performance():
    """Enables cuDNN benchmarking and checks RTX CUDA availability."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available! An NVIDIA GPU is required for this configuration.")
    
    # Speed up convolution autotuning on NVIDIA RTX Tensor Cores
    torch.backends.cudnn.benchmark = True
    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    logger.info(f"⚡ GPU Acceleration Active: {gpu_name} ({vram:.2f} GB VRAM)")

def create_gpu_dataloader(X, y, batch_size, shuffle=True):
    """Creates an RTX-optimized DataLoader with pinned memory and multi-worker prefetching."""
    # Strict validation: Ensure shape (N, SEQ_LEN, NUM_FEATURES) before permute
    assert X.ndim == 3, f"Expected 3D input (N, L, C), got shape {X.shape}"
    assert X.shape[2] == config.NUM_FEATURES, f"Target leakage check failed: Expected {config.NUM_FEATURES} features, got {X.shape[2]}"
    
    X_tensor = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)  # (B, C, L) for 1D-Conv
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    
    num_workers = min(4, os.cpu_count() or 1)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if num_workers > 0 else False
    )

def load_swansf_data():
    """Load all training partitions and the separate untouched test partitions."""
    return load_train_test_partitions()

def run_stratified_kfold_pipeline(n_splits=5, epochs_per_fold=None):
    if epochs_per_fold is None:
        epochs_per_fold = config.DL_CONFIG.get("epochs", 60)
    # optimize_gpu_performance() must be called BEFORE set_seed() to prevent
    # it from overwriting torch.backends.cudnn.benchmark back to True, which
    # would invalidate the deterministic cuDNN flag set by set_seed().
    optimize_gpu_performance()
    set_seed(42)

    logger.info("=========================================================")
    logger.info(f"STARTING {n_splits}-FOLD INTERNAL STRATIFIED CV")
    logger.info("=========================================================")

    X, y, X_test, y_test = load_swansf_data()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_tss_scores = []
    fold_accuracies = []
    best_overall_tss = -1.0
    dl_probs = np.zeros((len(y), config.NUM_CLASSES), dtype=np.float32)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        logger.info(f"\n--- FOLD {fold}/{n_splits} ---")
        X_train_fold, y_train_fold = X[train_idx], y[train_idx]
        X_val_fold, y_val_fold = X[val_idx], y[val_idx]

        # Calculate class weights strictly on current training fold (prevent data leakage)
        classes = np.unique(y_train_fold)
        weights = compute_class_weight('balanced', classes=classes, y=y_train_fold)
        full_weights = np.ones(config.NUM_CLASSES)
        for cls, w in zip(classes, weights):
            if int(cls) < config.NUM_CLASSES:
                full_weights[int(cls)] = w
                
        class_weights_tensor = torch.tensor(full_weights, dtype=torch.float32).to(config.DEVICE)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
        
        train_loader = create_gpu_dataloader(X_train_fold, y_train_fold, config.DL_CONFIG["batch_size"], shuffle=True)
        val_loader = create_gpu_dataloader(X_val_fold, y_val_fold, config.DL_CONFIG["batch_size"], shuffle=False)

        model = SolarFlareCNNLSTM(num_classes=config.NUM_CLASSES)
        trainer = CNNLSTMTrainer(model, device=config.DEVICE, criterion=criterion)

        for epoch in range(1, epochs_per_fold + 1):
            train_loss, train_acc = trainer.train_epoch(train_loader)
            
            # Periodic validation during training
            if epoch % 10 == 0 or epoch == epochs_per_fold:
                val_loss, val_acc, val_preds, _, _ = trainer.evaluate(val_loader)
                val_tss_epoch = calculate_tss(y_val_fold, val_preds)
                mean_val_tss = float(np.mean(list(val_tss_epoch.values())))
                logger.info(
                    f"[Fold {fold} | Epoch {epoch:02d}/{epochs_per_fold}] "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%, Val TSS: {mean_val_tss:.4f}"
                )
            gc.collect()
            torch.cuda.empty_cache()

        # Strict out-of-fold evaluation on validation set
        val_loss, val_acc, val_preds, val_targets, val_fold_probs = trainer.evaluate(val_loader)
        dl_probs[val_idx] = val_fold_probs
        
        # Verify validation targets match ground truth
        assert np.array_equal(val_targets, y_val_fold), "Validation target mismatch detected in evaluation loop!"
        
        tss_dict = calculate_tss(y_val_fold, val_preds)
        mean_tss = float(np.mean(list(tss_dict.values())))
        fold_acc = accuracy_score(y_val_fold, val_preds)
        
        fold_tss_scores.append(tss_dict)
        fold_accuracies.append(fold_acc)

        if config.NUM_CLASSES == 2:
            logger.info(f"Fold {fold} Validation -> Acc: {fold_acc*100:.2f}% | Mean TSS: {mean_tss:.4f} | Flare TSS: {tss_dict.get(1, 0.0):.4f}")
        elif config.NUM_CLASSES == 3:
            logger.info(f"Fold {fold} Validation -> Acc: {fold_acc*100:.2f}% | Mean TSS: {mean_tss:.4f} | M: {tss_dict.get(1, 0.0):.4f} | X: {tss_dict.get(2, 0.0):.4f}")
        else:
            logger.info(f"Fold {fold} Validation -> Acc: {fold_acc*100:.2f}% | Mean TSS: {mean_tss:.4f} | C: {tss_dict.get(1, 0.0):.4f} | M: {tss_dict.get(2, 0.0):.4f} | X: {tss_dict.get(3, 0.0):.4f}")

        if mean_tss > best_overall_tss:
            best_overall_tss = mean_tss
            trainer.save_checkpoint("best_solar_flare_model.pt")
            logger.info(f"🏆 New Best Model Saved with Validation Mean TSS: {mean_tss:.4f}")
            
        gc.collect()
        torch.cuda.empty_cache()

    # Aggregated CV Benchmark Report
    logger.info("\n" + "="*80)
    logger.info(" INTERNAL CV REPORT (TRAINING PARTITIONS ONLY; NOT TEST PERFORMANCE) ")
    logger.info("="*80)
    
    mean_cv_acc = np.mean(fold_accuracies) * 100.0
    overall_mean_tss = np.mean([np.mean(list(f.values())) for f in fold_tss_scores])
    avg_q_tss = np.mean([f[0] for f in fold_tss_scores])

    logger.info(f"Mean Internal CV Accuracy: {mean_cv_acc:.2f}%")
    logger.info(f"Quiet Class TSS         : {avg_q_tss:.4f}")
    if config.NUM_CLASSES == 2:
        avg_flare_tss = np.mean([f.get(1, 0.0) for f in fold_tss_scores])
        logger.info(f"Flare Class TSS         : {avg_flare_tss:.4f}")
    elif config.NUM_CLASSES == 3:
        avg_m_tss = np.mean([f.get(1, 0.0) for f in fold_tss_scores])
        avg_x_tss = np.mean([f.get(2, 0.0) for f in fold_tss_scores])
        logger.info(f"M-Class Flare TSS       : {avg_m_tss:.4f}")
        logger.info(f"X-Class Flare TSS       : {avg_x_tss:.4f}")
    else:
        avg_c_tss = np.mean([f.get(1, 0.0) for f in fold_tss_scores])
        avg_m_tss = np.mean([f.get(2, 0.0) for f in fold_tss_scores])
        avg_x_tss = np.mean([f.get(3, 0.0) for f in fold_tss_scores])
        logger.info(f"C-Class Flare TSS       : {avg_c_tss:.4f}")
        logger.info(f"M-Class Flare TSS       : {avg_m_tss:.4f}")
        logger.info(f"X-Class Flare TSS       : {avg_x_tss:.4f}")
        
    logger.info(f"Overall Internal CV Mean TSS: {overall_mean_tss:.4f}")
    logger.info("="*80)

    checkpoint_path = os.path.join(config.MODEL_SAVE_DIR, "best_solar_flare_model.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Best DL checkpoint was not created: {checkpoint_path}")

    best_model = SolarFlareCNNLSTM(num_classes=config.NUM_CLASSES)
    best_model.load_state_dict(torch.load(checkpoint_path, map_location=config.DEVICE))
    test_loader = create_gpu_dataloader(X_test, y_test, config.DL_CONFIG["batch_size"], shuffle=False)
    test_trainer = CNNLSTMTrainer(best_model, device=config.DEVICE)
    test_loss, test_acc, test_preds, test_targets, test_probs = test_trainer.evaluate(test_loader)
    if not np.array_equal(test_targets, y_test):
        raise RuntimeError("Held-out test targets do not match the test partition order.")

    test_tss = calculate_tss(y_test, test_preds)
    logger.info("\n" + "=" * 80)
    logger.info("HELD-OUT TEST REPORT")
    logger.info(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc * 100:.2f}%")
    logger.info(f"Test Mean TSS: {np.mean(list(test_tss.values())):.4f} | Flare TSS: {test_tss.get(1, 0.0):.4f}")
    logger.info("=" * 80)

    return test_probs

if __name__ == "__main__":
    run_stratified_kfold_pipeline(n_splits=5, epochs_per_fold=config.DL_CONFIG.get("epochs", 60))
