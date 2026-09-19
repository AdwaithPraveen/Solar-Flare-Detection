import gc
import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

import config
from data_loader import available_partition_ids, load_all_training_partitions, load_partition_pair
from models.cnn_lstm_model import CNNLSTMTrainer, SolarFlareCNNLSTM
from utils import calculate_tss, calculate_all_metrics, set_seed, setup_logger

logger = setup_logger("TrainDL")


def optimize_gpu_performance():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This DL configuration requires an NVIDIA GPU.")
    torch.backends.cudnn.benchmark = True
    logger.info("NVIDIA is being used.")


def create_gpu_dataloader(X, y, batch_size, shuffle):
    if X.ndim != 3 or X.shape[1:] != (config.SEQ_LEN, config.NUM_FEATURES):
        raise ValueError(f"Expected input shape (N, {config.SEQ_LEN}, {config.NUM_FEATURES}), got {X.shape}")
    if len(X) != len(y):
        raise ValueError("Feature and label counts do not match.")

    X_tensor = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)
    y_tensor = torch.tensor(y, dtype=torch.long)
    workers = min(4, os.cpu_count() or 1)
    return DataLoader(
        TensorDataset(X_tensor, y_tensor),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )


def class_weighted_loss(y_train):
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    full_weights = np.ones(config.NUM_CLASSES, dtype=np.float32)
    for label, weight in zip(classes, weights):
        if 0 <= int(label) < config.NUM_CLASSES:
            full_weights[int(label)] = weight
    return torch.nn.CrossEntropyLoss(
        weight=torch.tensor(full_weights, dtype=torch.float32, device=config.DEVICE)
    )


def print_detailed_evaluation(partition_id, model_name, metrics):
    cm = metrics["confusion_matrix"]
    raw_acc = metrics["raw_accuracy"] * 100.0
    bal_acc = metrics["balanced_accuracy"] * 100.0
    tss_flare = metrics["tss"].get(1, 0.0)
    hss = metrics["hss"]
    rec_flare = metrics["recall"].get(1, 0.0) * 100.0
    spec_quiet = metrics["specificity"].get(0, 0.0) * 100.0
    prec_flare = metrics["precision"].get(1, 0.0) * 100.0
    f1_flare = metrics["f1"].get(1, 0.0)

    part_str = f"Partition {partition_id}" if partition_id is not None else "Final Deployment Model"
    logger.info("-" * 78)
    logger.info(f"  DETAILED METRICS: {model_name} on {part_str}")
    logger.info("-" * 78)
    logger.info(f"  * Raw Accuracy       : {raw_acc:6.2f}%  (Reflects class imbalance: ~99% Quiet)")
    logger.info(f"  * Balanced Accuracy  : {bal_acc:6.2f}%  (Macro-averaged recall across classes)")
    logger.info(f"  * Flare TSS          : {tss_flare:6.4f}   (True Skill Statistic = TPR - FPR)")
    logger.info(f"  * Heidke Skill Score : {hss:6.4f}   (HSS skill relative to random chance)")
    logger.info(f"  * Flare Recall (TPR) : {rec_flare:6.2f}%  (Detection rate of actual flares)")
    logger.info(f"  * Quiet Specificity  : {spec_quiet:6.2f}%  (Correct identification of non-flares)")
    logger.info(f"  * Flare Precision    : {prec_flare:6.2f}%")
    logger.info(f"  * Flare F1-Score     : {f1_flare:6.4f}")
    if "roc_auc" in metrics:
        logger.info(f"  * ROC-AUC Score      : {metrics['roc_auc']:6.4f}")
    if "pr_auc" in metrics:
        logger.info(f"  * PR-AUC Score       : {metrics['pr_auc']:6.4f}")
    tn, fp = cm[0,0], cm[0,1]
    fn, tp = cm[1,0], cm[1,1]
    logger.info(f"  * Confusion Matrix   : TN={tn}, FP={fp} | FN={fn}, TP={tp}")
    logger.info("-" * 78)


def run_partition_pipeline(epochs=None):
    """Train and evaluate a fresh CNN-LSTM on each supplied partition pair."""
    if epochs is None:
        epochs = config.DL_CONFIG["epochs"]

    optimize_gpu_performance()
    set_seed(42)
    results = []

    for partition_id in available_partition_ids():
        X_train, y_train, X_test, y_test = load_partition_pair(partition_id)
        logger.info("=" * 78)
        logger.info(f"PARTITION {partition_id}: Train {X_train.shape}, Test {X_test.shape}")

        train_loader = create_gpu_dataloader(
            X_train, y_train, config.DL_CONFIG["batch_size"], shuffle=True
        )
        test_loader = create_gpu_dataloader(
            X_test, y_test, config.DL_CONFIG["batch_size"], shuffle=False
        )
        model = SolarFlareCNNLSTM(num_classes=config.NUM_CLASSES)
        trainer = CNNLSTMTrainer(model, device=config.DEVICE, criterion=class_weighted_loss(y_train))

        for epoch in range(1, epochs + 1):
            train_loss, train_accuracy = trainer.train_epoch(train_loader)
            if epoch % 10 == 0 or epoch == epochs:
                logger.info(
                    f"[Partition {partition_id} | Epoch {epoch:02d}/{epochs}] "
                    f"Train Loss: {train_loss:.4f}, Train Balanced/Raw Acc: {train_accuracy * 100:.2f}%"
                )
            gc.collect()
            torch.cuda.empty_cache()

        test_loss, _, _, test_targets, test_probs = trainer.evaluate(test_loader)
        if not np.array_equal(test_targets, y_test):
            raise RuntimeError("Test target order changed during evaluation.")

        threshold = getattr(config, "OPERATIONAL_THRESHOLD", 0.5)
        if config.NUM_CLASSES == 2:
            test_preds = (test_probs[:, 1] >= threshold).astype(int)
        else:
            test_preds = np.argmax(test_probs, axis=1)

        metrics = calculate_all_metrics(y_test, test_preds, y_probs=test_probs)
        print_detailed_evaluation(partition_id, "CNN-LSTM", metrics)

        trainer.save_checkpoint(f"cnn_lstm_partition_{partition_id}.pt")
        results.append({"partition_id": partition_id, "metrics": metrics})

        del model, trainer, train_loader, test_loader
        gc.collect()
        torch.cuda.empty_cache()

    mean_raw_acc = np.mean([r['metrics']['raw_accuracy'] for r in results]) * 100.0
    mean_bal_acc = np.mean([r['metrics']['balanced_accuracy'] for r in results]) * 100.0
    mean_tss = np.mean([r['metrics']['tss'].get(1, 0.0) for r in results])
    mean_hss = np.mean([r['metrics']['hss'] for r in results])

    logger.info("=" * 78)
    logger.info("CNN-LSTM SUMMARY ACROSS ALL PARTITIONS:")
    logger.info(f"  Mean Raw Accuracy      : {mean_raw_acc:.2f}%")
    logger.info(f"  Mean Balanced Accuracy : {mean_bal_acc:.2f}%")
    logger.info(f"  Mean Flare TSS         : {mean_tss:.4f}")
    logger.info(f"  Mean Heidke Skill Score: {mean_hss:.4f}")
    logger.info("=" * 78)
    return results


def train_final_dl_model(epochs=None):
    """Fit one deployment CNN-LSTM model on all training partitions."""
    if epochs is None:
        epochs = config.DL_CONFIG["epochs"]

    X_train, y_train = load_all_training_partitions()
    train_loader = create_gpu_dataloader(
        X_train, y_train, config.DL_CONFIG["batch_size"], shuffle=True
    )
    model = SolarFlareCNNLSTM(num_classes=config.NUM_CLASSES)
    trainer = CNNLSTMTrainer(model, device=config.DEVICE, criterion=class_weighted_loss(y_train))

    logger.info(f"Training final CNN-LSTM deployment model on {len(X_train)} sequences...")
    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = trainer.train_epoch(train_loader)
        if epoch % 10 == 0 or epoch == epochs:
            logger.info(
                f"[Final CNN-LSTM | Epoch {epoch:02d}/{epochs}] "
                f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy * 100:.2f}%"
            )
        gc.collect()
        torch.cuda.empty_cache()

    trainer.save_checkpoint("cnn_lstm_final.pt")
    logger.info("Saved final CNN-LSTM deployment model.")
    return trainer


if __name__ == "__main__":
    run_partition_pipeline()
