import os
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import config
from utils import setup_logger, set_seed, verify_hardware_acceleration, calculate_tss, calculate_all_metrics
from data_loader import (
    available_partition_ids,
    extract_ml_features,
    load_all_training_partitions,
    load_partition_pair,
)
from models.xgboost_model import SolarXGBoost
from presentation import SolarPresentationEngine

logger = setup_logger("TrainML")

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

def run_ml_pipeline():
    set_seed(42)
    verify_hardware_acceleration(logger)
    logger.info("Starting XGBoost evaluation across matched train/test partitions...")

    results = []
    for partition_id in available_partition_ids():
        X_train, y_train, X_test, y_test = load_partition_pair(partition_id)
        logger.info(f"Partition {partition_id}: Train {X_train.shape}, Test {X_test.shape}")

        X_train_2d = extract_ml_features(X_train)
        X_test_2d = extract_ml_features(X_test)

        xgb_classifier = SolarXGBoost()
        xgb_classifier.build_model()
        xgb_classifier.train(X_train_2d, y_train)
        test_probs = xgb_classifier.predict_proba(X_test_2d)
        if config.NUM_CLASSES == 2 and test_probs.ndim == 1:
            test_probs = np.column_stack([1.0 - test_probs, test_probs])

        threshold = getattr(config, "OPERATIONAL_THRESHOLD", 0.5)
        if config.NUM_CLASSES == 2:
            test_preds = (test_probs[:, 1] >= threshold).astype(int)
        else:
            test_preds = np.argmax(test_probs, axis=1)

        metrics = calculate_all_metrics(y_test, test_preds, y_probs=test_probs)
        print_detailed_evaluation(partition_id, "XGBoost", metrics)

        xgb_classifier.save_model(f"xgboost_partition_{partition_id}.joblib")
        results.append({"partition_id": partition_id, "metrics": metrics})

    mean_raw_acc = np.mean([r['metrics']['raw_accuracy'] for r in results]) * 100.0
    mean_bal_acc = np.mean([r['metrics']['balanced_accuracy'] for r in results]) * 100.0
    mean_tss = np.mean([r['metrics']['tss'].get(1, 0.0) for r in results])
    mean_hss = np.mean([r['metrics']['hss'] for r in results])

    logger.info("=" * 78)
    logger.info("XGBoost SUMMARY ACROSS ALL PARTITIONS:")
    logger.info(f"  Mean Raw Accuracy      : {mean_raw_acc:.2f}%")
    logger.info(f"  Mean Balanced Accuracy : {mean_bal_acc:.2f}%")
    logger.info(f"  Mean Flare TSS         : {mean_tss:.4f}")
    logger.info(f"  Mean Heidke Skill Score: {mean_hss:.4f}")
    logger.info("=" * 78)
    return results

def train_final_ml_model():
    """Fit one deployment XGBoost model on all training partitions."""
    X_train, y_train = load_all_training_partitions()
    X_train_2d = extract_ml_features(X_train)
    classifier = SolarXGBoost()
    classifier.build_model()
    classifier.train(X_train_2d, y_train)
    classifier.save_model("xgboost_final.joblib")
    logger.info("Saved final XGBoost deployment model.")
    return classifier

if __name__ == "__main__":
    run_ml_pipeline()
