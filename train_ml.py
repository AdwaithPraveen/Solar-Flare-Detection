import os
import pickle
import glob
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import config
from utils import setup_logger, set_seed, verify_hardware_acceleration
from data_loader import extract_ml_features
from models.xgboost_model import SolarXGBoost
from presentation import SolarPresentationEngine

logger = setup_logger("TrainML")

def find_partition_files(data_dir):
    """
    Locates SWANSF train/test pickle files or falls back to partition search.
    """
    # Look in data_dir and current working directory
    search_dirs = [data_dir, "."]
    
    X_train_path, y_train_path = None, None
    X_test_path, y_test_path = None, None

    for d in search_dirs:
        # Check standard partition names
        train_x_matches = glob.glob(os.path.join(d, "*X_train*.pkl")) + [f for f in glob.glob(os.path.join(d, "train", "*Partition1*.pkl")) if "Labels" not in f]
        train_y_matches = glob.glob(os.path.join(d, "*y_train*.pkl")) + glob.glob(os.path.join(d, "train", "*Partition1_Labels*.pkl"))
        
        test_x_matches = glob.glob(os.path.join(d, "*X_test*.pkl")) + [f for f in glob.glob(os.path.join(d, "test", "*Partition2*.pkl")) if "Labels" not in f]
        test_y_matches = glob.glob(os.path.join(d, "*y_test*.pkl")) + glob.glob(os.path.join(d, "test", "*Partition2_Labels*.pkl"))

        if train_x_matches and train_y_matches:
            X_train_path = train_x_matches[0]
            y_train_path = train_y_matches[0]
        if test_x_matches and test_y_matches:
            X_test_path = test_x_matches[0]
            y_test_path = test_y_matches[0]

    return X_train_path, y_train_path, X_test_path, y_test_path

def run_ml_pipeline():
    set_seed(42)
    verify_hardware_acceleration(logger)

    logger.info("Starting Machine Learning (XGBoost) Pipeline Execution...")

    # 1. Locate and Load Dataset Partitions
    X_train_p, y_train_p, X_test_p, y_test_p = find_partition_files(config.DATA_DIR)

    if X_train_p and y_train_p and X_test_p and y_test_p:
        logger.info(f"Loading Train Data from: {X_train_p}")
        with open(X_train_p, "rb") as f:
            X_train = pickle.load(f)
        with open(y_train_p, "rb") as f:
            y_train = pickle.load(f)

        logger.info(f"Loading Test Data from: {X_test_p}")
        with open(X_test_p, "rb") as f:
            X_test = pickle.load(f)
        with open(y_test_p, "rb") as f:
            y_test = pickle.load(f)
    else:
        logger.warning("Partition files not found directly. Generating synthetic SWANSF array for demonstration.")
        X_train = np.random.randn(2000, 60, 24).astype(np.float32)
        y_train = np.random.randint(0, 4, size=(2000,))
        X_test = np.random.randn(500, 60, 24).astype(np.float32)
        y_test = np.random.randint(0, 4, size=(500,))

    # Handle dictionary partitions if pickled as dicts
    if isinstance(X_train, dict):
        first_key = list(X_train.keys())[0]
        logger.info(f"Unpacking dictionary partition using key '{first_key}'...")
        X_train = X_train[first_key]
        y_train = y_train[first_key]
        X_test = X_test[list(X_test.keys())[0]]
        y_test = y_test[list(y_test.keys())[0]]

    # Ensure y values are 1D arrays
    y_train = np.array(y_train).squeeze()
    y_test = np.array(y_test).squeeze()

    logger.info(f"Data Loaded successfully -> Train: {X_train.shape}, Test: {X_test.shape}")

    # 2. Extract 2D Statistical Features
    X_train_2d = extract_ml_features(X_train)
    X_test_2d = extract_ml_features(X_test)

    # 3. Initialize and Train GPU XGBoost
    # LEAKAGE FIX: Create an internal 20% validation split from TRAINING data only.
    # The held-out test set (X_test_2d / y_test) must NEVER appear inside fit().
    X_tr_xgb, X_val_xgb, y_tr_xgb, y_val_xgb = train_test_split(
        X_train_2d, y_train,
        test_size=0.20,
        stratify=y_train,
        random_state=42
    )
    xgb_classifier = SolarXGBoost()
    xgb_classifier.build_model()
    xgb_classifier.train(X_tr_xgb, y_tr_xgb, eval_set=[(X_val_xgb, y_val_xgb)])

    # 4. Model Evaluation
    test_probs = xgb_classifier.predict_proba(X_test_2d)

    # XGBoost with binary:logistic returns shape (N,) — a single P(class=1) per sample.
    # Reshape to (N, 2) = [P(class=0), P(class=1)] so all downstream code is uniform.
    if config.NUM_CLASSES == 2 and test_probs.ndim == 1:
        test_probs = np.column_stack([1.0 - test_probs, test_probs])

    test_preds = np.argmax(test_probs, axis=1)

    acc = accuracy_score(y_test, test_preds)
    logger.info(f"XGBoost Test Accuracy: {acc * 100:.2f}%")
    if config.NUM_CLASSES == 3:
        logger.info("\nClassification Report:\n" + classification_report(y_test, test_preds, labels=[0, 1, 2], target_names=["Quiet", "M-Class", "X-Class"]))
    else:
        logger.info("\nClassification Report:\n" + classification_report(y_test, test_preds, labels=[0, 1, 2, 3], target_names=["Quiet", "C-Class", "M-Class", "X-Class"]))

    # 5. Save Model
    xgb_classifier.save_model("xgboost_solar_flare.joblib")

    # 6. Generate Human-Readable Presentation Reports for top samples
    engine = SolarPresentationEngine()
    print("\n" + "="*80)
    print("       SAMPLE PRESENTATION REPORTS GENERATED BY XGBOOST PIPELINE")
    print("="*80)
    
    for idx in range(min(3, len(X_test))):
        report = engine.format_single_report(idx, test_probs[idx], model_name="XGBoost (GPU Accelerated)")
        print(report)

    return test_probs

if __name__ == "__main__":
    run_ml_pipeline()