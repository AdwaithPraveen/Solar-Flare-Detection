import os
import sys
import numpy as np
import config
from utils import setup_logger, set_seed, verify_hardware_acceleration
from train_ml import run_ml_pipeline
from train_dl import run_stratified_kfold_pipeline
from ensemble import run_ensemble_pipeline
from presentation import SolarPresentationEngine

logger = setup_logger("MainPipeline")

def main():
    """
    Executes the complete Space Weather Analytics for Solar Flares (SWANSF) Pipeline:
    1. Verifies NVIDIA CUDA hardware acceleration.
    2. Runs Machine Learning (XGBoost GPU) training & evaluation.
    3. Runs Deep Learning (PyTorch CNN-LSTM GPU) 5-Fold Stratified Training.
    4. Runs GPU Soft-Voting Ensemble (PyTorch + XGBoost blending).
    5. Generates side-by-side model comparison presentation reports.
    """
    print("\n" + "="*80)
    print("      SPACE WEATHER ANALYTICS FOR SOLAR FLARES (SWANSF) PREDICTION SYSTEM")
    print("="*80 + "\n")

    # 1. Environment & Diagnostics Verification
    set_seed(42)
    verify_hardware_acceleration(logger)

    # 2. Machine Learning Execution (XGBoost)
    logger.info("Executing Machine Learning Pipeline Phase...")
    try:
        ml_probs = run_ml_pipeline()
    except Exception as e:
        logger.error(f"Machine Learning pipeline failed with error: {str(e)}")
        sys.exit(1)

    # 3. Deep Learning Execution (5-Fold Stratified PyTorch CNN-LSTM)
    logger.info("Executing Deep Learning Pipeline Phase (~45-60 min)...")
    try:
        dl_probs = run_stratified_kfold_pipeline(
            n_splits=5, 
            epochs_per_fold=config.DL_CONFIG.get("epochs", 60)
        )
    except Exception as e:
        logger.error(f"Deep Learning pipeline failed with error: {str(e)}")
        sys.exit(1)

    # 4. Ensemble Pipeline Phase (PyTorch + XGBoost Soft-Voting)
    logger.info("Executing Ensemble Pipeline Phase...")
    try:
        ensemble_probs = run_ensemble_pipeline()
    except Exception as e:
        logger.error(f"Ensemble pipeline failed with error: {str(e)}")
        sys.exit(1)

    # 5. Comparative Presentation Output Generation
    logger.info("Generating Side-by-Side Model Comparison Reports...")
    engine = SolarPresentationEngine()
    
    if ml_probs is not None and dl_probs is not None and len(ml_probs) > 0:
        num_samples_to_report = min(5, len(ml_probs))
        
        print("\n" + "#"*80)
        print("        EXECUTIVE PRESENTATION SUMMARY: MODEL COMPARISON REPORTS")
        print("#"*80)

        for sample_idx in range(num_samples_to_report):
            comparison_text = engine.compare_models_report(
                sample_idx=sample_idx,
                ml_probs=ml_probs[sample_idx],
                dl_probs=dl_probs[sample_idx],
                ensemble_probs=ensemble_probs[sample_idx] if ensemble_probs is not None else None
            )
            print(comparison_text)

    logger.info("Entire pipeline execution completed successfully!")

if __name__ == "__main__":
    main()