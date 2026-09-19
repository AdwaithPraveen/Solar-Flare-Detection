import os
import sys
import config
from utils import setup_logger, set_seed, verify_hardware_acceleration
from train_ml import run_ml_pipeline, train_final_ml_model
from train_dl import run_partition_pipeline, train_final_dl_model

logger = setup_logger("MainPipeline")

def main():
    """
    Evaluates ML and DL models across the five supplied SWANSF partition pairs:
    1. Verifies NVIDIA CUDA hardware acceleration.
    2. Trains and evaluates XGBoost on each matched train/test partition.
    3. Trains and evaluates CNN-LSTM on each matched train/test partition.
    4. Trains final deployment models on all training partitions.
    """
    print("\n" + "="*80)
    print("      SPACE WEATHER ANALYTICS FOR SOLAR FLARES (SWANSF) PREDICTION SYSTEM")
    print("="*80 + "\n")

    # 1. Environment & Diagnostics Verification
    set_seed(42)
    verify_hardware_acceleration(logger)

    # 2. Machine Learning Execution (XGBoost)
    logger.info("Executing Machine Learning Pipeline Phase (XGBoost)...")
    try:
        ml_results = run_ml_pipeline()
    except Exception as e:
        logger.error(f"Machine Learning pipeline failed with error: {str(e)}")
        sys.exit(1)

    # 3. Deep Learning Execution (CNN-LSTM)
    logger.info("Executing Deep Learning Pipeline Phase (PyTorch CNN-LSTM)...")
    try:
        dl_results = run_partition_pipeline(epochs=config.DL_CONFIG.get("epochs", 40))
    except Exception as e:
        logger.error(f"Deep Learning pipeline failed with error: {str(e)}")
        sys.exit(1)

    logger.info(
        f"Completed {len(ml_results)} XGBoost and {len(dl_results)} CNN-LSTM partition evaluations."
    )

    # 4. Final Deployment Models
    logger.info("Training final deployment models on all training partitions...")
    try:
        train_final_ml_model()
        train_final_dl_model(epochs=config.DL_CONFIG.get("epochs", 40))
    except Exception as e:
        logger.error(f"Final deployment-model training failed with error: {str(e)}")
        sys.exit(1)

    logger.info("Entire pipeline execution completed successfully!")

if __name__ == "__main__":
    main()
