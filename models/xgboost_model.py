import os
import joblib
import numpy as np
import xgboost as xgb
import config
from utils import setup_logger

logger = setup_logger("XGBoostModel")

class SolarXGBoost:
    """
    XGBoost Machine Learning Classifier optimized for GPU acceleration.
    """
    def __init__(self, params=None):
        self.params = params if params is not None else config.ML_CONFIG
        self.model = None

    def build_model(self):
        """Initializes the XGBoost Classifier with GPU tree parameters."""
        logger.info("Initializing XGBoost Classifier with NVIDIA CUDA acceleration...")
        self.model = xgb.XGBClassifier(**self.params)
        return self.model

    def train(self, X_train, y_train, eval_set=None):
        """
        Trains the XGBoost model on 2D extracted features using GPU.
        """
        if self.model is None:
            self.build_model()

        logger.info(f"Training XGBoost model on {X_train.shape[0]} samples with {X_train.shape[1]} features...")
        
        fit_params = {}
        if eval_set is not None:
            fit_params["eval_set"] = eval_set
            fit_params["verbose"] = True

        self.model.fit(X_train, y_train, **fit_params)
        logger.info("XGBoost GPU Training complete.")

    def predict_proba(self, X):
        """
        Generates probability distributions across the 4 solar flare classes.
        Output shape: (N, 4)
        """
        if self.model is None:
            raise ValueError("Model is not trained or loaded yet.")
        
        orig_device = self.model.get_params().get('device', 'cpu')
        self.model.set_params(device='cpu')
        res = self.model.predict_proba(X)
        self.model.set_params(device=orig_device)
        return res

    def predict(self, X):
        """Generates hard class predictions (0, 1, 2, or 3)."""
        if self.model is None:
            raise ValueError("Model is not trained or loaded yet.")
        
        orig_device = self.model.get_params().get('device', 'cpu')
        self.model.set_params(device='cpu')
        res = self.model.predict(X)
        self.model.set_params(device=orig_device)
        return res

    def save_model(self, filename: str = "xgboost_solar_flare.joblib") -> None:
        """Saves the trained model to disk."""
        try:
            filepath = os.path.join(config.MODEL_SAVE_DIR, filename)
            joblib.dump(self.model, filepath)
            logger.info(f"Saved XGBoost model to: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save XGBoost model: {e}")

    def load_model(self, filename: str = "xgboost_solar_flare.joblib") -> None:
        """Loads a pre-trained model from disk."""
        filepath = os.path.join(config.MODEL_SAVE_DIR, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No saved model found at {filepath}")
        
        try:
            self.model = joblib.load(filepath)
            logger.info(f"Loaded XGBoost model from: {filepath}")
        except Exception as e:
            logger.error(f"Failed to load XGBoost model: {e}")
            raise

if __name__ == "__main__":
    # Dry-run test with dummy 2D data
    dummy_X = np.random.randn(200, 120)
    dummy_y = np.random.randint(0, 4, size=(200,))
    
    xgb_runner = SolarXGBoost()
    xgb_runner.build_model()
    xgb_runner.train(dummy_X, dummy_y)
    probs = xgb_runner.predict_proba(dummy_X[:5])
    logger.info(f"Dry-run prediction probabilities shape: {probs.shape}")   