import logging
import os
import pickle

logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/models/model.pkl")


def load_model():
    """
    Load the trained sklearn pipeline from a local pickle file.

    The model is saved during the training pipeline execution to
    artifacts/models/model.pkl. This approach avoids the need for
    a running MLflow tracking server at inference time.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at '{MODEL_PATH}'. "
            "Run the training pipeline first: python -m src.pipeline.train_pipeline"
        )

    logger.info(f"Loading model from {MODEL_PATH} ...")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    logger.info("Model loaded successfully.")
    return model
