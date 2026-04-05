import os
import pickle

import mlflow
import yaml

from src.components.evaluation import Evaluator
from src.components.model_trainer import ModelTrainer
from src.components.preprocessing import Preprocessor
from src.utils.logger import get_logger


def load_config(path="configs/config.yaml"):
    if not os.path.exists(path):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = os.path.join(base_dir, "configs", "config.yaml")

    with open(path, "r") as file:
        return yaml.safe_load(file)


def save_model_pickle(model, path="artifacts/models/model.pkl"):
    """Persist model as a plain pickle so the API can load it without an MLflow server."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"✓ Model saved to {path}")


def run_pipeline():
    logger = get_logger()
    logger.info("Starting training pipeline...")

    config = load_config()

    # File-based MLflow tracking — works locally and in CI without a server
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = config["experiment"]["name"]
    mlflow.set_experiment(experiment_name)

    logger.info(f"MLflow tracking URI: {tracking_uri} | Experiment: '{experiment_name}'")

    with mlflow.start_run():
        preprocessor = Preprocessor(config)
        X_train, X_test, y_train, y_test = preprocessor.process()

        trainer = ModelTrainer(config)
        model = trainer.train(X_train, y_train)

        evaluator = Evaluator(config)
        evaluator.evaluate(model, X_test, y_test)

        # Also save as plain pickle for the API to load without an MLflow server
        save_model_pickle(model)

    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    run_pipeline()
