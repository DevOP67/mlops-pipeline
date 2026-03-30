import yaml
import mlflow
import os

from src.components.model_trainer import ModelTrainer
from src.components.preprocessing import Preprocessor
from src.components.evaluation import Evaluator
from src.utils.logger import get_logger

def load_config(path="configs/config.yaml"):
    # Ensure relative path resolves correctly based on execution dir
    if not os.path.exists(path):
        # fallback to looking relative to this script
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = os.path.join(base_dir, "configs", "config.yaml") # actually joining with configs/config.yaml is safer
        
    with open(path, "r") as file:
        return yaml.safe_load(file)

def run_pipeline():
    logger = get_logger()
    logger.info("Starting training pipeline...")

    config = load_config()

    # Setup MLflow Tracking (can be overridden by environment variable, falling back to localhost)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_uri)
    
    experiment_name = config["experiment"]["name"]
    mlflow.set_experiment(experiment_name)

    logger.info(f"Targeting MLflow Tracking server at {tracking_uri} for experiment '{experiment_name}'")

    with mlflow.start_run():
        preprocessor = Preprocessor(config)
        X_train, X_test, y_train, y_test = preprocessor.process()

        trainer = ModelTrainer(config)
        model = trainer.train(X_train, y_train)

        evaluator = Evaluator(config)
        evaluator.evaluate(model, X_test, y_test)

    logger.info("Pipeline completed.")

if __name__ == "__main__":
    run_pipeline()
