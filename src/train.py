import os

import mlflow
import mlflow.sklearn
import yaml
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.model import build_model
from src.utils.logger import get_logger


def load_config(path="configs/config.yaml"):
    with open(path, "r") as file:
        return yaml.safe_load(file)


def train():
    logger = get_logger()
    config = load_config()

    experiment_name = config["experiment"]["name"]
    model_config = config["model"]
    training_config = config["training"]

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        logger.info("Loading dataset...")
        X, y = load_iris(return_X_y=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=training_config["test_size"],
            random_state=model_config["random_state"],
        )

        logger.info("Building model...")
        model = build_model(
            n_estimators=model_config["n_estimators"],
            max_depth=model_config["max_depth"],
            random_state=model_config["random_state"],
        )

        logger.info("Training model...")
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        # ✅ FIXED NumPy float → Python float
        accuracy = float(accuracy_score(y_test, predictions))

        logger.info(f"Accuracy: {accuracy}")

        mlflow.log_params(model_config)
        mlflow.log_metric("accuracy", accuracy)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="IrisClassifier",
        )


if __name__ == "__main__":
    train()
