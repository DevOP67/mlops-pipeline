import yaml
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.model import build_model
from src.logger import get_logger


def load_config(path="configs/config.yaml"):
    with open(path, "r") as file:
        return yaml.safe_load(file)


def train():

    logger = get_logger()
    config = load_config()

    experiment_name = config["experiment"]["name"]
    model_config = config["model"]
    training_config = config["training"]

    mlflow.set_tracking_uri("file:./mlruns")
    
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():

        logger.info("Loading dataset...")
        data = load_iris()

        X_train, X_test, y_train, y_test = train_test_split(
            data.data,
            data.target,
            test_size=training_config["test_size"],
            random_state=model_config["random_state"]
        )

        logger.info("Building model...")
        model = build_model(
            n_estimators=model_config["n_estimators"],
            max_depth=model_config["max_depth"],
            random_state=model_config["random_state"]
        )

        logger.info("Training model...")
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        logger.info(f"Accuracy: {accuracy}")

        # Log everything to MLflow
        mlflow.log_params(model_config)
        mlflow.log_metric("accuracy", accuracy)

        model_name = "IrisClassifier"
        mlflow.sklearn.log_model(
            sk_model = model,
            artifact_path = "model",
            registered_model_name = model_name
        )


if __name__ == "__main__":
    train()
