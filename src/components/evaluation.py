import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score

from src.utils.logger import get_logger


class Evaluator:
    def __init__(self, config: dict):
        self.config = config
        self.logger = get_logger()

    def evaluate(self, model, X_test, y_test):
        self.logger.info("Evaluating model...")
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        self.logger.info(f"Test Accuracy: {accuracy:.4f}")

        # Log tracked hyperparams & metric results to MLflow
        mlflow.log_params(self.config["model"])
        mlflow.log_metric("accuracy", accuracy)

        # Log trained model pipeline safely inside an MLflow artifact
        model_name = "IrisClassifier"
        mlflow.sklearn.log_model(
            sk_model=model, artifact_path="model", registered_model_name=model_name)
        self.logger.info(f"Model logged to MLflow with name: {model_name}")

        return accuracy
