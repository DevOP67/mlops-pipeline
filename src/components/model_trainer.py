from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.logger import get_logger


class ModelTrainer:
    def __init__(self, config: dict):
        self.config = config
        self.logger = get_logger()

    def train(self, X_train, y_train):
        self.logger.info("Building model pipeline (Scaler + RandomForest)...")
        model_config = self.config["model"]

        # Using a unified pipeline scales seamlessly to new, unpredictable data distributions automatically!
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=model_config["n_estimators"],
                        max_depth=model_config["max_depth"],
                        random_state=model_config["random_state"],
                    ),
                ),
            ]
        )

        self.logger.info("Training pipeline...")
        pipeline.fit(X_train, y_train)

        return pipeline
