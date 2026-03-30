from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from src.utils.logger import get_logger

class Preprocessor:
    def __init__(self, config: dict):
        self.config = config
        self.logger = get_logger()

    def process(self):
        self.logger.info("Loading dataset...")
        data = load_iris()

        test_size = self.config["training"]["test_size"]
        random_state = self.config["model"]["random_state"]

        self.logger.info(f"Splitting data with test_size={test_size}")
        X_train, X_test, y_train, y_test = train_test_split(
            data.data,
            data.target,
            test_size=test_size,
            random_state=random_state
        )

        return X_train, X_test, y_train, y_test
