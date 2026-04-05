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
        return
