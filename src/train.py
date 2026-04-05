import yaml
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.model import build_model
from src.utils.logger import get_logger


def load_config(path="configs/config.yaml"):
    with open(path, "r") as file:
        return
