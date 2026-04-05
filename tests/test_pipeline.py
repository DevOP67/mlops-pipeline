"""
tests/test_pipeline.py
Unit tests for ML pipeline components (preprocessor, trainer, evaluator).
All tests use file-based MLflow to avoid needing a running server.
"""

import os

import numpy as np
import pytest

# Use file-based MLflow for tests — no server needed
os.environ.setdefault("MLFLOW_TRACKING_URI", "file:./mlruns")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def config():
    """Load the real project config."""
    import yaml

    config_path = os.path.join(os.path.dirname(
        __file__), "..", "configs", "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def preprocessed_data(config):
    """Run the preprocessor once and share across tests."""
    from src.components.preprocessing import Preprocessor

    prep = Preprocessor(config)
    return prep.process()


# ---------------------------------------------------------------------------
# Imports & Dependencies
# ---------------------------------------------------------------------------


class TestImports:
    def test_sklearn_importable(self):
        import sklearn

        assert sklearn.__version__ is not None

    def test_mlflow_importable(self):
        import mlflow

        assert mlflow.__version__ is not None

    def test_pandas_importable(self):
        import pandas as pd

        assert pd.__version__ is not None

    def test_numpy_importable(self):
        assert np.__version__ is not None

    def test_fastapi_importable(self):
        import fastapi

        assert fastapi.__version__ is not None


# ---------------------------------------------------------------------------
# Config Loading
# ---------------------------------------------------------------------------


class TestConfig:
    def test_config_loads(self, config):
        assert config is not None

    def test_config_has_experiment(self, config):
        assert "experiment" in config
        assert "name" in config["experiment"]

    def test_config_has_model(self, config):
        assert "model" in config
        assert "n_estimators" in config["model"]
        assert "max_depth" in config["model"]

    def test_config_has_training(self, config):
        assert "training" in config
        assert "test_size" in config["training"]
        assert 0 < config["training"]["test_size"] < 1


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------


class TestPreprocessor:
    def test_process_returns_four_arrays(self, preprocessed_data):
        assert len(preprocessed_data) == 4

    def test_train_test_split_shapes(self, preprocessed_data):
        X_train, X_test, y_train, y_test = preprocessed_data
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)

    def test_feature_dimensions(self, preprocessed_data):
        X_train, X_test, y_train, y_test = preprocessed_data
        # Iris has 4 features
        assert X_train.shape[1] == 4
        assert X_test.shape[1] == 4

    def test_no_nan_in_features(self, preprocessed_data):
        X_train, X_test, _, _ = preprocessed_data
        assert not np.isnan(X_train).any(), "NaN values found in X_train"
        assert not np.isnan(X_test).any(), "NaN values found in X_test"

    def test_labels_are_integer(self, preprocessed_data):
        _, _, y_train, y_test = preprocessed_data
        assert y_train.dtype in (np.int32, np.int64, int)

    def test_correct_split_ratio(self, config, preprocessed_data):
        X_train, X_test, _, _ = preprocessed_data
        total = len(X_train) + len(X_test)
        expected_test = config["training"]["test_size"]
        actual_test = len(X_test) / total
        assert abs(actual_test - expected_test) < 0.05


# ---------------------------------------------------------------------------
# Model Trainer
# ---------------------------------------------------------------------------


class TestModelTrainer:
    @pytest.fixture(scope="class")
    def trained_model(self, config, preprocessed_data):
        from src.components.model_trainer import ModelTrainer

        X_train, _, y_train, _ = preprocessed_data
        trainer = ModelTrainer(config)
        return trainer.train(X_train, y_train)

    def test_model_not_none(self, trained_model):
        assert trained_model is not None

    def test_model_has_predict(self, trained_model):
        assert hasattr(trained_model, "predict")

    def test_model_has_predict_proba(self, trained_model):
        assert hasattr(trained_model, "predict_proba")

    def test_model_predicts_on_train_data(self, trained_model, preprocessed_data):
        X_train, _, _, _ = preprocessed_data
        preds = trained_model.predict(X_train)
        assert len(preds) == len(X_train)

    def test_model_predicts_on_test_data(self, trained_model, preprocessed_data):
        _, X_test, _, _ = preprocessed_data
        preds = trained_model.predict(X_test)
        assert len(preds) == len(X_test)

    def test_predictions_are_valid_classes(self, trained_model, preprocessed_data):
        _, X_test, _, y_test = preprocessed_data
        preds = trained_model.predict(X_test)
        valid_classes = set(np.unique(y_test))
        for p in preds:
            assert p in valid_classes, f"Unexpected class {p}"

    def test_training_accuracy_above_threshold(self, trained_model, preprocessed_data):
        from sklearn.metrics import accuracy_score

        X_train, _, y_train, _ = preprocessed_data
        preds = trained_model.predict(X_train)
        acc = accuracy_score(y_train, preds)
        assert acc >= 0.80, f"Training accuracy too low: {acc:.2%}"


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class TestEvaluator:
    def test_evaluator_runs_without_error(self, config, preprocessed_data):
        import mlflow

        from src.components.evaluation import Evaluator
        from src.components.model_trainer import ModelTrainer

        X_train, X_test, y_train, y_test = preprocessed_data
        trainer = ModelTrainer(config)
        model = trainer.train(X_train, y_train)

        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment("test_evaluation")

        with mlflow.start_run():
            evaluator = Evaluator(config)
            evaluator.evaluate(model, X_test, y_test)  # should not raise


# ---------------------------------------------------------------------------
# Full Pipeline Integration (lightweight, no server)
# ---------------------------------------------------------------------------


class TestFullPipeline:
    def test_pipeline_runs_end_to_end(self):
        """Run the complete pipeline using file-based MLflow."""
        import mlflow

        from src.pipeline.train_pipeline import run_pipeline

        mlflow.set_tracking_uri("file:./mlruns")
        # Should complete without exceptions
        run_pipeline()
