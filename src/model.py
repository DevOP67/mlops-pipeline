from sklearn.ensemble import RandomForestClassifier

from src.utils.logger import get_logger

logger = get_logger()


def build_model(n_estimators=100, max_depth=5, random_state=42):
    """
    Build RandomForest model matching pipeline config.
    Note: Pipeline uses scaler+RF, this is simplified.
    """
    logger.info(f"Building model: RF(n_estimators={n_estimators}, max_depth={max_depth})")
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    return model
