from sklearn.ensemble import RandomForestClassifier  # pyright: ignore[reportMissingImports]

def build_model(n_estimators, max_depth, random_state):
    return RandomForestClassifier(
        n_estimators = n_estimators,
        max_depth = max_depth,
        random_state = random_state
    )