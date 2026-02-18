import mlflow.pyfunc

def load_production_model():
    model = mlflow.pyfunc.load_model(
        model_uri = "models:/IrisClassifier/Production"
    )
    return model