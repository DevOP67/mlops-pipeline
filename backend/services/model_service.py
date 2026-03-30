import os
import time
import mlflow # pyright: ignore[reportMissingImports]
import requests

def wait_for_mlflow():
    mlflow_url = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    for attempt in range(30):
        try:
            response = requests.get(mlflow_url, timeout=2)
            if response.status_code in [200, 404]:  # 404 is fine, server is responding
                print("✓ MLflow is ready.")
                return
        except requests.exceptions.RequestException:
            pass
        
        print(f"Waiting for MLflow... (attempt {attempt + 1}/30)")
        time.sleep(2)
    
    raise Exception("MLflow server not reachable after 60 seconds")

def load_model():
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    
    # Wait for MLflow to be ready
    wait_for_mlflow()
    
    # Set tracking URI
    mlflow.set_tracking_uri(mlflow_uri)
    
    # Load model with retries
    max_retries = 5
    for attempt in range(max_retries):
        try:
            model_uri = "models:/IrisClassifier@production"
            model = mlflow.pyfunc.load_model(model_uri)
            print("✓ Model loaded successfully")
            return model
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Failed to load model (attempt {attempt + 1}/{max_retries}): {str(e)}")
                time.sleep(2)
            else:
                raise Exception(f"Failed to load model after {max_retries} attempts: {str(e)}")
