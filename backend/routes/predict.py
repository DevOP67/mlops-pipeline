import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel

from backend.services.model_service import load_model

router = APIRouter()

# Lazy load model - only load when first prediction is needed
model = None


def get_model():
    global model
    if model is None:
        model = load_model()
    return model


class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@router.post("/predict")
def predict_endpoint(data: IrisInput):
    current_model = get_model()

    # Pack feature list into 2D array matrix required by Sklearn Pipeline API
    features = np.array(
        [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])

    prediction = current_model.predict(features)

    return {"prediction": int(prediction[0])}
