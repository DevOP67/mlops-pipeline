from fastapi import FastAPI
from backend.routes import predict

app = FastAPI()

app.include_router(predict.router)

@app.get("/")
def home():
    return {"message": "Iris ML Model Pipeline API Running (Modular Architecture)"}