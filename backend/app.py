from fastapi import FastAPI

from backend.routes import predict

app = FastAPI(
    title="Iris Intelligence API",
    description="MLOps inference API for Iris species classification.",
    version="1.0.0",
)

app.include_router(predict.router)


@app.get("/")
def root():
    return {"message": "Iris Intelligence API is running"}
