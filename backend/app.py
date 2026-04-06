from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from backend.routes import predict

app = FastAPI(
    title="Iris Intelligence API",
    description="MLOps inference API for Iris species classification.",
    version="1.0.0",
)

app.include_router(predict.router)


# Root endpoint (for API + tests)
@app.get("/")
def root():
    return {"message": "Iris Intelligence API is running"}


# ✅ UI Route (THIS WAS MISSING)
@app.get("/ui", response_class=HTMLResponse)
def ui():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Iris ML UI</title>
</head>
<body>
    <h1>Iris Intelligence 🚀</h1>
    <p>Your ML API is live!</p>
    <p>Go to <a href="/docs">/docs</a> to test the model</p>
</body>
</html>
"""
