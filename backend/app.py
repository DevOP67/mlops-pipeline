from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from backend.routes import predict

app = FastAPI(
    title="🌌 Iris Intelligence — MLOps API",
    description="High-performance machine learning inference gateway for Iris-species classification.",
    version="1.0.0"
)

app.include_router(predict.router)

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🌌 Iris Intelligence API</title>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;600&family=Inter:wght@400;500&display=swap');
            
            :root {
                --primary: #6366f1;
                --secondary: #a855f7;
                --bg: #0f172a;
                --card-bg: rgba(30, 41, 59, 0.7);
                --text: #f8fafc;
            }

            body {
                margin: 0;
                padding: 0;
                font-family: 'Inter', sans-serif;
                background-color: var(--bg);
                color: var(--text);
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
                background-image: 
                    radial-gradient(circle at 10% 20%, rgba(99, 102, 241, 0.15) 0%, transparent 40%),
                    radial-gradient(circle at 90% 80%, rgba(168, 85, 247, 0.15) 0%, transparent 40%);
            }

            .container {
                max-width: 800px;
                padding: 3rem;
                background: var(--card-bg);
                backdrop-filter: blur(20px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 2rem;
                box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
                text-align: center;
                animation: fadeIn 0.8s ease-out;
            }

            h1 {
                font-family: 'Space Grotesk', sans-serif;
                font-size: 4rem;
                margin: 0 0 1rem;
                background: linear-gradient(135deg, var(--primary), var(--secondary));
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }

            p {
                font-size: 1.25rem;
                line-height: 1.6;
                color: #94a3b8;
                margin-bottom: 2rem;
            }

            .badge {
                display: inline-block;
                padding: 0.5rem 1.25rem;
                background: rgba(99, 102, 241, 0.1);
                border: 1px solid var(--primary);
                color: var(--primary);
                border-radius: 9999px;
                font-weight: 600;
                font-size: 0.875rem;
                margin-bottom: 1.5rem;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }

            .button-group {
                display: flex;
                gap: 1.5rem;
                justify-content: center;
            }

            .btn {
                padding: 1rem 2rem;
                border-radius: 1rem;
                font-weight: 600;
                text-decoration: none;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                font-family: 'Space Grotesk', sans-serif;
                font-size: 1.1rem;
            }

            .btn-primary {
                background: var(--primary);
                color: white;
                box-shadow: 0 10px 15px -3px rgba(99, 102, 241, 0.4);
            }

            .btn-primary:hover {
                transform: translateY(-3px) scale(1.02);
                box-shadow: 0 20px 25px -5px rgba(99, 102, 241, 0.5);
            }

            .btn-secondary {
                background: rgba(148, 163, 184, 0.1);
                color: #f8fafc;
                border: 1px solid rgba(148, 163, 184, 0.2);
            }

            .btn-secondary:hover {
                background: rgba(148, 163, 184, 0.15);
                transform: translateY(-2px);
            }

            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="badge">🔥 MLOps Enabled</div>
            <h1>Iris Intelligence</h1>
            <p>
                The <b>Iris Intelligence</b> API is actively serving production-grade 
                machine learning predictions for botanical species classification. 
                Built with FastAPI and monitored by MLflow.
            </p>
            <div class="button-group">
                <a href="/docs" class="btn btn-primary">📖 Explore API Docs</a>
                <a href="https://github.com/DevOP6/mlops-pipeline" class="btn btn-secondary">📦 Source Code</a>
            </div>
        </div>
    </body>
    </html>
    """