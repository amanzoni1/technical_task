from fastapi import FastAPI
from routes import generation, classification, training

app = FastAPI(title="LoRA Image Generation & Training API")

# Include router from each module
app.include_router(generation.router, prefix="/generate", tags=["Image Generation"])
app.include_router(training.router, prefix="/train", tags=["Training"])
app.include_router(classification.router, prefix="/classify", tags=["Classification"])

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
