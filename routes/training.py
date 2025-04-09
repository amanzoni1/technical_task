import os
import time
import shutil
import zipfile
from typing import List
from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException
from pydantic import BaseModel
import json
# from utils.training_diffusers import train_lora_model
from utils.training_transformers import train_lora_model

router = APIRouter()


class TrainingRequest(BaseModel):
    keywords: List[str]


@router.post(
    "/", summary="Train a LoRA adapter using the provided dataset and keywords"
)
def train_model(
    background_tasks: BackgroundTasks,
    dataset: UploadFile = File(...),
    keywords: str = Form(...),  # JSON string of keyword list
):
    try:
        # Parse keywords
        keyword_list = json.loads(keywords)

        # Create unique identifier for this training run
        model_id = f"lora_{int(time.time())}"
        output_dir = f"storage/trained_models/{model_id}"
        dataset_dir = f"storage/datasets/{model_id}"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(dataset_dir, exist_ok=True)

        # Save the uploaded dataset
        dataset_path = f"{dataset_dir}/dataset.zip"
        with open(dataset_path, "wb") as buffer:
            shutil.copyfileobj(dataset.file, buffer)

        # Extract dataset
        with zipfile.ZipFile(dataset_path, "r") as zip_ref:
            zip_ref.extractall(dataset_dir)

        # Start training in background
        background_tasks.add_task(
            train_lora_model,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            keywords=keyword_list,
        )

        return {
            "status": "Training started",
            "model_id": model_id,
            "keywords": keyword_list,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{model_id}", summary="Check training status")
def check_status(model_id: str):
    model_dir = f"storage/trained_models/{model_id}"

    if not os.path.exists(model_dir):
        raise HTTPException(status_code=404, detail="Model not found")

    # Simple status check
    status = "in_progress"

    if os.path.exists(f"{model_dir}/pytorch_lora_weights.bin") or os.path.exists(
        f"{model_dir}/pytorch_lora_weights.safetensors"
    ):
        status = "complete"

    if os.path.exists(f"{model_dir}/error.txt"):
        status = "failed"

    return {"status": status, "model_id": model_id}
