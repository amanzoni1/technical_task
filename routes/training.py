from fastapi import APIRouter, UploadFile, File, BackgroundTasks
from typing import List

router = APIRouter()


@router.post("/", summary="Train the model with provided prompt keywords and dataset")
async def train_model(
    prompt_keywords: List[str],
    dataset: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
):

    return {

    }
