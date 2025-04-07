from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class ClassifyRequest(BaseModel):
    prompt: str


@router.post("/", summary="Classify prompt as image-generation or casual conversation")
async def classify_prompt(request: ClassifyRequest):
    prompt = request.prompt.lower()
    # Simple rule-based classification. Consider using a pre-trained model for robust results.
    if any(keyword in prompt for keyword in ["kitchen", "paint", "draw", "sketch"]):
        classification = "image_generation"
    else:
        classification = "casual"
    return {"classification": classification}
