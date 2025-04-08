from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from transformers import pipeline

router = APIRouter()

# Request model for the classification endpoint
class Prompt(BaseModel):
    prompt: str

# Response model for API documentation
class ClassificationResponse(BaseModel):
    prompt: str
    label: str
    score: float
    message: str

# Assumption that the environment supports CUDA
#  Initialize the text-classification pipeline using the fine-tuned model
classifier = pipeline(
    "text-classification",
    model="AManzoni/prompt-classifier",
    device=0,
)

# Mapping to convert the raw model output to human-readable labels.
id2label = {
    "0": "request for visual content creation",
    "1": "conversational message",
    "LABEL_0": "request for visual content creation",
    "LABEL_1": "conversational message",
}

@router.post("/", response_model=ClassificationResponse)
async def classify_prompt(prompt_obj: Prompt):
    prompt_text = prompt_obj.prompt
    try:
        # Get the classification result.
        results = classifier(prompt_text)
        if not results or not isinstance(results, list) or not results[0]:
            raise ValueError("No output returned by the classifier.")

        top_result = results[0]
        top_label = top_result["label"]
        top_score = top_result["score"]

        # Map the model output to a human-readable label.
        readable_label = id2label.get(top_label, top_label)
        message = "Classification successful."

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification error: {e}")

    return ClassificationResponse(
        prompt=prompt_text,
        label=readable_label,
        score=top_score,
        message=message,
    )
