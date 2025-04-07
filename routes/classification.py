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
# Initialize the zero-shot classification pipeline with GPU acceleration
classifier = pipeline(
    # "text-classification",
    # model="utils/fine-tuned-classifier",
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=0,
)


@router.post("/", response_model=ClassificationResponse)
async def classify_prompt(prompt_obj: Prompt):
    prompt_text = prompt_obj.prompt
    candidate_labels = ["request for visual content creation", "conversational message"]

    try:
        # Classify the prompt using zero-shot classification with a hypothesis template.
        result = classifier(
            prompt_text,
            candidate_labels,
            hypothesis_template="The user is asking for {}",
        )
        top_label = result["labels"][0]
        top_score = result["scores"][0]
        second_score = result["scores"][1]
        score_diff = top_score - second_score
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification error: {e}")

    # Define a threshold for ambiguity. For example, if the difference is less than 0.1
    ambiguity_threshold = 0.15
    if score_diff < ambiguity_threshold:
        return ClassificationResponse(
            prompt=prompt_text,
            label=top_label,
            score=top_score,
            message="Sorry, I'm not sure I've properly understood. Could you please reformulate your question?",
        )
    else:
        return ClassificationResponse(
            prompt=prompt_text,
            label=top_label,
            score=top_score,
            message="Classification successful.",
        )
