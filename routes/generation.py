import io
import base64
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

# Assumption that the environment supports CUDA
# Global Model Setup (runs once at startup)
base_model = "black-forest-labs/FLUX.1-dev"
lora_repo = "strangerzonehf/Flux-Super-Realism-LoRA"
trigger_word = "Super Realism"

pipeline = DiffusionPipeline.from_pretrained(base_model, torch_dtype=torch.bfloat16)
pipeline.load_lora_weights(lora_repo)
device = torch.device("cuda")
pipeline.to(device)

############################################
# Optimization Block: Replace default scheduler and enable GPU-specific optimizations
# Replace default scheduler with an optimized DPMSolverMultistepScheduler that uses trailing timestep spacing
# and Karras sigmas to improve image quality with fewer inference steps.
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
    pipeline.scheduler.config,
    algorithm_type="sde-dpmsolver++",
    timestep_spacing="trailing",
    use_karras_sigmas=True,
)

# Enable attention slicing to reduce memory usage
pipeline.enable_attention_slicing(slice_size="auto")

# Enable memory efficient attention (if available)
try:
    pipeline.enable_xformers_memory_efficient_attention()
except Exception as e:
    print(f"Warning: xformers memory efficient attention could not be enabled: {e}")
############################################

# Request model for input
class GenerateRequest(BaseModel):
    prompt: str


# Endpoint
@router.post("/", summary="Generate an image from a text prompt")
async def generate_image(request: GenerateRequest):
    try:
        prompt_text = request.prompt
        full_prompt = f"{trigger_word}, {prompt_text}"

        # Generate the image
        result = pipeline(
            prompt=full_prompt,
            # num_inference_steps=40,
            num_inference_steps=20,
            width=1024,
            height=1024,
            guidance_scale=6,
        )
        image = result.images[0]

        # Convert the image to a base64-encoded string
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {"image": img_str, "prompt_used": full_prompt}

    except Exception as e:
        return {"error": str(e)}, 500
