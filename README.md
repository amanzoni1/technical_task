# Technical Tasks

## Quick Start & Testing

### Important Note on Hardware Compatibility

A GPU is required.
This project has been developed and tested on NVIDIA H100 GPUs x86 architecture.
The current implementation is not compatible with ARM-based GPUs like GH200.

### Setup

Ensure you have an `.env` file in the base directory containing your Hugging Face token:

```
HF_TOKEN=${HF_TOKEN}
```

### Testing the Endpoints

There are two options for testing:

1. **Using Test Files:**
   Run the test scripts available in the `test/` directory:

   - `test/generation/test_generation.py`
   - `test/training/test_training.py`
   - `test/classification/test_classification.py`

2. **Using cURL Commands:**
   Examples:

   - **Test image generation endpoint:**

     ```
     curl -X POST http://localhost:8000/generate/ \
       -H "Content-Type: application/json" \
       -d '{"prompt": "a beautiful woman"}' -o response.json
     ```

     This command sends a prompt to the `/generate/` endpoint and writes the JSON response (including the base64-encoded image) to `response.json`.

   - **Test classification endpoint:**

     ```
     curl -X POST http://localhost:8000/classify/ \
       -H "Content-Type: application/json" \
       -d '{"prompt": "can I see you in the kitchen?"}'
     ```

     - **Test training endpoint:**

     First, prepare a ZIP file of your dataset (for example, `test_dataset.zip`), then run:

     ```
     curl -X POST http://localhost:8000/train/ \
       -F "dataset=@test_dataset.zip" \
       -F 'keywords=["realistic", "detailed", "high quality"]'
     ```

     This command sends the training request with the uploaded dataset and keywords. The response will include a model identifier (`model_id`).

     You can check the training status with:

     ```
     curl -X GET http://localhost:8000/train/status/<model_id>
     ```

     Replace `<model_id>` with the actual model ID returned from the training request.

---

## Repository Structure

- `main.py`: FastAPI application entry point
- `routes/`: Contains endpoint modules
- `utils/`: Helper functions and fine-tuning scripts
- `test/`: Scripts to test the endpoints and review results

## Endpoints

The system is built using FastAPI and is designed to run on GPU for optimal performance. It includes:

- `/generate/`: Endpoint for image generation
- `/train/`: Endpoint for model training (to be implemented)
- `/classify/`: Endpoint for prompt classification

## Task Overview

I approached the assigned tasks as follows:

### 1. Image Generation API

For this component, I implemented an endpoint that generates high-quality images based on user prompts using the Flux model with Super-Realism LoRA fine-tuning.

**Technical details:**

- Base model: `black-forest-labs/FLUX.1-dev`
- LoRA weights: `strangerzonehf/Flux-Super-Realism-LoRA`
- Trigger word: "Super Realism"
- Resolution: 1024×1024

### 2. Model Training API

This API accepts prompt keywords and a dataset (uploaded as a ZIP file) to fine-tune the image generation model.

Two versions are implemented in the utils folder:

- training_diffusers.py:
  A simpler training implementation using the Diffusers framework.
  This version is working and leverages the train_text_to_image_lora.py file from the Diffusers GitHub repository.
- training_transformers.py:
  A more complex, custom training loop using Transformers.
  Adapting many components was required, especially in building the compute_loss function—
  to correctly handle data collation, precision/device management, and the hidden input
  signature of PEFT-wrapped models. It’s not working yet.

### 3. Prompt Classification API

For the prompt classification task, I initially tried a zero-shot classification approach but found that it struggled with ambiguous prompts. To address this, I fine-tuned a BART model (utils/finetune_classifier.py) specifically for distinguishing between image requests and casual conversation.

**Fine-tuning process:**

1. Created a dataset of labeled examples
2. Fine-tuned a BART model (facebook/bart-large-mnli)
3. Achieved 95.24% accuracy on the test set
4. Deployed the model to Hugging Face Hub for easy access

Test results:

```
Evaluating model on test set...
Evaluation results: {
    'eval_loss': 0.14287129044532776,
    'eval_accuracy': 0.9523809523809523,
    'eval_f1': 0.9530864197530863,
    'eval_runtime': 0.8195,
    'eval_samples_per_second': 25.624,
    'eval_steps_per_second': 7.321,
    'epoch': 4.0
}
```

After more testing with various prompts, the classifier has proven to be quite accurate. Seems that only a few ambiguous cases remain that may create doubts or lead to misclassification.
