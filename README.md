# Technical Tasks

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

### 2. Model Training API (To Be Implemented)

This API accepts prompt keywords and a dataset (uploaded as a ZIP file) to fine-tune the image generation model.

Two versions are implemented in the utils folder:
• training_lora.py:
A simpler training implementation using the Diffusers framework. This version is stable, although it may not provide the granularity of a full-from-scratch solution.
• training_transformers.py:
A more complex, custom training loop implemented using Transformers. This approach aims to train all components from scratch but has proven more challenging in terms of memory and precision management.

### 3. Prompt Classification API

For the prompt classification task, I initially tried a zero-shot classification approach but found that it struggled with ambiguous prompts. To address this, I fine-tuned a BART model specifically for distinguishing between image requests and casual conversation.

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
