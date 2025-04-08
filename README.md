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

The second task involves creating an API to train models with custom datasets and keywords. This component will be implemented to allow custom fine-tuning of the image generation model based on user-specified data.

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

ssh -i /Users/andreamanzoni/Desktop/code/lambda-ssh.pem ubuntu@209.20.158.250

git clone <your-repo-url>
cd <your-repo-directory>
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

huggingface-cli login

uvicorn main:app --host 0.0.0.0 --port 8000
