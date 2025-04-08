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

The training job is initiated asynchronously, immediately returning a job ID, while the actual training runs in the background (handled via utilities in the utils/ directory).

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

ou should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Map: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 2476.77 examples/s]
Map: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 21/21 [00:00<00:00, 1538.82 examples/s]
Starting training...
{'loss': 0.8197, 'grad_norm': 12.770651817321777, 'learning_rate': 1.9062500000000003e-05, 'epoch': 0.25}
{'loss': 0.5597, 'grad_norm': 13.700414657592773, 'learning_rate': 1.7812500000000003e-05, 'epoch': 0.5}
{'loss': 0.3305, 'grad_norm': 5.97348690032959, 'learning_rate': 1.6562500000000003e-05, 'epoch': 0.75}
{'loss': 0.2035, 'grad_norm': 42.699981689453125, 'learning_rate': 1.5312500000000003e-05, 'epoch': 1.0}
{'loss': 0.3411, 'grad_norm': 1.5188775062561035, 'learning_rate': 1.4062500000000001e-05, 'epoch': 1.25}
{'loss': 0.0201, 'grad_norm': 0.28559166193008423, 'learning_rate': 1.2812500000000001e-05, 'epoch': 1.5}
{'loss': 0.0031, 'grad_norm': 0.24607136845588684, 'learning_rate': 1.1562500000000002e-05, 'epoch': 1.75}
{'loss': 0.0023, 'grad_norm': 0.030707012861967087, 'learning_rate': 1.0312500000000002e-05, 'epoch': 2.0}
{'loss': 0.0011, 'grad_norm': 0.04747402295470238, 'learning_rate': 9.0625e-06, 'epoch': 2.25}
{'loss': 0.0004, 'grad_norm': 0.019453033804893494, 'learning_rate': 7.8125e-06, 'epoch': 2.5}
{'loss': 0.0004, 'grad_norm': 0.012176189571619034, 'learning_rate': 6.5625e-06, 'epoch': 2.75}
{'loss': 0.0004, 'grad_norm': 0.012131319381296635, 'learning_rate': 5.3125e-06, 'epoch': 3.0}
{'loss': 0.0002, 'grad_norm': 0.013541527092456818, 'learning_rate': 4.0625000000000005e-06, 'epoch': 3.25}
{'loss': 0.0003, 'grad_norm': 0.007716039661318064, 'learning_rate': 2.8125e-06, 'epoch': 3.5}
{'loss': 0.0003, 'grad_norm': 0.01869775727391243, 'learning_rate': 1.5625e-06, 'epoch': 3.75}
{'loss': 0.0002, 'grad_norm': 0.006826460361480713, 'learning_rate': 3.125e-07, 'epoch': 4.0}
{'train_runtime': 34.763, 'train_samples_per_second': 7.249, 'train_steps_per_second': 1.841, 'train_loss': 0.14271790210477775, 'epoch': 4.0}
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 64/64 [00:34<00:00, 1.84it/s]

Evaluating on test set...
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 8.89it/s]
Evaluation results: {'eval_loss': 0.14287129044532776, 'eval_accuracy': 0.9523809523809523, 'eval_f1': 0.9530864197530863, 'eval_runtime': 0.8195, 'eval_samples_per_second': 25.624, 'eval_steps_per_second': 7.321, 'epoch': 4.0}
Model saved to ./fine-tuned-classifier
Label mappings saved to ./fine-tuned-classifier/label_mapping.json

Testing specific examples:
Text: 'can I see you in the kitchen?'
Prediction: request for visual content creation (class 0)
Confidence: 0.9957

Text: 'see you tomorrow'
Prediction: conversational message (class 1)
Confidence: 0.9979

Text: 'make an image of you in Paris'
Prediction: request for visual content creation (class 0)
Confidence: 0.9999
