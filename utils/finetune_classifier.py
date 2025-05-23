import torch
import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, f1_score

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Base model
base_model = "facebook/bart-large-mnli"

# Define the label mappings explicitly
id2label = {0: "request for visual content creation", 1: "conversational message"}
label2id = {"request for visual content creation": 0, "conversational message": 1}

# Training data
training_data = [
    # Clear requests for visual content (label 0)
    {"text": "can you show me a picture of you on the beach?", "label": 0},
    {"text": "generate an image of you in a red dress", "label": 0},
    {"text": "I'd like to see you in a kitchen setting", "label": 0},
    {"text": "show me how you would look in a business suit", "label": 0},
    {"text": "create a photo of you sitting in a cafe", "label": 0},
    {"text": "can I see you in a winter outfit?", "label": 0},
    {"text": "make an image of you hiking in the mountains", "label": 0},
    {"text": "render yourself in a futuristic cityscape", "label": 0},
    {"text": "generate a picture of you with a sunset background", "label": 0},
    {"text": "can you create an image of yourself reading a book?", "label": 0},
    {"text": "I want to see you in a vintage 80s style", "label": 0},
    {"text": "show me what you'd look like as a superhero", "label": 0},
    {"text": "create an image of you at the beach", "label": 0},
    {"text": "how would you look with short hair? can you show me?", "label": 0},
    {"text": "draw yourself in an anime style", "label": 0},
    {"text": "can you make a picture where you're playing piano?", "label": 0},
    {"text": "generate an image of you in a garden", "label": 0},
    {"text": "show me a picture of you in formal attire", "label": 0},
    {"text": "create a portrait of you in renaissance style", "label": 0},
    {"text": "I'd like to see how you'd look in a summer dress", "label": 0},
    {"text": "can you generate an image of yourself with a pet?", "label": 0},
    {"text": "make a picture of you cooking in a kitchen", "label": 0},
    {"text": "show yourself in a professional office setting", "label": 0},
    {"text": "can you create a picture of you in Paris?", "label": 0},
    {"text": "generate an image where you're wearing sunglasses", "label": 0},
    # Clear conversational messages (label 1)
    {"text": "how was your day?", "label": 1},
    {"text": "what's your favorite movie?", "label": 1},
    {"text": "do you have any advice about my relationship?", "label": 1},
    {"text": "tell me a funny story", "label": 1},
    {"text": "how do you feel about climate change?", "label": 1},
    {"text": "what's the best book you've read recently?", "label": 1},
    {"text": "can you help me with my homework?", "label": 1},
    {"text": "I'm feeling sad today", "label": 1},
    {"text": "what do you think about the latest political news?", "label": 1},
    {"text": "tell me about your interests", "label": 1},
    {"text": "what's your opinion on artificial intelligence?", "label": 1},
    {"text": "how can I improve my cooking skills?", "label": 1},
    {"text": "I think the project deadline is too tight", "label": 1},
    {"text": "have you ever traveled to another country?", "label": 1},
    {"text": "what music do you listen to?", "label": 1},
    {"text": "can you recommend a good restaurant?", "label": 1},
    {"text": "how's the weather where you are?", "label": 1},
    {"text": "do you have siblings?", "label": 1},
    {"text": "what time should we meet tomorrow?", "label": 1},
    {"text": "I'm thinking about changing careers", "label": 1},
    {"text": "good morning, did you sleep well?", "label": 1},
    {"text": "can you give me advice on investment?", "label": 1},
    {"text": "I'm planning my vacation", "label": 1},
    {"text": "what's your favorite food?", "label": 1},
    {"text": "do you think I should get a dog?", "label": 1},
    # Ambiguous but more likely visual content (label 0)
    {"text": "can I see you in a red dress?", "label": 0},
    {"text": "show me what you look like", "label": 0},
    {"text": "I wonder how you'd appear in casual clothes", "label": 0},
    {"text": "let me see you in a different setting", "label": 0},
    {"text": "picture yourself at the beach", "label": 0},
    {"text": "how would you look with blonde hair?", "label": 0},
    {"text": "can you show yourself in a different outfit?", "label": 0},
    {"text": "what would you look like in winter clothes?", "label": 0},
    {"text": "I'm curious to see you in a historical setting", "label": 0},
    {"text": "show me a different version of you", "label": 0},
    {"text": "can you appear in a fantasy world?", "label": 0},
    {"text": "let me see what you'd look like as a doctor", "label": 0},
    {"text": "picture this: you in a space station", "label": 0},
    # Ambiguous but more likely conversational (label 1)
    {"text": "see you on Monday", "label": 1},
    {"text": "can you explain how this works?", "label": 1},
    {"text": "show me what you mean by that", "label": 1},
    {"text": "I'd like to see your perspective on this issue", "label": 1},
    {"text": "can I see what you think about this problem?", "label": 1},
    {"text": "picture this scenario: you're stranded on an island", "label": 1},
    {"text": "let me know your thoughts", "label": 1},
    {"text": "can you show me how to solve this equation?", "label": 1},
    {"text": "I'd like to see how you would approach this situation", "label": 1},
    {"text": "imagine we're planning a trip together", "label": 1},
    {"text": "help me see this from your point of view", "label": 1},
    {"text": "can you make sense of these instructions?", "label": 1},
    {"text": "what would it look like if we try something different?", "label": 1},
    {"text": "can you picture a world without poverty?", "label": 1},
    {"text": "I see you're good at math", "label": 1},
    {"text": "show me how this recipe works", "label": 1},
    {"text": "can you create a story about dragons?", "label": 1},
    {"text": "draw your own conclusions from this data", "label": 1},
    {"text": "I'm trying to visualize the solution", "label": 1},
    {"text": "can you see yourself doing this job?", "label": 1},
    {"text": "make yourself comfortable and let's chat", "label": 1},
]

# Convert to dataset and split
dataset = Dataset.from_list(training_data)
train_test_split = dataset.train_test_split(test_size=0.25, seed=42)

# Print dataset statistics
print(f"Training examples: {len(train_test_split['train'])}")
print(f"Testing examples: {len(train_test_split['test'])}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Load model with explicit label mappings
model = AutoModelForSequenceClassification.from_pretrained(
    base_model,
    num_labels=2,
    problem_type="single_label_classification",
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)


# Tokenize datasets
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_train = train_test_split["train"].map(tokenize_function, batched=True)
tokenized_test = train_test_split["test"].map(tokenize_function, batched=True)


# Define evaluation function
def compute_metrics(eval_pred):
    if isinstance(eval_pred[0], tuple):
        # If logits is a tuple, use the first element
        logits = eval_pred[0][0]  # Extract from nested tuple
    else:
        # Otherwise, use it directly
        logits = eval_pred[0]

    labels = eval_pred[1]

    # Get the predicted class from the logits
    predictions = logits.argmax(axis=1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted"),
    }


# Training arguments
output_dir = "./fine-tuned-classifier"
training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=4,
    weight_decay=0.01,
    logging_steps=4,
    save_total_limit=2,
)

# Create trainer with evaluation
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics,
)

# Train the model
print("Starting training...")
trainer.train()

# Evaluate the model
print("\nEvaluating on test set...")
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# Save the model
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model saved to {output_dir}")

# Save label mappings to a file
with open(f"{output_dir}/label_mapping.json", "w") as f:
    json.dump({"id2label": id2label, "label2id": label2id}, f)
print(f"Label mappings saved to {output_dir}/label_mapping.json")

# Test a few examples
examples = [
    "can I see you in the kitchen?",
    "see you tomorrow",
    "make an image of you in Paris",
]

print("\nTesting specific examples:")
model.eval()
for example in examples:
    inputs = tokenizer(example, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][predicted_class].item()

    print(f"Text: '{example}'")
    print(f"Prediction: {id2label[predicted_class]} (class {predicted_class})")
    print(f"Confidence: {confidence:.4f}\n")
