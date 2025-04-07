project/
├── main.py # Application entry point
├── routes/
│ ├── **init**.py # Makes the endpoints folder a package
│ ├── generation.py # Image generation endpoint
│ ├── training.py # Fine-tuning endpoint
│ └── classification.py # Prompt classification endpoint
├── utils/
├── test/
├── requirements.txt
└── README.md # Project documentation

ssh -i /Users/andreamanzoni/Desktop/code/lambda-ssh.pem ubuntu@209.20.158.250

git clone <your-repo-url>
cd <your-repo-directory>
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

huggingface-cli login

uvicorn main:app --host 0.0.0.0 --port 8000

Starting training...
{'loss': 0.6394, 'grad_norm': 11.438572883605957, 'learning_rate': 1.775e-05, 'epoch': 0.62}
{'loss': 0.3823, 'grad_norm': 0.9362940192222595, 'learning_rate': 1.525e-05, 'epoch': 1.25}
{'loss': 0.0093, 'grad_norm': 0.21372751891613007, 'learning_rate': 1.275e-05, 'epoch': 1.88}
{'loss': 0.0026, 'grad_norm': 0.008792843669652939, 'learning_rate': 1.025e-05, 'epoch': 2.5}
{'loss': 0.0726, 'grad_norm': 0.012034337036311626, 'learning_rate': 7.75e-06, 'epoch': 3.12}
{'loss': 0.0002, 'grad_norm': 0.010561075061559677, 'learning_rate': 5.2500000000000006e-06, 'epoch': 3.75}
{'loss': 0.0002, 'grad_norm': 0.013754589483141899, 'learning_rate': 2.7500000000000004e-06, 'epoch': 4.38}
{'loss': 0.0002, 'grad_norm': 0.03263351321220398, 'learning_rate': 2.5000000000000004e-07, 'epoch': 5.0}
{'train_runtime': 37.1329, 'train_samples_per_second': 8.483, 'train_steps_per_second': 2.154, 'train_loss': 0.13835361970413942, 'epoch': 5.0}
100%|███████████████████████████████████████████████████████████████████████████████████████████████| 80/80 [00:37<00:00, 2.15it/s]
Model saved to ./fine-tuned-classifier

Evaluating model on test set...
Test Accuracy: 0.9524

Testing specific examples:
Text: 'can I see you in the kitchen?'
Prediction: visual content
Confidence: 0.9984

Text: 'see you tomorrow'
Prediction: conversation
Confidence: 0.9964

Text: 'make an image of you in Paris'
Prediction: visual content
Confidence: 0.9999

python test/test_generation.py

===== TESTING IMAGE GENERATION ENDPOINT =====

Generating image 1/3:
• Prompt: 'portrait of you with long dark hair against a sunset'
• Exception: HTTPConnectionPool(host='localhost', port=8000): Max retries exceeded with url: /generate/ (Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 0x7eb7b6a91540>: Failed to establish a new connection: [Errno 111] Connection refused'))

Generating image 2/3:
• Prompt: 'you in a professional business suit in an office'
• Exception: HTTPConnectionPool(host='localhost', port=8000): Max retries exceeded with url: /generate/ (Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 0x7eb7b6a91f00>: Failed to establish a new connection: [Errno 111] Connection refused'))

Generating image 3/3:
• Prompt: 'casual photo of you enjoying a day at the beach'
• Exception: HTTPConnectionPool(host='localhost', port=8000): Max retries exceeded with url: /generate/ (Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 0x7eb7b6a92800>: Failed to establish a new connection: [Errno 111] Connection refused'))

Image generation results saved to test_results/images/results.json

===== IMAGE GENERATION TEST SUMMARY =====

No images were successfully generated.
