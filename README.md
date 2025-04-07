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
