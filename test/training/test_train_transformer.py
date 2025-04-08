import requests
import json
import os
import time
import zipfile
from PIL import Image
import numpy as np

# API endpoint URL – adjust if necessary
BASE_URL = "http://localhost:8000"
TRAIN_URL = f"{BASE_URL}/train/"


def create_test_dataset():
    """Create a small test dataset with a few images."""
    print("Creating test dataset...")

    # Create a directory for test images
    os.makedirs("test_dataset", exist_ok=True)

    # Generate 3 sample images (colored squares)
    for i in range(3):
        # Create a random colored image of size 512x512
        img_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(f"test_dataset/image_{i}.jpg")

    # Create a zip file of the test dataset directory
    zip_path = "test_dataset.zip"
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for file in os.listdir("test_dataset"):
            zipf.write(os.path.join("test_dataset", file), file)

    print("Created test dataset with 3 images")
    return zip_path


def test_training_endpoint():
    print("\n===== TESTING TRAINING ENDPOINT =====\n")

    # 1. Create a simple test dataset
    dataset_path = create_test_dataset()

    # 2. Define keywords for training
    keywords = ["realistic", "detailed", "high quality"]

    # 3. Send request to training endpoint
    print(f"Sending training request to {TRAIN_URL}")
    print(f"Keywords: {keywords}")

    try:
        files = {"dataset": open(dataset_path, "rb")}
        data = {"keywords": json.dumps(keywords)}

        response = requests.post(TRAIN_URL, files=files, data=data)

        if response.status_code == 200:
            result = response.json()
            model_id = result.get("model_id")
            print("Training request successful!")
            print(f"Model ID: {model_id}")

            # 4. Check training status until complete or failed
            status_url = f"{TRAIN_URL}status/{model_id}"
            print("\nMonitoring training status...")
            poll_count = 0

            while True:
                status_response = requests.get(status_url)
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    current_status = status_data.get("status", "unknown")
                    print(f"Poll {poll_count + 1}: Status = {current_status}")

                    if current_status == "complete":
                        print("\nTraining completed successfully!")
                        break
                    elif current_status == "failed":
                        print("\nTraining failed. Check server logs.")
                        break
                else:
                    print(f"Error checking status: {status_response.status_code}")

                poll_count += 1
                time.sleep(10)  # Check every 10 seconds

                if poll_count >= 30:
                    print("\nReached maximum polls. Training may still be running.")
                    break
        else:
            print(f"Error: {response.status_code}")
            print(f"Response: {response.text}")

    except Exception as e:
        print(f"Exception when sending request: {str(e)}")
    finally:
        if "files" in locals() and "dataset" in files:
            files["dataset"].close()


if __name__ == "__main__":
    test_training_endpoint()
