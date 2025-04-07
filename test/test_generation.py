import requests
import json
import base64
from PIL import Image
import io
import os
import time

# API endpoint
base_url = "http://localhost:8000"
generate_url = f"{base_url}/generate/"

# Create directory for outputs
os.makedirs("test_results/images", exist_ok=True)

# Test prompts for image generation
generation_test_prompts = [
    "portrait of a beautiful woman with long dark hair against a sunset",
    "photo of a beautiful woman in a professional business suit in an office",
    "casual photo of a beautiful woman enjoying a day at the beach",
    "close-up portrait of a beautiful woman with a soft smile and elegant makeup",
    "artistic photo of a beautiful woman in a vintage dress with a scenic background",
]

def test_generation():
    print("\n===== TESTING IMAGE GENERATION ENDPOINT =====\n")
    generation_results = []

    for i, prompt in enumerate(generation_test_prompts):
        print(f"\nGenerating image {i+1}/{len(generation_test_prompts)}:")
        print(f"  • Prompt: '{prompt}'")

        payload = {"prompt": prompt}

        try:
            start_time = time.time()
            response = requests.post(generate_url, json=payload)
            response_time = time.time() - start_time

            if response.status_code == 200:
                result = response.json()
                image_b64 = result.get("image")
                full_prompt = result.get("prompt_used", prompt)

                if image_b64:
                    # Decode and save the image
                    image_data = base64.b64decode(image_b64)
                    image = Image.open(io.BytesIO(image_data))

                    # Create a filename
                    filename = f"test_results/images/image_{i+1}_{prompt.replace(' ', '_')[:30]}.png"
                    image.save(filename)

                    print(f"  • Image saved to {filename}")
                    print(f"  • Full prompt used: {full_prompt}")
                    print(f"  • Response time: {response_time:.2f}s")

                    # Store result
                    generation_results.append(
                        {
                            "prompt": prompt,
                            "full_prompt": full_prompt,
                            "image_path": filename,
                            "response_time": response_time,
                        }
                    )
                else:
                    print(f"  • Error: No image data in response")
            else:
                print(f"  • Error: {response.status_code}")
                print(f"  • Response: {response.text}")

        except Exception as e:
            print(f"  • Exception: {e}")

    # Save generation results
    with open("test_results/images/results.json", "w") as f:
        json.dump(generation_results, f, indent=2)

    print(f"\nImage generation results saved to test_results/images/results.json")

    # Print summary
    print_summary(generation_results)


def print_summary(generation_results):
    print("\n===== IMAGE GENERATION TEST SUMMARY =====\n")

    if generation_results:
        total_gen_time = sum(r["response_time"] for r in generation_results)
        avg_gen_time = (
            total_gen_time / len(generation_results) if generation_results else 0
        )

        print("Image Generation Results:")
        print(
            f"  • Images generated: {len(generation_results)}/{len(generation_test_prompts)}"
        )
        print(f"  • Average generation time: {avg_gen_time:.2f}s")

        for i, result in enumerate(generation_results):
            print(
                f"  • Image {i+1}: {result['image_path']} ({result['response_time']:.2f}s)"
            )
    else:
        print("No images were successfully generated.")


if __name__ == "__main__":
    test_generation()
