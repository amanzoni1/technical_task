import requests
import json
import os
import time

# API endpoint
base_url = "http://localhost:8000"
classify_url = f"{base_url}/classify/"

# Create directory for outputs
os.makedirs("test_results/classification", exist_ok=True)

# Test prompts for classification
classification_test_prompts = {
    "Image Requests": [
        "can you show me a picture of you on the beach?",
        "generate an image of you in a red dress",
        "I'd like to see you in a kitchen setting",
        "can I see you in a winter outfit?",
        "picture yourself at the beach",
    ],
    "Conversation": [
        "how was your day?",
        "what's your favorite movie?",
        "I think the project deadline is too tight",
        "tell me a funny story",
        "see you on Monday",
    ],
    "Ambiguous": [
        "can I see you in the kitchen?",
        "show me what you think",
        "picture this scenario",
        "what would it look like if we try something different?",
        "can you make something nice",
    ],
}


def test_classification():
    print("\n===== TESTING CLASSIFICATION ENDPOINT =====\n")
    all_results = []

    for category, prompts in classification_test_prompts.items():
        print(f"\nTesting {category} prompts:")

        for prompt in prompts:
            payload = {"prompt": prompt}

            try:
                start_time = time.time()
                response = requests.post(classify_url, json=payload)
                response_time = time.time() - start_time

                if response.status_code == 200:
                    result = response.json()
                    print(f"  • '{prompt}'")
                    print(
                        f"    → Classified as: {result['label']} (score: {result['score']:.4f})"
                    )
                    print(f"    → Message: {result['message']}")
                    print(f"    → Response time: {response_time:.2f}s")

                    # Store result
                    all_results.append(
                        {
                            "category": category,
                            "prompt": prompt,
                            "classification": result["label"],
                            "score": result["score"],
                            "message": result["message"],
                            "response_time": response_time,
                        }
                    )
                else:
                    print(f"  • Error for '{prompt}': {response.status_code}")
                    print(f"    → {response.text}")

            except Exception as e:
                print(f"  • Exception for '{prompt}': {e}")

    # Save all results to a file
    with open("test_results/classification/results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(
        f"\nClassification test results saved to test_results/classification/results.json"
    )

    # Print summary
    print_summary(all_results)


def print_summary(classification_results):
    print("\n===== CLASSIFICATION TEST SUMMARY =====\n")

    if classification_results:
        category_stats = {}
        total_time = 0

        for result in classification_results:
            category = result["category"]
            if category not in category_stats:
                category_stats[category] = {"count": 0, "correct": 0}

            category_stats[category]["count"] += 1

            # For image requests and conversation, we can check if classification matches category
            if (
                category == "Image Requests"
                and result["classification"] == "request for visual content creation"
            ) or (
                category == "Conversation"
                and result["classification"] == "conversational message"
            ):
                category_stats[category]["correct"] += 1

            total_time += result["response_time"]

        print("Classification Results:")
        for category, stats in category_stats.items():
            if category != "Ambiguous":
                accuracy = (
                    (stats["correct"] / stats["count"]) * 100
                    if stats["count"] > 0
                    else 0
                )
                print(
                    f"  • {category}: {stats['correct']}/{stats['count']} correct ({accuracy:.1f}%)"
                )
            else:
                print(
                    f"  • {category}: {stats['count']} tested (correctness not measured)"
                )

        avg_time = (
            total_time / len(classification_results) if classification_results else 0
        )
        print(f"  • Average response time: {avg_time:.2f}s")


if __name__ == "__main__":
    test_classification()
