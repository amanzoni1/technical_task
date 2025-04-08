import os
import glob
import json
import subprocess


def train_lora_model(dataset_dir, output_dir, keywords):
    """
    Trains a LoRA model

    Args:
        dataset_dir (str): Directory containing training images
        output_dir (str): Directory to save trained model
        keywords (list): List of keywords/captions for the images

    Returns:
        bool: True if training successful, False otherwise
    """
    try:
        # Find all image files
        image_files = []
        for ext in ["jpg", "jpeg", "png"]:
            image_files.extend(
                glob.glob(os.path.join(dataset_dir, f"**/*.{ext}"), recursive=True)
            )
            image_files.extend(
                glob.glob(
                    os.path.join(dataset_dir, f"**/*.{ext.upper()}"), recursive=True
                )
            )

        if not image_files:
            raise ValueError("No image files found in the dataset")

        # Create caption file for each image using the keywords
        caption = ", ".join(keywords)
        for img_path in image_files:
            caption_file = os.path.splitext(img_path)[0] + ".txt"
            with open(caption_file, "w") as f:
                f.write(caption)

        # Create metadata.jsonl file
        with open(os.path.join(dataset_dir, "metadata.jsonl"), "w") as f:
            for img_path in image_files:
                rel_path = os.path.relpath(img_path, dataset_dir)
                entry = {"file_name": rel_path, "text": caption}
                f.write(json.dumps(entry) + "\n")

        # Define the training settings
        base_model_id = "black-forest-labs/FLUX.1-dev"
        lora_model_id = "strangerzonehf/Flux-Super-Realism-LoRA"

        # Construct the absolute path for the training script
        script_path = os.path.join(
            "/home/ubuntu/technical_task/utils", "train_text_to_image_lora.py"
        )

        # Construct training command
        cmd = [
            "accelerate",
            "launch",
            "--num_processes",
            "1",
            "--num_machines",
            "1",
            "--dynamo_backend",
            "no",
            "--mixed_precision",
            "fp16",
            script_path,
            f"--pretrained_model_name_or_path={base_model_id}",
            f"--lora_model_path={lora_model_id}",  # Start from checkpoint
            f"--train_data_dir={dataset_dir}",
            f"--output_dir={output_dir}",
            f"--resolution=512",
            f"--train_batch_size=1",
            f"--gradient_accumulation_steps=1",
            f"--max_train_steps=5",
            f"--learning_rate=1e-4",
            f"--lr_scheduler=constant",
            f"--lora_rank=16",
            f"--lora_alpha=32",
            f"--seed=42",
            "--center_crop",
            "--validation_prompt=High quality image generation",
            "--num_validation_images=1",
        ]

        # Run the training process
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )

        # Log output
        log_file = os.path.join(output_dir, "training_log.txt")
        with open(log_file, "w") as f:
            for line in process.stdout:
                f.write(line)
                f.flush()

        # Check result
        if process.wait() != 0:
            with open(os.path.join(output_dir, "error.txt"), "w") as f:
                f.write("Training failed with non-zero exit code")
            return False

        return True

    except Exception as e:
        # Log error
        with open(os.path.join(output_dir, "error.txt"), "w") as f:
            f.write(str(e))
        return False
