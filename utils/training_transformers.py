import os
import zipfile
import torch
from diffusers import DiffusionPipeline, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from peft import LoraConfig, get_peft_model
from PIL import Image
import torch.nn.functional as F
from datasets import Dataset
import json
from tqdm import tqdm
import glob


# ------------------------------------------------------------------
# Configuration class for training hyperparameters
# ------------------------------------------------------------------
class TrainingConfig:

    def __init__(
        self,
        base_model_id="black-forest-labs/FLUX.1-dev",
        lora_model_id="prithivMLmods/SD3.5-Large-Photorealistic-LoRA",
        prompt_keywords=None,
        caption_prefix="",
        resolution=512,
        batch_size=1,
        gradient_accumulation_steps=1,
        max_train_steps=5,
        learning_rate=1e-4,
        lr_scheduler="constant",
        seed=42,
        lora_rank=16,
        num_epochs=2,
        train_text_encoder=False,
    ):
        self.base_model_id = base_model_id
        self.lora_model_id = lora_model_id
        self.prompt_keywords = prompt_keywords or []
        self.caption_prefix = caption_prefix
        self.resolution = resolution
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_train_steps = max_train_steps
        self.learning_rate = learning_rate
        self.lr_scheduler = lr_scheduler
        self.seed = seed
        self.lora_rank = lora_rank
        self.num_epochs = num_epochs
        self.train_text_encoder = train_text_encoder


# ------------------------------------------------------------------
# Step 1: Extract and prepare the dataset
# ------------------------------------------------------------------
def prepare_dataset(
    dataset_zip_path, extraction_dir, keywords, resolution=512, caption_prefix=""
):
    """
    Extracts the dataset zip and prepares the dataset for training.

    Args:
        dataset_zip_path (str): Path to the zipped dataset.
        extraction_dir (str): Directory where the dataset will be extracted.
        keywords (list): Default keywords used for captions if no individual caption exists.
        resolution (int): Target resolution for images.
        caption_prefix (str): Optional prefix for captions.

    Returns:
        dict: A dictionary with keys "image", "text", and "pixel_values" representing the dataset.
    """
    # Extract the zip file
    os.makedirs(extraction_dir, exist_ok=True)
    with zipfile.ZipFile(dataset_zip_path, "r") as zip_ref:
        zip_ref.extractall(extraction_dir)

    # Find image files
    image_files = []
    for ext in ["jpg", "jpeg", "png"]:
        image_files.extend(
            glob.glob(os.path.join(extraction_dir, f"**/*.{ext}"), recursive=True)
        )
        image_files.extend(
            glob.glob(
                os.path.join(extraction_dir, f"**/*.{ext.upper()}"), recursive=True
            )
        )

    if not image_files:
        raise ValueError("No image files found in the dataset")

    # Create a default caption using keywords
    default_caption = ", ".join(keywords)
    if caption_prefix:
        default_caption = f"{caption_prefix}, {default_caption}"

    images = []
    captions = []
    pixel_values = []

    for img_path in image_files:
        caption_file = os.path.splitext(img_path)[0] + ".txt"
        if os.path.exists(caption_file):
            with open(caption_file, "r") as f:
                caption = f.read().strip()
            if caption_prefix:
                caption = f"{caption_prefix}, {caption}"
        else:
            caption = default_caption

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            continue

        # Resize the image to the target resolution
        img = img.resize((resolution, resolution))

        # Convert image to a tensor and normalize to [-1, 1]
        tensor = torch.tensor(list(img.getdata()), dtype=torch.float32).reshape(
            resolution, resolution, 3
        )
        tensor = (
            tensor / 127.5
        ) - 1.0  # Normalize assuming original pixel values are [0, 255]
        # Permute dimensions to (C, H, W)
        tensor = tensor.permute(2, 0, 1)

        images.append(img)
        captions.append(caption)
        pixel_values.append(tensor)

    # Save metadata for debugging
    meta_path = os.path.join(extraction_dir, "metadata.jsonl")
    with open(meta_path, "w") as f:
        for img_path in image_files:
            rel_path = os.path.relpath(img_path, extraction_dir)
            entry = {"file_name": rel_path, "text": default_caption}
            f.write(json.dumps(entry) + "\n")

    return {"image": images, "text": captions, "pixel_values": pixel_values}


# ------------------------------------------------------------------
# Utility: Update progress (writes a JSON file)
# ------------------------------------------------------------------
def update_progress(output_dir, step, total_steps, status, metrics=None):
    data = {
        "step": step,
        "total_steps": total_steps,
        "status": status,
        "metrics": metrics or {},
    }
    with open(os.path.join(output_dir, "progress.json"), "w") as f:
        json.dump(data, f)


# ------------------------------------------------------------------
# Utility: Save a checkpoint for the UNet (LoRA weights)
# ------------------------------------------------------------------
def save_checkpoint(unet, output_dir, step, final=False):
    save_dir = output_dir if final else os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(save_dir, exist_ok=True)
    unet.save_pretrained(os.path.join(save_dir, "unet"))


# ------------------------------------------------------------------
# Main training function
# ------------------------------------------------------------------
def train_lora(dataset_zip_path, output_dir, config):
    try:
        # Prepare dataset: extract zip and build data dict.
        extraction_dir = os.path.dirname(dataset_zip_path)
        dataset_dict = prepare_dataset(
            dataset_zip_path,
            extraction_dir,
            config.prompt_keywords,
            resolution=config.resolution,
            caption_prefix=config.caption_prefix,
        )
        dataset = Dataset.from_dict(dataset_dict)

        # Load base model components
        tokenizer = CLIPTokenizer.from_pretrained(
            config.base_model_id, subfolder="tokenizer"
        )
        text_encoder = CLIPTextModel.from_pretrained(
            config.base_model_id, subfolder="text_encoder"
        )
        unet = UNet2DConditionModel.from_pretrained(
            config.base_model_id, subfolder="unet"
        )
        noise_scheduler = DDPMScheduler.from_pretrained(
            config.base_model_id, subfolder="scheduler"
        )

        # Optionally, load from an existing LoRA adapter.
        # If you plan to train a new adapter from scratch, leave lora_model_id as None.
        if config.lora_model_id:
            pipeline = DiffusionPipeline.from_pretrained(
                config.base_model_id, torch_dtype=torch.float16
            )
            pipeline.load_lora_weights(config.lora_model_id)
            unet = pipeline.unet
            if config.train_text_encoder:
                text_encoder = pipeline.text_encoder

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Configure and apply LoRA to UNet
        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_rank,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            lora_dropout=0.1,
        )
        unet = get_peft_model(unet, lora_config)
        text_encoder.requires_grad_(False)

        # Setup training dataloader
        train_dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=config.batch_size, shuffle=True
        )
        unet = unet.to(device)
        text_encoder = text_encoder.to(device)
        optimizer = torch.optim.AdamW(unet.parameters(), lr=config.learning_rate)

        update_progress(output_dir, 0, config.max_train_steps, "training")
        unet.train()
        global_step = 0

        for epoch in range(config.num_epochs):
            for batch in train_dataloader:
                if global_step >= config.max_train_steps:
                    break

                # Use precomputed tensor values from the dataset
                pixel_values = torch.stack(batch["pixel_values"]).to(device)

                text_inputs = tokenizer(
                    batch["text"],
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                ).to(device)
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(text_inputs.input_ids)[0]

                batch_size = pixel_values.size(0)
                latents = torch.randn(
                    (batch_size, 4, config.resolution // 8, config.resolution // 8),
                    device=device,
                    dtype=torch.float16,
                )
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (batch_size,),
                    device=device,
                )
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                noise_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states
                ).sample
                loss = F.mse_loss(noise_pred, noise)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                update_progress(
                    output_dir,
                    global_step,
                    config.max_train_steps,
                    "training",
                    {"loss": loss.item()},
                )

                if global_step % 100 == 0:
                    save_checkpoint(unet, output_dir, global_step)

        update_progress(
            output_dir, config.max_train_steps, config.max_train_steps, "saving"
        )
        save_checkpoint(unet, output_dir, global_step, final=True)

        with open(os.path.join(output_dir, "README.md"), "w") as f:
            f.write(f"# Fine-tuned LoRA Adapter\n\n")
            f.write(f"Base model: {config.base_model_id}\n")
            if config.lora_model_id:
                f.write(f"Starting checkpoint: {config.lora_model_id}\n")
            f.write(f"Keywords: {', '.join(config.prompt_keywords)}\n")
            f.write(f"Training steps: {global_step}\n")

        update_progress(
            output_dir, config.max_train_steps, config.max_train_steps, "complete"
        )
        return True

    except Exception as e:
        print(f"Training error: {str(e)}")
        with open(os.path.join(output_dir, "error.txt"), "w") as f:
            f.write(str(e))
        update_progress(
            output_dir, 0, config.max_train_steps, "failed", {"error": str(e)}
        )
        return False


# ------------------------------------------------------------------
# Wrapper function to be called by the API route
# ------------------------------------------------------------------
def train_lora_model(dataset_dir, output_dir, keywords):
    """
    Wrapper function that creates a default training configuration and starts training.
    Assumes the uploaded zip file is stored as `dataset.zip` in dataset_dir.
    """
    config = TrainingConfig(
        base_model_id="black-forest-labs/FLUX.1-dev",  # Change if needed for your Flux model
        lora_model_id=None,  # Set to None to train a new adapter from scratch, or provide a valid adapter ID.
        prompt_keywords=keywords,
        caption_prefix="",
        resolution=512,
        batch_size=1,
        gradient_accumulation_steps=1,
        max_train_steps=5,
        learning_rate=1e-4,
        lr_scheduler="constant",
        seed=42,
        lora_rank=16,
        num_epochs=2,
        train_text_encoder=False,
    )
    dataset_zip_path = os.path.join(dataset_dir, "dataset.zip")
    return train_lora(dataset_zip_path, output_dir, config)
