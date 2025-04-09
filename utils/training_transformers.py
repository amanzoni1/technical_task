import os
import zipfile
import torch
from diffusers import (
    FluxTransformer2DModel,
    FlowMatchEulerDiscreteScheduler,
)
from transformers import CLIPTokenizer, CLIPTextModel, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from PIL import Image
import torch.nn.functional as F
from datasets import Dataset
import json
import glob


# ------------------------------------------------------------------
# Configuration class for training hyperparameters
# ------------------------------------------------------------------
class TrainingConfig:
    def __init__(
        self,
        base_model_id="black-forest-labs/FLUX.1-dev",
        prompt_keywords=None,
        resolution=512,
        batch_size=1,
        max_train_steps=5,
        learning_rate=1e-4,
        lora_rank=16,
    ):
        self.base_model_id = base_model_id
        self.prompt_keywords = prompt_keywords or []
        self.resolution = resolution
        self.batch_size = batch_size
        self.max_train_steps = max_train_steps
        self.learning_rate = learning_rate
        self.lora_rank = lora_rank


# ------------------------------------------------------------------
# Extract and prepare the dataset
# ------------------------------------------------------------------
def prepare_dataset(dataset_zip_path, extraction_dir, keywords, resolution=256):
    """Simplified dataset preparation function"""
    os.makedirs(extraction_dir, exist_ok=True)
    with zipfile.ZipFile(dataset_zip_path, "r") as zip_ref:
        zip_ref.extractall(extraction_dir)

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

    default_caption = ", ".join(keywords)

    images = []
    captions = []
    pixel_values = []
    for img_path in image_files:
        caption_file = os.path.splitext(img_path)[0] + ".txt"
        if os.path.exists(caption_file):
            with open(caption_file, "r") as f:
                caption = f.read().strip()
        else:
            caption = default_caption

        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((resolution, resolution))
            tensor = torch.tensor(list(img.getdata()), dtype=torch.float32).reshape(
                resolution, resolution, 3
            )
            tensor = (tensor / 127.5) - 1.0
            tensor = tensor.permute(2, 0, 1)

            images.append(img)
            captions.append(caption)
            pixel_values.append(tensor)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            continue

    return {"image": images, "text": captions, "pixel_values": pixel_values}


# ------------------------------------------------------------------
# Update progress
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
# Trainer for Flux models
# ------------------------------------------------------------------
class SimpleFluxTrainer(Trainer):
    def __init__(self, noise_scheduler, text_encoder, tokenizer, **kwargs):
        super().__init__(**kwargs)
        self.noise_scheduler = noise_scheduler
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer

    def compute_loss(self, model, inputs, return_outputs=False):
        # Get text encodings
        text_inputs = self.tokenizer(
            inputs["text"],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            encoder_hidden_states = self.text_encoder(text_inputs.input_ids)[0]

        # Get image tensors
        pixel_values = inputs["pixel_values"].to(self.model.device)

        # Sample noise and timesteps
        batch_size = pixel_values.shape[0]
        noise = torch.randn_like(pixel_values)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=self.model.device,
        )

        # Add noise based on scheduler
        sigma_t = self.noise_scheduler.config.base_shift + (
            timesteps.float() / self.noise_scheduler.config.num_train_timesteps
        ) * (
            self.noise_scheduler.config.max_shift
            - self.noise_scheduler.config.base_shift
        )
        sigma_t = sigma_t[:, None, None, None]
        noisy_images = pixel_values + noise * sigma_t

        # Predict noise
        noise_pred = model(noisy_images, timesteps, encoder_hidden_states).sample

        # Calculate loss
        loss = F.mse_loss(noise_pred, noise)

        return (loss, {"loss": loss}) if return_outputs else loss

    def on_log(self, args, state, control, logs=None, **kwargs):
        super().on_log(args, state, control, logs, **kwargs)
        # Update progress tracker
        update_progress(
            args.output_dir,
            state.global_step,
            args.max_steps,
            "training",
            {"loss": logs.get("loss", 0) if logs else 0},
        )


# ------------------------------------------------------------------
# Main training function
# ------------------------------------------------------------------
def train_lora_model(dataset_dir, output_dir, keywords):
    """
    Simplified training function using Trainer API.
    """
    try:
        # Basic setup
        os.makedirs(output_dir, exist_ok=True)
        update_progress(output_dir, 0, 1, "starting")

        # Config
        config = TrainingConfig(prompt_keywords=keywords)

        # Prepare dataset
        dataset_zip_path = os.path.join(dataset_dir, "dataset.zip")
        extraction_dir = os.path.dirname(dataset_zip_path)
        dataset_dict = prepare_dataset(
            dataset_zip_path,
            extraction_dir,
            config.prompt_keywords,
            resolution=config.resolution,
        )

        # Convert to dataset object
        dataset = Dataset.from_dict(dataset_dict)
        if "image" in dataset.column_names:
            dataset = dataset.remove_columns(["image"])

        # Load model components
        tokenizer = CLIPTokenizer.from_pretrained(
            config.base_model_id, subfolder="tokenizer"
        )
        text_encoder = CLIPTextModel.from_pretrained(
            config.base_model_id, subfolder="text_encoder"
        )
        flux_model = FluxTransformer2DModel.from_pretrained(
            config.base_model_id, subfolder="transformer"
        )
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            config.base_model_id, subfolder="scheduler"
        )

        # Apply LoRA
        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_rank,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            lora_dropout=0.1,
        )
        flux_model = get_peft_model(flux_model, lora_config)

        # Freeze text encoder
        text_encoder.requires_grad_(False)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            max_steps=config.max_train_steps,
            save_steps=config.max_train_steps,
            logging_steps=1,
            remove_unused_columns=False,
        )

        # Create trainer
        trainer = SimpleFluxTrainer(
            model=flux_model,
            args=training_args,
            train_dataset=dataset,
            noise_scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )

        # Train
        update_progress(output_dir, 0, config.max_train_steps, "training")
        trainer.train()

        # Save model
        update_progress(
            output_dir, config.max_train_steps, config.max_train_steps, "saving"
        )
        trainer.save_model(os.path.join(output_dir, "flux_model"))

        # Complete
        update_progress(
            output_dir, config.max_train_steps, config.max_train_steps, "complete"
        )
        return True

    except Exception as e:
        print(f"Training error: {str(e)}")
        with open(os.path.join(output_dir, "error.txt"), "w") as f:
            f.write(str(e))
        update_progress(output_dir, 0, 1, "failed", {"error": str(e)})
        return False
