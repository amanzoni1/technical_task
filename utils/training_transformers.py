import os
import glob
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from diffusers import (
    FluxTransformer2DModel,
    FlowMatchEulerDiscreteScheduler,
)
from transformers import CLIPTokenizer, CLIPTextModel, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model


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
# Simple PyTorch Dataset Class
# ------------------------------------------------------------------
class SimpleImageDataset(Dataset):
    def __init__(self, folder_path, resolution=256, default_caption="a photo"):
        """
        Args:
            folder_path (str): Path to the folder containing images (and optional .txt caption files).
            resolution (int): The resolution to which images will be resized (assumed square).
            default_caption (str): Caption to use if no caption file is found.
        """
        self.folder_path = folder_path
        self.resolution = resolution
        self.default_caption = default_caption
        # Gather image file paths for supported formats
        self.image_paths = []
        for ext in ["jpg", "jpeg", "png"]:
            self.image_paths.extend(
                glob.glob(os.path.join(folder_path, f"**/*.{ext}"), recursive=True)
            )
            self.image_paths.extend(
                glob.glob(
                    os.path.join(folder_path, f"**/*.{ext.upper()}"), recursive=True
                )
            )

        if not self.image_paths:
            raise ValueError("No images found in the given folder.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # Look for a matching caption file
        caption_file = os.path.splitext(img_path)[0] + ".txt"
        if os.path.exists(caption_file):
            with open(caption_file, "r") as f:
                caption = f.read().strip()
        else:
            caption = self.default_caption

        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((self.resolution, self.resolution))
            tensor = torch.tensor(list(img.getdata()), dtype=torch.float32).reshape(
                self.resolution, self.resolution, 3
            )
            tensor = (tensor / 127.5) - 1.0
            tensor = tensor.permute(2, 0, 1)  # Convert to (C, H, W)
        except Exception as e:
            raise RuntimeError(f"Error processing {img_path}: {e}")

        return {"pixel_values": tensor, "text": caption}


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
# Simple Data Collator
# ------------------------------------------------------------------
def simple_data_collator(features):
    # First, make a simple check to ensure features is a list
    if not isinstance(features, list) or len(features) == 0:
        print(f"Warning: features is not a list or is empty: {type(features)}")
        # Return an empty dict or handle appropriately
        return {}

    try:
        collated = {}
        for key in features[0].keys():
            if key == "pixel_values" and isinstance(features[0][key], torch.Tensor):
                collated[key] = torch.stack([feature[key] for feature in features])
            else:
                collated[key] = [feature[key] for feature in features]
        return collated
    except Exception as e:
        print(f"Error in collator: {str(e)}, features: {type(features)}")
        # Return a minimal valid batch
        return {"pixel_values": torch.zeros(1, 3, 256, 256), "text": ["a photo"]}


# ------------------------------------------------------------------
# Trainer for Flux models
# ------------------------------------------------------------------
class SimpleFluxTrainer(Trainer):
    def __init__(self, noise_scheduler, text_encoder, tokenizer, **kwargs):
        super().__init__(**kwargs)
        self.noise_scheduler = noise_scheduler
        self.text_encoder = text_encoder
        self.my_tokenizer = tokenizer

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        try:
            # Print input info for debugging
            print(f"Input type: {type(inputs)}")
            if isinstance(inputs, dict):
                print(f"Input keys: {list(inputs.keys())}")

            # If inputs is a list, collate it
            if isinstance(inputs, list):
                inputs = simple_data_collator(inputs)
                print("Collated inputs from list")

            # Check if text exists and is valid
            if "text" not in inputs:
                print("No text key in inputs")
                if isinstance(inputs.get("pixel_values", []), list):
                    n = len(inputs["pixel_values"])
                else:
                    n = inputs["pixel_values"].shape[0]
                inputs["text"] = ["a photo"] * n
            else:
                if not isinstance(inputs["text"], list):
                    inputs["text"] = [inputs["text"]]

            # Check tokenizer
            if self.my_tokenizer is None:
                raise ValueError("Tokenizer is None")

            print(f"Text input: {inputs['text']}")

            # Tokenize with error checking
            text_encoding = self.my_tokenizer(
                inputs["text"],
                padding="max_length",
                max_length=self.my_tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )

            if text_encoding is None:
                raise ValueError("text_encoding is None")

            # Check input_ids
            if "input_ids" not in text_encoding:
                raise ValueError("input_ids missing from text_encoding")

            # Use device derived from model parameters
            device = next(model.parameters()).device

            # Move to device with careful checking
            text_inputs = {}
            for k, v in text_encoding.items():
                if v is None:
                    raise ValueError(f"text_encoding[{k}] is None")
                text_inputs[k] = v.to(device)

            # Process text encoder outputs
            with torch.no_grad():
                text_encoder_output = self.text_encoder(text_inputs["input_ids"])
                if text_encoder_output is None or len(text_encoder_output) == 0:
                    raise ValueError("Text encoder returned None or empty output")
                encoder_hidden_states = text_encoder_output[0]
                if encoder_hidden_states is None:
                    raise ValueError("encoder_hidden_states is None")

                # Ensure encoder_hidden_states is the correct shape and type
                print(f"encoder_hidden_states shape: {encoder_hidden_states.shape}")
                print(f"encoder_hidden_states dtype: {encoder_hidden_states.dtype}")

                # Make sure it's on the same device as the model
                encoder_hidden_states = encoder_hidden_states.to(device)

                # If using fp16, ensure encoder_hidden_states is also fp16
                if next(model.parameters()).dtype == torch.float16:
                    encoder_hidden_states = encoder_hidden_states.to(torch.float16)

            # Ensure "pixel_values" exists and is a tensor
            if "pixel_values" not in inputs:
                raise ValueError("pixel_values missing from inputs")
            if isinstance(inputs["pixel_values"], list):
                try:
                    pixel_values = torch.stack([x.to(device) for x in inputs["pixel_values"]])
                except Exception as e:
                    raise ValueError(f"Error stacking pixel_values: {e}")
            elif isinstance(inputs["pixel_values"], torch.Tensor):
                pixel_values = inputs["pixel_values"].to(device)
            else:
                raise ValueError("pixel_values must be a list or a tensor")

            batch_size = pixel_values.shape[0]
            resolution = pixel_values.shape[2]

            # Create latents (FLUX expects 64 channels latent)
            latents = torch.randn((batch_size, 64, resolution // 8, resolution // 8), device=device)
            noise = torch.randn_like(latents)

            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (batch_size,),
                device=device,
                dtype=torch.long,
            )

            sigma_t = self.noise_scheduler.config.base_shift + (
                timesteps.float() / self.noise_scheduler.config.num_train_timesteps
            ) * (
                self.noise_scheduler.config.max_shift - self.noise_scheduler.config.base_shift
            )
            sigma_t = sigma_t[:, None, None, None]
            noisy_latents = latents + noise * sigma_t

            model_output = model(
                noisy_latents,
                timesteps,
                encoder_hidden_states,
            )
            if model_output is None:
                raise ValueError("Model returned None")
            if not hasattr(model_output, "sample") or model_output.sample is None:
                raise ValueError("Model output doesn't have valid 'sample' attribute")

            noise_pred = model_output.sample
            loss = F.mse_loss(noise_pred, noise)
            return (loss, {"loss": loss}) if return_outputs else loss

        except Exception as e:
            print(f"Error in compute_loss: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise e

    def on_log(self, args, state, control, logs=None, **kwargs):
        super().on_log(args, state, control, logs, **kwargs)
        update_progress(
            args.output_dir,
            state.global_step,
            args.max_steps,
            "training",
            {"loss": logs.get("loss", 0) if logs else 0},
        )


# ------------------------------------------------------------------
# Main training function using the simple dataset
# ------------------------------------------------------------------
def train_lora_model(dataset_dir, output_dir, keywords):
    """
    Training function using a standard PyTorch Dataset.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        update_progress(output_dir, 0, 1, "starting")

        config = TrainingConfig(prompt_keywords=keywords)

        # Use the simple dataset class; here dataset_dir is the folder containing your images and captions.
        default_caption = ", ".join(config.prompt_keywords)
        simple_dataset = SimpleImageDataset(
            folder_path=dataset_dir,
            resolution=config.resolution,
            default_caption=default_caption,
        )

        # You can use the Hugging Face Trainer with a PyTorch Dataset directly.
        train_dataset = simple_dataset

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

        # Determine device, then move models to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        text_encoder = text_encoder.to(device)
        flux_model = flux_model.to(device)

        # Apply LoRA modifications
        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_rank,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            lora_dropout=0.1,
        )
        flux_model = get_peft_model(flux_model, lora_config)

        # Freeze text encoder parameters
        text_encoder.requires_grad_(False)

        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            max_steps=config.max_train_steps,
            save_steps=config.max_train_steps,
            logging_steps=1,
            remove_unused_columns=False,
            report_to=[],
            fp16=True,
        )

        trainer = SimpleFluxTrainer(
            model=flux_model,
            args=training_args,
            train_dataset=train_dataset,
            noise_scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            data_collator=simple_data_collator,
        )

        torch.cuda.empty_cache()
        update_progress(output_dir, 0, config.max_train_steps, "training")
        trainer.train()

        update_progress(
            output_dir, config.max_train_steps, config.max_train_steps, "saving"
        )
        trainer.save_model(os.path.join(output_dir, "flux_model"))
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
