import argparse
import yaml
from diffusers import DiffusionPipeline
import torch
from utils.image_utils import save_image

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def generate_image(prompt, model_name, device):
    # Conditionally set torch_dtype
    if device == "cuda":
        dtype = torch.float16
    else:
        # CPU does not support float16 in this pipeline, so use float32
        dtype = torch.float32

    # Load the pipeline with the specified model
    pipeline = DiffusionPipeline.from_pretrained(model_name, torch_dtype=dtype)
    pipeline.to(device)

    result = pipeline(prompt)
    return result.images[0]

def main():
    parser = argparse.ArgumentParser(description="Generate an image using Stable Diffusion")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation")
    args = parser.parse_args()
    
    config = load_config()
    model_name = config.get("model_name", "stable-diffusion-v1-5/stable-diffusion-v1-5")
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    
    image = generate_image(args.prompt, model_name, device)
    # Save the generated image to the output folder
    save_image(image, "output/generated_image.png")
    print("Image generated and saved to output/generated_image.png")

if __name__ == "__main__":
    main()
