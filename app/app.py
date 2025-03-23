import gradio as gr
import yaml
from diffusers import DiffusionPipeline
import torch

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

# Load configuration and set up the pipeline
config = load_config()
model_name = config.get("model_name", "stable-diffusion-v1-5/stable-diffusion-v1-5")
device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

# Conditionally set torch_dtype
if device == "cuda":
    dtype = torch.float16
else:
    dtype = torch.float32

pipeline = DiffusionPipeline.from_pretrained(model_name, torch_dtype=dtype)
pipeline.to(device)

def generate_image(prompt):
    result = pipeline(prompt)
    return result.images[0]

iface = gr.Interface(
    fn=generate_image,
    inputs="text",
    outputs="image",
    title="Stable Diffusion Image Generator",
    description="Enter a prompt to generate an image using Stable Diffusion."
)

if __name__ == "__main__":
    iface.launch()
