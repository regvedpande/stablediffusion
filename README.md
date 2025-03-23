
# Stable Diffusion Image Generator

This project is a Stable Diffusion-based image generator using `diffusers`, `Gradio`, and `PyTorch`.

## Features
- Generates images from text prompts using Stable Diffusion.
- Uses a configurable model from `config.yaml`.
- Supports both CPU and GPU (CUDA).
- Interactive UI using Gradio.
- Command-line usage for batch generation.
- Saves generated images automatically.

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/stablediffusion.git
   cd stablediffusion
   ```

2. Create and activate a virtual environment:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

### Using Gradio UI
Run the Gradio app:
```sh
python app/app.py
```
Then open the provided URL in your browser to enter a prompt and generate images.

### Command-Line Usage
Generate an image from the command line:
```sh
python main.py --prompt "A futuristic cityscape at night"
```
The generated image will be saved in the `output/` directory.

## Configuration
Edit `config/config.yaml` to change the model name or device:
```yaml
model_name: "stabilityai/stable-diffusion-2"
device: "cuda"  # Change to "cpu" if you don't have a GPU
```

## Saving Images
Generated images are automatically saved in `output/`.  
You can modify `utils/image_utils.py` to change the saving format.

## Dependencies
- `diffusers`
- `torch`
- `gradio`
- `pyyaml`
- `Pillow`

Install all dependencies via:
```sh
pip install -r requirements.txt
```
