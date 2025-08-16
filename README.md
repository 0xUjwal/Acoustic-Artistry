🎨 Acoustic Artistry - Voice to Image Generator
===============================================

An AI-powered application that converts your voice descriptions into stunning images using Stable Diffusion and speech recognition technology.

* * * * *

✨ Features
----------

-   🎤 **Voice Input**: Record your voice to describe images

-   ⌨️ **Text Input**: Alternative text input option

-   🎨 **AI Image Generation**: Powered by Stable Diffusion v1.5

-   🔧 **Customizable Settings**: Adjust image dimensions and quality

-   🌐 **Web Interface**: Beautiful, responsive Gradio interface

-   🚀 **Real-time Processing**: Fast speech recognition and image generation

* * * * *

🖼️ Sample Outputs
------------------

Try these example prompts:

-   "A serene lake surrounded by mountains at sunset"

-   "A magical forest with glowing mushrooms"

-   "A futuristic city with flying cars"

-   "A majestic lion in the African savanna"

* * * * *

🚀 Quick Start
--------------

### Option 1: Run on Hugging Face Spaces (Recommended)

Click here to run the app: [🤗 Hugging Face Space](https://huggingface.co/spaces/yourusername/acoustic-artistry)

### Option 2: Local Installation

1.  Clone the repository:

    `git clone https://github.com/yourusername/acoustic-artistry-voice-to-image.git
    cd acoustic-artistry-voice-to-image`

2.  Install dependencies:

    `pip install -r requirements.txt`

3.  Run the application:

    `python app.py`

4.  Open your browser and go to:\
    👉 <http://localhost:7860>

* * * * *

💻 System Requirements
----------------------

-   **RAM**: 8GB minimum (16GB recommended for CUDA)

-   **GPU**: NVIDIA GPU with 6GB+ VRAM (optional, but recommended)

-   **Storage**: ~5GB for model files

-   **Python**: 3.8 or higher

* * * * *

🛠️ Technologies Used
---------------------

-   **Gradio**: Web interface framework

-   **Stable Diffusion**: AI image generation

-   **SpeechRecognition**: Voice-to-text conversion

-   **PyTorch**: Deep learning framework

-   **Diffusers**: Hugging Face diffusion models

* * * * *

🎯 How It Works
---------------

1.  **Voice Input** → Record your voice description

2.  **Speech Recognition** → Convert audio to text using Google Speech API

3.  **Text Processing** → Enhance prompt for better image generation

4.  **AI Generation** → Generate image using Stable Diffusion

5.  **Display Results** → Show both recognized text and generated image

* * * * *

🌐 Deployment Options
---------------------

### 🔹 Google Colab (Free GPU)

Create a Colab notebook with the following setup:

`# Install dependencies
!pip install gradio diffusers torch transformers accelerate speechrecognition pillow

# Clone your repo
!git clone https://github.com/0xUjwal/Acoustic-Artistry.git
%cd Acoustic-Artistry

# Run the app
!python app.py`
