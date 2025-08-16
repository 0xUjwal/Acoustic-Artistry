import gradio as gr
import speech_recognition as sr
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import os
import warnings
import logging

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
logging.getLogger("diffusers").setLevel(logging.WARNING)

class VoiceToImageGenerator:
    def __init__(self):
        self.pipe = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_loaded = False
        print(f"Using device: {self.device}")
        
    def load_model(self):
        """Load the Stable Diffusion model"""
        if self.model_loaded:
            return
            
        try:
            print("Loading Stable Diffusion model... This may take a few minutes.")
            model_id = "runwayml/stable-diffusion-v1-5"  # More stable version
            
            # Load with appropriate settings
            if self.device == "cuda":
                self.pipe = StableDiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    safety_checker=None,
                    requires_safety_checker=False
                )
                self.pipe.enable_attention_slicing()  # Memory optimization
            else:
                self.pipe = StableDiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False
                )
            
            self.pipe = self.pipe.to(self.device)
            self.model_loaded = True
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e
    
    def recognize_speech(self, audio_file_path):
        """Convert audio to text using speech recognition"""
        if not audio_file_path:
            return "No audio file provided"
        
        recognizer = sr.Recognizer()
        
        try:
            with sr.AudioFile(audio_file_path) as source:
                # Adjust for ambient noise
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = recognizer.record(source)
            
            # Try multiple recognition services
            try:
                text = recognizer.recognize_google(audio)
                print(f"Recognized text: {text}")
                return text
            except sr.UnknownValueError:
                try:
                    # Fallback to Sphinx (offline)
                    text = recognizer.recognize_sphinx(audio)
                    print(f"Recognized text (offline): {text}")
                    return text
                except:
                    return "Could not understand audio. Please try speaking clearly."
            except sr.RequestError:
                return "Speech recognition service unavailable. Please try again."
                
        except Exception as e:
            return f"Error processing audio: {str(e)}"
    
    def generate_image(self, text, width=512, height=512, num_inference_steps=20):
        """Generate image from text using Stable Diffusion"""
        if not self.model_loaded:
            self.load_model()
        
        if not text or text.strip() == "":
            return None, "Please provide text to generate an image"
        
        try:
            # Enhance prompt for better results
            enhanced_prompt = f"{text}, high quality, detailed, artistic"
            
            print(f"Generating image for: {enhanced_prompt}")
            
            # Generate image
            if self.device == "cuda":
                with torch.cuda.amp.autocast():
                    result = self.pipe(
                        enhanced_prompt,
                        width=width,
                        height=height,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=7.5
                    )
            else:
                result = self.pipe(
                    enhanced_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=7.5
                )
            
            image = result.images[0]
            return image, f"Image generated successfully for: '{text}'"
            
        except Exception as e:
            error_msg = f"Error generating image: {str(e)}"
            print(error_msg)
            return None, error_msg

# Initialize the generator
generator = VoiceToImageGenerator()

def process_voice_to_image(audio_file, manual_text="", image_width=512, image_height=512, steps=20):
    """Main function to process voice input and generate image"""
    
    # Use manual text if provided, otherwise process audio
    if manual_text.strip():
        recognized_text = manual_text.strip()
        status = "Using manually entered text"
    elif audio_file:
        recognized_text = generator.recognize_speech(audio_file)
        status = "Text recognized from audio"
    else:
        return "No input provided", None, "Please provide either audio or text input"
    
    if not recognized_text or "Error" in recognized_text or "Could not" in recognized_text:
        return recognized_text, None, "Speech recognition failed"
    
    # Generate image
    image, generation_status = generator.generate_image(
        recognized_text, 
        width=image_width, 
        height=image_height, 
        num_inference_steps=steps
    )
    
    final_status = f"{status}\n{generation_status}"
    
    return recognized_text, image, final_status

# Create the Gradio interface
def create_interface():
    with gr.Blocks(
        theme=gr.themes.Soft(),
        title="🎨 Acoustic Artistry - Voice to Image Generator"
    ) as interface:
        
        gr.HTML("""
        <div style="text-align: center; padding: 20px;">
            <h1>🎨 Acoustic Artistry</h1>
            <h2>Voice to Image Generator</h2>
            <p style="font-size: 18px; color: #666;">
                Speak your imagination into reality! Record your voice or type text to generate stunning AI images.
            </p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("<h3>🎤 Input</h3>")
                
                audio_input = gr.Audio(
                    label="Record your voice",
                    type="filepath",
                    elem_id="audio_input"
                )
                
                gr.HTML("<p style='text-align: center; margin: 10px;'><strong>OR</strong></p>")
                
                text_input = gr.Textbox(
                    label="Type your description manually",
                    placeholder="e.g., 'a beautiful sunset over mountains'",
                    lines=2
                )
                
                gr.HTML("<h4>🎛️ Advanced Settings</h4>")
                
                with gr.Row():
                    width_slider = gr.Slider(
                        minimum=256, maximum=768, value=512, step=64,
                        label="Image Width"
                    )
                    height_slider = gr.Slider(
                        minimum=256, maximum=768, value=512, step=64,
                        label="Image Height"
                    )
                
                steps_slider = gr.Slider(
                    minimum=10, maximum=50, value=20, step=5,
                    label="Generation Steps (higher = better quality, slower)"
                )
                
                generate_btn = gr.Button(
                    "🚀 Generate Image", 
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=2):
                gr.HTML("<h3>📋 Results</h3>")
                
                recognized_text_output = gr.Textbox(
                    label="Recognized/Input Text",
                    interactive=False
                )
                
                generated_image_output = gr.Image(
                    label="Generated Image",
                    height=400
                )
                
                status_output = gr.Textbox(
                    label="Status",
                    interactive=False
                )
        
        # Example prompts
        gr.HTML("""
        <div style="margin-top: 30px; padding: 20px; border-top: 1px solid #ddd;">
            <h3>💡 Example Prompts to Try:</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; margin-top: 10px;">
                <div style="padding: 10px; background: #f0f0f0; border-radius: 8px;">
                    <strong>Nature:</strong> "A serene lake surrounded by mountains at sunset"
                </div>
                <div style="padding: 10px; background: #f0f0f0; border-radius: 8px;">
                    <strong>Fantasy:</strong> "A magical forest with glowing mushrooms"
                </div>
                <div style="padding: 10px; background: #f0f0f0; border-radius: 8px;">
                    <strong>Architecture:</strong> "A futuristic city with flying cars"
                </div>
                <div style="padding: 10px; background: #f0f0f0; border-radius: 8px;">
                    <strong>Animals:</strong> "A majestic lion in the African savanna"
                </div>
            </div>
        </div>
        """)
        
        # Event handlers
        generate_btn.click(
            fn=process_voice_to_image,
            inputs=[audio_input, text_input, width_slider, height_slider, steps_slider],
            outputs=[recognized_text_output, generated_image_output, status_output]
        )
        
        return interface

# Launch the app
if __name__ == "__main__":
    print("Starting Acoustic Artistry - Voice to Image Generator...")
    interface = create_interface()
    
    # Launch with public sharing
    interface.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        debug=False,
        show_error=True
    )
