"""
YouTube Automation Pipeline Web Interface
A Gradio-based web interface for running the YouTube automation pipeline in Google Colab.
"""
import os
import sys
import json
import logging
import argparse
import gradio as gr
from pathlib import Path
from youtube_automation_pipeline import YouTubeAutomationPipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("webapp.log"), logging.StreamHandler()]
)
logger = logging.getLogger("YouTube-Pipeline-WebApp")

class YouTubeAutomationWebApp:
    """Web interface for the YouTube automation pipeline."""
    
    def __init__(self, config_path="config.yaml"):
        """Initialize the web app with configuration."""
        self.config_path = config_path
        self.pipeline = None
        self.initialize_pipeline()
        logger.info("YouTube Automation WebApp initialized")
    
    def initialize_pipeline(self):
        """Initialize the pipeline instance."""
        try:
            self.pipeline = YouTubeAutomationPipeline(config_path=self.config_path)
            logger.info("Pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing pipeline: {e}")
            self.pipeline = None
    
    def create_interface(self):
        """Create and configure the Gradio interface."""
        # Define the interface
        with gr.Blocks(title="AI YouTube Automation Pipeline", theme="default") as interface:
            gr.Markdown("# 🔥 Advanced AI-Powered YouTube Automation Pipeline 🎬")
            gr.Markdown("""
            This pipeline combines multiple AI models to generate creative videos for YouTube:
            - 🧠 Multi-Model Prompt Generation with Gemini 2.0 & LLama 3
            - 🖼️ High-Quality Image Generation with Gemini & SDXL
            - 🎭 Hyper-Realistic Animation with AnimateDiff
            - 🔊 AI Voiceover with XTTS and Bark
            - 🎥 Advanced Video Compilation
            - 📤 Google Drive Integration
            """)
            
            with gr.Tab("Generate Content"):
                with gr.Row():
                    with gr.Column():
                        theme_input = gr.Textbox(
                            label="Theme", 
                            placeholder="Enter a theme for your video (e.g., space exploration, underwater world)",
                            info="Provide a general theme to guide the AI's creativity"
                        )
                        
                        style_input = gr.Textbox(
                            label="Visual Style", 
                            placeholder="Enter a visual style (e.g., cyberpunk, watercolor, cinematic)",
                            info="Define the aesthetic style for the visuals"
                        )
                        
                        duration_input = gr.Slider(
                            label="Target Duration (seconds)",
                            minimum=15,
                            maximum=180,
                            value=60,
                            step=15,
                            info="Longer videos will take more time to generate"
                        )
                        
                        output_dir_input = gr.Textbox(
                            label="Custom Output Directory (optional)",
                            placeholder="Leave empty for default output location",
                            info="Custom location to save all generated files"
                        )
                    
                    with gr.Column():
                        # Advanced options
                        with gr.Accordion("Advanced Options", open=False):
                            num_prompts_input = gr.Slider(
                                label="Number of Prompts",
                                minimum=1,
                                maximum=5,
                                value=3,
                                step=1,
                                info="Number of creative prompts to generate"
                            )
                            
                            preferred_model_input = gr.Dropdown(
                                label="Preferred Image Model",
                                choices=["gemini", "sdxl", "alternate"],
                                value="alternate",
                                info="Which AI model to prioritize for image generation"
                            )
                            
                            tts_engine_input = gr.Dropdown(
                                label="Voice Generation Engine",
                                choices=["xtts", "bark"],
                                value="xtts",
                                info="Engine used for generating voiceovers"
                            )
                
                generate_btn = gr.Button("Generate YouTube Video", variant="primary")
                
                with gr.Row():
                    status_output = gr.Textbox(label="Status", interactive=False)
                
                with gr.Row():
                    output_gallery = gr.Gallery(label="Generated Images", columns=3, height=400)
                    video_output = gr.Video(label="Generated Video")
                
                with gr.Row():
                    json_output = gr.JSON(label="Pipeline Results")
            
            with gr.Tab("Configuration"):
                config_text = gr.TextArea(
                    label="YAML Configuration",
                    value=self._load_config_as_text(),
                    info="Edit and save to modify pipeline configuration",
                    height=600
                )
                
                config_save_btn = gr.Button("Save Configuration")
                config_status = gr.Textbox(label="Config Status", interactive=False)
                
                config_save_btn.click(
                    fn=self._save_config_text,
                    inputs=[config_text],
                    outputs=[config_status]
                )
            
            with gr.Tab("Help & Info"):
                gr.Markdown("""
                ## How to Use This Tool
                
                ### Basic Usage
                1. Go to the "Generate Content" tab
                2. Enter a theme (e.g., "space exploration", "underwater world")
                3. Choose a visual style (e.g., "cyberpunk", "watercolor")
                4. Set your desired video duration
                5. Click "Generate YouTube Video"
                6. Wait for the pipeline to complete
                
                ### Advanced Options
                - **Number of Prompts**: Controls how many different creative prompts to generate
                - **Preferred Image Model**: Choose between Gemini, Stable Diffusion XL, or alternating between both
                - **Voice Generation Engine**: Select XTTS or Bark for voiceover generation
                
                ### Output
                - Generated images will appear in the gallery
                - The final video will be displayed in the video player
                - Complete results and file paths will be shown in the JSON output
                - All files are saved to the output directory (default: `output/`)
                
                ### Google Drive Integration
                To enable Google Drive uploads:
                1. Place your Google Drive API credentials in a file named `credentials.json`
                2. Set the `GOOGLE_DRIVE_API_KEY` environment variable
                """)
            
            # Set up event handlers
            generate_btn.click(
                fn=self._run_pipeline,
                inputs=[
                    theme_input, style_input, duration_input, 
                    output_dir_input, num_prompts_input, 
                    preferred_model_input, tts_engine_input
                ],
                outputs=[status_output, output_gallery, video_output, json_output]
            )
        
        return interface
    
    def _run_pipeline(self, theme, style, duration, output_dir, num_prompts, preferred_model, tts_engine):
        """Run the automation pipeline with the provided parameters."""
        try:
            if not self.pipeline:
                self.initialize_pipeline()
                if not self.pipeline:
                    return "Failed to initialize pipeline", None, None, None
            
            # Update configuration based on inputs
            self._update_dynamic_config(num_prompts, preferred_model, tts_engine)
            
            # Override output directory if specified
            if output_dir:
                self.pipeline.output_dir = Path(output_dir)
                os.makedirs(self.pipeline.output_dir, exist_ok=True)
            
            # Initialize components
            status = "Initializing pipeline components..."
            yield status, None, None, None
            
            self.pipeline.initialize_components()
            
            # Run the pipeline
            status = f"Running pipeline with theme: {theme}, style: {style}, duration: {duration}s..."
            yield status, None, None, None
            
            result = self.pipeline.run(
                theme=theme,
                style=style,
                duration=int(duration)
            )
            
            # Process and display results
            if result["status"] == "success":
                status = f"Pipeline completed successfully! Output in: {result['output_directory']}"
                
                # Get generated images to display in gallery
                images = self._collect_generated_images(result["output_directory"])
                
                # Get the video path
                video_path = result["video_path"]
                
                # Return results
                yield status, images, video_path, result
            else:
                status = f"Pipeline failed: {result.get('error', 'Unknown error')}"
                yield status, None, None, result
        
        except Exception as e:
            logger.error(f"Error running pipeline: {e}", exc_info=True)
            yield f"Error: {str(e)}", None, None, {"status": "error", "error": str(e)}
    
    def _collect_generated_images(self, output_dir):
        """Collect generated images to display in the gallery."""
        images = []
        
        try:
            # Look for images in the images directory
            images_dir = Path(output_dir) / "images"
            if images_dir.exists():
                for img_file in images_dir.glob("*.png"):
                    images.append(str(img_file))
            
            # If no images found, check the main output directory
            if not images:
                output_path = Path(output_dir)
                for img_file in output_path.glob("**/*.png"):
                    images.append(str(img_file))
        
        except Exception as e:
            logger.error(f"Error collecting images: {e}")
        
        return images
    
    def _update_dynamic_config(self, num_prompts, preferred_model, tts_engine):
        """Update configuration based on user inputs."""
        if self.pipeline and self.pipeline.config:
            # Update prompt generator config
            if "prompt_generator" in self.pipeline.config:
                self.pipeline.config["prompt_generator"]["num_prompts"] = int(num_prompts)
            
            # Update image generator config
            if "image_generator" in self.pipeline.config:
                self.pipeline.config["image_generator"]["preferred_model"] = preferred_model
            
            # Update voiceover config
            if "voiceover" in self.pipeline.config:
                self.pipeline.config["voiceover"]["tts_engine"] = tts_engine
    
    def _load_config_as_text(self):
        """Load the configuration file as text."""
        try:
            with open(self.config_path, 'r') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            return f"# Error loading config file: {e}\n# Creating a default config"
    
    def _save_config_text(self, config_text):
        """Save the edited configuration text to file."""
        try:
            with open(self.config_path, 'w') as f:
                f.write(config_text)
            
            # Reinitialize pipeline with new config
            self.initialize_pipeline()
            
            return f"Configuration saved successfully to {self.config_path}"
        except Exception as e:
            logger.error(f"Error saving config file: {e}")
            return f"Error saving config file: {e}"

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="YouTube Automation Pipeline Web Interface")
    parser.add_argument("--config", type=str, default="config.yaml", 
                        help="Path to configuration file")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Host address to run the server on")
    parser.add_argument("--port", type=int, default=7860,
                        help="Port to run the server on")
    parser.add_argument("--share", action="store_true",
                        help="Create a shareable link (useful for Colab)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # Create and launch the web app
    app = YouTubeAutomationWebApp(config_path=args.config)
    interface = app.create_interface()
    
    # Launch the interface
    interface.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share
    )
