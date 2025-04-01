"""
Advanced YouTube Automation Pipeline with Multi-Model Integration
This script orchestrates the complete pipeline for generating AI-powered YouTube content.
"""
import os
import argparse
import logging
import json
import uuid
import yaml
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("pipeline.log"), logging.StreamHandler()]
)
logger = logging.getLogger("YouTube-Pipeline")

class YouTubeAutomationPipeline:
    """Main class orchestrating the YouTube content automation pipeline."""
    
    def __init__(self, config_path="config.yaml"):
        """Initialize the pipeline with configuration."""
        self.config = self._load_config(config_path)
        self.output_dir = self._create_output_directory()
        self.pipeline_id = str(uuid.uuid4())[:8]
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize components
        self.prompt_generator = None
        self.image_generator = None
        self.animator = None
        self.voiceover_generator = None
        self.video_compiler = None
        self.drive_uploader = None
        
        logger.info(f"Pipeline initialized with ID: {self.pipeline_id}")
    
    def _load_config(self, config_path):
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def _create_output_directory(self):
        """Create directory structure for outputs."""
        base_dir = Path(self.config.get("output_directory", "output"))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = base_dir / f"generation_{timestamp}"
        
        # Create subdirectories
        for subdir in ["prompts", "images", "animations", "audio", "video", "metadata"]:
            (output_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created output directory: {output_dir}")
        return output_dir
    
    def initialize_components(self):
        """Initialize all pipeline components based on configuration."""
        from components.prompt_generator import MultiModelPromptGenerator
        from components.image_generator import MultiModalImageGenerator
        from components.animator import HyperRealisticAnimator
        from components.voiceover import AIVoiceoverGenerator
        from components.video_compiler import AdvancedVideoCompiler
        from components.drive_uploader import GoogleDriveUploader
        
        # Initialize components with their respective configurations
        self.prompt_generator = MultiModelPromptGenerator(
            self.config.get("prompt_generator", {})
        )
        
        self.image_generator = MultiModalImageGenerator(
            self.config.get("image_generator", {})
        )
        
        self.animator = HyperRealisticAnimator(
            self.config.get("animator", {})
        )
        
        self.voiceover_generator = AIVoiceoverGenerator(
            self.config.get("voiceover", {})
        )
        
        self.video_compiler = AdvancedVideoCompiler(
            self.config.get("video_compiler", {})
        )
        
        self.drive_uploader = GoogleDriveUploader(
            self.config.get("drive_uploader", {})
        )
        
        logger.info("All pipeline components initialized")
    
    def run(self, theme=None, style=None, duration=None):
        """Run the complete pipeline."""
        try:
            logger.info(f"Starting pipeline execution with theme: {theme}")
            
            # Override config with provided parameters
            params = {
                "theme": theme,
                "style": style,
                "duration": duration
            }
            
            # 1. Generate creative prompts
            prompts = self.prompt_generator.generate(params)
            self._save_output("prompts", "prompts.json", prompts)
            logger.info(f"Generated {len(prompts)} creative prompts")
            
            # 2. Generate images from prompts
            images_info = self.image_generator.generate(prompts)
            self._save_output("images", "images_info.json", images_info)
            logger.info(f"Generated {len(images_info)} images")
            
            # 3. Animate the images
            animation_info = self.animator.animate(images_info)
            self._save_output("animations", "animation_info.json", animation_info)
            logger.info("Created animations from images")
            
            # 4. Generate voiceover
            voiceover_info = self.voiceover_generator.generate(prompts, animation_info)
            self._save_output("audio", "voiceover_info.json", voiceover_info)
            logger.info("Generated AI voiceover")
            
            # 5. Compile video with effects
            video_info = self.video_compiler.compile(
                animation_info, voiceover_info, params
            )
            self._save_output("video", "video_info.json", video_info)
            logger.info(f"Compiled final video: {video_info.get('output_path')}")
            
            # 6. Generate metadata and upload to Drive
            metadata = self._generate_metadata(prompts, video_info)
            self._save_output("metadata", "metadata.json", metadata)
            
            upload_info = self.drive_uploader.upload(
                video_path=video_info.get("output_path"),
                thumbnail_path=video_info.get("thumbnail_path"),
                metadata=metadata
            )
            self._save_output("metadata", "upload_info.json", upload_info)
            logger.info(f"Uploaded content to Google Drive: {upload_info.get('url')}")
            
            return {
                "status": "success",
                "pipeline_id": self.pipeline_id,
                "output_directory": str(self.output_dir),
                "video_path": video_info.get("output_path"),
                "drive_url": upload_info.get("url")
            }
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            return {
                "status": "error",
                "pipeline_id": self.pipeline_id,
                "error": str(e)
            }
    
    def _save_output(self, subdir, filename, data):
        """Save output data to a JSON file."""
        output_path = self.output_dir / subdir / filename
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        return str(output_path)
    
    def _generate_metadata(self, prompts, video_info):
        """Generate SEO-optimized metadata for the video."""
        # This would normally use AI to generate optimized metadata
        # For now we'll use a simple approach
        return {
            "title": f"AI Generated Story: {prompts[0]['title'] if prompts else 'Untitled'}",
            "description": self._create_description(prompts),
            "tags": self._extract_tags(prompts),
            "thumbnail_path": video_info.get("thumbnail_path"),
            "generated_at": self.timestamp
        }
    
    def _create_description(self, prompts):
        """Create a video description from prompts."""
        if not prompts:
            return "AI-generated video created with advanced multi-model pipeline."
        
        main_prompt = prompts[0]
        description = f"{main_prompt.get('title', 'AI Story')}\n\n"
        description += main_prompt.get('description', '')
        description += "\n\nThis video was created entirely with AI using multiple models including "
        description += "Gemini, Stable Diffusion XL, AnimateDiff, and advanced voice synthesis."
        return description
    
    def _extract_tags(self, prompts):
        """Extract relevant tags from prompts."""
        tags = ["AI", "AI Art", "AI Video", "AI Story", "AI Animation"]
        
        if prompts:
            # Extract keywords from title and description
            for prompt in prompts:
                title = prompt.get("title", "")
                desc = prompt.get("description", "")
                
                # Extract single words and two-word phrases as potential tags
                words = set((title + " " + desc).replace(",", " ").replace(".", " ").split())
                for word in words:
                    if len(word) > 3 and word.lower() not in [t.lower() for t in tags]:
                        tags.append(word)
                
                # Add any explicitly defined tags from the prompt
                if "tags" in prompt and isinstance(prompt["tags"], list):
                    for tag in prompt["tags"]:
                        if tag not in tags:
                            tags.append(tag)
        
        return tags[:20]  # YouTube allows up to 500 characters of tags


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="YouTube Automation Pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", 
                        help="Path to configuration file")
    parser.add_argument("--theme", type=str, help="Theme for content generation")
    parser.add_argument("--style", type=str, help="Visual style for generation")
    parser.add_argument("--duration", type=int, help="Target video duration in seconds")
    parser.add_argument("--output-dir", type=str, help="Custom output directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    # Create pipeline instance
    pipeline = YouTubeAutomationPipeline(config_path=args.config)
    
    # Override output directory if specified
    if args.output_dir:
        pipeline.output_dir = Path(args.output_dir)
        os.makedirs(pipeline.output_dir, exist_ok=True)
    
    # Initialize all components
    pipeline.initialize_components()
    
    # Run the pipeline
    result = pipeline.run(
        theme=args.theme,
        style=args.style,
        duration=args.duration
    )
    
    # Print final result
    print(json.dumps(result, indent=2))
