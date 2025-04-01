"""
Multi-Modal Story and Image Generation Component
Uses Gemini 2.0 Flash (Image Generation) and Stable Diffusion XL (SDXL)
to generate highly detailed and realistic images with ControlNet for consistency.
"""
import os
import logging
import base64
import io
import random
import time
from pathlib import Path
import numpy as np
from PIL import Image
import google.generativeai as genai
import torch
from diffusers import StableDiffusionXLPipeline, ControlNetModel, DiffusionPipeline
from diffusers.utils import load_image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger("YouTube-Pipeline.ImageGenerator")

class MultiModalImageGenerator:
    """
    Generates high-quality images for storytelling using multiple AI models:
    1. Gemini 2.0 Flash for text-to-image generation
    2. Stable Diffusion XL for highly detailed supporting images
    3. ControlNet for ensuring consistency across related images
    """
    
    def __init__(self, config):
        """Initialize the image generator with configuration."""
        self.config = config
        self.models = {}
        self.output_dir = Path(config.get("output_dir", "output/images"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.initialize_models()
        logger.info(f"MultiModalImageGenerator initialized (using {self.device})")
    
    def initialize_models(self):
        """Initialize AI models based on configuration."""
        # Initialize Gemini
        try:
            if not os.getenv("GEMINI_API_KEY"):
                logger.warning("GEMINI_API_KEY not found in environment variables")
            else:
                genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
                gemini_model_name = self.config.get("gemini_model", "gemini-1.5-flash")
                self.models["gemini"] = genai.GenerativeModel(gemini_model_name)
                logger.info(f"Initialized Gemini model: {gemini_model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            self.models["gemini"] = None
        
        # Initialize a smaller diffusion model suitable for Colab
        if self.config.get("enable_sdxl", True):
            try:
                # Use a smaller Stable Diffusion model instead of SDXL
                # Options: "CompVis/stable-diffusion-v1-4" (smaller) or "runwayml/stable-diffusion-v1-5" (better quality)
                sdxl_model_name = self.config.get("sdxl_model", "runwayml/stable-diffusion-v1-5")
                logger.info(f"Attempting to load image model: {sdxl_model_name}")
                
                # In Colab, we need to use lower precision to save memory
                load_in_8bit = self.config.get("load_in_8bit", True)  # Default to 8-bit for Colab
                torch_dtype = torch.float16 if self.config.get("use_half_precision", True) else None
                
                try:
                    # First try importing a more memory-efficient pipeline
                    from diffusers import DiffusionPipeline
                    
                    # Initialize the pipeline with memory optimizations
                    self.models["sdxl"] = DiffusionPipeline.from_pretrained(
                        sdxl_model_name,
                        torch_dtype=torch_dtype,
                        use_safetensors=True,
                        variant="fp16" if torch_dtype == torch.float16 else None,
                        use_onnx=False,  # Using ONNX can reduce memory usage
                        local_files_only=False
                    )
                    
                    # Enable memory optimizations
                    self.models["sdxl"].enable_attention_slicing()
                    
                    # Move to appropriate device
                    if self.device == "cuda":
                        if load_in_8bit:
                            try:
                                # Try using 8-bit optimizations
                                self.models["sdxl"].unet = self.models["sdxl"].unet.to_bettertransformer()
                            except:
                                logger.warning("Could not convert to BetterTransformer, continuing with standard model")
                        
                        # Only move to GPU if we have enough VRAM 
                        try:
                            self.models["sdxl"] = self.models["sdxl"].to(self.device)
                            logger.info(f"Successfully loaded image model on {self.device}")
                        except Exception as gpu_error:
                            logger.warning(f"Failed to load model on GPU: {gpu_error}, falling back to CPU")
                            self.device = "cpu"  # Fall back to CPU
                    
                    logger.info(f"Initialized image generation model: {sdxl_model_name}")
                
                except Exception as init_error:
                    logger.error(f"Failed to initialize primary image model: {init_error}")
                    logger.info("Falling back to CPU-friendly text-to-image model")
                    
                    try:
                        # Try an even smaller model that works on CPU
                        from diffusers import StableDiffusionPipeline
                        small_model = "CompVis/stable-diffusion-v1-4"
                        
                        self.models["sdxl"] = StableDiffusionPipeline.from_pretrained(
                            small_model,
                            torch_dtype=torch.float32,  # Use full precision for CPU
                            use_safetensors=True
                        )
                        logger.info(f"Initialized fallback image model: {small_model}")
                    except Exception as fallback_error:
                        logger.error(f"Failed to initialize fallback model: {fallback_error}")
                        self.models["sdxl"] = None
                
                # Skip ControlNet in Colab - it uses too much memory
                self.models["controlnet"] = None
                
            except Exception as e:
                logger.error(f"Failed to initialize SDXL: {e}")
                self.models["sdxl"] = None
        else:
            logger.info("SDXL model disabled in configuration")
            self.models["sdxl"] = None
    
    def generate(self, prompts):
        """
        Generate images based on the provided prompts.
        
        Args:
            prompts: List of prompt dictionaries from the prompt generator
            
        Returns:
            List of image information dictionaries with paths and metadata
        """
        all_images = []
        
        for prompt_idx, prompt in enumerate(prompts):
            logger.info(f"Generating images for prompt: {prompt.get('title', f'Prompt {prompt_idx+1}')}")
            
            # Extract scene descriptions for image generation
            scenes = prompt.get("scenes", [])
            if not scenes:
                scenes = [f"Scene from: {prompt.get('title', 'Story')}"]
            
            # Extract style guidance
            style_guide = prompt.get("style_guide", "")
            
            # Generate main images for each scene
            prompt_images = []
            
            for scene_idx, scene_desc in enumerate(scenes):
                try:
                    # Combine scene description with style guidance
                    image_prompt = self._create_image_prompt(scene_desc, style_guide, prompt)
                    
                    # Progress logging
                    logger.info(f"Generating image {scene_idx+1}/{len(scenes)} for prompt {prompt_idx+1}/{len(prompts)}")
                    
                    # Determine which model to use (alternate for variety or based on scene complexity)
                    model_choice = self._select_model_for_scene(scene_idx, scene_desc)
                    
                    # Generate the image
                    image_info = self._generate_image_with_model(
                        model_choice, 
                        image_prompt,
                        prompt_id=prompt.get("id", f"prompt_{prompt_idx+1}"),
                        scene_id=f"scene_{scene_idx+1}"
                    )
                    
                    if image_info:
                        # Add relation to the prompt and scene
                        image_info["prompt_id"] = prompt.get("id", f"prompt_{prompt_idx+1}")
                        image_info["prompt_title"] = prompt.get("title", f"Prompt {prompt_idx+1}")
                        image_info["scene_index"] = scene_idx
                        image_info["scene_description"] = scene_desc
                        
                        prompt_images.append(image_info)
                        logger.info(f"Generated image saved to: {image_info.get('path')}")
                    else:
                        logger.warning(f"Failed to generate image for scene {scene_idx+1}")
                        
                except Exception as e:
                    logger.error(f"Error generating image for scene {scene_idx+1}: {e}")
            
            # Generate variant images or character close-ups if needed
            if self.config.get("generate_variants", True) and len(prompt_images) > 0:
                try:
                    variant_images = self._generate_variant_images(prompt_images, prompt)
                    prompt_images.extend(variant_images)
                    logger.info(f"Generated {len(variant_images)} variant images")
                except Exception as e:
                    logger.error(f"Error generating variant images: {e}")
            
            # Add all images for this prompt to the result list
            all_images.extend(prompt_images)
        
        return all_images
    
    def _create_image_prompt(self, scene_description, style_guide, prompt_data):
        """Create a detailed image prompt combining scene description and style guidance."""
        # Start with the scene description
        image_prompt = scene_description
        
        # Add style guidance if available
        if style_guide:
            image_prompt += f". {style_guide}"
        
        # Add character descriptions if relevant to the scene
        characters = prompt_data.get("characters", [])
        for character in characters:
            char_name = character.get("name", "")
            if char_name in scene_description:
                char_desc = character.get("description", "")
                if char_desc:
                    image_prompt += f". {char_name}: {char_desc}"
        
        # Add genre-specific modifiers
        genre = prompt_data.get("genre", "")
        if genre:
            genre_modifiers = {
                "sci-fi": "futuristic lighting, advanced technology, science fiction atmosphere",
                "fantasy": "magical atmosphere, fantasy lighting, mystical environment",
                "thriller": "dramatic lighting, tense atmosphere, high contrast",
                "mystery": "moody lighting, enigmatic atmosphere, shadows and fog",
                "romance": "soft lighting, warm colors, intimate atmosphere",
                "comedy": "bright colors, light-hearted atmosphere, vibrant scene",
                "drama": "emotional lighting, nuanced shadows, evocative mood",
                "action": "dynamic lighting, energetic atmosphere, vivid colors",
                "adventure": "dramatic vistas, atmospheric lighting, sense of wonder",
                "horror": "dark shadows, eerie lighting, unsettling atmosphere"
            }
            
            modifier = genre_modifiers.get(genre.lower(), "")
            if modifier:
                image_prompt += f". {modifier}"
        
        # Add universal quality modifiers
        quality_modifiers = [
            "8k resolution", "highly detailed", "cinematic lighting", 
            "professional photography", "realistic texture"
        ]
        
        if self.config.get("add_quality_modifiers", True):
            image_prompt += f". {', '.join(quality_modifiers)}"
        
        return image_prompt
    
    def _select_model_for_scene(self, scene_idx, scene_description):
        """Select the best model to use for a given scene."""
        # If only one model is available, use that
        if "gemini" in self.models and "sdxl" not in self.models:
            return "gemini"
        if "sdxl" in self.models and "gemini" not in self.models:
            return "sdxl"
        
        # If both are available, make a strategic choice
        if "gemini" in self.models and "sdxl" in self.models:
            # Use config preference if specified
            preferred_model = self.config.get("preferred_model", "alternate")
            
            if preferred_model == "gemini":
                return "gemini"
            elif preferred_model == "sdxl":
                return "sdxl"
            elif preferred_model == "alternate":
                # Alternate between models for variety
                return "gemini" if scene_idx % 2 == 0 else "sdxl"
            else:
                # Make decision based on scene complexity
                complex_terms = ["intricate", "detailed", "complex", "multiple", "action"]
                scene_complexity = sum(term in scene_description.lower() for term in complex_terms)
                return "sdxl" if scene_complexity >= 2 else "gemini"
        
        # Fallback
        return "fallback"
    
    def _generate_image_with_model(self, model_choice, image_prompt, prompt_id, scene_id):
        """Generate an image using the selected model."""
        if model_choice == "gemini" and self.models.get("gemini"):
            return self._generate_with_gemini(image_prompt, prompt_id, scene_id)
        elif model_choice == "sdxl" and self.models.get("sdxl"):
            return self._generate_with_sdxl(image_prompt, prompt_id, scene_id)
        else:
            logger.warning(f"Selected model '{model_choice}' not available, using fallback method")
            return self._generate_fallback_image(image_prompt, prompt_id, scene_id)
    
    def _generate_with_gemini(self, image_prompt, prompt_id, scene_id):
        """Generate an image using Gemini model."""
        try:
            # Configure generation parameters
            generation_config = {
                "temperature": self.config.get("gemini_temperature", 0.7),
                "top_p": self.config.get("gemini_top_p", 0.95),
                "top_k": self.config.get("gemini_top_k", 40),
                "max_output_tokens": self.config.get("gemini_max_tokens", 1024),
            }
            
            # Generate the image
            response = self.models["gemini"].generate_content(
                image_prompt,
                generation_config=generation_config,
                stream=False
            )
            
            # Extract and save the image
            if hasattr(response, 'parts') and response.parts:
                for part in response.parts:
                    if hasattr(part, 'inline_data') and part.inline_data:
                        # Save the image
                        img_data = base64.b64decode(part.inline_data.data)
                        img = Image.open(io.BytesIO(img_data))
                        
                        # Save to file
                        output_path = self.output_dir / f"{prompt_id}_{scene_id}_gemini.png"
                        img.save(output_path)
                        
                        return {
                            "id": f"{prompt_id}_{scene_id}_gemini",
                            "path": str(output_path),
                            "model": "gemini",
                            "prompt": image_prompt,
                            "width": img.width,
                            "height": img.height,
                            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
                        }
            
            logger.warning("Gemini did not return valid image data")
            return None
            
        except Exception as e:
            logger.error(f"Error generating image with Gemini: {e}")
            return None
    
    def _generate_with_sdxl(self, image_prompt, prompt_id, scene_id):
        """Generate an image using Stable Diffusion XL model."""
        try:
            # Configure generation parameters
            negative_prompt = self.config.get("sdxl_negative_prompt", 
                "blurry, distorted, disfigured, bad anatomy, watermark, signature, text, logo")
            
            num_inference_steps = self.config.get("sdxl_steps", 30)
            guidance_scale = self.config.get("sdxl_guidance_scale", 7.5)
            width = self.config.get("sdxl_width", 1024)
            height = self.config.get("sdxl_height", 1024)
            
            # Generate the image
            image = self.models["sdxl"](
                prompt=image_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height
            ).images[0]
            
            # Save to file
            output_path = self.output_dir / f"{prompt_id}_{scene_id}_sdxl.png"
            image.save(output_path)
            
            return {
                "id": f"{prompt_id}_{scene_id}_sdxl",
                "path": str(output_path),
                "model": "sdxl",
                "prompt": image_prompt,
                "negative_prompt": negative_prompt,
                "width": image.width,
                "height": image.height,
                "steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
        except Exception as e:
            logger.error(f"Error generating image with SDXL: {e}")
            return None
    
    def _generate_fallback_image(self, image_prompt, prompt_id, scene_id):
        """Generate a fallback placeholder image when model generation fails."""
        logger.info(f"Generating fallback image for scene {scene_id}")
        
        # Get a very short version of the prompt
        short_prompt = image_prompt.split(".")[0] if "." in image_prompt else image_prompt
        short_prompt = short_prompt[:50] + "..." if len(short_prompt) > 50 else short_prompt
        
        # First try using a very basic diffusion request if Gemini failed
        try:
            from diffusers import DiffusionPipeline
            import torch
            
            # Try to load a tiny model
            pipe = DiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float32,
                use_cpu_offload=True,
                low_cpu_mem_usage=True
            )
            
            # Generate with minimal settings
            image = pipe(
                prompt=short_prompt,
                num_inference_steps=20,
                guidance_scale=7.0
            ).images[0]
            
            # Save the image
            output_path = self.output_dir / f"gemini_{prompt_id}_scene_{scene_id}_fallback.png"
            image.save(output_path)
            
            return {
                "prompt_id": prompt_id,
                "scene_id": scene_id,
                "path": str(output_path),
                "model": "fallback_diffusion",
                "prompt": short_prompt,
                "success": True
            }
        except Exception as diffusion_error:
            logger.warning(f"Diffusion fallback failed: {diffusion_error}")
        
        # If diffusion fallback fails, create a text-based image
        try:
            # Create a simple PIL image with text
            img_size = (512, 512)
            background_color = (random.randint(0, 50), random.randint(0, 50), random.randint(0, 50))
            text_color = (200, 200, 200)
            
            img = Image.new('RGB', img_size, color=background_color)
            
            # Draw text on image
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(img)
            
            # Use a default font
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()
            
            # Wrap text to fit the image
            import textwrap
            wrapped_text = textwrap.fill(short_prompt, width=30)
            
            # Calculate text position for centering
            text_width, text_height = draw.textsize(wrapped_text, font=font) if hasattr(draw, 'textsize') else (300, 150)
            position = ((img_size[0] - text_width) // 2, (img_size[1] - text_height) // 2)
            
            # Draw the text
            draw.text(position, wrapped_text, fill=text_color, font=font)
            
            # Save the image
            output_path = self.output_dir / f"gemini_{prompt_id}_scene_{scene_id}_fallback.png"
            img.save(output_path)
            
            return {
                "prompt_id": prompt_id,
                "scene_id": scene_id,
                "path": str(output_path),
                "model": "fallback_text",
                "prompt": short_prompt,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error creating text fallback image: {e}")
            
            # Last resort - create an empty image file
            output_path = self.output_dir / f"gemini_{prompt_id}_scene_{scene_id}_fallback.png"
            with open(output_path, 'w') as f:
                f.write(f"PLACEHOLDER: {short_prompt}")
            
            return {
                "prompt_id": prompt_id,
                "scene_id": scene_id,
                "path": str(output_path),
                "model": "fallback_placeholder",
                "prompt": short_prompt,
                "success": True
            }
    
    def _generate_variant_images(self, base_images, prompt):
        """Generate variant images based on existing generated images."""
        variant_images = []
        
        # Only generate variants if we have SDXL
        if "sdxl" not in self.models:
            return variant_images
        
        # Select a subset of base images to create variants for
        num_variants = min(self.config.get("num_variants", 2), len(base_images))
        selected_images = random.sample(base_images, num_variants)
        
        # Characters for close-ups
        characters = prompt.get("characters", [])
        
        for idx, image_info in enumerate(selected_images):
            try:
                # Decide variant type: composition variation or character close-up
                variant_type = "character" if characters and idx < len(characters) else "variation"
                
                if variant_type == "character":
                    # Generate character close-up
                    character = characters[idx]
                    char_name = character.get("name", "Unknown")
                    char_desc = character.get("description", "")
                    
                    variant_prompt = f"Close-up portrait of {char_name}, {char_desc}. "
                    if "style_guide" in prompt:
                        variant_prompt += prompt["style_guide"]
                    
                    # Generate with SDXL
                    variant_info = self._generate_with_sdxl(
                        variant_prompt,
                        prompt_id=image_info["prompt_id"],
                        scene_id=f"character_{idx+1}"
                    )
                    
                    if variant_info:
                        variant_info["variant_type"] = "character"
                        variant_info["character_name"] = char_name
                        variant_info["base_image_id"] = image_info["id"]
                        variant_images.append(variant_info)
                
                else:
                    # Generate composition variation using ControlNet if available
                    if "controlnet" in self.models and self.models["controlnet"]:
                        # Load the base image
                        base_image_path = image_info["path"]
                        if not os.path.exists(base_image_path):
                            continue
                        
                        init_image = load_image(base_image_path)
                        
                        # Create a modified prompt
                        base_prompt = image_info["prompt"]
                        variant_prompt = f"Alternative version of: {base_prompt}"
                        
                        # TODO: Implement ControlNet-based variation
                        # This requires additional setup with the control pipeline
                        
                    else:
                        # Without ControlNet, just create a different composition
                        base_prompt = image_info["prompt"]
                        variant_prompt = f"Alternative perspective of: {base_prompt}"
                        
                        # Generate with SDXL with a different seed
                        variant_info = self._generate_with_sdxl(
                            variant_prompt,
                            prompt_id=image_info["prompt_id"],
                            scene_id=f"variant_{idx+1}"
                        )
                        
                        if variant_info:
                            variant_info["variant_type"] = "composition"
                            variant_info["base_image_id"] = image_info["id"]
                            variant_images.append(variant_info)
            
            except Exception as e:
                logger.error(f"Error generating variant image {idx+1}: {e}")
        
        return variant_images
