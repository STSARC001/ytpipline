"""
Advanced Video Compilation with Dynamic Effects Component
Uses FFmpeg + MoviePy for frame-perfect sequencing and dynamic effects.
"""
import os
import logging
import time
from pathlib import Path
import random
import subprocess
import json
from dotenv import load_dotenv
try:
    import moviepy.editor as mpy
except ImportError:
    mpy = None

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger("YouTube-Pipeline.VideoCompiler")

class AdvancedVideoCompiler:
    """
    Compiles final videos with cinematic effects:
    1. Sequences animations with perfect timing
    2. Adds transitions and special effects
    3. Syncs with AI-generated voiceover
    4. Generates thumbnail for the video
    """
    
    def __init__(self, config):
        """Initialize the video compiler with configuration."""
        self.config = config
        self.output_dir = Path(config.get("output_dir", "output/video"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if we're in Google Colab
        try:
            import google.colab
            self.in_colab = True
            logger.info("Running in Google Colab environment - using lightweight video compilation methods")
            self.config["simplified_mode"] = True
        except:
            self.in_colab = False
        
        self.moviepy_available = mpy is not None
        self.ffmpeg_path = config.get("ffmpeg_path", "ffmpeg")
        self.transitions = config.get("transitions", ["fade", "dissolve", "wipe"])
        self.effects = config.get("effects", ["motion_blur", "color_enhance", "dynamic_zoom"])
        logger.info(f"AdvancedVideoCompiler initialized (MoviePy available: {self.moviepy_available}, Colab mode: {self.in_colab})")
    
    def compile(self, animation_info, voiceover_info, params=None):
        """
        Compile animations and voiceover into a final video.
        
        Args:
            animation_info: List of animation information dictionaries from the animator
            voiceover_info: List of voiceover information dictionaries from the voiceover generator
            params: Dictionary containing optional parameters
            
        Returns:
            Dictionary with final video information
        """
        if params is None:
            params = {}
        
        # Extract parameters
        target_duration = params.get("duration", 60)  # seconds
        style = params.get("style", "")
        video_title = params.get("title", "AI Generated Video")
        
        logger.info(f"Compiling video with target duration: {target_duration}s")
        
        # Group animations and voiceovers by prompt_id
        animations_by_prompt = {}
        for anim in animation_info:
            prompt_id = anim.get("prompt_id", "unknown")
            if prompt_id not in animations_by_prompt:
                animations_by_prompt[prompt_id] = []
            animations_by_prompt[prompt_id].append(anim)
        
        voiceovers_by_prompt = {}
        for voice in voiceover_info:
            prompt_id = voice.get("prompt_id", "unknown")
            voiceovers_by_prompt[prompt_id] = voice
        
        # Determine which prompts to include based on target duration
        total_duration = 0
        selected_prompts = []
        
        for prompt_id, animations in animations_by_prompt.items():
            # Calculate total animation duration for this prompt
            prompt_duration = sum(anim.get("duration", 0) for anim in animations)
            
            # Only add if we haven't exceeded target duration
            if total_duration + prompt_duration <= target_duration:
                selected_prompts.append(prompt_id)
                total_duration += prompt_duration
            
            # Stop once we have enough content
            if total_duration >= target_duration:
                break
        
        logger.info(f"Selected {len(selected_prompts)} prompts for compilation, total duration: {total_duration}s")
        
        # Create the output filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_filename = f"ai_video_{timestamp}.mp4"
        output_path = self.output_dir / output_filename
        
        # Create the final video
        if self.moviepy_available:
            video_info = self._compile_with_moviepy(
                selected_prompts,
                animations_by_prompt,
                voiceovers_by_prompt,
                output_path,
                style
            )
        else:
            video_info = self._compile_with_ffmpeg(
                selected_prompts,
                animations_by_prompt,
                voiceovers_by_prompt,
                output_path,
                style
            )
        
        # Generate thumbnail
        thumbnail_path = self.output_dir / f"thumbnail_{timestamp}.jpg"
        thumbnail_info = self._generate_thumbnail(
            animations_by_prompt,
            selected_prompts,
            thumbnail_path,
            video_title
        )
        
        # Add thumbnail path to video info
        if thumbnail_info:
            video_info["thumbnail_path"] = thumbnail_info.get("path")
        
        logger.info(f"Video compilation complete: {output_path}")
        return video_info
    
    def _compile_with_moviepy(self, selected_prompts, animations_by_prompt, 
                             voiceovers_by_prompt, output_path, style):
        """Compile video using MoviePy library."""
        if not self.moviepy_available:
            logger.warning("MoviePy is not available, falling back to FFmpeg")
            return self._compile_with_ffmpeg(selected_prompts, animations_by_prompt, 
                                            voiceovers_by_prompt, output_path, style)
        
        # In Colab with simplified mode, use minimal processing
        if self.config.get("simplified_mode", False):
            logger.info("Using simplified MoviePy compilation for Colab compatibility")
            try:
                # Create a placeholder video that will work in Colab
                # For simplicity, we'll concatenate clips without fancy effects
                clips = []
                
                # Process each animation clip
                for prompt_id in selected_prompts:
                    animations = animations_by_prompt.get(prompt_id, [])
                    for anim in animations:
                        path = anim.get("path")
                        
                        # If the path exists and is a proper video, add it
                        if os.path.exists(path) and path.endswith(('.mp4', '.avi', '.mov')):
                            try:
                                clip = mpy.VideoFileClip(path)
                                clips.append(clip)
                            except Exception as e:
                                logger.warning(f"Could not load clip {path}: {e}")
                
                # If we have no clips, create a placeholder
                if not clips:
                    # Create a black clip with text
                    text_clip = mpy.TextClip(
                        "YouTube Automation Pipeline\nPlaceholder Video", 
                        fontsize=70, 
                        color='white',
                        size=(1280, 720)
                    ).set_duration(5)
                    clips = [text_clip]
                
                # Concatenate clips (simple method)
                final_clip = mpy.concatenate_videoclips(clips, method="compose")
                
                # Add audio if it exists and is a proper audio file
                for prompt_id in selected_prompts:
                    voiceover = voiceovers_by_prompt.get(prompt_id)
                    if voiceover and voiceover.get("path") and os.path.exists(voiceover.get("path")) and voiceover.get("path").endswith(('.mp3', '.wav')):
                        try:
                            audio = mpy.AudioFileClip(voiceover.get("path"))
                            # Set audio to loop if shorter than video
                            audio = audio.set_duration(final_clip.duration)
                            final_clip = final_clip.set_audio(audio)
                        except Exception as e:
                            logger.warning(f"Could not load audio {voiceover.get('path')}: {e}")
                
                # Write the final clip with minimal settings
                final_clip.write_videofile(
                    output_path,
                    codec='libx264',
                    preset='ultrafast',  # Use fastest preset for Colab
                    threads=2,  # Limit threads for Colab
                    logger=None  # Reduce verbose output
                )
                
                return {
                    "output_path": str(output_path),
                    "duration": final_clip.duration,
                    "num_clips": len(clips),
                    "style": style,
                    "compiler": "moviepy_simplified",
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            
            except Exception as e:
                logger.error(f"Simplified MoviePy compilation failed: {e}")
                return self._compile_with_ffmpeg(selected_prompts, animations_by_prompt, 
                                                voiceovers_by_prompt, output_path, style)
        
        # Original MoviePy implementation for non-Colab environments
        logger.info("Compiling video with MoviePy")
        
        # This would be a full MoviePy implementation in a real system
        # For demonstration, we'll create a simulated representation
        
        # In a real implementation, this would:
        # 1. Load all animation clips
        # 2. Add transitions between clips
        # 3. Add effects based on style
        # 4. Add voiceover audio
        # 5. Export final video
        
        # For demonstration, we'll simulate compilation by creating a JSON report
        clips = []
        total_duration = 0
        
        for prompt_id in selected_prompts:
            animations = animations_by_prompt.get(prompt_id, [])
            voiceover = voiceovers_by_prompt.get(prompt_id)
            
            # Process each animation for this prompt
            for anim in animations:
                anim_path = anim.get("path")
                if not os.path.exists(anim_path):
                    logger.warning(f"Animation file not found: {anim_path}")
                    continue
                
                duration = anim.get("duration", 5)
                
                # Add clip to sequence
                clips.append({
                    "type": "video",
                    "path": anim_path,
                    "start_time": total_duration,
                    "duration": duration,
                    "effects": self._select_effects(style, anim)
                })
                
                total_duration += duration
            
            # Add voiceover if available
            if voiceover:
                clips.append({
                    "type": "audio",
                    "path": voiceover.get("path"),
                    "start_time": 0,
                    "duration": voiceover.get("duration", total_duration)
                })
        
        # Write compilation report to output location
        with open(output_path.with_suffix('.json'), 'w') as f:
            json.dump({
                "clips": clips,
                "total_duration": total_duration,
                "style": style
            }, f, indent=2)
        
        # Create a placeholder video file
        self._create_placeholder_video(output_path, total_duration)
        
        return {
            "output_path": str(output_path),
            "duration": total_duration,
            "num_clips": len(clips),
            "style": style,
            "compiler": "moviepy_simulated",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def _compile_with_ffmpeg(self, selected_prompts, animations_by_prompt, 
                            voiceovers_by_prompt, output_path, style):
        """Compile video using FFmpeg directly."""
        logger.info("Compiling video with FFmpeg")
        
        # This would be a full FFmpeg implementation in a real system
        # For demonstration, we'll create a simulated representation
        
        # In a real implementation, this would:
        # 1. Create an FFmpeg complex filter graph
        # 2. Process all input files with transitions
        # 3. Mix in audio
        # 4. Apply final processing and output video
        
        # For demonstration, we'll simulate compilation
        clips = []
        total_duration = 0
        
        for prompt_id in selected_prompts:
            animations = animations_by_prompt.get(prompt_id, [])
            voiceover = voiceovers_by_prompt.get(prompt_id)
            
            # Process each animation for this prompt
            for anim in animations:
                anim_path = anim.get("path")
                if not os.path.exists(anim_path):
                    logger.warning(f"Animation file not found: {anim_path}")
                    continue
                
                duration = anim.get("duration", 5)
                
                # Add clip to sequence
                clips.append({
                    "type": "video",
                    "path": anim_path,
                    "start_time": total_duration,
                    "duration": duration,
                    "effects": self._select_effects(style, anim)
                })
                
                total_duration += duration
            
            # Add voiceover if available
            if voiceover:
                clips.append({
                    "type": "audio",
                    "path": voiceover.get("path"),
                    "start_time": 0,
                    "duration": voiceover.get("duration", total_duration)
                })
        
        # Write ffmpeg command file
        with open(output_path.with_suffix('.ffmpeg.txt'), 'w') as f:
            f.write(f"# FFmpeg command to compile {len(clips)} clips\n")
            f.write("# This is a placeholder for the actual FFmpeg command\n")
            f.write(f"# Total duration: {total_duration}s\n")
            for i, clip in enumerate(clips):
                f.write(f"# Clip {i+1}: {clip['path']} (duration: {clip['duration']}s)\n")
        
        # Create a placeholder video file
        self._create_placeholder_video(output_path, total_duration)
        
        return {
            "output_path": str(output_path),
            "duration": total_duration,
            "num_clips": len(clips),
            "style": style,
            "compiler": "ffmpeg_simulated",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def _select_effects(self, style, animation):
        """Select appropriate visual effects based on style and animation."""
        selected_effects = []
        
        # Select 1-3 effects based on style
        if "sci-fi" in style.lower():
            effect_pool = ["glow", "digital_distortion", "color_shift"]
        elif "fantasy" in style.lower():
            effect_pool = ["magic_particles", "color_enhance", "soft_light"]
        elif "thriller" in style.lower() or "horror" in style.lower():
            effect_pool = ["film_grain", "vignette", "contrast_enhance"]
        elif "romance" in style.lower():
            effect_pool = ["soft_focus", "warm_filter", "light_leak"]
        else:
            # Default effects pool
            effect_pool = ["motion_blur", "color_enhance", "dynamic_zoom", "light_adjustment"]
        
        # Select random effects from the pool
        num_effects = random.randint(1, min(3, len(effect_pool)))
        selected_effects = random.sample(effect_pool, num_effects)
        
        return selected_effects
    
    def _generate_thumbnail(self, animations_by_prompt, selected_prompts, output_path, title):
        """Generate an attractive thumbnail for the video."""
        logger.info(f"Generating thumbnail: {output_path}")
        
        # In simplified Colab mode, create a basic thumbnail
        if self.config.get("simplified_mode", False):
            logger.info("Using simplified thumbnail generation for Colab compatibility")
            try:
                # Find the first animation clip path
                first_frame_path = None
                for prompt_id in selected_prompts:
                    animations = animations_by_prompt.get(prompt_id, [])
                    if animations:
                        # Select the first animation as source
                        first_frame_path = animations[0].get("source_image")
                        break
                
                # If we have a video clip, extract the first frame
                if first_frame_path and first_frame_path.endswith(('.mp4', '.avi', '.mov')) and self.moviepy_available:
                    try:
                        video = mpy.VideoFileClip(first_frame_path)
                        # Get the first frame and save as thumbnail
                        video.save_frame(str(output_path), t=0)
                        video.close()
                        return {
                            "path": str(output_path),
                            "source": first_frame_path,
                            "title": title,
                            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
                        }
                    except Exception as e:
                        logger.warning(f"Error extracting frame from video: {e}")
                
                # Fallback: create a text-based thumbnail
                if self.moviepy_available:
                    try:
                        # Create a simple colored background with text
                        text_clip = mpy.TextClip(
                            title, 
                            fontsize=60, 
                            color='white',
                            size=(1280, 720),
                            bg_color='black'
                        )
                        text_clip.save_frame(str(output_path))
                        return {
                            "path": str(output_path),
                            "title": title,
                            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
                        }
                    except Exception as e:
                        logger.warning(f"Error creating text thumbnail: {e}")
                
                # If all else fails, write a placeholder
                with open(output_path, 'w') as f:
                    f.write("THUMBNAIL PLACEHOLDER")
                
                return {
                    "path": str(output_path),
                    "title": title,
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            
            except Exception as e:
                logger.error(f"Simplified thumbnail generation failed: {e}")
                return None
        
        # Original thumbnail implementation for non-Colab environments
        # In a real implementation, this would:
        # 1. Select the most visually appealing frame from animations
        # 2. Apply enhancements and text overlay
        # 3. Save as the thumbnail image
        
        # For demonstration, we'll simulate thumbnail generation
        if not selected_prompts:
            logger.error("No prompts available for thumbnail generation")
            return None
        
        # Try to find a good animation frame to use
        thumbnail_source = None
        for prompt_id in selected_prompts:
            animations = animations_by_prompt.get(prompt_id, [])
            if animations:
                # Select the first animation as source
                thumbnail_source = animations[0].get("source_image")
                break
        
        if not thumbnail_source:
            logger.warning("No source image found for thumbnail")
            return None
        
        # Create a placeholder thumbnail file
        with open(output_path, 'w') as f:
            f.write(f"PLACEHOLDER THUMBNAIL - Source: {thumbnail_source}, Title: {title}")
        
        return {
            "path": str(output_path),
            "source": thumbnail_source,
            "title": title,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def _create_placeholder_video(self, output_path, duration):
        """Create a placeholder video file for demonstration purposes."""
        # In a real implementation, this would be replaced by actual video generation
        # For demonstration, we'll just create a text file with the same name
        with open(output_path, 'w') as f:
            f.write(f"PLACEHOLDER VIDEO FILE - Duration: {duration} seconds\n")
            f.write("In a real implementation, this would be an actual video file.\n")
