"""
AI Voiceover with Emotion and Dynamic Effects Component
Uses XTTS and Bark AI for multi-speaker, emotion-rich voiceovers with real-time audio effects.
"""
import os
import logging
import json
import time
import random
from pathlib import Path
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger("YouTube-Pipeline.Voiceover")

class AIVoiceoverGenerator:
    """
    Generates realistic AI voiceovers with emotion and audio effects:
    1. XTTS for high-quality, controllable speech synthesis
    2. Bark AI for expressive, multi-speaker narration
    3. Audio effects for cinematic depth
    4. Auto-synchronization with animation timing
    """
    
    def __init__(self, config):
        """Initialize the voiceover generator with configuration."""
        self.config = config
        self.output_dir = Path(config.get("output_dir", "output/audio"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tts_engine = config.get("tts_engine", "xtts")
        self.models = {}
        self.initialize_models()
        logger.info(f"AIVoiceoverGenerator initialized with TTS engine: {self.tts_engine}")
    
    def initialize_models(self):
        """Initialize TTS models based on configuration."""
        # XTTS would be initialized here in a real implementation
        self.models["xtts"] = self.config.get("enable_xtts", True)
        
        # Bark would be initialized here in a real implementation
        self.models["bark"] = self.config.get("enable_bark", True)
        
        logger.info(f"TTS models initialized: XTTS={self.models['xtts']}, Bark={self.models['bark']}")
    
    def generate(self, prompts, animation_info):
        """
        Generate voiceovers based on the provided prompts and animations.
        
        Args:
            prompts: List of prompt dictionaries from the prompt generator
            animation_info: List of animation information dictionaries from the animator
            
        Returns:
            List of voiceover information dictionaries with paths and metadata
        """
        voiceovers = []
        
        # Group animations by prompt_id
        grouped_animations = {}
        for anim_info in animation_info:
            prompt_id = anim_info.get("prompt_id", "unknown")
            if prompt_id not in grouped_animations:
                grouped_animations[prompt_id] = []
            grouped_animations[prompt_id].append(anim_info)
        
        # Process each prompt
        for prompt in prompts:
            prompt_id = prompt.get("id", "unknown")
            logger.info(f"Generating voiceover for prompt: {prompt.get('title', prompt_id)}")
            
            # Get animations for this prompt
            animations = grouped_animations.get(prompt_id, [])
            
            # Generate script from prompt
            script = self._generate_script(prompt, animations)
            
            # Generate voiceover audio
            voiceover_info = self._generate_voiceover(
                script, 
                prompt_id, 
                animations
            )
            
            if voiceover_info:
                voiceovers.append(voiceover_info)
                logger.info(f"Generated voiceover: {voiceover_info.get('path')}")
        
        return voiceovers
    
    def _generate_script(self, prompt, animations):
        """Generate a narration script from the prompt data."""
        # Extract relevant content from the prompt
        title = prompt.get("title", "Untitled")
        description = prompt.get("description", "")
        scenes = prompt.get("scenes", [])
        characters = prompt.get("characters", [])
        
        # Create script structure
        script = {
            "title": title,
            "intro": self._create_intro(title, description),
            "segments": []
        }
        
        # Calculate available speaking time for each scene based on animations
        scene_durations = {}
        total_animation_duration = 0
        
        for anim in animations:
            scene_idx = anim.get("scene_index", -1)
            duration = anim.get("duration", 5)
            
            if scene_idx >= 0:
                scene_durations[scene_idx] = duration
                total_animation_duration += duration
        
        # Fill in scene narrations
        for i, scene in enumerate(scenes):
            # Get scene duration (if available)
            duration = scene_durations.get(i, 5)
            
            # Create scene narration, adapted to fit the duration
            narration = self._create_scene_narration(scene, duration, characters)
            
            script["segments"].append({
                "scene_index": i,
                "duration": duration,
                "text": narration,
                "emotion": self._determine_emotion(scene)
            })
        
        # Add outro if total narration is short
        if total_animation_duration > 0 and len(script["segments"]) > 0:
            script["outro"] = self._create_outro(title, description)
            
        return script
    
    def _create_intro(self, title, description):
        """Create introductory narration."""
        intro_templates = [
            f"Welcome to {title}.",
            f"Presenting {title}.",
            f"Experience {title}.",
            f"Journey into {title}."
        ]
        
        intro = random.choice(intro_templates)
        
        # Add a sentence from the description if available
        if description:
            sentences = description.split('.')
            for sentence in sentences:
                if sentence and len(sentence.split()) > 3:
                    intro += f" {sentence.strip()}."
                    break
        
        return intro
    
    def _create_scene_narration(self, scene, duration, characters):
        """Create narration for a scene within time constraints."""
        # Estimate words per second (average speaking rate)
        words_per_second = 2.5
        
        # Calculate target word count based on duration (leaving buffer for pauses)
        buffer_seconds = 0.5
        target_words = int((duration - buffer_seconds) * words_per_second)
        
        # Use the scene description as starting point
        narration = scene
        
        # Check if any character names are in the scene
        for character in characters:
            char_name = character.get("name", "")
            if char_name in scene and character.get("description", ""):
                # Add a brief description if character is mentioned
                narration += f" {char_name} is {character.get('description', '')}."
        
        # Adjust length to fit target word count
        words = narration.split()
        if len(words) > target_words:
            # Truncate and add ellipsis if too long
            narration = " ".join(words[:target_words]) + "..."
        elif len(words) < target_words * 0.7:
            # Expand if too short (in real implementation, would use AI to expand)
            narration += " The scene unfolds with remarkable detail and emotion."
        
        return narration
    
    def _create_outro(self, title, description):
        """Create concluding narration."""
        outro_templates = [
            f"Thank you for experiencing {title}.",
            f"That concludes our journey through {title}.",
            f"We hope you enjoyed {title}."
        ]
        
        return random.choice(outro_templates)
    
    def _determine_emotion(self, scene):
        """Determine appropriate emotion for the scene narration."""
        # Simple keyword-based emotion detection
        emotion_keywords = {
            "happy": ["happy", "joy", "celebration", "smile", "laugh", "excitement"],
            "sad": ["sad", "sorrow", "grief", "tear", "cry", "misery", "melancholy"],
            "angry": ["angry", "rage", "fury", "mad", "outrage", "violent"],
            "fear": ["fear", "terror", "horror", "fright", "dread", "panic"],
            "surprise": ["surprise", "astonish", "amaze", "shock", "unexpected"],
            "neutral": []  # fallback
        }
        
        # Count emotion keywords in the scene
        scene_lower = scene.lower()
        counts = {emotion: 0 for emotion in emotion_keywords}
        
        for emotion, keywords in emotion_keywords.items():
            for keyword in keywords:
                if keyword in scene_lower:
                    counts[emotion] += 1
        
        # Determine primary emotion (highest count)
        primary_emotion = max(counts, key=counts.get)
        
        # If no emotion detected, use neutral
        if counts[primary_emotion] == 0:
            primary_emotion = "neutral"
        
        return primary_emotion
    
    def _generate_voiceover(self, script, prompt_id, animations):
        """Generate voiceover audio based on the script."""
        # Determine which TTS engine to use
        if self.tts_engine == "xtts" and self.models["xtts"]:
            return self._generate_with_xtts(script, prompt_id, animations)
        elif self.tts_engine == "bark" or (not self.models["xtts"] and self.models["bark"]):
            return self._generate_with_bark(script, prompt_id, animations)
        else:
            logger.warning("No TTS engine available, using fallback method")
            return self._generate_fallback(script, prompt_id, animations)
    
    def _generate_with_xtts(self, script, prompt_id, animations):
        """Generate voiceover using XTTS."""
        # In a real implementation, this would use XTTS
        # For demonstration purposes, we'll simulate the output
        
        logger.info(f"Generating voiceover with XTTS for: {prompt_id}")
        
        # Calculate total duration
        total_duration = sum(segment.get("duration", 5) for segment in script.get("segments", []))
        
        # Ensure output directory exists
        output_path = self.output_dir / f"{prompt_id}_voiceover.mp3"
        
        # In a real implementation, we would:
        # 1. Convert script to SSML with emotion markers
        # 2. Call XTTS to generate audio
        # 3. Add effects and process audio
        
        # Simulate processing time
        time.sleep(1)
        
        # Create a placeholder audio file (silent)
        self._create_silent_audio(output_path, total_duration)
        
        # Create timestamps for synchronization
        timestamps = self._create_timestamps(script, animations)
        
        return {
            "id": f"{prompt_id}_voiceover",
            "path": str(output_path),
            "engine": "xtts_simulated",
            "prompt_id": prompt_id,
            "script": script,
            "timestamps": timestamps,
            "duration": total_duration,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def _generate_with_bark(self, script, prompt_id, animations):
        """Generate voiceover using Bark AI."""
        # In a real implementation, this would use Bark
        # For demonstration purposes, we'll simulate the output
        
        logger.info(f"Generating voiceover with Bark for: {prompt_id}")
        
        # Calculate total duration
        total_duration = sum(segment.get("duration", 5) for segment in script.get("segments", []))
        
        # Ensure output directory exists
        output_path = self.output_dir / f"{prompt_id}_voiceover.mp3"
        
        # In a real implementation, we would:
        # 1. Convert script to Bark's format
        # 2. Call Bark to generate audio with appropriate voice and style
        # 3. Add effects and process audio
        
        # Simulate processing time
        time.sleep(1)
        
        # Create a placeholder audio file (silent)
        self._create_silent_audio(output_path, total_duration)
        
        # Create timestamps for synchronization
        timestamps = self._create_timestamps(script, animations)
        
        return {
            "id": f"{prompt_id}_voiceover",
            "path": str(output_path),
            "engine": "bark_simulated",
            "prompt_id": prompt_id,
            "script": script,
            "timestamps": timestamps,
            "duration": total_duration,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def _generate_fallback(self, script, prompt_id, animations):
        """Generate a simple fallback voiceover when TTS engines are unavailable."""
        logger.info(f"Generating fallback voiceover for: {prompt_id}")
        
        # Calculate total duration
        total_duration = sum(segment.get("duration", 5) for segment in script.get("segments", []))
        
        # Ensure output directory exists
        output_path = self.output_dir / f"{prompt_id}_voiceover.mp3"
        
        # Create a placeholder audio file (silent)
        self._create_silent_audio(output_path, total_duration)
        
        # Create timestamps for synchronization
        timestamps = self._create_timestamps(script, animations)
        
        return {
            "id": f"{prompt_id}_voiceover",
            "path": str(output_path),
            "engine": "fallback",
            "prompt_id": prompt_id,
            "script": script,
            "timestamps": timestamps,
            "duration": total_duration,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def _create_silent_audio(self, output_path, duration):
        """Create a silent audio file for demonstration purposes."""
        # In a real implementation, this would use a library like pydub
        # For demonstration, we'll just create a placeholder file
        with open(output_path, 'w') as f:
            f.write(f"PLACEHOLDER AUDIO FILE - Duration: {duration} seconds")
    
    def _create_timestamps(self, script, animations):
        """Create timestamps for synchronizing audio with animations."""
        timestamps = {
            "intro": {"start": 0, "end": 0},
            "segments": [],
            "outro": {"start": 0, "end": 0}
        }
        
        current_time = 0
        
        # Calculate intro duration (roughly based on word count)
        intro_text = script.get("intro", "")
        intro_words = len(intro_text.split())
        intro_duration = intro_words / 2.5  # Assuming 2.5 words per second
        
        timestamps["intro"]["end"] = intro_duration
        current_time += intro_duration
        
        # Calculate segment timestamps
        for segment in script.get("segments", []):
            start_time = current_time
            duration = segment.get("duration", 5)
            end_time = start_time + duration
            
            timestamps["segments"].append({
                "scene_index": segment.get("scene_index", 0),
                "start": start_time,
                "end": end_time
            })
            
            current_time = end_time
        
        # Calculate outro timestamp
        outro_text = script.get("outro", "")
        if outro_text:
            outro_words = len(outro_text.split())
            outro_duration = outro_words / 2.5
            
            timestamps["outro"]["start"] = current_time
            timestamps["outro"]["end"] = current_time + outro_duration
        
        return timestamps
