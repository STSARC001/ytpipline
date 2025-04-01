"""
Multi-Model Prompt Generation Component
Uses Gemini 2.0 Flash and LLama 3 for creative, diverse, and multi-layered prompt generation.
"""
import os
import logging
import json
import random
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger("YouTube-Pipeline.PromptGenerator")

class MultiModelPromptGenerator:
    """
    Generates creative prompts using multiple AI models:
    1. Gemini 2.0 Flash for primary prompts
    2. LLama 3 or other open-source LLMs for alternative viewpoints
    """
    
    def __init__(self, config):
        """Initialize the prompt generator with configuration."""
        self.config = config
        self.models = {}
        self.initialize_models()
        logger.info("MultiModelPromptGenerator initialized")
    
    def initialize_models(self):
        """Initialize AI models based on configuration."""
        # Initialize Gemini
        try:
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            gemini_model_name = self.config.get("gemini_model", "gemini-1.5-flash")
            self.models["gemini"] = genai.GenerativeModel(gemini_model_name)
            logger.info(f"Initialized Gemini model: {gemini_model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            self.models["gemini"] = None
        
        # Initialize a smaller, Colab-compatible LLM
        try:
            # Use a smaller model instead of Llama 3 that works in Colab
            # Options: "google/flan-t5-large", "bigscience/bloom-560m", or "EleutherAI/gpt-neo-125M"
            llm_model_name = self.config.get("llm_model", "google/flan-t5-large")
            logger.info(f"Attempting to initialize LLM: {llm_model_name}")
            
            # Only load model if config specifies to do so (might be resource intensive)
            if self.config.get("load_llm_model", False):
                try:
                    llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
                    llm_model = AutoModelForCausalLM.from_pretrained(
                        llm_model_name, 
                        device_map="auto",
                        load_in_8bit=self.config.get("load_in_8bit", True)
                    )
                    self.models["llm"] = pipeline(
                        "text-generation",
                        model=llm_model,
                        tokenizer=llm_tokenizer
                    )
                    logger.info(f"Initialized LLM model: {llm_model_name}")
                except Exception as e:
                    logger.warning(f"Failed to load local LLM model, trying API approach: {e}")
                    # Try to use the model through the Hugging Face API instead
                    try:
                        from transformers import pipeline
                        self.models["llm"] = pipeline(
                            "text-generation",
                            model=llm_model_name
                        )
                        logger.info(f"Initialized LLM via pipeline API: {llm_model_name}")
                    except Exception as e2:
                        logger.error(f"Failed to initialize LLM via API: {e2}")
                        self.models["llm"] = None
            else:
                # Use API endpoint if provided
                api_endpoint = self.config.get("llm_api_endpoint", "")
                if api_endpoint:
                    # This would be an implementation to use a remote API
                    logger.info(f"Using API endpoint for LLM: {api_endpoint}")
                    self.models["llm"] = {"endpoint": api_endpoint}
                else:
                    logger.warning("No local model loading or API endpoint specified for LLM")
                    self.models["llm"] = None
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            self.models["llm"] = None
    
    def generate(self, params=None):
        """
        Generate creative prompts using multiple models.
        
        Args:
            params: Dictionary containing parameters like theme, style, etc.
            
        Returns:
            List of prompt dictionaries with fields like title, description, etc.
        """
        if params is None:
            params = {}
        
        theme = params.get("theme", "")
        style = params.get("style", "")
        num_prompts = self.config.get("num_prompts", 3)
        
        logger.info(f"Generating {num_prompts} prompts with theme: {theme}, style: {style}")
        
        # Prepare base prompts for each model
        gemini_system_prompt = self._create_gemini_system_prompt(theme, style)
        llm_system_prompt = self._create_llm_system_prompt(theme, style)
        
        prompts = []
        
        # Generate primary prompt with Gemini
        if self.models["gemini"]:
            try:
                gemini_prompts = self._generate_with_gemini(gemini_system_prompt, num_prompts)
                prompts.extend(gemini_prompts)
                logger.info(f"Generated {len(gemini_prompts)} prompts using Gemini")
            except Exception as e:
                logger.error(f"Error generating Gemini prompts: {e}")
        
        # If Gemini failed or we need more prompts, use LLM
        if len(prompts) < num_prompts and self.models["llm"]:
            try:
                num_additional = num_prompts - len(prompts)
                llm_prompts = self._generate_with_llm(llm_system_prompt, num_additional)
                prompts.extend(llm_prompts)
                logger.info(f"Generated {len(llm_prompts)} prompts using LLM")
            except Exception as e:
                logger.error(f"Error generating LLM prompts: {e}")
        
        # Ensure we have the minimum number of prompts (fallback to template if needed)
        while len(prompts) < num_prompts:
            prompts.append(self._create_fallback_prompt(theme, style))
            logger.warning("Added fallback prompt due to model generation failures")
        
        # Add additional metadata to each prompt
        for i, prompt in enumerate(prompts):
            if "id" not in prompt:
                prompt["id"] = f"prompt_{i+1}"
            if "tags" not in prompt:
                prompt["tags"] = self._generate_tags(prompt)
            if "genre" not in prompt:
                prompt["genre"] = self._assign_genre(prompt)
        
        return prompts
    
    def _create_gemini_system_prompt(self, theme, style):
        """Create system prompt for Gemini."""
        prompt = """
        Generate a creative and engaging story concept for a YouTube short video.
        
        Your task is to create:
        1. A captivating title (10 words or less)
        2. A detailed description (3-5 paragraphs)
        3. A list of 5-10 key scenes for visualization
        4. Main characters with brief descriptions
        5. Suggested style guidance for visuals
        
        Make the story emotionally engaging with a clear narrative arc.
        """
        
        if theme:
            prompt += f"\nIncorporate the theme: {theme}"
        if style:
            prompt += f"\nUse the visual style: {style}"
        
        return prompt
    
    def _create_llm_system_prompt(self, theme, style):
        """Create system prompt for LLama or other LLMs."""
        prompt = """
        <SYSTEM>
        You are a creative AI tasked with generating an engaging story concept for a YouTube short video.
        </SYSTEM>
        
        <USER>
        Generate a creative and engaging story concept with:
        1. A captivating title (10 words or less)
        2. A detailed description (3-5 paragraphs)
        3. A list of 5-10 key scenes for visualization
        4. Main characters with brief descriptions
        5. Suggested style guidance for visuals
        
        Format your response as a JSON object with keys: title, description, scenes, characters, and style_guide.
        """
        
        if theme:
            prompt += f"\nIncorporate the theme: {theme}"
        if style:
            prompt += f"\nUse the visual style: {style}"
        
        prompt += "\n</USER>"
        return prompt
    
    def _generate_with_gemini(self, system_prompt, num_prompts):
        """Generate prompts using Gemini model."""
        prompts = []
        
        for i in range(num_prompts):
            # Add some randomness to each generation
            variant_prompt = system_prompt
            if i > 0:
                variant_types = ["alternative perspective", "different ending", 
                                "opposite emotion", "different time period"]
                variant = random.choice(variant_types)
                variant_prompt += f"\n\nCreate a variant with a {variant}."
            
            response = self.models["gemini"].generate_content(variant_prompt)
            
            try:
                # Try to parse as JSON first
                content = response.text
                result = self._extract_structured_data(content)
                
                prompts.append({
                    "id": f"gemini_{i+1}",
                    "title": result.get("title", f"Gemini Story {i+1}"),
                    "description": result.get("description", ""),
                    "scenes": result.get("scenes", []),
                    "characters": result.get("characters", []),
                    "style_guide": result.get("style_guide", ""),
                    "source": "gemini"
                })
            except Exception as e:
                logger.warning(f"Failed to parse Gemini output as structured data: {e}")
                
                # Fallback to parsing the text more liberally
                content = response.text
                title = self._extract_title(content) or f"Gemini Story {i+1}"
                description = self._extract_paragraphs(content)
                
                prompts.append({
                    "id": f"gemini_{i+1}",
                    "title": title,
                    "description": description,
                    "scenes": self._extract_scenes(content),
                    "characters": self._extract_characters(content),
                    "style_guide": self._extract_style_guide(content),
                    "source": "gemini"
                })
        
        return prompts
    
    def _generate_with_llm(self, system_prompt, num_prompts):
        """Generate prompts using LLama or other LLM."""
        prompts = []
        
        for i in range(num_prompts):
            # Add some variability to each prompt
            variant_prompt = system_prompt
            if i > 0:
                genres = ["sci-fi", "fantasy", "thriller", "romance", "comedy", "drama"]
                genre = random.choice(genres)
                variant_prompt = variant_prompt.replace("</USER>", 
                                                    f"\nMake this a {genre} story.\n</USER>")
            
            if callable(self.models["llm"]):
                # API-based access
                response = self.models["llm"](variant_prompt)
                content = response.get("generated_text", "")
            else:
                # Direct model access
                response = self.models["llm"](
                    variant_prompt,
                    max_length=2048,
                    num_return_sequences=1
                )
                content = response[0]["generated_text"]
            
            try:
                # Try to parse as JSON first
                result = self._extract_structured_data(content)
                
                prompts.append({
                    "id": f"llm_{i+1}",
                    "title": result.get("title", f"LLM Story {i+1}"),
                    "description": result.get("description", ""),
                    "scenes": result.get("scenes", []),
                    "characters": result.get("characters", []),
                    "style_guide": result.get("style_guide", ""),
                    "source": "llm"
                })
            except Exception as e:
                logger.warning(f"Failed to parse LLM output as JSON: {e}")
                
                # Fallback to parsing the text more liberally
                title = self._extract_title(content) or f"LLM Story {i+1}"
                description = self._extract_paragraphs(content)
                
                prompts.append({
                    "id": f"llm_{i+1}",
                    "title": title,
                    "description": description,
                    "scenes": self._extract_scenes(content),
                    "characters": self._extract_characters(content),
                    "style_guide": self._extract_style_guide(content),
                    "source": "llm"
                })
        
        return prompts
    
    def _create_fallback_prompt(self, theme, style):
        """Create a fallback prompt when model generation fails."""
        title = f"Adventure in the {theme or 'Unknown'}"
        description = f"""
        A captivating journey through a world of wonder and excitement.
        Our protagonist discovers new challenges and overcomes obstacles.
        The story reaches its climax with an unexpected revelation.
        """
        
        scenes = [
            "Opening scene introducing the main character",
            "Discovery of a strange artifact or clue",
            "Meeting a mysterious guide or companion",
            "Confrontation with an obstacle or adversary",
            "Climactic resolution and revelation"
        ]
        
        characters = [
            {"name": "The Protagonist", "description": "Curious and determined"},
            {"name": "The Guide", "description": "Wise and enigmatic"}
        ]
        
        style_guide = style or "Vibrant colors with dramatic lighting"
        
        return {
            "id": "fallback_1",
            "title": title,
            "description": description,
            "scenes": scenes,
            "characters": characters,
            "style_guide": style_guide,
            "source": "fallback"
        }
    
    def _extract_structured_data(self, text):
        """Attempt to extract JSON from the model output."""
        try:
            # Look for JSON blocks in the text
            start_idx = text.find('{')
            end_idx = text.rfind('}')
            
            if start_idx != -1 and end_idx != -1:
                json_str = text[start_idx:end_idx+1]
                return json.loads(json_str)
            else:
                raise ValueError("No JSON block found in text")
        except Exception as e:
            logger.debug(f"JSON extraction error: {e}")
            raise
    
    def _extract_title(self, text):
        """Extract title from unstructured text."""
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            # Look for lines that seem like titles (short, no punctuation at end)
            if 3 <= len(line) <= 50 and not line.endswith(('.', ':', '?', '!')):
                return line
            # Look for explicit title markers
            if line.lower().startswith(('title:', '# ', 'title -')):
                return line.split(':', 1)[1].strip() if ':' in line else line[2:].strip()
        return None
    
    def _extract_paragraphs(self, text):
        """Extract description paragraphs from unstructured text."""
        lines = text.split('\n')
        paragraphs = []
        current_paragraph = []
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
            elif not line.startswith(('#', '1.', '2.', '- ', '*', 'Scene', 'Character')):
                current_paragraph.append(line)
        
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
        
        # Join paragraphs with newlines and return
        return '\n\n'.join(paragraphs)
    
    def _extract_scenes(self, text):
        """Extract scene descriptions from unstructured text."""
        scenes = []
        in_scenes_section = False
        
        lines = text.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Check if we're entering a scenes section
            if 'scene' in line.lower() and (':' in line or line.endswith(':')):
                in_scenes_section = True
                continue
            
            # If we're in a scenes section, look for numbered or bulleted items
            if in_scenes_section:
                if line.startswith(('- ', '* ', '• ')):
                    scenes.append(line[2:].strip())
                elif line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')):
                    scene_text = line.split('.', 1)[1].strip()
                    scenes.append(scene_text)
                # Exit scenes section if we hit another heading or empty line
                elif not line or line.endswith(':') or (i < len(lines)-1 and not lines[i+1].strip()):
                    in_scenes_section = False
        
        # If no scenes were found using the structured approach, try a more liberal extraction
        if not scenes:
            text_lower = text.lower()
            if 'scene' in text_lower:
                # Find paragraphs that mention scene or scenes
                for paragraph in text.split('\n\n'):
                    if 'scene' in paragraph.lower():
                        scene_lines = paragraph.split('\n')
                        for line in scene_lines:
                            if line.strip() and not line.endswith(':'):
                                scenes.append(line.strip())
        
        # If we still have no scenes, extract some meaningful sentences as scenes
        if not scenes:
            sentences = [s.strip() for s in text.replace('\n', ' ').split('.') if s.strip()]
            scenes = sentences[:min(5, len(sentences))]
        
        return scenes[:10]  # Return at most 10 scenes
    
    def _extract_characters(self, text):
        """Extract character descriptions from unstructured text."""
        characters = []
        in_character_section = False
        current_character = {}
        
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            
            # Check if we're entering a characters section
            if 'character' in line.lower() and (':' in line or line.endswith(':')):
                in_character_section = True
                continue
            
            # If we're in a characters section, look for character entries
            if in_character_section:
                if line.startswith(('- ', '* ', '• ')):
                    line = line[2:].strip()
                    if ':' in line:
                        name, desc = line.split(':', 1)
                        characters.append({
                            "name": name.strip(),
                            "description": desc.strip()
                        })
                    else:
                        if current_character and 'name' in current_character:
                            characters.append(current_character)
                        current_character = {"name": line, "description": ""}
                
                # Exit character section if we hit another heading
                elif not line or line.endswith(':'):
                    if current_character and 'name' in current_character:
                        characters.append(current_character)
                    current_character = {}
                    in_character_section = False
        
        # Add final character if needed
        if current_character and 'name' in current_character:
            characters.append(current_character)
        
        # If no characters were found, try to extract them from the text
        if not characters:
            text_lower = text.lower()
            if 'character' in text_lower:
                # Try to identify character mentions
                for paragraph in text.split('\n\n'):
                    if 'character' in paragraph.lower():
                        lines = paragraph.split('\n')
                        for line in lines[1:]:  # Skip the heading
                            if ':' in line:
                                name, desc = line.split(':', 1)
                                characters.append({
                                    "name": name.strip(),
                                    "description": desc.strip()
                                })
        
        # If still no characters, create generic ones based on nouns in the text
        if not characters:
            characters = [
                {"name": "Protagonist", "description": "The main character of the story"},
                {"name": "Companion", "description": "A supporting character who aids the protagonist"}
            ]
        
        return characters
    
    def _extract_style_guide(self, text):
        """Extract style guidance from unstructured text."""
        style_guide = ""
        in_style_section = False
        
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            
            # Check if we're entering a style section
            if any(s in line.lower() for s in ['style guide', 'visual style', 'art style']) and (':' in line or line.endswith(':')):
                in_style_section = True
                if ':' in line:
                    style_guide = line.split(':', 1)[1].strip()
                continue
            
            # If we're in a style section, collect text
            if in_style_section:
                if line and not line.endswith(':'):
                    style_guide += " " + line
                # Exit style section if we hit another heading
                elif not line or line.endswith(':'):
                    in_style_section = False
        
        # If no style guide was found, look for style-related sentences
        if not style_guide:
            text_lower = text.lower()
            for term in ['style', 'visual', 'aesthetic', 'color', 'atmosphere']:
                if term in text_lower:
                    # Find sentences containing the term
                    sentences = text.replace('\n', ' ').split('.')
                    for sentence in sentences:
                        if term in sentence.lower():
                            style_guide = sentence.strip()
                            break
                    if style_guide:
                        break
        
        # If still no style guide, create a generic one
        if not style_guide:
            style_guide = "Vibrant colors with dynamic composition and cinematic lighting"
        
        return style_guide
    
    def _generate_tags(self, prompt):
        """Generate relevant tags based on the prompt content."""
        tags = []
        title = prompt.get("title", "")
        description = prompt.get("description", "")
        
        # Extract meaningful words from title and description
        content = title + " " + description
        words = content.replace(",", " ").replace(".", " ").split()
        
        # Filter out common words and keep meaningful ones
        stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with"}
        for word in words:
            word = word.lower()
            if len(word) > 3 and word not in stopwords and word not in tags:
                tags.append(word)
        
        # Add genre as a tag
        genre = prompt.get("genre", self._assign_genre(prompt))
        if genre and genre not in tags:
            tags.append(genre.lower())
        
        # Add source as a tag
        source = prompt.get("source", "")
        if source and source not in tags:
            tags.append(source)
        
        return tags[:20]  # Limit to 20 tags
    
    def _assign_genre(self, prompt):
        """Assign a genre based on prompt content."""
        genres = [
            "sci-fi", "fantasy", "thriller", "mystery", "romance", 
            "comedy", "drama", "action", "adventure", "horror"
        ]
        
        # Check title and description for genre keywords
        title = prompt.get("title", "").lower()
        description = prompt.get("description", "").lower()
        
        for genre in genres:
            if genre in title or genre in description:
                return genre
        
        # Use a simple keyword matching approach
        keywords = {
            "sci-fi": ["space", "robot", "alien", "future", "technology"],
            "fantasy": ["magic", "dragon", "wizard", "spell", "kingdom"],
            "thriller": ["danger", "suspense", "mystery", "threat", "chase"],
            "mystery": ["detective", "solve", "clue", "puzzle", "unknown"],
            "romance": ["love", "heart", "relationship", "passion", "emotion"],
            "comedy": ["funny", "laugh", "humor", "joke", "hilarious"],
            "drama": ["emotional", "conflict", "struggle", "intense", "relationship"],
            "action": ["fight", "battle", "mission", "explosive", "combat"],
            "adventure": ["journey", "quest", "explore", "discovery", "expedition"],
            "horror": ["fear", "terrify", "monster", "dark", "nightmare"]
        }
        
        content = title + " " + description
        scores = {genre: 0 for genre in genres}
        
        for genre, words in keywords.items():
            for word in words:
                if word in content:
                    scores[genre] += 1
        
        # Return highest-scoring genre, or "adventure" as fallback
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
        else:
            return "adventure"
