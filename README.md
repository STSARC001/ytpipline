# Advanced AI-Powered YouTube Automation Pipeline 🔥 ⚙️

A comprehensive multi-model pipeline that automates YouTube content creation using the latest AI technologies.

## 🌟 Features

### 1. Multi-Model Prompt Generation
- Uses Gemini 2.0 Flash and LLama 3 for creative, diverse, and multi-layered prompt generation
- Generates multiple prompts with divergent storytelling elements (e.g., different character arcs, alternate endings)
- Includes dynamic genre-based prompts (e.g., Sci-fi, Fantasy, Thriller)

### 2. Multi-Modal Story and Image Generation
- Uses Gemini 2.0 Flash for text-to-image generation
- Adds Stable Diffusion XL (SDXL) to generate highly detailed and realistic supporting images
- Incorporates ControlNet for precise image consistency, ensuring continuity across frames

### 3. Hyper-Realistic Image Animation
- Uses AnimateDiff and EBSynth for multi-frame animation, creating seamless character motion and facial expressions
- Implements frame interpolation with RIFE (Real-Time Intermediate Flow Estimation) for ultra-smooth transitions
- Adds depth mapping for parallax effects, making animations feel more dynamic and lifelike

### 4. AI Voiceover with Emotion and Dynamic Effects
- Uses XTTS and Bark AI for multi-speaker, emotion-rich voiceovers
- Applies Real-Time Audio Effects (reverb, pitch variation) for cinematic depth
- Auto-syncs voiceover with animation timing

### 5. Advanced Video Compilation with Dynamic Effects
- Uses FFmpeg + MoviePy for frame-perfect sequencing
- Adds AI-generated transitions and special effects (e.g., motion blur, dynamic lighting)
- Implements dynamic zoom and pan effects for enhanced visual storytelling

### 6. Google Drive Storage and Metadata Generation
- Automatically saves the video, images, and text in a unique Google Drive folder
- Generates SEO-optimized metadata:
  - AI-generated title, description, and tags
  - Includes relevant keywords and hashtags
  - Creates a thumbnail image using Gemini + SDXL for visually appealing clickbait

## 🚀 Getting Started in Google Colab

### Option 1: Use the Quick Setup Notebook

1. Create a new Google Colab notebook
2. Add and run the following code to set up the pipeline:

```python
# Clone the repository
!git clone https://github.com/yourusername/ytpipline.git
%cd ytpipline

# Install dependencies
!pip install -r requirements.txt

# Launch the web interface
!python webapp.py --share
```

### Option 2: Manual Setup in Colab

1. Create a new cell and run:
```python
# Install dependencies
!pip install google-generativeai>=0.3.0 transformers>=4.30.0 torch>=2.0.0 torchvision>=0.15.0 diffusers>=0.21.0 accelerate>=0.21.0 moviepy>=1.0.3 tensorflow>=2.12.0 opencv-python>=4.7.0 ftfy>=6.1.1 scipy>=1.10.1 tqdm>=4.65.0 pydrive>=1.3.1 gradio>=3.36.1 numpy>=1.24.3 Pillow>=9.5.0 matplotlib>=3.7.2 python-dotenv>=1.0.0 omegaconf>=2.3.0 einops>=0.6.1 pyngrok>=6.0.0 ffmpeg-python>=0.2.0 google-auth>=2.22.0 google-auth-oauthlib>=1.0.0 google-auth-httplib2>=0.1.0 google-api-python-client>=2.97.0 PyYAML>=6.0.1
```

2. Upload all the project files to Colab or copy them using magic commands:
```python
%%writefile youtube_automation_pipeline.py
# Paste the contents of youtube_automation_pipeline.py here
```

3. Run the web interface:
```python
!python webapp.py --share
```

## 💻 Local Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/ytpipline.git
cd ytpipline
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Configure API keys:
Create a `.env` file with your API keys:
```
GEMINI_API_KEY=your_gemini_api_key
LLM_API_KEY=your_llm_api_key
GOOGLE_DRIVE_API_KEY=your_google_drive_api_key
```

4. Run the web interface:
```
python webapp.py
```

## 📖 Usage

### Web Interface

The web interface provides an easy way to use the pipeline:

1. Enter a theme for your video (e.g., "space exploration", "underwater world")
2. Choose a visual style (e.g., "cyberpunk", "watercolor", "cinematic")
3. Set your desired video duration
4. Click "Generate YouTube Video"
5. Wait for the pipeline to complete

### Command Line

You can also run the pipeline from the command line:

```
python youtube_automation_pipeline.py --theme "space exploration" --style "cinematic" --duration 60
```

## ⚙️ Configuration

The pipeline is configured using `config.yaml`. Key settings include:

- `prompt_generator`: Settings for the creative prompt generation
- `image_generator`: Settings for image generation models 
- `animator`: Animation settings including FPS and duration
- `voiceover`: Voice generation settings
- `video_compiler`: Video compilation settings
- `drive_uploader`: Google Drive integration settings

## 🔄 Pipeline Flow

1. **Prompt Generation**: Creates multiple creative story concepts
2. **Image Generation**: Generates visual scenes based on prompts
3. **Animation**: Animates static images into dynamic sequences
4. **Voiceover**: Creates narration synchronized with animations
5. **Video Compilation**: Combines animations and audio into a final video
6. **Upload & Metadata**: Uploads to Google Drive with SEO metadata

## 📝 Notes for Google Colab Users

- The pipeline is designed to work optimally in Google Colab's environment with GPU acceleration
- When using in Colab, enable GPU runtime for faster processing
- The shared link feature allows you to access the web interface from any device
- Large models like Stable Diffusion XL may require high-memory Colab instances

## 🔑 API Keys

To use all features, you'll need:

- **Gemini API Key**: For prompt and image generation
- **Google Drive API Credentials**: For uploading content (optional)

## 📋 Dependencies

See `requirements.txt` for a full list of dependencies.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
