"""
Google Drive Storage and Metadata Generation Component
Automatically saves video, images, and text in a unique Google Drive folder with SEO metadata.
"""
import os
import logging
import time
from pathlib import Path
import json
import io
from dotenv import load_dotenv
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseUpload

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger("YouTube-Pipeline.DriveUploader")

# Define the scopes needed for Google Drive access
SCOPES = ['https://www.googleapis.com/auth/drive.file']

class GoogleDriveUploader:
    """
    Handles video and asset upload to Google Drive with SEO metadata:
    1. Creates dedicated folders for each generation
    2. Uploads video, images, and text assets
    3. Generates SEO-optimized metadata
    4. Creates clickbait thumbnails
    """
    
    def __init__(self, config):
        """Initialize the Google Drive uploader with configuration."""
        self.config = config
        self.output_dir = Path(config.get("output_dir", "output/metadata"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.credentials_file = config.get("credentials_file", "credentials.json")
        self.service = None
        self.initialize_drive_api()
        logger.info("GoogleDriveUploader initialized")
    
    def initialize_drive_api(self):
        """Initialize connection to Google Drive API."""
        # Check if credentials file exists
        if not os.path.exists(self.credentials_file):
            logger.warning(f"Google Drive credentials file not found: {self.credentials_file}")
            self.drive_connected = False
            return
        
        try:
            creds = None
            # The file token.json stores the user's access and refresh tokens
            token_path = os.path.join(os.path.dirname(self.credentials_file), 'token.json')
            
            # If token.json exists, load credentials from it
            if os.path.exists(token_path):
                creds = Credentials.from_authorized_user_info(
                    json.load(open(token_path)), SCOPES)
            
            # If there are no valid credentials, let the user log in
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.credentials_file, SCOPES)
                    creds = flow.run_local_server(port=0)
                
                # Save the credentials for the next run
                with open(token_path, 'w') as token:
                    token.write(creds.to_json())
            
            # Build the Drive service
            self.service = build('drive', 'v3', credentials=creds)
            self.drive_connected = True
            logger.info("Google Drive API initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Google Drive API: {e}")
            self.service = None
            self.drive_connected = False
            logger.info("Google Drive uploads will be simulated")
    
    def upload(self, video_path, thumbnail_path, metadata):
        """
        Upload video, thumbnail, and metadata to Google Drive.
        
        Args:
            video_path: Path to the final video file
            thumbnail_path: Path to the thumbnail image
            metadata: Dictionary containing video metadata (title, description, tags)
            
        Returns:
            Dictionary with upload information
        """
        logger.info(f"Uploading to Google Drive: {metadata.get('title', 'Untitled Video')}")
        
        # Create a unique folder name for this upload
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        folder_name = f"AI_Video_{timestamp}"
        
        # Check if files exist
        if not os.path.exists(video_path):
            logger.warning(f"Video file not found: {video_path}")
        
        if not os.path.exists(thumbnail_path):
            logger.warning(f"Thumbnail file not found: {thumbnail_path}")
        
        # Generate enhanced SEO metadata
        enhanced_metadata = self._enhance_metadata(metadata)
        
        # Save enhanced metadata locally
        metadata_path = self.output_dir / f"metadata_{timestamp}.json"
        with open(metadata_path, 'w') as f:
            json.dump(enhanced_metadata, f, indent=2)
        
        folder_id = None
        video_id = None
        thumbnail_id = None
        metadata_id = None
        drive_url = None
        
        if self.drive_connected and self.service:
            try:
                # 1. Create a folder on Google Drive
                folder_id = self._create_drive_folder(folder_name)
                
                # 2. Upload the video file
                if os.path.exists(video_path):
                    video_id = self._upload_file(
                        video_path, 
                        os.path.basename(video_path),
                        "video/mp4",
                        folder_id,
                        enhanced_metadata.get("description", "")
                    )
                
                # 3. Upload the thumbnail
                if os.path.exists(thumbnail_path):
                    thumbnail_id = self._upload_file(
                        thumbnail_path,
                        os.path.basename(thumbnail_path),
                        "image/jpeg",
                        folder_id,
                        "Thumbnail for " + enhanced_metadata.get("title", "video")
                    )
                
                # 4. Upload the metadata file
                if os.path.exists(metadata_path):
                    metadata_id = self._upload_file(
                        str(metadata_path),
                        os.path.basename(metadata_path),
                        "application/json",
                        folder_id,
                        "Metadata for " + enhanced_metadata.get("title", "video")
                    )
                
                # Get the URL to the folder
                if folder_id:
                    drive_url = f"https://drive.google.com/drive/folders/{folder_id}"
                    logger.info(f"Uploaded content to Google Drive folder: {drive_url}")
                
                upload_status = "uploaded"
            except Exception as e:
                logger.error(f"Error uploading to Google Drive: {e}")
                upload_status = "error"
                drive_url = None
        else:
            # Simulate the upload for demo purposes
            logger.info("Simulating Google Drive upload (no credentials or connection)")
            upload_status = "simulated"
            drive_url = f"https://drive.google.com/simulated/{folder_name}"
        
        upload_info = {
            "folder_name": folder_name,
            "folder_id": folder_id,
            "video_path": video_path,
            "video_id": video_id,
            "thumbnail_path": thumbnail_path,
            "thumbnail_id": thumbnail_id,
            "metadata_path": str(metadata_path),
            "metadata_id": metadata_id,
            "uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": upload_status,
            "url": drive_url,
            "title": enhanced_metadata.get("title"),
            "description": enhanced_metadata.get("description")[:100] + "..." if len(enhanced_metadata.get("description", "")) > 100 else enhanced_metadata.get("description"),
            "tags": enhanced_metadata.get("tags")
        }
        
        logger.info(f"Video upload {upload_status}: {upload_info.get('title')}")
        return upload_info
    
    def _create_drive_folder(self, folder_name):
        """Create a folder on Google Drive."""
        if not self.service:
            return None
            
        try:
            file_metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder'
            }
            
            folder = self.service.files().create(
                body=file_metadata,
                fields='id'
            ).execute()
            
            logger.info(f"Created folder: {folder_name} with ID: {folder.get('id')}")
            return folder.get('id')
        except Exception as e:
            logger.error(f"Error creating folder: {e}")
            return None
    
    def _upload_file(self, file_path, file_name, mime_type, parent_folder_id=None, description=None):
        """Upload a file to Google Drive."""
        if not self.service:
            return None
            
        try:
            file_metadata = {
                'name': file_name,
                'description': description or f"Uploaded by YouTube Automation Pipeline on {time.strftime('%Y-%m-%d')}"
            }
            
            # Add to folder if specified
            if parent_folder_id:
                file_metadata['parents'] = [parent_folder_id]
            
            # Upload the file
            media = MediaFileUpload(
                file_path,
                mimetype=mime_type,
                resumable=True
            )
            
            file = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()
            
            logger.info(f"Uploaded file: {file_name} with ID: {file.get('id')}")
            return file.get('id')
        except Exception as e:
            logger.error(f"Error uploading file: {e}")
            return None
    
    def _enhance_metadata(self, metadata):
        """Enhance metadata with SEO optimization."""
        enhanced = metadata.copy()
        
        # Ensure title has attention-grabbing elements
        title = enhanced.get("title", "AI Generated Video")
        if not any(term in title for term in ["Amazing", "Incredible", "Stunning", "Mind-Blowing", "Unbelievable"]):
            enhanced["title"] = self._enhance_title(title)
        
        # Enhance description with keywords
        description = enhanced.get("description", "")
        enhanced["description"] = self._enhance_description(description)
        
        # Expand tags with trending and relevant terms
        tags = enhanced.get("tags", [])
        enhanced["tags"] = self._enhance_tags(tags)
        
        # Add YouTube-specific metadata
        enhanced["youtube_metadata"] = {
            "category": self._determine_category(description, tags),
            "privacy_status": "private",  # Default to private for safety
            "made_for_kids": False,
            "language": "en",
            "license": "youtube"
        }
        
        return enhanced
    
    def _enhance_title(self, title):
        """Add attention-grabbing elements to the title."""
        attention_grabbers = [
            "Stunning", "Amazing", "Incredible", "Mind-Blowing", 
            "You Won't Believe", "Must See", "Unbelievable",
            "AI Creates", "AI-Generated"
        ]
        
        # If title is already long, don't make it longer
        if len(title) > 50:
            return title
        
        # Add an attention grabber
        grabber = attention_grabbers[hash(title) % len(attention_grabbers)]
        
        # Various title formats
        title_formats = [
            f"{grabber}: {title}",
            f"{title} - {grabber} AI Creation",
            f"{grabber} {title} [AI-Generated]"
        ]
        
        enhanced_title = title_formats[hash(title) % len(title_formats)]
        
        # Ensure title isn't too long for YouTube (100 char limit)
        if len(enhanced_title) > 100:
            enhanced_title = enhanced_title[:97] + "..."
        
        return enhanced_title
    
    def _enhance_description(self, description):
        """Enhance the description with keywords and calls to action."""
        # If description is empty, create a basic one
        if not description:
            description = "An amazing AI-generated video created with cutting-edge technology."
        
        # Add keywords and calls to action
        enhancements = [
            "\n\nThis video was created entirely by AI using advanced machine learning models.",
            "\n\nLike and subscribe for more AI-generated content!",
            "\n\n#AIart #AIvideo #MachineLearning #ArtificialIntelligence",
            "\n\nCheck out my channel for more amazing AI creations!"
        ]
        
        # Add enhancements
        for enhancement in enhancements:
            if enhancement not in description:
                description += enhancement
        
        return description
    
    def _enhance_tags(self, tags):
        """Enhance tags with trending and relevant terms."""
        # Always add these essential tags
        essential_tags = [
            "AI", "artificial intelligence", "ai art", "ai video", 
            "ai animation", "ai generated", "machine learning",
            "generative ai", "text to video", "neural network"
        ]
        
        # Add additional trending tags based on content
        trending_tags = [
            "trending", "viral", "amazing", "beautiful", 
            "satisfying", "oddly satisfying", "mindblowing"
        ]
        
        # Combine all tags
        all_tags = list(tags)
        
        # Add essential tags if not already present
        for tag in essential_tags:
            if tag.lower() not in [t.lower() for t in all_tags]:
                all_tags.append(tag)
        
        # Add some trending tags
        num_trending = min(3, len(trending_tags))
        selected_trending = trending_tags[:num_trending]
        for tag in selected_trending:
            if tag.lower() not in [t.lower() for t in all_tags]:
                all_tags.append(tag)
        
        # Limit to 500 characters total (YouTube limit for all tags combined)
        total_length = sum(len(tag) for tag in all_tags) + len(all_tags) - 1  # account for commas
        if total_length > 500:
            # Remove tags until under limit
            while total_length > 500 and len(all_tags) > len(essential_tags):
                # Remove the longest non-essential tag
                non_essential = [t for t in all_tags if t.lower() not in [e.lower() for e in essential_tags]]
                if non_essential:
                    longest = max(non_essential, key=len)
                    all_tags.remove(longest)
                    total_length = sum(len(tag) for tag in all_tags) + len(all_tags) - 1
                else:
                    break
        
        return all_tags
    
    def _determine_category(self, description, tags):
        """Determine the best YouTube category for the content."""
        # YouTube category IDs
        categories = {
            "Film & Animation": 1,
            "Autos & Vehicles": 2,
            "Music": 10,
            "Pets & Animals": 15,
            "Sports": 17,
            "Travel & Events": 19,
            "Gaming": 20,
            "People & Blogs": 22,
            "Comedy": 23,
            "Entertainment": 24,
            "News & Politics": 25,
            "Howto & Style": 26,
            "Education": 27,
            "Science & Technology": 28
        }
        
        # Score each category based on tags and description
        scores = {category: 0 for category in categories}
        
        # Category keywords
        keywords = {
            "Film & Animation": ["animation", "film", "movie", "cinematic", "3d", "character"],
            "Music": ["music", "song", "audio", "sound", "beat", "melody"],
            "Gaming": ["game", "gaming", "player", "level", "character", "fantasy"],
            "Entertainment": ["entertainment", "funny", "amazing", "cool", "awesome"],
            "Science & Technology": ["tech", "ai", "science", "technology", "artificial intelligence", "neural"],
            "Education": ["education", "learn", "knowledge", "tutorial", "how to", "explain"],
            "Comedy": ["comedy", "funny", "laugh", "humor", "joke", "hilarious"]
        }
        
        # Score based on tags
        for tag in tags:
            tag_lower = tag.lower()
            for category, terms in keywords.items():
                if any(term in tag_lower for term in terms):
                    scores[category] += 1
        
        # Score based on description
        description_lower = description.lower()
        for category, terms in keywords.items():
            for term in terms:
                if term in description_lower:
                    scores[category] += 0.5
        
        # Always give some points to general categories
        scores["Entertainment"] += 1
        scores["Science & Technology"] += 2  # Bias toward tech since it's AI-generated
        
        # Get category with highest score
        if max(scores.values()) > 0:
            top_category = max(scores, key=scores.get)
            return categories[top_category]
        else:
            # Default to Entertainment if no clear category
            return categories["Entertainment"]
