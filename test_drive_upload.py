"""
Test script for Google Drive upload functionality
This script creates a test file and uploads it to Google Drive to verify API integration.
"""
import os
import sys
import logging
import time
from pathlib import Path
import json
from datetime import datetime

# Set up logging to console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("DriveUploadTest")

# Add the project directory to Python path to import components
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the GoogleDriveUploader
from components.drive_uploader import GoogleDriveUploader

def create_test_files():
    """Create test files for upload"""
    # Create test output directory
    output_dir = Path("output/test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a test text file
    test_text_path = output_dir / "test_text.txt"
    with open(test_text_path, "w") as f:
        f.write(f"This is a test file for Google Drive upload.\nCreated at: {datetime.now()}")
    
    # Create a test JSON file (simulating metadata)
    test_json_path = output_dir / "test_metadata.json"
    test_metadata = {
        "title": "Test Upload",
        "description": "This is a test upload to verify Google Drive integration",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test": True
    }
    with open(test_json_path, "w") as f:
        json.dump(test_metadata, f, indent=2)
    
    # Create a simple image file (small colored rectangle)
    try:
        from PIL import Image
        test_image_path = output_dir / "test_image.jpg"
        img = Image.new('RGB', (100, 100), color = (73, 109, 137))
        img.save(test_image_path)
        logger.info("Created test image file")
    except ImportError:
        logger.warning("PIL not available, skipping test image creation")
        test_image_path = None
    
    return test_text_path, test_json_path, test_image_path, test_metadata

def test_drive_upload():
    """Test the Google Drive upload functionality"""
    logger.info("Starting Google Drive upload test")
    
    # Check for credentials file
    credentials_file = "credentials.json"
    if not os.path.exists(credentials_file):
        logger.error(f"Credentials file not found: {credentials_file}")
        return False
    
    # Create test files
    logger.info("Creating test files")
    text_file, json_file, image_file, metadata = create_test_files()
    
    # Initialize the uploader with default configuration
    logger.info("Initializing GoogleDriveUploader")
    config = {
        "credentials_file": credentials_file,
        "output_dir": "output/test"
    }
    uploader = GoogleDriveUploader(config)
    
    # Attempt to upload the files
    logger.info("Uploading test files to Google Drive")
    result = uploader.upload(
        video_path=str(text_file),  # Using text file as "video" for testing
        thumbnail_path=str(image_file) if image_file else str(text_file),
        metadata=metadata
    )
    
    # Check the result
    if result.get("status") == "uploaded" and result.get("url"):
        logger.info("✅ Test PASSED: Files uploaded successfully to Google Drive")
        logger.info(f"Folder URL: {result.get('url')}")
        return True
    elif result.get("status") == "error":
        logger.error("❌ Test FAILED: Error occurred during upload")
        return False
    else:
        logger.warning("⚠️ Test PARTIAL: Upload was simulated but not actually performed")
        logger.info("This could mean the credentials are invalid or Google Drive API is not properly initialized")
        return False

if __name__ == "__main__":
    success = test_drive_upload()
    sys.exit(0 if success else 1)
