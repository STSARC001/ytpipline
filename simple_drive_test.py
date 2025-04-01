"""
Simple Google Drive API Test Script
This script tests if the provided credentials.json file works correctly for Google Drive uploads.
"""
import os
import sys
import logging
import json
from datetime import datetime
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SimpleDriveTest")

# Google Drive API scopes
SCOPES = ['https://www.googleapis.com/auth/drive.file']

def create_test_file():
    """Create a simple test file"""
    # Create test directory if it doesn't exist
    os.makedirs('output/test', exist_ok=True)
    
    # Create a test text file
    test_file_path = 'output/test/drive_test.txt'
    with open(test_file_path, 'w') as f:
        f.write(f"This is a test file for Google Drive upload.\nCreated at: {datetime.now()}")
    
    return test_file_path

def authenticate_drive():
    """Authenticate with Google Drive API"""
    creds = None
    # The token.json file stores user's access and refresh tokens
    token_path = 'token.json'
    credentials_file = 'credentials.json'
    
    # Check if credentials file exists
    if not os.path.exists(credentials_file):
        logger.error(f"Credentials file not found: {credentials_file}")
        return None
    
    # Load credentials from token file if it exists
    if os.path.exists(token_path):
        try:
            creds = Credentials.from_authorized_user_info(
                json.load(open(token_path)), SCOPES)
        except Exception as e:
            logger.error(f"Error loading token file: {e}")
    
    # If credentials don't exist or are invalid, get new ones
    if not creds or not creds.valid:
        try:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    credentials_file, SCOPES)
                # Use a specific port that should match what's in your Google Cloud Console
                logger.info("Opening browser for authentication...")
                logger.info("Make sure http://localhost:8080 is an authorized redirect URI in your Google Cloud Console")
                try:
                    creds = flow.run_local_server(port=8080)
                except Exception as port_error:
                    logger.warning(f"Failed with port 8080: {port_error}. Trying alternative ports...")
                    try:
                        # Try with port 0 (let the OS choose)
                        logger.info("Trying with automatic port selection...")
                        creds = flow.run_local_server(port=0)
                    except Exception as backup_port_error:
                        logger.error(f"Authentication failed with automatic port: {backup_port_error}")
                        return None
            
            # Save the credentials for next run
            with open(token_path, 'w') as token:
                token.write(creds.to_json())
        except Exception as e:
            logger.error(f"Error during authentication: {e}")
            return None
    
    return creds

def upload_test_file(service, file_path):
    """Upload a test file to Google Drive"""
    file_name = os.path.basename(file_path)
    
    # File metadata
    file_metadata = {
        'name': file_name,
        'description': 'Test file uploaded by YouTube Automation Pipeline'
    }
    
    # Upload media
    media = MediaFileUpload(
        file_path,
        mimetype='text/plain',
        resumable=True
    )
    
    # Create the file on Google Drive
    file = service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id,webViewLink'
    ).execute()
    
    return file

def main():
    """Main function to test Google Drive upload"""
    logger.info("Starting Google Drive API test")
    
    # Create test file
    test_file_path = create_test_file()
    logger.info(f"Created test file: {test_file_path}")
    
    # Authenticate with Google Drive
    logger.info("Authenticating with Google Drive...")
    creds = authenticate_drive()
    
    if not creds:
        logger.error("❌ Authentication failed. Please check your credentials file.")
        return False
    
    # Create Drive service
    try:
        drive_service = build('drive', 'v3', credentials=creds)
        logger.info("Successfully created Drive service")
    except Exception as e:
        logger.error(f"❌ Failed to create Drive service: {e}")
        return False
    
    # Upload test file
    try:
        logger.info("Uploading test file to Google Drive...")
        file = upload_test_file(drive_service, test_file_path)
        
        logger.info(f"✅ File uploaded successfully!")
        logger.info(f"File ID: {file.get('id')}")
        logger.info(f"View URL: {file.get('webViewLink')}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Failed to upload file: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
