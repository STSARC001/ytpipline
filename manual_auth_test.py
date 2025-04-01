"""
Manual Google Drive API Test Script
This script uses an alternative authentication method for Google Drive API.
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
logger = logging.getLogger("ManualAuthTest")

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

def authorize_with_manual_code():
    """Authenticate with Google Drive API using manual code entry"""
    creds = None
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
            if creds and creds.valid:
                logger.info("Loaded existing valid credentials")
                return creds
        except Exception as e:
            logger.error(f"Error loading token file: {e}")
    
    # If no valid credentials, do manual auth
    try:
        flow = InstalledAppFlow.from_client_secrets_file(
            credentials_file, SCOPES)
        
        # Use manual authorization
        auth_url = flow.authorization_url(prompt='consent')[0]
        
        print("\n\n")
        print("=" * 80)
        print("Please go to this URL to authorize the application:")
        print(f"\n{auth_url}\n")
        print("After authorization, you will be redirected to a page that says 'The authentication flow has completed.'")
        print("The URL of that page will contain a 'code' parameter.")
        print("Copy the entire URL from your browser and paste it below.")
        print("=" * 80)
        
        # Get authorization code from user
        auth_code = input("\nEnter the full URL you were redirected to: ")
        
        # Extract code from URL if needed
        if "?" in auth_code and "code=" in auth_code:
            try:
                auth_code = auth_code.split("code=")[1].split("&")[0]
            except:
                # If parsing fails, use the input as is
                pass
        
        # Exchange auth code for credentials
        flow.fetch_token(code=auth_code)
        creds = flow.credentials
        
        # Save the credentials for next run
        with open(token_path, 'w') as token:
            token.write(creds.to_json())
            logger.info(f"Saved credentials to {token_path}")
        
        return creds
        
    except Exception as e:
        logger.error(f"Error during authentication: {e}")
        return None

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
    logger.info("Starting Google Drive API test with manual authentication")
    
    # Create test file
    test_file_path = create_test_file()
    logger.info(f"Created test file: {test_file_path}")
    
    # Authenticate with Google Drive
    logger.info("Authenticating with Google Drive...")
    creds = authorize_with_manual_code()
    
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
