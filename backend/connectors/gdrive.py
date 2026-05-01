import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

class GoogleDriveConnector:
    def __init__(self, credentials_path: str):
        """
        Initialize the Google Drive connector with service account credentials.
        :param credentials_path: Path to the service account JSON key file.
        """
        self.scopes = ['https://www.googleapis.com/auth/drive.readonly']
        if not os.path.exists(credentials_path):
            raise FileNotFoundError(f"Credentials file not found at {credentials_path}")
        
        self.creds = service_account.Credentials.from_service_account_file(
            credentials_path, scopes=self.scopes
        )
        self.service = build('drive', 'v3', credentials=self.creds)

    def list_files(self, folder_id: str = None):
        """
        List files in Google Drive. Optionally filter by folder ID.
        Filters for PDF, Google Docs, and TXT files.
        """
        try:
            # Query for specific mimeTypes
            query = (
                "mimeType = 'application/pdf' or "
                "mimeType = 'application/vnd.google-apps.document' or "
                "mimeType = 'text/plain'"
            )
            
            if folder_id:
                query = f"('{folder_id}' in parents) and ({query})"

            results = self.service.files().list(
                q=query,
                pageSize=100,
                fields="nextPageToken, files(id, name, mimeType)"
            ).execute()
            
            return results.get('files', [])

        except HttpError as error:
            print(f"An error occurred: {error}")
            return []
