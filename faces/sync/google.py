"""
Google Photos Sync
"""

from pathlib import Path
import logging
import pickle
from urllib.request import urlretrieve as download
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from .sync_interface import SyncInterface

__SCOPES__ = 'https://www.googleapis.com/auth/photoslibrary'
__GTOKEN__ = './google_token.pickle'

class GooglePhotos(SyncInterface):
    """
    GooglePhotos
    """
    def __init__(self, sync_path, google_cred, album_id):
        self.logger = logging.getLogger(__name__)
        self.sync_path = sync_path
        self.google_cred = google_cred
        self.album_id = album_id

    def _request_service(self):
        creds = None
        if Path(__GTOKEN__).exists():
            with open(__GTOKEN__, 'rb') as token:
                creds = pickle.load(token)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(self.google_cred, __SCOPES__)
                creds = flow.run_local_server()

        with open(__GTOKEN__, 'wb') as token:
            pickle.dump(creds, token)

        return build('photoslibrary',
                     'v1',
                     credentials=creds,
                     cache_discovery=False)

    def sync(self):
        """
        sync
        """
        if not Path(self.google_cred).exists():
            self.logger.info('Google credentials file not found. File: %s.', self.google_cred)
            return []

        photos_service = None
        try:
            photos_service = self._request_service()
        except:
            self.logger.error('Failed to retrieve google photo service')
            return []

        self.logger.info('Downloading google album photos.')
        results = (photos_service
                   .mediaItems()
                   .search(body={'albumId': self.album_id, 'pageSize': 100})
                   .execute())
        items = results.get('mediaItems', [])
        photos = self._download_items(items)
        next_page_token = next_page_token = results.get('nextPageToken', None)

        while next_page_token is not None:
            results = (photos_service
                       .mediaItems()
                       .search(body={'albumId': self.album_id,
                                     'pageSize': 100, 'pageToken': next_page_token})
                       .execute())
            items = results.get('mediaItems', [])
            photos += self._download_items(items)
            next_page_token = results.get('nextPageToken', None)

        return photos

    def _download_items(self, items):
        """
        download items
        """
        if not Path(self.sync_path).exists():
            self.logger.error('Output path %s does not exists.', self.sync_path)
            return []

        photos = []
        for image_info in items:
            if 'image' in image_info['mimeType']:
                url = image_info['baseUrl'] + "=d"
                filename = image_info['filename']
                fullpath = Path(self.sync_path).joinpath(filename)
                if fullpath.exists():
                    self.logger.info('File %s already exists, skiping.', filename)
                    continue
                download(url, str(fullpath))
                photos.append(str(fullpath))
                self.logger.info('Download Item %s', filename)

        return photos

    def move(self, src: str, dst: str) -> str:
        """
        move
        """

        if not Path(src).exists():
            self.logger.error('Cannot save %s. File does not exist.',
                              src)
            return None

        photos_service = None
        try:
            photos_service = self._request_service()
        except:
            self.logger.error('Failed to retrieve google photo service')
            return None

        filename = Path(src).name
        media_file = MediaFileUpload(src, mimetype='video/mp4', resumable=True)

        self.logger.info('Uploading %s to google photos.', filename)
        request = (photos_service
                   .mediaItems()
                   .insert(body={'name': filename}, media_body=media_file))
        response = None

        while response is None:
            status, response = request.next_chunk()
            if status:
                self.logger.info('Uploaded %d%%.', int(status.progress()) * 100)

        return dst
