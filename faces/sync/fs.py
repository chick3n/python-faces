"""
File System sync
"""

from pathlib import Path
import glob
import os
import logging
from .sync_interface import SyncInterface

class Fs(SyncInterface):
    """
    Fs
    """
    def __init__(self, path: str):
        self.path = path
        self.logger = logging.getLogger(__name__)

    def sync(self):
        """
        sync
        """
        if not os.path.exists(self.path):
            self.logger.info("folder path %s doesn't exist.", self.path)
            return

        photos = self.get_photos()

        return photos

    def get_photos(self):
        """
        get_photos
        """
        valid_images = [".jpg", ".jpeg", ".gif", ".tga", ".png", ".bmp"]
        photos = []
        path = os.path.join(self.path, "*")
        files = glob.glob(path)
        photos = [fi for fi in files
                  if os.path.isfile(fi) and os.path.splitext(fi)[1] in valid_images]

        return photos

    def move(self, src: str, dst: str) -> str:
        """
        move
        """
        src_path = Path(src)
        dst_path = Path(dst)

        if not src_path.exists():
            self.logger.error('Cannot save %s. File does not exist.',
                              src)
            return None

        if not dst_path.exists():
            self.logger.error('Cannot save %s to %s. Path does not exist.',
                              src_path.name,
                              dst)
            return None

        filename = src_path.name
        final_dst_path = str(dst_path.joinpath(filename))

        if str(src_path) == final_dst_path:
            return dst

        src_path.rename(final_dst_path)
        return final_dst_path
