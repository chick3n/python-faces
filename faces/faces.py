"""
faces
"""

import logging
from datetime import datetime, timedelta
import os
from pathlib import Path
from PIL import Image
import ffmpeg
from . import local_lazy_config as _confuse, sync, photo as _photo

__version__ = '0.1.0'

class Faces():
    """
    faces
    """
    def __init__(self):
        self.config = _confuse.LocalLazyConfig('faces')
        self.setup_logging()
        self.photo_synchronizer = None

        border = {
            'width':self.config['photos']['border']['width'].get(int),
            'height':self.config['photos']['border']['height'].get(int),
            'blur':self.config['photos']['border']['blur'].get(bool),
            'color':self.config['photos']['border']['color'].get()
        }

        skip_manual_select = self.config['photos']['skipManualSelect'].get(bool)
        self.photo_handler = _photo.Photo(self.config['photos']['output'].get(),
                                          self.config['photos']['template'].get(),
                                          width=self.config['video']['width'].get(int),
                                          height=self.config['video']['height'].get(int),
                                          overwrite=self.config['photos']['overwrite'].get(bool),
                                          debug=self.config['debug'].get(bool),
                                          skip_manual_select=skip_manual_select,
                                          border=border)

    def run(self):
        """
        run
        """
        print("Version: {0} of faces.".format(__version__))
        photos = self.sync()
        missing, duplicates = self.analyze(photos)

        if missing > 0 or duplicates > 0:
            print("Continue processing? [Y]es/[N]o Default: No")
            response = str(input()).upper()
            if response != 'Y':
                self.logger.info(("User chose to terminate program after "
                                  "missing and/or duplicate photos were identified."))
                exit()

        self.photo_handler.detect_faces(photos)
        self._create_video()

    def sync(self) -> tuple:
        """
        sync
        """
        sync_method = self.config['photos']['sync'].get()
        sync_path = self.config['photos']['path'].get()
        photos = []
        self.logger.info('Syncing photos using %s method.', sync_method)
        if sync_method == 'fs':
            self.photo_synchronizer = sync.Fs(sync_path)
        elif sync_method == 'google':
            google_cred = self.config['photos']['googleCredentials'].get()
            album_id = self.config['photos']['albumId'].get()
            self.photo_synchronizer = sync.GooglePhotos(sync_path,
                                                        google_cred,
                                                        album_id)

        if self.photo_synchronizer is not None:
            photos = self.photo_synchronizer.sync()

        self.logger.info('%s photos retrieved.', len(photos))

        return photos

    def analyze(self, photos):
        """
        analyze
        """
        print('Gathering photo data')

        self.photos_dates(photos)
        photos.sort(key=lambda tup: tup[1])
        print(f"Photos sequence between {photos[0][1].date()} to {photos[len(photos)-1][1].date()}")

        if not self.config['photos']['skip']:
            missing_dates = self.missing_photos_dates(photos)
            if len(missing_dates) > 0:
                print("Missing dates in photos sequence.")
                _ = [print(f" - {missing_date}") for missing_date in missing_dates]

            duplicate_dates = self.duplicate_photo_dates(photos)
            if len(duplicate_dates) > 0:
                print("Duplicate dates in photos sequence.")
                _ = [print(f" - {duplicate_date}") for duplicate_date in duplicate_dates]

            return (len(missing_dates), len(duplicate_dates))

        return (0, 0)


    def photos_dates(self, photos):
        """
        photos_dates
        """
        # sort oldest first
        # list start date, end date, missing items
        image_exif_date_tags = ['DateTimeOriginal', 'DateTime', 'DateTimeDigitized']

        for i, photo in enumerate(photos):
            image = Image.open(photo)
            image_exif = image.getexif()

            if image_exif is None:
                self.logger.warning("Photo %s has no available exif data.", os.path.split(photo)[1])
            else:
                image_exif_dic = dict(image_exif)
                image_datetime = None
                image_exif_tags = image_exif_dic.keys()

                if any(tag in image_exif_date_tags for tag in image_exif_tags):
                    for tag in image_exif_date_tags:
                        if tag in image_exif_tags:
                            exif_value = image_exif_dic[tag]
                            date_value = exif_value.split(" ")[0]
                            date_components = date_value.split(':')
                            image_datetime = datetime.date(int(date_components[0]),
                                                           int(date_components[1]),
                                                           int(date_components[2]))
                            break
                else:
                    timestamp = os.path.getctime(photo)
                    image_datetime = datetime.fromtimestamp(timestamp)

                if not image_datetime is None:
                    photos[i] = (photo, image_datetime)
                else:
                    self.logger.warning("Unable to extract date from photo %s", photo)

    def missing_photos_dates(self, photos) -> list:
        """
        missing_photos_dates
        """
        if not photos:
            return []

        missing_dates = []
        dates = [date.date() for _, date in photos]
        start_date = dates[0]
        delta = dates[len(dates)-1] - dates[0]
        date_seq = [start_date + timedelta(days=day) for day in range(delta.days+1)]
        missing_dates = set(date_seq) - set(dates)

        return sorted(missing_dates)

    def duplicate_photo_dates(self, photos) -> list:
        """
        duplicate_photo_dates
        """
        if not photos:
            return {}

        dates = [date.date() for _, date in photos]
        duplicate_dates = [date for date in dates if dates.count(date) > 1]
        return sorted(set(duplicate_dates))


    def setup_logging(self):
        """
        setup_logging
        """
        log_levels = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARN': logging.WARN,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL,
            'NONE': logging.CRITICAL + 1
        }
        log_level = log_levels['NONE']
        if(self.config['logLevel'] and self.config['logLevel'].get() in log_levels):
            log_level = log_levels[self.config['logLevel'].get()]

        log_formatter = '%(asctime)s - %(levelname)s - %(message)s'
        logging.basicConfig(format=log_formatter, level=log_level)

        self.logger = logging.getLogger(__name__)

    def _create_video(self):
        """
        _create_video
        """
        output_path = self.config['photos']['output'].get()
        filter_ext = r'image%05d.jpg'
        filter_input_path = str(Path(output_path).joinpath(filter_ext))
        filename = str(Path(output_path).joinpath('vid.mp4'))
        ffmpeg_path = self.config['ffmpegPath'].get()
        video_opts = {
            'framerate': self.config['video']['framerate'].get(),
            'scale': ('{}x{}'
                      .format(self.config['video']['width'].get(),
                              self.config['video']['height'].get())
                     ),
            'codec': self.config['video']['codec'].get()
        }

        if Path(ffmpeg_path).exists():
            (
                ffmpeg.input(filter_input_path, f='image2', framerate=video_opts['framerate'])
                .output(filename, vcodec=video_opts['codec'],
                        s=video_opts['scale'], vb='100M')
                .run(cmd=ffmpeg_path, capture_stdout=True, overwrite_output=True)
            )
        else:
            self.logger.critical('ffmpeg executable not found at %s', {ffmpeg_path})

        self.photo_synchronizer.move(filename,
                                     self.config['video']['output'].get())
