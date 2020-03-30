"""
Photo Handler
"""

import logging
import math
from pathlib import Path
import cv2
import dlib
import numpy as np

class Photo(object):
    """
    Photo
    """
    def __init__(self, output, template_image, width: 600, height: 600,
                 overwrite: False, debug: None, skip_manual_select: False,
                 border: None):
        self.logger = logging.getLogger(__name__)
        self.debug = debug
        self.output = output
        self.output_size = (width, height)
        self.output_size_mid = math.floor(width/2), math.floor(height/2)
        self.overwrite = overwrite
        self.template_image = template_image
        self.shape_predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
        self.detector = dlib.get_frontal_face_detector()
        self.skip_manual_select = skip_manual_select
        self.border = border

        self.template_face_bb = None
        self.template_dim = None
        self.template_image_size = None
        self.template_face_midpoint = None
        self.template_output = None

    def _set_template(self):
        """
        _set_template
        """
        image, face = self.detect_face(self.template_image)
        self.template_face_bb = (face.rect.left(), face.rect.top(),
                                 face.rect.right(), face.rect.bottom())
        self.template_dim = face.rect.right()-face.rect.left(), face.rect.bottom() - face.rect.top()
        self.template_image_size = (image.shape[1], image.shape[0])
        self.template_face_midpoint = (math.floor((face.rect.right()+face.rect.left())/2),
                                       math.floor((face.rect.bottom()+face.rect.top())/2))

        left, top, right, bottom = 0, 0, 0, 0
        inc_x, inc_y = math.floor(self.output_size[0]/2), math.floor(self.output_size[1]/2)

        rem_x = self.template_face_midpoint[0] - inc_x
        left = rem_x if rem_x > 0 else 0
        if rem_x < 0:
            inc_x += rem_x

        rem_x = self.template_face_midpoint[0] + inc_x
        right = rem_x if rem_x < image.shape[1] else image.shape[1]

        rem_y = self.template_face_midpoint[1] - inc_y
        top = rem_y if rem_y > 0 else 0
        if rem_y < 0:
            inc_y += abs(rem_y)
        rem_y = self.template_face_midpoint[1] + inc_y
        bottom = rem_y if rem_y < image.shape[0] else image.shape[0]

        self.template_output = left, top, right, bottom # l t r b

    def detect_faces(self, photos):
        """
        detect_faces
        """
        self._set_template()

        for i, photo in enumerate(photos):
            print(f"{i+1}/{len(photos)} transforming {photo[0]}.")
            tmp_filename = "image{:05d}{}".format(i, Path(photo[0]).suffix)

            if Path(self.output).joinpath(tmp_filename).exists():
                self.logger.info("%s exists in output path, skipping.", tmp_filename)
                continue

            image, face = self.detect_face(photo[0])
            if image is None or face is None:
                continue

            image, face_rect = self._transform_photo(image, face)
            image = self._insert_border(image)
            opath = self.save_photo(image, tmp_filename, self.template_output)
            photos[i] = (*photo, opath, face_rect)

    def _insert_border(self, image):
        """
        _insert_border
        """
        if self.border is None:
            return image

        width = self.border['width']
        height = self.border['height']
        blur = self.border['blur']
        hexcolor = self.border['color']
        color = (0, 0, 0)
        left, top, right, bottom = self.template_output

        if hexcolor is not None:
            hexcolor = hexcolor.lstrip('#')
            color = tuple(int(hexcolor[i:i+2], 16) for i in (0, 2, 4))

        if blur:
            self._border_blur(image, (width, height), self.template_output)
        else:
            if width > 0:
                cv2.rectangle(image, (left, top), (left+width, bottom), color, -1)
                cv2.rectangle(image, (right, top), (right-width, bottom), color, -1)
            if height > 0:
                cv2.rectangle(image, (left, top), (right, top+height), color, -1)
                cv2.rectangle(image, (left, bottom-height), (right, bottom), color, -1)

        return image

    def _border_blur(self, image, size, roi):
        """
        _border_blur
        """
        blurred_img = cv2.GaussianBlur(image[roi[1]:roi[3], roi[0]:roi[2]],
                                       (21, 21),
                                       cv2.BORDER_DEFAULT)
        center = math.floor(blurred_img.shape[1]/2), math.floor(blurred_img.shape[0]/2)
        if size[0] > 0:
            image[roi[1]:roi[3], roi[0]:roi[0]+size[0]] = blurred_img[0:blurred_img.shape[0],
                                                                      center[0]-size[0]:center[0]]
            image[roi[1]:roi[3], roi[2]-size[0]:roi[2]] = blurred_img[0:blurred_img.shape[0],
                                                                      center[0]:center[0]+size[0]]
        if size[1] > 0:
            image[roi[1]:roi[1]+size[1], roi[0]:roi[2]] = blurred_img[center[1]-size[1]:center[1],
                                                                      0:blurred_img.shape[1]]
            image[roi[3]-size[1]:roi[3], roi[0]:roi[2]] = blurred_img[center[1]:center[1]+size[1],
                                                                      0:blurred_img.shape[1]]

        return image

    def _transform_photo(self, image, face):
        """
        Adjust face to rotate them straight
        Adjust photo to mimic positions in template photo
        """
        if image is None or face is None:
            return image, face

        face_rect = face.rect

        if not isinstance(face, MockDlibFace):
            eyes, jaw = self._get_face_features(face)
            if eyes is None or jaw is None:
                self.logger.critical("Missing facial features.")
                return image, face

            left_eye, right_eye = eyes

            right_eye_mid_x = math.ceil((right_eye[0][0] + right_eye[3][0])/2)
            right_eye_mid_y = math.ceil((((right_eye[1][1] + right_eye[2][1])/2) +
                                         ((right_eye[4][1] + right_eye[5][1])/2))/2)

            left_eye_mid_x = math.ceil((left_eye[0][0] + left_eye[3][0])/2)
            left_eye_mid_y = math.ceil((((left_eye[1][1] + left_eye[2][1])/2) +
                                        ((left_eye[4][1] + left_eye[5][1])/2))/2)

            eyes_midpoint = (math.ceil((left_eye_mid_x + right_eye_mid_x)/2),
                             math.ceil((left_eye_mid_y + right_eye_mid_y)/2))
            jaw_midpoint = jaw[8]
            image_midpoint = (math.ceil(image.shape[1]/2), math.ceil(image.shape[0]/2))

            horizantal = (0, jaw_midpoint[1]), (image.shape[1], jaw_midpoint[1])
            vertical = (image_midpoint[0], 0), (image_midpoint[0], horizantal[0][1])

            face_line = (eyes_midpoint[0], eyes_midpoint[1]), (jaw_midpoint[0], jaw_midpoint[1])
            center_line = (image_midpoint[0]+1, 0), (image_midpoint[0], horizantal[0][1])
            center = (eyes_midpoint[0] + jaw_midpoint[0])/2, (eyes_midpoint[1] + jaw_midpoint[1])/2

            image = Photo.rotate_photo(image, center, face_line, center_line)

            if self.debug:
                img_c = image.copy()
                cv2.line(img_c, eyes_midpoint, jaw_midpoint, (0, 255, 134), 2)
                cv2.line(img_c, horizantal[0], horizantal[1], (0, 0, 0), 2) # horizantal line
                cv2.line(img_c, vertical[0], vertical[1], (0, 0, 0), 2) # vertical line
                cv2.line(img_c, vertical[0], vertical[1], (255, 0, 0), 2) # vertical line

                self._debug_window(img_c, 'Rotated Image')

        image, face_rect = self._scale_photo2(image, face_rect)
        image, face_rect = self._shift_photo(image, face_rect)

        return image, face_rect

    @staticmethod
    def rotate_photo(image, center, line_a, line_b):
        """
        rotate_photo
        """
        a_pt2, a_pt1 = line_a
        b_pt2, b_pt1 = line_b

        try:
            slope_face = (a_pt2[1] - a_pt1[1])/(a_pt2[0] - a_pt1[0])
            slope_center = (b_pt2[1] - b_pt1[1])/(b_pt2[0] - b_pt1[0])
        except ZeroDivisionError:
            return image

        angle = math.degrees(math.atan((slope_center-slope_face)/(1+(slope_face*slope_center))))
        rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
        return cv2.warpAffine(image, rotation_matrix, image.shape[1::-1],
                              flags=cv2.INTER_LINEAR)

    def _scale_photo2(self, image, face):
        """
        Rescale photo
        """
        current_width, current_height = face.right()-face.left(), face.bottom()-face.top()
        scale_x = abs(self.template_dim[0] / current_width)
        scale_y = abs(self.template_dim[1] / current_height)

        if(scale_x != 1 or scale_y != 1):
            left = math.floor(face.left()*scale_x)
            right = math.floor(face.right()*scale_x)
            top = math.floor(face.top()*scale_y)
            bottom = math.floor(face.bottom()*scale_y)
            template_mid_w, template_mid_h = (math.floor(self.template_image_size[0]/2),
                                              math.floor(self.template_image_size[1]/2))
            current_mid_w, current_mid_h = math.floor((left+right)/2), math.floor((top+bottom)/2)

            image = cv2.resize(image, (0, 0), fx=scale_x, fy=scale_y)
            blank_image = np.zeros((self.template_image_size[1], self.template_image_size[0], 3),
                                   np.uint8)
            if self.debug:
                img_c = image.copy()
                cv2.rectangle(img_c, (left, top), (right, bottom), (255, 0, 0), 1)
                self._debug_window(img_c, 'Scaled Image')

            x1_pos = current_mid_w - template_mid_w if (current_mid_w - template_mid_w) >= 0 else 0
            y1_pos = current_mid_h - template_mid_h if (current_mid_h - template_mid_h) >= 0 else 0
            x_pos = image.shape[1]
            if x_pos-x1_pos > self.template_image_size[0]:
                x_pos -= (x_pos-x1_pos-self.template_image_size[0])
            y_pos = image.shape[0]
            if y_pos-y1_pos > self.template_image_size[1]:
                y_pos -= (y_pos-y1_pos-self.template_image_size[1])

            # adjust left top right bottom to compensate
            left -= x1_pos
            right -= x1_pos
            top -= y1_pos
            bottom -= y1_pos

            blank_image[0:y_pos-y1_pos, 0:x_pos-x1_pos] = image[y1_pos:y_pos, x1_pos:x_pos]
            image = blank_image

            if self.debug:
                img_c = image.copy()
                cv2.rectangle(img_c, (left, top), (right, bottom), (255, 0, 0), 1)
                self._debug_window(img_c, 'Adjusted Scaled Image')

            face = dlib.rectangle(left, top, right, bottom)

        return image, face

    def _debug_window(self, image, title='debug'):
        """
        _debug_window
        """
        cv2.imshow(title, image)
        cv2.waitKey(0)
        cv2.destroyWindow(title)

    def _shift_photo(self, image, rect):
        """
        shift image
        """
        face_midpoint = (math.floor((rect.right()+rect.left())/2),
                         math.floor((rect.bottom()+rect.top())/2))

        dir_x = 1 if face_midpoint[0] < self.template_face_midpoint[0] else -1
        dir_y = 1 if face_midpoint[1] < self.template_face_midpoint[1] else -1
        shift_x = abs(face_midpoint[0] - self.template_face_midpoint[0]) * dir_x
        shift_y = abs(face_midpoint[1] - self.template_face_midpoint[1]) * dir_y

        if shift_x != 0 or shift_y != 0:
            t_m = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            image = cv2.warpAffine(image, t_m, image.shape[1::-1])
            rect = dlib.rectangle(rect.left()+shift_x,
                                  rect.top()+shift_y,
                                  rect.right()+shift_x,
                                  rect.bottom()+shift_y)

            if self.debug:
                img_c = image.copy()
                cv2.rectangle(img_c,
                              (rect.left(), rect.top()),
                              (rect.right(), rect.bottom()),
                              (255, 0, 0), 2)
                self._debug_window(img_c, 'Shifted Image')

        return image, rect

    def save_photo(self, image, filename, rect):
        """
        Save photo
        """
        if not Path(self.output).exists():
            Path(self.output).mkdir(parents=True, exist_ok=True)

        output_path = Path(self.output).joinpath(filename)
        output_path_str = str(output_path)
        if not self.overwrite and output_path.exists():
            print(f"A file already exists at {output_path_str}. Overwrite [Y]es or [N]o.")
            overwrite = input()
            if overwrite == 'Y' or overwrite == 'y':
                output_path.unlink()

        cv2.imwrite(output_path_str, image[rect[1]:rect[3], rect[0]:rect[2]])

        if self.debug:
            self._debug_window(image[rect[1]:rect[3], rect[0]:rect[2]], 'Saved Image')

        return output_path_str

    def detect_face(self, photo, scale_fx=0.2, scale_fy=0.2, image=None):
        """
        Save photo
        """
        face = None

        if image is None:
            image = cv2.imread(photo)
        if scale_fx > 0 or scale_fy > 0:
            image = cv2.resize(image, (0, 0), fx=scale_fx, fy=scale_fy)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        detections = self.detector(img, 1)
        faces = dlib.full_object_detections()
        for det in detections:
            faces.append(self.shape_predictor(img, det))

        if len(faces) == 0:
            print(f"No face found for photo {photo}. Manual selection used.")
            if not self.skip_manual_select:
                face = self._select_face(image)
            else:
                return image, None
        elif len(faces) != 1:
            face = faces[0] #todo select prominent face
        else:
            face = faces[0]

        if self.debug:
            img_c = image.copy()
            cv2.rectangle(img_c,
                          (face.rect.left(), face.rect.top()),
                          (face.rect.right(), face.rect.bottom()),
                          (255, 0, 0),
                          2)
            self._debug_window(img_c, 'Original Image')

        return image, face

    def _select_face(self, image):
        """
        _select_face
        """
        window_name = 'SelectFace'
        roi = [0, 0, 0, 0]
        drawing = False
        done = False

        def _select_face_handler(event, x_pos, y_pos, flags, params):
            """
            _select_face_handler
            """
            nonlocal roi, drawing, done

            if event == cv2.EVENT_LBUTTONDOWN: #left mouse down
                drawing = True
                roi[0] = x_pos
                roi[1] = y_pos
                roi[2] = x_pos
                roi[3] = y_pos
            elif event == cv2.EVENT_MOUSEMOVE: #mouse move
                if drawing:
                    roi[2] = x_pos
                    roi[3] = y_pos
            elif event == cv2.EVENT_LBUTTONUP: #left mouse up
                drawing = False
                done = True
                roi[2] = x_pos
                roi[3] = y_pos

        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, _select_face_handler)
        while True:
            if not drawing:
                cv2.imshow(window_name, image)
            elif drawing and not done:
                img_c = image.copy()
                cv2.rectangle(img_c, (roi[0], roi[1]), (roi[2], roi[3]), (255, 0, 0), 2)
                cv2.imshow(window_name, img_c)
            elif done:
                break

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or done:
                break
        cv2.destroyWindow(window_name)

        return MockDlibFace(roi[0], roi[1], roi[2], roi[3])

    def _face_transformations(self, photo, image, face):
        """
        _face_transformations
        """
        if image is None or face is None:
            return image

        eyes, jaw = self._get_face_features(face)
        if eyes is None or jaw is None:
            self.logger.warning("Missing facial features for %s", photo)
            return image

        left_eye, right_eye = eyes

        right_eye_mid_x = math.ceil((right_eye[0][0] + right_eye[3][0])/2)
        right_eye_mid_y = math.ceil((((right_eye[1][1] + right_eye[2][1])/2) +
                                     ((right_eye[4][1] + right_eye[5][1])/2))/2)

        left_eye_mid_x = math.ceil((left_eye[0][0] + left_eye[3][0])/2)
        left_eye_mid_y = math.ceil((((left_eye[1][1] + left_eye[2][1])/2) +
                                    ((left_eye[4][1] + left_eye[5][1])/2))/2)

        eyes_midpoint = (math.ceil((left_eye_mid_x + right_eye_mid_x)/2),
                         math.ceil((left_eye_mid_y + right_eye_mid_y)/2))
        jaw_midpoint = jaw[8]
        face_midpoint = ((eyes_midpoint[0] + jaw_midpoint[0])/2,
                         (eyes_midpoint[1] + jaw_midpoint[1])/2)
        image_center = (math.floor(image.shape[1]/2), math.floor(image.shape[0]/2))

        horizantal = (0, jaw_midpoint[1]), (image.shape[1], jaw_midpoint[1])
        vertical = (image_center[0], 0), (image_center[0], horizantal[0][1])

        slope_face = (eyes_midpoint[1] - jaw_midpoint[1])/(eyes_midpoint[0] - jaw_midpoint[0])
        slope_center = (0 - horizantal[0][1])/((image_center[0]+1)-image_center[0])
        angle = math.degrees(math.atan((slope_center-slope_face)/(1+(slope_face*slope_center))))
        rotation_matrix = cv2.getRotationMatrix2D(face_midpoint, -angle, 1.0)
        image = cv2.warpAffine(image, rotation_matrix, image.shape[1::-1], flags=cv2.INTER_LINEAR)

        #shift image
        #if(angle < 1):
        #    shift_x = image_center[0] - eyes_midpoint[0]
        #    t_m = np.float32([[1,0,shift_x], [0,1,0]])
        #    image = cv2.warpAffine(image, t_m, image.shape[1::-1])

        if self.debug:
            cv2.line(image, eyes_midpoint, jaw_midpoint, (0, 255, 134), 2)
            cv2.line(image, horizantal[0], horizantal[1], (0, 0, 0), 2) # horizantal line
            cv2.line(image, vertical[0], vertical[1], (0, 0, 0), 2) # vertical line
            cv2.line(image, vertical[0], vertical[1], (255, 0, 0), 2) # vertical line

        return image

    def _get_face_features(self, face) -> tuple:
        """
        _get_face_features
        """
        if face is None:
            return None, None

        right_eye = [face.part(i) for i in range(36, 42)]
        right_eye = [(i.x, i.y) for i in right_eye]

        left_eye = [face.part(i) for i in range(42, 48)]
        left_eye = [(i.x, i.y) for i in left_eye]

        jaw = [face.part(i) for i in range(0, 17)]
        jaw = [(i.x, i.y) for i in jaw]

        return (left_eye, right_eye), jaw

class MockDlibFace(object):
    """
    MockDlibFace
    """
    def __init__(self, left, top, right, bottom):
        self._left = left
        self._top = top
        self._right = right
        self._bottom = bottom

    def left(self):
        """left"""
        return self._left

    def right(self):
        """right"""
        return self._right

    def top(self):
        """top"""
        return self._top

    def bottom(self):
        """bottom"""
        return self._bottom

    @property
    def rect(self):
        """rect"""
        return dlib.rectangle(self._left, self._top, self._right, self._bottom)
