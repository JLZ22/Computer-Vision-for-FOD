from ultralytics import YOLO
from typing import Generator
import cv2
import math

class Detector:
    '''
    Wrapper class for the YOLO model to detect objects in
    images, videos, or camera streams.
    '''

    def __init__(self, model):
        '''
        Initialize the Detector with the YOLO model.
        - - -
        model: The YOLO model to use for object detection.
        '''
        if model is None:
            self.model = YOLO('../models/yolov8n.pt')
        self.model = model

    def interpret_frame_result(self, 
                               results: Generator, 
                               frame: cv2.typing.MatLike, 
                               show: bool, 
                               input_type: str, 
                               win_name: str):
        '''
        Shows the results of the detection on the frame and highlights objects 
        that are within a certain space.

        TODO: interpret the frame in the context of the FOD problem
        '''
        # get the next result from the generator
        r = next(results, None)
        if r:
            boxes = r.boxes

            # for each box in the frame, draw the box and label on the frame
            for box in boxes:
                # bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

                # draw the bounding box on the frame
                cv2.rectangle(frame, 
                              (x1, y1), 
                              (x2, y2), 
                              (255, 0, 255), 
                              3)

                # calculate confidence
                confidence = math.ceil((box.conf[0]*100))/100

                # get class name
                cls = int(box.cls[0])

                # specify text details
                org = [x1 + 5, y1+25]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (0, 255, 255)
                thickness = 2

                # write the class name and confidence on the frame
                cv2.putText(frame, self.model.names[cls] + ' ' + str(confidence), org, font, fontScale, color, thickness)
        
        # if the input type is an image, show the image with a loop
        # video and camera streams are shown with a single frame because
        # the loop to display is outside of this function
        if input_type == 'Image' and show:
            while True:
                cv2.imshow(win_name, frame)
                if cv2.waitKey(1) == ord('q'):
                    break
            cv2.destroyWindow(win_name)
        elif show:
            cv2.imshow(win_name, frame)

    def detect_camera(self, 
                     confidence=    0.7,
                     camera=        0,
                     show=          True
                     ):
        '''
        Detect objects in a camera stream and highlights objects 
        that are not supposed to be in a certain space.
        - - -
        confidence: The confidence threshold for the model to detect an object.\n
        camera:     The camera number to use for the stream.\n
        show:       Boolean value to show the camera stream or not.
        '''
        cap = cv2.VideoCapture(camera)
        win_name = f'Camera {camera}'

        # loop to read the camera stream and detect objects
        while True:
            ret, frame = cap.read()

            # if the frame is not read, break the loop
            if not ret:
                break

            # detect objects in the frame
            results = self.model.predict(frame, 
                                         show=      False, 
                                         conf=      confidence, 
                                         stream=    True)
            
            # show bounding boxes and highlight objects that are not supposed to be in the space
            self.interpret_frame_result(results, frame, show, 'Camera', win_name)
            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        cv2.destroyWindow(win_name)

    def detect_media(self, 
                    confidence,
                    media_paths,
                    show):  
        '''
        Detect objects in images or videos. Highlights objects that are 
        not supposed to be in a certain space.
        - - -
        confidence:     The confidence threshold for the model to detect an object.\n
        media_paths:    A list of paths to the media files to detect objects in.\n
        show:           Boolean value to show the media files or not.\n
        '''
        # loop through the media paths and detect objects in the media
        for media_path in media_paths:
            # read media as an image
            # if the media is not an image, read it as a video

            frame = cv2.imread(media_path)
            if frame is not None:
                # detect objects in the image
                results = self.model.predict(frame, 
                                                show=      False, 
                                                conf=      confidence, 
                                                stream=    True)
                # show bounding boxes and highlight objects that are not supposed to be in the space
                self.interpret_frame_result(results, frame, show, 'Image', media_path)
            else:
                # read the media as a video
                cap = cv2.VideoCapture(media_path)

                # loop to read the video stream and detect objects
                while True:
                    ret, frame = cap.read()

                    # if the frame is not read, break the loop
                    if not ret:
                        break

                    # detect objects in the frame
                    results = self.model.predict(frame, 
                                                    show=      False, 
                                                    conf=      confidence, 
                                                    stream=    True)
                    
                    # show bounding boxes and highlight objects that are not supposed to be in the space
                    self.interpret_frame_result(results, frame, show, 'Video', media_path)
                    if cv2.waitKey(1) == ord('q'):
                        break
                cv2.destroyWindow(media_path)
            
    def detect(self,  
               input_type,
               confidence=  0.7,
               media_paths= [],
               camera=      0,
               save_path=   None,
               show=        True):
        '''
        Detect objects in images, videos, or camera streams. Saves the 
        results if a save path is provided. If the input type is 'media',
        the media paths should be provided. If the input type is 'camera',
        the camera number should be provided.
        
        TODO: Add the ability to save the results to a file.
        - - -
        input_type:     The type of input to detect objects in.\n
        confidence:     The confidence threshold for the model to detect an object.\n
        media_paths:    A list of paths to the media files to detect objects in.\n
        camera:         The camera number to use for the stream.\n
        save_path:      The path to save the results to.\n
        show:           Boolean value to show the media files or not.\n
        '''
        if input_type == 'media':
            media_paths = [str(media_path) for media_path in media_paths]
            self.detect_media(confidence, media_paths, show)
        elif input_type == 'camera':
            self.detect_camera(confidence, camera, show)
        else:
            raise ValueError("Invalid input type. Please choose either 'media' or 'camera'.")