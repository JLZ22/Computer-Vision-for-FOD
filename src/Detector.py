from ultralytics import YOLO
import cv2
import math

class Detector:
    '''
    Wrapper class for the YOLO model to detect objects in
    images, videos, or camera streams.
    '''

    def __init__(self, model= YOLO("../models/yolov8n.pt"), class_names=None):
        
        self.model=model
        if class_names is None:
            self.class_names = model.names

    def interpret_frame_result(self, results, frame, show, input_type, win_name):
        '''
        Show function to display the bounding boxes and class names
        along with other FOD details and custom functionalities. 
        TODO: interpret the frame in the context of the FOD problem
        '''
        r = next(results, None)
        if r:
            boxes = r.boxes
            for box in boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

                # put box in cam
                cv2.rectangle(frame, 
                              (x1, y1), 
                              (x2, y2), 
                              (255, 0, 255), 
                              3)

                # confidence
                confidence = math.ceil((box.conf[0]*100))/100

                # class name
                cls = int(box.cls[0])

                # object details
                org = [x1 + 5, y1+25]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (0, 255, 255)
                thickness = 2

                cv2.putText(frame, self.class_names[cls] + ' ' + str(confidence), org, font, fontScale, color, thickness)
        
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
        '''
        cap = cv2.VideoCapture(camera)
        win_name = f'Camera {camera}'
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = self.model.predict(frame, 
                                         show=      False, 
                                         conf=      confidence, 
                                         stream=    True)
            
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
        Detect objects in images or videos. Shows results using 
        custom function interpret_frame_result.
        '''
        for media_path in media_paths:
            # read media_path and get the frame
            frame = cv2.imread(media_path)
            if frame is not None:
                results = self.model.predict(frame, 
                                                show=      False, 
                                                conf=      confidence, 
                                                stream=    True)
                self.interpret_frame_result(results, frame, show, 'Image', media_path)
            else:
                cap = cv2.VideoCapture(media_path)
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    results = self.model.predict(frame, 
                                                    show=      False, 
                                                    conf=      confidence, 
                                                    stream=    True)
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
        '''
        if input_type == 'media':
            media_paths = [str(media_path) for media_path in media_paths]
            self.detect_media(confidence, media_paths, show)
        elif input_type == 'camera':
            self.detect_camera(confidence, camera, show)
        else:
            raise ValueError("Invalid input type. Please choose either 'media' or 'camera'.")