from ultralytics import YOLO
import cv2
import math

allNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

FODNames = ['allen wrench', 'pencil', 'screwdriver', 'tool bit', 'wrench']

class Detector:
    '''
    Wrapper class for the YOLO model to detect objects in
    images, videos, or camera streams.
    '''

    def __init__(self, 
                 model=         YOLO("../models/yolov8n.pt"),
                 class_names=   allNames):
        
        self.model=model
        self.classNames=class_names

    def show_results(self, results, frame):
        '''
        Show function to display the bounding boxes and class names
        along with other FOD details and custom functionalities. 
        '''
        for r in results:
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
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(frame, self.classNames[cls] + ' ' + str(confidence), org, font, fontScale, color, thickness)
            cv2.imshow("Image", frame)

    def detect_camera(self, 
                     confidence=    0.7,
                     camera=        0
                     ):
        '''
        Detect objects in a camera stream.
        '''
        cap = cv2.VideoCapture(camera)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = self.model.predict(frame, 
                                         show=      False, 
                                         conf=      confidence, 
                                         stream=    True)
            
            self.showResults(results, frame)
            if cv2.waitKey(1) == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def detect_media(self, 
                    confidence,
                    media_paths):  
        '''
        Detect objects in images or videos.
        '''
        for vid in media_paths:
            self.model.predict(vid, 
                               show=        True, 
                               conf=        confidence,)
    
    def detect(self,  
               input_type,
               confidence=  0.7,
               media_paths= [],
               camera=      0,
               save_path=   None):
        '''
        Detect objects in images, videos, or camera streams. Saves the 
        results if a save path is provided.
        '''
        if input_type == 'media':
            self.detect_media(confidence, media_paths)
        elif input_type == 'camera':
            self.detect_camera(confidence, camera)
        else:
            raise ValueError("Invalid input type. Please choose either 'media' or 'camera'.")