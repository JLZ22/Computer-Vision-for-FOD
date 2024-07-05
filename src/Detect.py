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

class Detect:
    '''
    model is a model from the ultralytics library.
    input is either "image", "video", or "camera".
    path is the path to the image file or video file
    camera is the camera index if input is "camera"
    '''
    def __init__(self, 
                 model = YOLO("models/yolov8n.pt"), 
                 input = "image", 
                 paths=[""],
                 camera = -1):
        self.model = model
        self.input_type = input
        self.paths = paths
        self.camera = camera
        self.classNames = ["person", "phone", "bottle"]

    def detectImage(self):
        results = self.model.predict(self.paths)
        for result in results:
            result.show()

    def showResults(self, results, frame):
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

                # put box in cam
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # confidence
                confidence = math.ceil((box.conf[0]*100))/100
                print("Confidence --->",confidence)

                # class name
                cls = int(box.cls[0])
                print("Class name -->", self.classNames[cls])

                # object details
                org = [x1 + 5, y1+25]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(frame, self.classNames[cls] + ' ' + str(confidence), org, font, fontScale, color, thickness)
            cv2.imshow("Image", frame)

    def detectCamera(self):
        cap = cv2.VideoCapture(self.camera)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = self.model.predict(frame, show=False, conf=0.7, stream=True)
            
            self.showResults(results, frame)
            if cv2.waitKey(1) == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def detectVideo(self):
        results = [self.model.predict(vid, show=True, conf=0.7) for vid in self.paths]
    
    def detect(self):
        if self.input_type == "image":
            self.detectImage()
        elif self.input_type == "video":
            self.detectVideo()
        elif self.input_type == "camera":
            self.detectCamera()
        else:
            print("Invalid input type. Please use 'image', 'video' or 'camera'.")