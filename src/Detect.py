from ultralytics import YOLO

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

    def detectImage(self):
        results = self.model(self.paths)
        for result in results:
            result.show()

    def detectCamera(self):
        ...

    def detectVideo(self):
        ...

    def detect(self):
        if self.input_type == "image":
            self.detectImage()
        elif self.input_type == "video":
            self.detectVideo()
        elif self.input_type == "camera":
            self.detectCamera()
        else:
            print("Invalid input type. Please use 'image', 'video' or 'camera'.")