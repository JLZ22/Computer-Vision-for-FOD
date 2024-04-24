import warnings
import platform
from ultralytics import YOLO
from Detect import Detect

'''
The different yolov8 models have varying levels of accuracy and speed.
This is the order of the models from fastest to slowest (and least to most accurate):
More details can be found here: https://docs.ultralytics.com/datasets/detect/coco/
YOLO will download the "*.pt" file if it is not found in the "models" directory.

yolov8n (Fastest and Least accurate)
yolov8s
yolov8m
yolov8l
yolov8x (Slowest and Most accurate)
'''
if __name__ == "__main__":
    # ignore "AVCaptureDeviceTypeExternal is deprecated" warning
    if platform.system() == "Darwin":
        warnings.filterwarnings("ignore", category=DeprecationWarning)

    model = YOLO("models/yolov8n.pt")
    detecter = Detect(model, "image", ["data/images/TestImage0.png"])
    detecter.detect()