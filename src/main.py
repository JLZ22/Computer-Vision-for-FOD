from ultralytics import YOLO
from Detector import Detector
import yaml


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
    with open('../config.yaml') as f:
        config = yaml.safe_load(f)
    detect = config['detect']
    model = YOLO(detect['model_path'])
    detector = Detector(model=model)
    detector.detect('camera')