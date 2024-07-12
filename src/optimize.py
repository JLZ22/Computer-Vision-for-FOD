from ultralytics import YOLO
import utils

if __name__ == '__main__':
    model = YOLO('../models/yolov8n.pt')
    best = model.tune(data="../test_data/dataset/dataset.yaml", 
                        epochs=1, 
                        iterations=10, 
                        imgsz=512,
                        optimizer="AdamW", 
                        plots=False, 
                        save=False, 
                        val=False,
                        device=utils.get_device())
    print(best)