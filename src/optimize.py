from ultralytics import YOLO
import utils
import pathlib as Path
import yaml

if __name__ == '__main__':
    data_path = Path("../test_data/dataset/dataset.yaml")
    with open("../env.yaml") as f:
        config = f.safe_load(f)
    if config:
        data_path = config['path']
    model = YOLO('../models/yolov8n.pt')
    best = model.tune(data=data_path, 
                        epochs=1, 
                        iterations=10, 
                        imgsz=512,
                        optimizer="AdamW", 
                        plots=False, 
                        save=False, 
                        val=False,
                        device=utils.get_device())
    print(best)