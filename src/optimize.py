from ultralytics import YOLO
import utils
from pathlib import Path
import yaml

if __name__ == '__main__':
    data_path = Path("../test_data/dataset/dataset.yaml")
    with open("../opt_config.yaml") as f:
        config = yaml.safe_load(f)
    if config:
        data_path = config['data_path']

    model = YOLO('../models/yolov8n.pt')
    best = model.tune(data=data_path, 
                        epochs=config['epochs'], 
                        iterations=config['iterations'], 
                        imgsz=config['imgsz'],
                        optimizer="AdamW", 
                        plots=False, 
                        save=False, 
                        val=False,
                        device=utils.get_device())
    print(best)