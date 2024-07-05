from ultralytics import YOLO

model = YOLO("../models/yolov8n.pt")
results = model.train(data="../test_data/data.yaml", 
                      epochs=3, 
                      batch_size=16, 
                      imgsz=640, 
                      device="0", 
                      weights="yolov5s.pt", 
                      project="runs/train", 
                      name="exp", 
                      exist_ok=True)