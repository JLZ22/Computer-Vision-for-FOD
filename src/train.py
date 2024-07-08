from ultralytics import YOLO
from clearml import Task
import argparse

task = Task.init(project_name="dev", 
                 task_name="first_task", 
                 task_type=Task.TaskTypes.training)

logger = task.get_logger()
logger.report_scalar("loss", "train", iteration=0, value=0.5)
logger.report_scalar("accuracy", "train", iteration=0, value=0.8)

parser = argparse.ArgumentParser()
parser.add_argument('--project', type=str, default="../models/dev")
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--img-size', type=int, default=512)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--momentum', type=float, default=0.937)
parser.add_argument('--weight-decay', type=float, default=0.0005)
parser.add_argument('--data-path', type=str, default="../test_data/dataset/dataset.yaml")
opt = parser.parse_args()

# Log hyperparameters to ClearML
hyperparameters = {
    'epochs': opt.epochs,
    'batch_size': opt.batch_size,
    'img_size': opt.img_size,
    'learning_rate': opt.lr,
    'momentum': opt.momentum,
    'weight_decay': opt.weight_decay,
    'data_path': opt.data_path,
    'project': opt.project
}
task.connect(hyperparameters)
model = YOLO("../models/yolov8n.pt")
results = model.train(data=opt.data_path, 
                      epochs=opt.epochs,
                      imgsz=opt.img_size,
                      project=opt.project,
                      batch=opt.batch_size,
                      lr0=opt.lr,
                      momentum=opt.momentum,
                      weight_decay=opt.weight_decay,
                      verbose=True)
task.close()