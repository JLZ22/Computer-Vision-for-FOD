from ultralytics import YOLO
from clearml import Task
import argparse
from pathlib import Path

def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=str, default="../models/dev")
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--img-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.937)
    parser.add_argument('--weight-decay', type=float, default=0.0005)
    parser.add_argument('--data-path', type=str, default="../test_data/dataset/dataset.yaml")
    parser.add_argument('--hyp', type=str, default="../models/hyp.yaml")
    parser.add_argument('--task-name', type=str, default='train')
    return parser.parse_args()

if __name__ == '__main__':
    opt = init_parser()

    task = Task.init(project_name="dev", 
                    task_name=opt.task_name, 
                    task_type=Task.TaskTypes.training)

    logger = task.get_logger()
    logger.report_scalar("loss", "train", iteration=0, value=0.5)
    logger.report_scalar("accuracy", "train", iteration=0, value=0.8)


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

    hyp_path = Path(opt.hyp)
    if hyp_path.exists():
        results = model.train(data=opt.data_path, 
                            epochs=opt.epochs,
                            batch=opt.batch_size,
                            imgsz=opt.img_size,
                            hyp=opt.hyp,
                            verbose=True)
    else:
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