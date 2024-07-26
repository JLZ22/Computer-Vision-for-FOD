from ultralytics import YOLO
from clearml import Task
import argparse
import yaml

def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-name', type=str, default='train')
    return parser.parse_args()

if __name__ == '__main__':
    opt = init_parser()

    # Initialize a new ClearML task
    task = Task.init(project_name="dev", 
                    task_name=opt.task_name, 
                    task_type=Task.TaskTypes.training)


    with open('../config.yaml') as f:
        config = yaml.safe_load(f)
        try:
            with open(config['train']['hyp']) as f:
                hyperparameters = yaml.safe_load(f)
            task.connect(hyperparameters)
            hyp_exists=True
        except:
            hyp_exists=False

    model = YOLO("../models/yolov8n.pt")

    if hyp_exists:
        model.train(
            data=config['data_path'],
            epochs=config['epochs'],
            batch=config['batch_size'],
            imgsz=config['imgsz'],
            cfg=config['hyp'],
            verbose=True,
            logger=task.get_logger(),
            patience=config['patience']
        )
    else:
        model.train(
            data=config['data_path'],
            epochs=config['epochs'],
            batch=config['batch_size'],
            imgsz=config['imgsz'],
            verbose=True,
            logger=task.get_logger(),
            patience=config['patience']
        )

    task.close()