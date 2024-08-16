from ultralytics import YOLO
import yaml
import Utils
import argparse
from clearml import Task

def parse_args() -> argparse.Namespace:
    '''
    Parse the command line arguments. The argument(s) are as follows:
    - `--config`: Path to the yaml config file which must have the following structure: 
        
        model_variant:  TEXT (e.g. yolov8n)
        
        train:
            data_path:  DIR
            epochs:     INT
            batch_size: INT
            imgsz:      INT
            hyp:        FILE
            patience:   INT

    You may optionally include the following if you are using ClearML:

    ```
    train:
        clear_ml:
            project_name: 
            task_name:
    ```
    - - -
    #####Return: `argparse.Namespace`
    An object that represents the parsed arguments.
    '''
    parser = argparse.ArgumentParser(description='Train a YOLO model')
    parser.add_argument('--config', type=str, required=True, help='Path to the config.yaml file', metavar='STR')
    return parser.parse_args()

def get_config_and_hyperparameters(args) -> tuple[dict, dict]:
   '''
    Load the configuration and hyperparameters from the command line arguments.
    If hyperparameters do not exist, return None.
    - - -
    `args`: command line arguments
    - - -
    #####Return: `dict`, `dict`
    A pair of dictionaries representing the configuration and hyperparameters.
   '''
   with open(args.config) as f:
        config = yaml.safe_load(f)
        try:
            with open(config['train']['hyp']) as f:
                hyperparameters = yaml.safe_load(f)
            return config, hyperparameters
        except:
            return config, None

def train(model, config: dict, hyp_exists: bool):
    '''
    Train the model using the configuration and hyperparameters. 
    - - -
    `model`:        YOLO model
    `config`:       dictionary with config parameters
    `hyp_exists`:   boolean indicating whether hyperparameters exist
    '''
    device = Utils.get_device()
    train = config['train']
    if hyp_exists:
        model.train(
            data=           train['data_path'],
            epochs=         train['epochs'],
            batch=          train['batch_size'],
            imgsz=          train['imgsz'],
            cfg=            train['hyp'],
            patience=       train['patience'],
            device=         device,
            verbose=        True,
            deterministic=  False
        )
    else:
        model.train(
            data=           train['data_path'],
            epochs=         train['epochs'],
            batch=          train['batch_size'],
            imgsz=          train['imgsz'],
            patience=       train['patience'],
            device=         device,
            verbose=        True,
            deterministic=  False
        )
    
def main():
    '''
    Train a YOLO model using the configuration and hyperparameters 
    specified in the yaml config file. This file must be given in 
    order to run this script.
    - - -
    ```
    usage: train.py [-h] --config STR
    ```
    '''
    # Parse command line arguments
    args = parse_args()

    # Get the configuration and hyperparameters from the config.yaml file
    config, hyperparameters = get_config_and_hyperparameters(args)
    hyp_exists = hyperparameters is not None

    # Initialize a new ClearML task
    task = None
    if 'clear_ml' in config:
        task = Task.init(project_name=  config['project_name'], 
                        task_name=      config['train']['task_name'], 
                        task_type=      Task.TaskTypes.training)

    # Connect the hyperparameters to the task if they exist
    if hyp_exists:
        task.connect(hyperparameters)
    
    # Load the model
    model = YOLO(f'../models/{config['model_variant']}.pt')

    # Connect the model to the task
    if task:
        task.connect(model)

    # Train the model
    train(model, config, hyp_exists)

    if task:
        task.close()

if __name__ == '__main__':
    main()