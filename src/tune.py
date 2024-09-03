from ultralytics import YOLO
import Utils
import yaml
import argparse
from pathlib import Path

def parse_args() -> argparse.Namespace:
    '''
    Parse the command line arguments. The argument(s) are as follows:
    - Path to the yaml config file which must have the following structure: 
        
        model_path:  FILE 
        
        tune:
            iterations: INT
            name:       TEXT
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
            project_name:   TEXT
            task_name:      TEXT
    ```
    - - -
    #####Return: `argparse.Namespace`
    An object that represents the parsed arguments.
    '''
    parser = argparse.ArgumentParser(description='Tune a YOLO model')
    parser.add_argument('config', type=str, help='Path to the config.yaml file', metavar='STR')
    return parser.parse_args()

def load_config(args) -> dict:
    '''
    Load the configuration from the command line arguments.
    - - -
    `args`: command line arguments
    - - -
    #####Return: `dict`
    A dictionary representing the configuration.
    '''
    with open(args.config) as f:
        return yaml.safe_load(f)
    
def main():
    '''
    Tune a YOLO model using the configuration specified in the config.yaml file 
    which is passed as a command line argument. 
    - - -
    ```
    usage: tune.py [-h] STR

    Tune a YOLO model

    positional arguments:
    STR         Path to the config.yaml file
    ```
    '''
    args = parse_args()
    config = load_config(args)

    # Load the model and check
    model_path = config['model_path']
    if not Path(model_path).exists():
        s = input("Cannot find model path. Download model path and continue? (Y/n)")
        if s.lower() == 'n':
            exit(1)
    model = YOLO(model_path)

    tune = config['tune']
    train = config['train']

    # check if automatically downlaoded weight is present in working directory
    yolov8n_path = Path('./yolov8n.pt')
    delete_yolov8n = not yolov8n_path.exists()

    model.tune(
        # Tuning params
        iterations= tune['iterations'],
        name=       tune['name'],       # technically a kwarg of train but it is appropriate to be a 
                                        # tuning param for naming here

        # Additional training params
        data=       train['data_path'], 
        epochs=     train['epochs'],
        batch=      train['batch_size'],
        imgsz=      train['imgsz'],
        plots=      True,               # Generate and display training plots
        val=        True,               # Evaluate the model on the validation set
        device=     Utils.get_device()  # Specify device for training based on availability
    )

    # delete automatically downloaded weights 
    if delete_yolov8n:
        yolov8n_path.unlink()

if __name__ == '__main__':
    main()