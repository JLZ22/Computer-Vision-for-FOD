from ultralytics import YOLO
from clearml import Task
import yaml
import Utils

def get_config_and_hyperparameters():
   '''
   Load the configuration and hyperparameters from the config.yaml file
   '''
   with open('../config.yaml') as f:
        config = yaml.safe_load(f)
        try:
            with open(config['train']['hyp']) as f:
                hyperparameters = yaml.safe_load(f)
            return config, hyperparameters
        except:
            return config, None

def train(model, config, hyperparameters):
    '''
    Train the model using the configuration and hyperparameters if they exist.
    Otherwise, train the model using the default YOLO hyperparameter configuration.
    '''
    device = Utils.get_device()
    train = config['train']
    if hyperparameters:
        return model.train(
            data=           train['data_path'],
            epochs=         train['epochs'],
            batch=          train['batch_size'],
            imgsz=          train['imgsz'],
            cfg=            train['hyp'],
            name=           train['name'],
            patience=       train['patience'],
            device=         device,
            verbose=        True,
            deterministic=  False
        )
    else:
        return model.train(
            data=           train['data_path'],
            epochs=         train['epochs'],
            batch=          train['batch_size'],
            imgsz=          train['imgsz'],
            patience=       train['patience'],
            device=         device,
            verbose=        True,
            deterministic=  False
        )

if __name__ == '__main__':
    # Get the configuration and hyperparameters from the config.yaml file
    config, hyperparameters = get_config_and_hyperparameters()

    # Initialize a new ClearML task
    task = Task.init(project_name=  config['project'], 
                    task_name=      config['train']['name'], 
                    task_type=      Task.TaskTypes.training)


    # Connect the hyperparameters to the task if they exist
    if hyperparameters:
        task.connect(hyperparameters)
    
    # Load the model
    model = YOLO(f'../models/{config['model_variant']}.pt')

    # Connect the model to the task
    task.connect(model)

    # Train the model
    train(model, config, hyperparameters)

    task.close()