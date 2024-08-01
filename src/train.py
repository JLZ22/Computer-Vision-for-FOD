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
    if hyperparameters:
        return model.train(
            data=config['data_path'],
            epochs=config['epochs'],
            batch=config['batch_size'],
            imgsz=config['imgsz'],
            cfg=config['train']['hyp'],
            name=config['name'],
            device=device,
            verbose=True,
            determinisitic=False
        )
    else:
        return model.train(
            data=config['data_path'],
            epochs=config['epochs'],
            batch=config['batch_size'],
            imgsz=config['imgsz'],
            patience=config['patience'],
            device=device,
            verbose=True,
            deterministic=False
        )

if __name__ == '__main__':
    # Get the configuration and hyperparameters from the config.yaml file
    config, hyperparameters = get_config_and_hyperparameters()

    # Initialize a new ClearML task
    task = Task.init(project_name=config['project'], 
                    task_name=config['train']['name'], 
                    task_type=Task.TaskTypes.training)


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