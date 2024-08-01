from ultralytics import YOLO
import Utils
from pathlib import Path
import yaml

if __name__ == '__main__':
    data_path = Path('../test_data/dataset/dataset.yaml')
    with open('../config.yaml') as f:
        config = yaml.safe_load(f)

    model = YOLO(f'../models/{config['model_variant']}.pt')

    tune = config['tune']
    train = config['train']
    results = model.tune(
        # Tuning params
        iterations= tune['iterations'],

        # Additional training params
        data=       train['data_path'], 
        epochs=     train['epochs'],
        batch=      train['batch_size'],
        imgsz=      train['imgsz'],
        plots=      True,               # Generate and display training plots
        val=        True,               # Evaluate the model on the validation set
        device=     Utils.get_device()  # Specify device for training based on availability
    )