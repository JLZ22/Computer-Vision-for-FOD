from ultralytics import YOLO
import Utils
from pathlib import Path
import yaml

if __name__ == '__main__':
    data_path = Path('../test_data/dataset/dataset.yaml')
    with open('../config.yaml') as f:
        config = yaml.safe_load(f)

    model = YOLO(f'../models/{config['model_variant']}.pt')
    results = model.tune(
        # tune parameters
        use_ray=True,
        grace_period=config['tune']['grace_period'],

        # train parameters
        data=config['data_path'], 
        epochs=config['epochs'],
        batch=config['batch_size'],
        imgsz=config['imgsz'],
        plots=True,    # Generate and display training plots
        val=True,      # Evaluate the model on the validation set
        device=Utils.get_device()  # Specify device for training based on availability
    )