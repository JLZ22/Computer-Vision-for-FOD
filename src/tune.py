from ultralytics import YOLO
import Utils
from pathlib import Path
import yaml

if __name__ == '__main__':
    data_path = Path("../test_data/dataset/dataset.yaml")
    with open("../config.yaml") as f:
        config = yaml.safe_load(f)

    model = YOLO('../models/yolov8n.pt')
    results = model.tune(
        data=config['data_path'], 
        epochs=config['epochs'], 
        iterations=config['iterations'], 
        imgsz=config['imgsz'],
        patience=config['patience'],
        deterministic=False,
        optimizer="AdamW", 
        name="tune",
        plots=True,    # Generate and display training plots
        save=True,     # Save model checkpoints and final weights
        val=True,      # Evaluate the model on the validation set
        device=Utils.get_device()  # Specify device for training based on availability
    )

    import matplotlib.pyplot as plt

    for i, result in enumerate(results):
        plt.plot(
            result.metrics_dataframe["training_iteration"],
            result.metrics_dataframe["mean_accuracy"],
            label=f"Trial {i+1}",
        )

    plt.xlabel("Training Iterations")
    plt.ylabel("Mean Accuracy")
    plt.legend()
    plt.show()