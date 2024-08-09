from ultralytics import YOLO
from Detector import Detector
import yaml
import argparse

'''
The different yolov8 models have varying levels of accuracy and speed.
This is the order of the models from fastest to slowest (and least to most accurate):
More details can be found here: https://docs.ultralytics.com/datasets/detect/coco/
YOLO will download the "*.pt" file if it is not found in the "models" directory.

yolov8n (Fastest and Least accurate)
yolov8s
yolov8m
yolov8l
yolov8x (Slowest and Most accurate)
'''

def parse_args():
    '''
    Parse command line arguments for the object detection script. The arguments are as follows:
    1. `--input-type`: Input type ("media" or "camera")
    2. `--confidence`: Confidence threshold for predictions
    3. `--media-paths`: List of media paths to predict on if --input-type is "media"
    4. `--camera-index`: Camera index if --input-type is "camera"
    5. `--save-path`: Path to save directory. Will not save if not specified.
    6. `--camera-save-name`: Name of the file the camera feed will save to if --input-type is
       "camera" and the save_path is given
    7. `--show`: Show the output
    8. `--config`: Path to config file which specifies arguments for this script if you choose 
       to use a config file instead of or with the other command line arguments
    '''
    parser = argparse.ArgumentParser(description='Detect objects in images or videos')
    parser.add_argument('--input-type',         type=str,   default=None,   help='Input type ("media" or "camera")', 
                        metavar='TEXT')
    parser.add_argument('--confidence',         type=float, default=None,   help='Confidence threshold for predictions', 
                        metavar='FLOAT')
    parser.add_argument('--media-paths',        type=str,   default=None,   help='List of media paths to predict on if --input-type is "media"', 
                        metavar='FILE', nargs='+')
    parser.add_argument('--camera-index',       type=int,   default=None,   help='Camera index if --input-type is "camera"',
                         metavar='INT')
    parser.add_argument('--save-path',          type=str,   default=None,   help='Path to save directory. Will not save if not specified.', 
                        metavar='DIR')
    parser.add_argument('--camera-save-name',   type=str,   default=None,   help='Name of the file the camera feed will save to if --input-type is "camera" and the save_path is given', 
                        metavar='TEXT')
    parser.add_argument('--show',               type=bool,  default=None,   help='Show the output', 
                        metavar='BOOL')
    parser.add_argument('--config',             type=str,   default=None,   help='Path to config file which specifies arguments for this script', 
                        metavar='FILE')
    parser.add_argument('--model-path',         type=str,   default=None,   help='Path to the model file', 
                        metavar='FILE')
    args = parser.parse_args()
    return args

def main():
    '''
    Main function to run a trained object detection model on either a list of media files 
    in a directory or a camera feed. The script will default to using a camera feed. If 
    you choose to use a config file, you can specify the arguments for this script in the
    'detect' section. If any arguments are specified in the command line, they will
    override the arguments in the config file.

    Below is an example of a config file:
    ```
    detect:
        confidence: 0.7
        media_paths: ['../test_data/vids/vid1.mp4', '../test_data/pascalvoc_pairs/3277.jpg']
        camera_index: 0
        save_path: '../test'
        show: True
    ```

    Below is the usage of the command line arguments:

    ```
    usage: main.py [-h] [--input-type TEXT] [--confidence FLOAT]
               [--media_paths FILE [FILE ...]] [--camera-index INT]
               [--save_path DIR] [--show BOOL] [--config FILE]
    ```
    '''
    args = parse_args()

    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
        detect = config['detect']
        for key in detect:
            if key in args and getattr(args, key) is None:
                setattr(args, key, detect[key]) 

    model = YOLO(args.model_path)
    detector = Detector(model=model)
    detector.detect(args.input_type, 
                    save_path=args.save_path, 
                    media_paths=args.media_paths, 
                    camera_index=args.camera_index, 
                    camera_save_name=args.camera_save_name, 
                    confidence=args.confidence, 
                    show=args.show)

if __name__ == "__main__":
    main()