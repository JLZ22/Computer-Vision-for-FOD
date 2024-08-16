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

def parse_args() -> argparse.Namespace:
    '''
    Parse command line arguments for the object detection script. The arguments are as follows:
    - `--input-type`: Input type ("media" or "camera")
    - `--confidence`: Confidence threshold for predictions
    - `--media-paths`: List of media paths to predict on if --input-type is "media"
    - `--camera-index`: Camera index if --input-type is "camera"    
    - `--save-path`: Path to save directory. Will not save if not specified.
    - `--camera-save-name`: Name of the file the camera feed will save to if --save is given
    - `--config`: Path to yaml config file which specifies arguments for this script
    - `--no-save`: Do not save the output
    - `--save`: Save the output
    - `--no-show`: Do not show the output
    - `--show`: Show the output
    - - -
    #####Return: `argparse.Namespace`
    An object that represents the parsed arguments.
    - - -

    Below is the format of the config file if you choose to use one:

    ```
    detect:
        input_type:         TEXT
        confidence:         FLOAT
        media_paths:        FILE [FILE ...]
        camera_index:       INT
        save_dir:           DIR
        save:               BOOL
        camera_save_name:   TEXT.mp4
        show:               BOOL
        model_path:         FILE
    ```
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
    parser.add_argument('--save',                                           help='Save the output',
                        action='store_true', default=None, dest='save')
    parser.add_argument('--no-save',                                        help='Do not save the output',
                        action='store_false', default=None, dest='save')
    parser.add_argument('--save-path',          type=str,   default=None,   help='Path to save directory. Will not save if not specified.', 
                        metavar='DIR')
    parser.add_argument('--camera-save-name',   type=str,   default=None,   help='Name of the file the camera feed will save to if --save is given', 
                        metavar='TEXT')
    parser.add_argument('--show',                                           help='Show the output',
                        action='store_true', default=None, dest='show')
    parser.add_argument('--no-show',                                        help='Do not show the output',
                        action='store_false', default=None, dest='show')
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
    - - -
    ```
    usage: main.py [-h] [--input-type TEXT] [--confidence FLOAT] 
                   [--media-paths FILE [FILE ...]] [--camera-index INT] 
                   [--save] [--no-save] [--save-path DIR] [--camera-save-name TEXT] 
                   [--show] [--no-show] [--config FILE] [--model-path FILE]
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
    
    if args.save is False:
        args.save_path = None
        args.camera_save_name = None

    model = YOLO(args.model_path if args.model_path else '../models/yolov8n.pt')
    detector = Detector(model=model)
    detector.detect(input_type=args.input_type if args.input_type else 'camera', 
                    save_dir=args.save_path, 
                    media_paths=args.media_paths, 
                    camera_index=args.camera_index, 
                    camera_save_name=args.camera_save_name, 
                    confidence=args.confidence, 
                    show=args.show)

if __name__ == "__main__":
    main()