from ultralytics import YOLO
from typing import Generator
import cv2
import math
from pathlib import Path

class Detector:
    '''
    Wrapper class for the YOLO model to detect objects in
    images, videos, or camera streams.
    '''

    def __init__(self, model):
        '''
        Initialize the Detector with the YOLO model.
        - - -
        `model`: The YOLO model to use for object detection.
        '''
        if model is None:
            self.model = YOLO('../models/yolov8n.pt')
        self.model = model
            
    def detect(self,  
               input_type: str,
               confidence=  0.7,
               media_paths= [],
               camera_index=      0,
               save_dir=   None,
               camera_save_name= None,
               show=        True):
        '''
        Detect objects in images, videos, or camera streams. Saves the 
        results if a save path is provided. If the input type is 'media',
        the media paths should be provided. If the input type is 'camera',
        the camera number should be provided. Can only save to mp4 format.
        If your video input is not in mp4 format, you cannot save the results.

        **TODO**: allow for multiple cameras to be used at once.
        - - -
        `input_type`:       The type of input to detect objects in.\n
        `confidence`:       The confidence threshold for the model to detect an object.\n
        `media_paths`:      A list of paths to the media files to detect objects in.\n
        `camera`:           The camera number to use for the stream.\n
        `save_dir`:        The path to save the results to. For media detection, this should
                            be a **/* directory.\n
        `camera_save_name`: The name of the video file to save the results to including the extension.\n
        `show`:             Boolean value to show the media files or not.\n
        '''
        if input_type == 'media':
            media_paths = [str(media_path) for media_path in media_paths]
            self.detect_media(confidence, media_paths, show, save_dir)
        elif input_type == 'camera':
            self.detect_camera(confidence, camera_index, show, save_dir, camera_save_name)
        else:
            raise ValueError("Invalid input type. Please choose either 'media' or 'camera'.")
    
    def detect_media(self, confidence: float, media_paths: list, show: bool, save_dir: Path):  
        '''
        Detect objects in images or videos. Highlights objects that are 
        not supposed to be in a certain space. Can only save to mp4 format.
        - - -
        `confidence`:     The confidence threshold for the model to detect an object.\n
        `media_paths`:    A list of paths to the media files to detect objects in.\n
        `show`:           Boolean value to show the media files or not.\n
        `save_dir`:      The path of the directory to save the results to.\n
        '''
        if save_dir and not save_dir.exists():
            save_dir.mkdir(parents=True)

        # loop through the media paths and detect objects in the media
        for media_path in media_paths:
            if not Path(media_path).exists():
                print(f"Invalid media path: {media_path}. Skipping this media file.")
                continue

            # read media as an image
            # if the media is not an image, read it as a video
            frame = cv2.imread(media_path)
            print('dafdfa')
            if frame is not None:
                # detect objects in the image
                results = self.model.predict(frame, 
                                                show=      False, 
                                                conf=      confidence, 
                                                stream=    True)
                # show bounding boxes and highlight objects that are not supposed to be in the space
                frame = self.interpret_frame_result(results, frame, show, 'Image', media_path)
                
                # save the results to a file
                if save_dir:
                    cv2.imwrite(str(save_dir / ('detect_' + Path(media_path).name)), frame)
            else:
                # read the media as a video
                cap = cv2.VideoCapture(media_path)
                
                # get the frame size and initialize the video writer
                if save_dir:
                    frame_size = (int(cap.get(3)), int(cap.get(4)))
                    if Path(media_path).suffix != '.mp4':
                        print(f"Cannot save the results of {media_path} to a file. Can only save to mp4 format. Performing detection only. Press 'enter' to continue.")
                        input()
                        out = None
                    else:
                        out = cv2.VideoWriter(str(save_dir / ('detect_' + Path(media_path).name)), cv2.VideoWriter_fourcc(*'mp4v'), 30.0, frame_size)
                
                # loop to read the video stream and detect objects
                while True:
                    ret, frame = cap.read()

                    # if the frame is not read, break the loop
                    if not ret:
                        break

                    # detect objects in the frame
                    results = self.model.predict(frame, 
                                                    show=      False, 
                                                    conf=      confidence, 
                                                    stream=    True)
                    
                    # show bounding boxes and highlight objects that are not supposed to be in the space
                    frame = self.interpret_frame_result(results, frame, show, 'Video', media_path)
                    
                    # save the results to a file
                    if out:
                        out.write(frame)

                    if cv2.waitKey(1) == ord('q'):
                        break
                if out:
                    out.release()
                cv2.destroyWindow(media_path)

    def detect_camera(self, 
                      confidence: float, 
                      camera: int, 
                      show: bool, 
                      save_dir: Path, 
                      save_name = None):
        '''
        Detect objects in a camera stream and highlights objects 
        that are not supposed to be in a certain space. Can 
        only save to mp4 format.
        - - -
        `confidence`: The confidence threshold for the model to detect an object.\n
        `camera`:     The camera number to use for the stream.\n
        `show`:       Boolean value to show the camera stream or not.\n
        `save_dir`:  A directory to save the video.\n
        `save_name`:  The name of the video file to save the results to including
                      the extension.\n
        '''
        # check if the save name is valid
        if save_name is None:
            save_name = f'predict_on_camera_{camera}.mp4'
        elif not save_name.endswith('.mp4'):
            raise ValueError("Can only save to mp4 format. Please provide a valid save name.")
        
        # check if the save path is valid
        if save_dir:
            save_dir = Path(save_dir)
            if not save_dir.exists():
                save_dir.mkdir(parents=True)

        # initialize the camera stream
        cap = cv2.VideoCapture(camera)
        win_name = f'Camera {camera}'

        # get the frame size and initialize the video writer
        if save_dir:
            # check if the save path exists and update save_name until it is unique
            save_path = save_dir / save_name
            i = 1
            while save_path.exists():
                save_path = save_dir / (save_path.stem + f'({i})' + save_path.suffix)
            frame_size = (int(cap.get(3)), int(cap.get(4))) 
            out = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*'mp4v'), 30.0, frame_size)

        # loop to read the camera stream and detect objects
        while True:
            ret, frame = cap.read()

            # if the frame is not read, break the loop
            if not ret:
                break

            # detect objects in the frame
            results = self.model.predict(frame, 
                                         show=      False, 
                                         conf=      confidence, 
                                         stream=    True)
            
            # show bounding boxes and highlight objects that are not supposed to be in the space
            frame = self.interpret_frame_result(results, frame, show, 'Camera', win_name)
            
            # save the results to a file
            if save_dir:
                out.write(frame)

            # break the loop if the 'q' key is pressed
            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        if save_dir:
            out.release()
        cv2.destroyWindow(win_name)

    def interpret_frame_result(self, 
                               results: Generator, 
                               frame: cv2.typing.MatLike, 
                               show: bool, 
                               input_type: str, 
                               win_name: str, 
                               roi= ((20, 40), (600, 700))):
        '''
        Shows the results of the detection on the frame and highlights objects 
        that are within a certain space.

        **TODO**: interpret the frame in the context of the FOD problem
        - - -
        `results`:      The results of the detection.\n
        `frame`:        The frame to show the results on.\n
        `show`:         Boolean value to show the frame or not.\n
        `input_type`:   The type of input the frame is from.\n
        `win_name`:     The name of the window to show the frame in.\n
        `roi`:          The region of interest to highlight objects in.\n
        '''
        # get the next result from the generator
        r = next(results, None)
        if r:
            # draw the boundary box on the frame
            frame = self.draw_box(frame, 
                                  roi[0], 
                                  roi[1], 
                                  'Assembly Boundary', 
                                  (0, 0, 255), 
                                  2, 
                                  text_color=(0, 0, 255), 
                                  text_thickness=2
            )

            # for each detected object in the frame, draw the box and label on the frame
            for box in r.boxes:
                # get the box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # get the class and confidence of the box
                conf = math.ceil((box.conf[0]*100))/100
                cls = int(box.cls[0])
                text = self.model.names[cls] + ' ' + str(conf)

                # draw the box on the frame
                frame = self.draw_box(frame, (x1, y1), (x2, y2), text)
        
        # if the input type is an image, show the image with a loop
        # video and camera streams are shown with a single frame because
        # the loop to display is outside of this function
        if input_type == 'Image' and show:
            while True:
                cv2.imshow(win_name, frame)
                if cv2.waitKey(1) == ord('q'):
                    break
            cv2.destroyWindow(win_name)
        elif show:
            cv2.imshow(win_name, frame)

        return frame

    def draw_box(self, 
                 frame:         cv2.typing.MatLike, 
                 pt1:           tuple,
                 pt2:           tuple,
                 text:          str,
                 edge_color=         (0, 255, 0),
                 edge_thickness=     1,
                 text_color=    (0, 255, 0),
                 text_thickness=1,
                 font_scale=    1.0,
                 font=          cv2.FONT_HERSHEY_SIMPLEX,
                 text_offset=   (5, 25)):
        '''
        Draw a bounding box on the frame.
        - - -
        `frame`:            The frame to draw the bounding box on.\n
        `box`:              The bounding box coordinates.\n
        `edge_color`:       The color of the bounding box.\n
        `edge_thickness`:   The thickness of the bounding box.\n
        `text_color`:       The color of the text.\n
        `text_thickness`:   The thickness of the text.\n
        `font_scale`:       The scale of the font.\n
        `font`:             The font to use for the text.\n
        `text_offset`:      The offset of the text from the top left corner of the bounding box.\n
        '''
        # draw the bounding box on the frame
        cv2.rectangle(frame, 
                        pt1, 
                        pt2, 
                        edge_color, 
                        edge_thickness
        )

        # specify text details
        org = [pt1[0] + text_offset[0], pt1[1] + text_offset[1]]

        # write the class name and confidence on the frame
        cv2.putText(frame, 
                    text, 
                    org, 
                    font, 
                    font_scale, 
                    text_color, 
                    text_thickness
        )

        return frame