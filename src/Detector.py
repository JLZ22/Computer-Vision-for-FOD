from ultralytics.engine.results import Results, Boxes
from ultralytics import YOLO
import cv2
from pathlib import Path
import time

class Box:
    '''
    Represents an object in an image or video frame.
    '''

    def __init__(self, 
                 label: str, 
                 confidence: float, 
                 xyxy: list,
                 id: int):
        '''
        Initialize the object with the label, confidence, and bounding box coordinates.
        - - -
        `label`:       The label of the object.\n
        `confidence`:  The confidence of the object.\n
        `xyxy`:        The bounding box coordinates of the object.\n
        `id`:          The id of the object.\n
        '''
        self.label = label
        self.confidence = confidence
        self.xyxy = xyxy
        self.id = id
        self.time_created = time.time()
        self.timestamp_of_exit_from_roi = None

    def get_age(self) -> float:
        '''
        Get the time (seconds) elapsed since the object was created.
        '''
        return time.time() - self.time_created
    
    def update(self, 
               label:       str = None,
               confidence:  float = None,
               xyxy:        list = None):
        '''
        Update the object with new values. If a value is not provided,
        the current value will remain the same.
        - - -
        `label`:       The label of the object.\n
        `confidence`:  The confidence of the object.\n
        `xyxy`:        The bounding box coordinates of the object.\n
        '''
        if label:
            self.label = label
        if confidence:
            self.confidence = confidence
        if xyxy:
            self.xyxy = xyxy

    def update_exit_timestamp(self):
        '''
        Update the time when the object was last not in the roi.
        '''
        self.timestamp_of_exit_from_roi = time.time()

    def get_time_outside_roi(self) -> float:
        '''
        Get the time (seconds) elapsed since the object exited the roi.
        '''
        if self.timestamp_of_exit_from_roi is None:
            return 0
        return time.time() - self.timestamp_of_exit_from_roi
    
    def reset_exit_timestamp(self):
        '''
        Reset the time when the object was last not in the roi.
        '''
        self.timestamp_of_exit_from_roi = None

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
        self.objects_in_roi = dict()

    def detect(self,  
               input_type:          str,
               confidence=          0.7,
               media_paths=         [],
               camera_index=        0,
               save_dir=            None,
               camera_save_name=    None,
               show=                True):
        '''
        Tract objects in videos or camera streams. Saves the 
        results if a save path is provided. If the input type is 'video',
        the media paths should be provided. If the input type is 'camera',
        the camera number should be provided. Can only save to mp4 format.
        If your video input is not in mp4 format, you cannot save the results.

        **TODO**: allow for multiple cameras to be used at once.
        - - -
        `input_type`:       The type of input to detect objects in.\n
        `confidence`:       The confidence threshold for the model to detect an object.\n
        `media_paths`:      A list of paths to the media files to detect objects in.\n
        `camera`:           The camera number to use for the stream.\n
        `save_dir`:         The path to save the results to. For media detection, this should
                            be a **/* directory.\n
        `camera_save_name`: The name of the video file to save the results to including the extension.\n
        `show`:             Boolean value to show the media files or not.\n
        `iou`:              The intersection over union threshold for the model to detect an object.\n
        '''
        if input_type == 'media':
            media_paths = [str(media_path) for media_path in media_paths]
            self.detect_media(confidence, media_paths, show, save_dir)
        elif input_type == 'camera':
            self.detect_camera(confidence, camera_index, show, save_dir, camera_save_name)
        else:
            raise ValueError("Invalid input type. Please choose either 'media' or 'camera'.")
    
    def detect_media(self, 
                     confidence: float, 
                     media_paths: list, 
                     show: bool, 
                     save_dir: Path, 
                     iou: float):  
        '''
        Detect objects in images or videos. Highlights objects that are 
        not supposed to be in a certain space. Can only save to mp4 format.
        - - -
        `confidence`:     The confidence threshold for the model to detect an object.\n
        `media_paths`:    A list of paths to the media files to detect objects in.\n
        `show`:           Boolean value to show the media files or not.\n
        `save_dir`:       The path of the directory to save the results to.\n
        `iou`:            The intersection over union threshold for the model to detect an object.\n
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
                results = self.model.track( frame, 
                                            persist=   True,
                                            conf=      confidence,
                                            iou=       0.5, # default value TODO: tune if necessary
            )
                # show bounding boxes and highlight objects that are not supposed to be in the space
                frame = self.interpret_frame_result(results[0], frame, show, 'Image', media_path)
                
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
                    results = self.model.track( frame, 
                                                persist=   True,
                                                conf=      confidence,
                                                iou=       0.5, # default value TODO: tune if necessary
                    )
                    
                    # show bounding boxes and highlight objects that are not supposed to be in the space
                    frame = self.interpret_frame_result(results[0], show, 'Video', media_path)
                    
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
                      iou: float,
                      save_name = None):
        '''
        Detect objects in a camera stream and highlights objects 
        that are not supposed to be in a certain space. Can 
        only save to mp4 format.
        - - -
        `confidence`: The confidence threshold for the model to detect an object.\n
        `camera`:     The camera number to use for the stream.\n
        `show`:       Boolean value to show the camera stream or not.\n
        `save_dir`:   A directory to save the video.\n
        `iou`:        The intersection over union threshold for the model to detect an object.\n
        `save_name`:  The name of the video file to save the results to including
                      the extension.\n
        '''
        # check if the save name is valid
        if save_name is None:
            save_name = f'track_on_camera_{camera}.mp4'
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
            results = self.model.track( frame, 
                                        persist=   True,
                                        conf=      confidence,
                                        iou=       0.5, # default value TODO: tune if necessary
            )
            
            # show bounding boxes and highlight objects that are not supposed to be in the space
            frame = self.interpret_frame_result(results[0], show, 'Camera', win_name)
            
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
                               results: Results, 
                               show: bool, 
                               input_type: str, 
                               win_name: str, 
                               roi= [50, 50, 50, 50]):
        '''
        Shows the results of the detection on the frame and highlights objects 
        that are within a certain space.

        **TODO**: highlight objects that are in the roi after a certain duration.
        - - -
        `results`:      The results of the detection.\n
        `frame`:        The frame to show the results on.\n
        `show`:         Boolean value to show the frame or not.\n
        `input_type`:   The type of input the frame is from.\n
        `win_name`:     The name of the window to show the frame in.\n
        `roi`:          The region of interest to highlight objects in.\n
        '''
        # get the boxes to highlight
        to_highlight = set()
        boxes = results.boxes
        try:
            for i in range(boxes.xyxy.shape[0]):
                # create a box object from the results for ease of access
                cls = boxes.cls[i]
                conf = boxes.conf[i]
                xyxy = boxes.xyxy[i]
                id = boxes.id[i]

                if id in self.objects_in_roi:
                    curr_obj = self.objects_in_roi[id]
                    if self.is_object_in_roi(xyxy, roi):
                        # update the object with the new detection values
                        curr_obj.update(cls, conf, xyxy)
                        # reset the exit time of the object to None
                        curr_obj.reset_exit_time()

                        # add the object to the set of objects to highlight if it has been in the roi for more than 3 seconds
                        if curr_obj.get_age() > 3:
                            to_highlight.add(curr_obj)

                    # remove the object from the roi if it has been outside the roi for more than 1 second
                    elif curr_obj.get_time_outside_roi() > 1:
                        self.objects_in_roi.pop(id)

                    # update the exit timestamp of the object if it is not in the roi and the exit timestamp has not already been set
                    elif curr_obj.timestamp_of_exit_from_roi is None:
                        curr_obj.update_exit_time()
                else:
                    # add the object to the roi if it is in the roi
                    if self.is_object_in_roi(xyxy, roi):
                        self.objects_in_roi[id] = Box(cls, conf, xyxy, id)
        except:
            pass
        
        # plot the results on the frame
        frame = results.plot()

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
    
    def is_object_in_roi(self,
                      xyxy: list,
                      roi: list,
                      percentage: float = 0.5) -> bool:
        '''
        Check if the xyxy coordinates of a bounding box overlap with
        the roi by at least the percentage specified. 
        - - -
        `xyxy`:       The bounding box coordinates.\n
        `roi`:        The region of interest to check the bounding box against.\n
        `percentage`: The percentage of the bounding box that must overlap with the roi.\n
        '''
        # calculate the area of the bounding box
        box_area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])

        # calculate the area of the roi
        roi_area = (roi[2] - roi[0]) * (roi[3] - roi[1])

        # calculate the area of the intersection
        intersection = ((min(xyxy[2], roi[2]) - max(xyxy[0], roi[0])) * 
                        (min(xyxy[3], roi[3]) - max(xyxy[1], roi[1])))
        
        # calculate the area of the union
        union = box_area + roi_area - intersection

        # calculate the percentage of the intersection
        percentage_overlap = intersection / union

        return percentage_overlap >= percentage