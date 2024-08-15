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
        self.timestamp_of_entering_roi = None
        self.timestamp_of_exit_from_roi = None
    
    def update_enter_timestamp(self):
        '''
        Update the time when the object entered the roi.
        '''
        self.timestamp_of_entering_roi = time.time()

    def get_time_elapsed_in_roi(self) -> float:
        '''
        Get the time (seconds) elapsed since the object was created.
        - - -
        #####Return: `float`
        The time elapsed since the object was created.
        '''
        if self.timestamp_of_entering_roi is None:
            return 0
        return time.time() - self.timestamp_of_entering_roi
    
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
        if label is not None:
            self.label = label
        if confidence is not None:
            self.confidence = confidence
        if xyxy is not None:
            self.xyxy = xyxy

    def update_exit_timestamp(self):
        '''
        Update the time when the object was last not in the roi.
        '''
        self.timestamp_of_exit_from_roi = time.time()

    def get_time_elapsed_outside_roi(self) -> float:
        '''
        Get the time (seconds) elapsed since the object exited the roi.
        - - -
        #####Return: `float`
        The time elapsed since the object exited the roi.
        '''
        if self.timestamp_of_exit_from_roi is None:
            return 0
        return time.time() - self.timestamp_of_exit_from_roi
    
    def reset_exit_timestamp(self):
        '''
        Reset the time when the object was last not in the roi.
        '''
        self.timestamp_of_exit_from_roi = None

    def reset_enter_timestamp(self):
        '''
        Reset the time when the object entered the roi.
        '''
        self.timestamp_of_entering_roi = None