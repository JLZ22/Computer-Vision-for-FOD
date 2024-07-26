import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import cv2
import xml.etree.ElementTree as ET
from pascal_voc_writer import Writer
import os
import multiprocessing
import time
import psutil
from tqdm import tqdm
import gc
from pathlib import Path
import Utils

class SimpleAugSeq:
    def __init__(self, read_path: Path, 
                 save_path: Path, 
                 num_copies: int, 
                 seed = 1, 
                 names = [], 
                 processes=1, 
                 check=True,
                 printSteps=False,
                 checkMem=False) -> None:
        self.read_path = Path(read_path) # read path
        self.save_path = Path(save_path) # save path
        self.seed = seed # seed for random augmentation generation
        self.num_copies = num_copies # number of augmented copies per original image
        self.names = names # an array of the names of the images to augment 
                           # excluding the file extension
        self.processes = processes # number of processes to use for multiprocessing
        self.check = check # true if user confirmation is required to start augmenting
        self.printSteps = printSteps # true if the steps of the augmentation process
                                     # should be printed
        self.duration = -1 # time taken to augment all images
        self.checkMem = checkMem # true if memory consumption should be checked
        ia.seed(self.seed)
        #Checks if the array that was passed in has a length of 0. If so it populates names array with every image name from read path
        if len(self.names) == 0:
            self.names = [i.stem for i in Utils.get_jpg_paths(self.read_path)]
        if not self.save_path.exists() or not self.save_path.is_dir():
            os.makedirs(self.save_path)
    
    # Return a Sequential object that is in charge of
    # augmenting the image
    def create_sequential(self) -> iaa.Sequential:
        return iaa.Sequential([  #randomly transforms the image
            iaa.Fliplr(0.5), # mirror image horizontally 50% of the time 

            iaa.Flipud(0.5), # mirror image vertically 50% of the time

            iaa.Crop(percent=(0, 0.1)), # random crops

            # Small gaussian blur with random sigma between 0 and 0.5.
            # But we only blur about 50% of all images.
            iaa.Sometimes(
                0.5,
                iaa.GaussianBlur(sigma=(0, 0.5))
            ),

            # Strengthen or weaken the contrast in each image.
            iaa.LinearContrast((0.75, 1.5)),

            # Add gaussian noise.
            # For 50% of all images, we sample the noise once per pixel.
            # For the other 50% of all images, we sample the noise per pixel AND
            # channel. This can change the color (not only brightness) of the
            # pixels.
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),

            # Make some images brighter and some darker.
            # In 20% of all cases, we sample the multiplier once per channel,
            # which can end up changing the color of the images.
            iaa.Multiply((0.8, 1.2), per_channel=0.2),

            # Apply affine transformations to each image.
            # Scale/zoom them, translate/move them.
            iaa.Affine(

                # zoom in or out
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, 
                
                # horizontal and vertical shifts
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            )
            ], 
            random_order=True) # apply augmenters in random order
    
    # Save the images and corresponding xml files 
    # 
    # Naming convention: name + '_aug_' + i + '.file_extension'
    # e.g. The third synthetic image of 18.jpg will be called
    #      `18_aug_2.jpg`. Its xml file will have the same name.
    def save_aug_pairs(self, 
                       imgs: np.array, 
                       bbss: np.array, 
                       original_name: str, 
                       height, 
                       width) -> None:
        print('Saving images and xml files for ' + original_name) if self.printSteps else None
        for i in range(imgs.shape[0]):
            img_path = str(self.save_path / (original_name + '_aug_' + str(i+1) + '.jpg'))
            xml_path = str(self.save_path / (original_name + '_aug_' + str(i+1) + '.xml'))
            cv2.imwrite(img_path, imgs[i])
            writer = Writer(img_path, width=width, height=height)
            for box in bbss[i]:
                writer.addObject(box.label, box.x1, box.y1, box.x2, box.y2)
            writer.save(xml_path)

    # The primary function in charge of 
    # This function creates the processes that are 
    # each in charge of augmenting one image
    def augment(self):
        # Prints conformation of read and write path
        print(f"Starting Augmentation...")
        print(f"\tRead Location: \"{self.read_path}\"")
        print(f"\tSave Location: \"{self.save_path}\"")
        print(f"\tNum Copies:   {self.num_copies}")
        print(f"\tNum Processes:   {self.processes}")
        
        #Requires user approval to start work
        if self.check:
            try:
                input("Press Enter to start augmenting images...")
            except SyntaxError or KeyboardInterrupt:
                Utils.print_red("Failed to augment images.")
                exit()

        start = time.time()
        mem = 0
        max_used = 0
        max_percent = 0.0

        # init pool and assign work 
        with multiprocessing.Pool(processes=self.processes) as pol:
            async_results = []
            for name in self.names:
                # if name contains the file extension, remove it
                if '.' in name:
                    name = name.split('.')[0]
                print(f'Augmenting {name}') if self.printSteps else None
                async_results.append(pol.apply_async(self.augstart, (name,)))

            # Display progress bar
            for async_result in tqdm(async_results, desc="Augmenting", total=len(async_results)):
                if self.checkMem: 
                    tempPoolMem = Utils.get_children_mem_consumption()
                    tempMaxUsed = psutil.virtual_memory().used
                    tempMaxPercent = psutil.virtual_memory().percent
                    max_percent = tempMaxPercent if tempMaxPercent > max_percent else max_percent
                    max_used = tempMaxUsed if tempMaxUsed > max_used else max_used
                    mem = tempPoolMem if tempPoolMem > mem else mem
                async_result.get()
                time.sleep(0.1)
        
        end = time.time()
        Utils.cut_off_bboxes_in_directory(self.save_path, progress=False)
        Utils.subtract_mean_in_directory(self.save_path, progress=False)
        self.duration = end - start
        if self.checkMem:
            print(f"Max Memory Consumption of Pool: {mem / 1024**2} MB")
            print(f"Max System Memory Used: {max_used / 1024**2} MB")
            print(f"Max System Memory Percent Used: {max_percent}%")
        print(f"Time to Augment: {self.duration} seconds")        

    # This function is the worker function and 
    # augments the image of name: "name" at 
    # save path and the coresponding xml file
    def augstart(self, name: str):
        tree = ET.parse(str(Path(self.read_path, name + '.xml'))) 
        root = tree.getroot()
        # make num_copies number of copies of the current image 
        images = Utils.make_copies_images(str(self.read_path / (name + '.jpg')), self.num_copies) 
        # create the BoundingBoxesOnImage object for the current image
        bbs = Utils.create_bbs(root, images[0].shape) 
        # make num_copies number of copies of the current image's corresponding xml file
        allbbs = Utils.make_copies_bboxes(bbs, self.num_copies) 

        seq = self.create_sequential() # create the sequential object in charge of the augmentation

        images_aug, bbs_aug = seq(images=images, bounding_boxes=allbbs)
        height = int(root.find("size").find("height").text)
        width = int(root.find("size").find("width").text)
        self.save_aug_pairs(images_aug, bbs_aug, name, height, width)
        # clean up arrays 
        del images
        del bbs
        del allbbs
        del images_aug
        del bbs_aug
        gc.collect()