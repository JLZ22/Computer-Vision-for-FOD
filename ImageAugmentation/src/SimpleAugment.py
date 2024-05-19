# Original code "https://imgaug.readthedocs.io/en/latest/source/examples_basics.html"
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import cv2
import xml.etree.ElementTree as ET
import json 
from pascal_voc_writer import Writer
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import os
import multiprocessing
from multiprocessing import pool
import time
import psutil
from tqdm import tqdm

def print_red(text):
    print("\033[91m{}\033[0m".format(text))

def print_green(text):
    print("\033[92m{}\033[0m".format(text))

class SimpleAugSeq:
    def __init__(self, path: str, 
                 save_path: str, 
                 seed: int, 
                 num_copies: int, 
                 names: list, 
                 processes=1, 
                 check=True,
                 printSteps=False,
                 checkMem=False) -> None:
        self.path = path # read path
        self.save_path = save_path # save path
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
        if (self.path[-1] != '/'):
            self.path += '/' 
        if (self.save_path[-1] != '/'):
            self.save_path += '/'
        #Checks if the array that was passed in has a length of 0. If so it populates names array with every image name from read path
        if len(self.names) == 0:
            self.names = self.getFileNames()
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    # Return an array of copies of the image stored at 
    # path/img. The array has num_copies number of copies.
    def make_copies_images(self, img: str) -> np.array:
        return np.array(
            [cv2.imread(os.path.join(self.path, img)) for _ in range(self.num_copies)],
            dtype=np.uint8
        )

    # Make num_copies number of the bbs object and return it 
    # in an array
    def make_copies_bboxes(self, bbs: BoundingBoxesOnImage) -> np.array:
        return [bbs for _ in range(self.num_copies)]
    
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
            # Scale/zoom them, translate/move them, rotate them and shear them.
            iaa.Affine(

                # zoom in or out
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, 
                
                # horizontal and vertical shifts
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # TODO: explore changing the fill color to something that looks like the work station 

                #horizontal and vertical distortion
                shear=(-8, 8)
            )
            ], 
            random_order=True) # apply augmenters in random order

    # Return a BoundingBoxesOnImage object with the 
    # given root and shape by automatically creating a 
    # new BoundingBox object for every object 
    # in the root
    def create_bbs(self, root, shape: int) -> BoundingBoxesOnImage:
        bboxes = []
        for member in root.findall('object'):
            xmin = int(float(member[4][0].text))
            ymin = int(float(member[4][1].text))
            xmax = int(float(member[4][2].text))
            ymax = int(float(member[4][3].text))
            bboxes.append(BoundingBox(x1=xmin, y1=ymin, x2=xmax, y2=ymax, label=member[0].text))
        return BoundingBoxesOnImage(bboxes, shape)
    
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
            curr_path = os.path.join(self.save_path, original_name)
            img_path = curr_path + '_aug_' + str(i) + '.jpg'
            xml_path = curr_path + '_aug_' + str(i) + '.xml'
            cv2.imwrite(img_path, imgs[i])
            writer = Writer(img_path, height, width)
            for box in bbss[i]:
                writer.addObject(box.label, box.x1, box.y1, box.x2, box.y2)

            writer.save(xml_path)

    # The primary function in charge of 
    # This function creates the process that are each in charge of augmenting one image
    def augment(self):
        # Prints conformation of read and write path
        print(f"Read Location: \"{self.path}\"")
        print(f"Save Location: \"{self.save_path}\"")
        print(f"Num Copies:   {self.num_copies}")
        print(f"Num Processes:   {self.processes}")
        
        #Requires user approval to start work
        if self.check:
            try:
                input("Press Enter to start augmenting images...")
            except SyntaxError or KeyboardInterrupt:
                print_red("Failed to augment images.")
                exit()

        start = time.time()
        mem = 0
        max_used = 0
        max_percent = 0.0

        # init pool and assign work 
        with multiprocessing.Pool(processes=self.processes) as pol:
            async_results = []
            for name in self.names:
                if name[-4] == '.':
                    name = name[:-4]
                print(f'Augmenting {name}') if self.printSteps else None
                async_results.append(pol.apply_async(self.augstart, (name,)))

            # Display progress bar
            for async_result in tqdm(async_results, desc="Processing", total=len(async_results)):
                if self.checkMem: 
                    tempPoolMem = self.get_mem_consumption()
                    tempMaxUsed = psutil.virtual_memory().used
                    tempMaxPercent = psutil.virtual_memory().percent
                    max_percent = tempMaxPercent if tempMaxPercent > max_percent else max_percent
                    max_used = tempMaxUsed if tempMaxUsed > max_used else max_used
                    mem = tempPoolMem if tempPoolMem > mem else mem
                async_result.get()
                time.sleep(0.1)

            # check memory consumption and system memory usage 
            # until all processes are done
            # while True:
            #     if all([result.ready() for result in async_results]):
            #         break
            #     time.sleep(1)
        
        end = time.time()
        self.duration = end - start
        if self.checkMem:
            print(f"Max Memory Consumption of Pool: {mem / 1024**2} MB")
            print(f"Max System Memory Used: {max_used / 1024**2} MB")
            print(f"Max System Memory Percent Used: {max_percent}%")
        print(f"Time to Augment: {self.duration} seconds")        
    
    def get_mem_consumption(self):
        pid = os.getpid()
        children = psutil.Process(pid).children(recursive=True)
        return sum([child.memory_info().rss for child in children])
        
    def resizeAndReplace(self, img, width: int, height: int, bbs: BoundingBoxesOnImage):
        seq = iaa.Sequential([
            iaa.Resize({"height": height, "width": width})
        ])

        resizedImage, newBbs = seq(images=[img], bounding_boxes = [bbs])
        # TODO: save the new image and xml file

    # This function is the worker function and 
    # augments the image of name: "name" at 
    # save path and the coresponding xml file
    def augstart(self, name: str):
        tree = ET.parse(self.path + name + '.xml') 
        root = tree.getroot()
        images = self.make_copies_images(name+'.jpg') # make num_copies number of copies of the current image 
        bbs = self.create_bbs(root, images[0].shape) # create the BoundingBoxesOnImage object for the current image
        allbbs = self.make_copies_bboxes(bbs) # make num_copies number of copies of the current image's corresponding xml file

        seq = self.create_sequential() # create the sequential object in charge of the augmentation

        images_aug, bbs_aug = seq(images=images, bounding_boxes=allbbs)
        height = int(root.find("size")[0].text)
        width = int(root.find("size")[1].text)
        self.save_aug_pairs(images_aug, bbs_aug, name, height, width)

    #gets all file names in the directory that end in .jpg
    def getFileNames(self):
        names = []
        names_Without = []
        #Populates the names array with every file name ending in .jpg from the path
        names = [f for f in os.listdir(self.path) if f.endswith('.jpg')]
        #removes the .jpg from the end of each name in the names array. The .jpg may be added back in later areas but only on a need basis.
        for f in names:
            names_Without.append(f[:-4])

        return names_Without
    
    def deleteFiles(self, path):
        for f in os.listdir(path):
            if os.path.isfile(os.path.join(path, f)):
                os.remove(os.path.join(path, f))
        print_green(f"Deleted all files in the directory: '{path}'")
    
if __name__ == '__main__':
    path = ''
    save_path = ''
    json_path = os.path.join('..','config.json')
    file_names = []

    path = '../test_data/raw/'
    save_path = '../test_data/aug/'
    path=os.path.abspath(path)
    save_path=os.path.abspath(save_path)
    # with open(json_path) as f:
    #     d = json.load(f)
    #     path = d["path"]
    #     save_path = d["save_path"]

    print(f"Available Physical Memory of System: {psutil.virtual_memory().available / 1024**2}MB")
    print(f"Total Physical Memory of System: {psutil.virtual_memory().total / 1024**2}MB")
    print(f"Percentage of Physical Memory Used: {psutil.virtual_memory().percent}%")
    simple_aug = SimpleAugSeq(path=path, 
                              save_path=save_path, 
                              seed=1, 
                              check=False,
                              num_copies=64, 
                              names=file_names,
                              processes=2,
                              checkMem=True) # 14 optimal for ecn-dec01 
    simple_aug.deleteFiles(save_path)
    simple_aug.augment()