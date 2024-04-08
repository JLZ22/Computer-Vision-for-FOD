#Original code "https://imgaug.readthedocs.io/en/latest/source/examples_basics.html"
#matplotlib version 3.5 and numpy version 1.2 and markupsafe version 2.0.1



import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import imageio as il
import os 
import xml.etree.ElementTree as ET
from pascal_voc_writer import Writer
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

class SimpleAugSeq:
    def __init__(self, path, save_path, seed, copies) -> None:
        self.path = path
        self.save_path = path
        self.seed = seed
        self.num_copies = num_copies
        self.contents = os.listdir(path) 
        ia.seed(self.seed)

    # Return an array of copies of the image stored at 
    # path/img. The array has num_copies number of copies.
    def make_copies(self, img: str, num_copies: int) -> np.array:
        return np.array(
            [il.imread(path + img) for _ in range(int)],
            dtype=np.uint8
        )
    
    def create_bbs(self, root, shape) -> BoundingBoxesOnImage:
        bboxes = []
        for member in root.findall('object'):
            xmin = int(member[4][0].text)
            ymin = int(member[4][1].text)
            xmax = int(member[4][2].text)
            ymax = int(member[4][3].text)
            bboxes.append(BoundingBox(x1=xmin, y1=ymin, x2=xmax, y2=ymax))
        return BoundingBoxesOnImage(bboxes, shape)
    
    def augment(self):
        for y in range(2): # Modifies 2 images starting at 18
            name = str(y+18) # name of the image/xml pair we are considering

            # get the tree for the current xml file as well as its root
            tree = ET.parse(path + name + '.xml') 
            root = tree.getroot()

            images = self.make_copies(name+'.jpg', self.num_copies) # make 64 copies of the current image and store it in a np.array
            bbs = self.create_bbs(root, image[0].shape) # create the BoundingBoxesOnImage object for the current image

# path on John laptop
path = "smb://ecn-techwin.ecn.purdue.edu/Research/PLM/Restricted/Research/Project Folders/Active/ADT - Assembly Digital Thread/FOD/Images/pencils/Synthetic images/"
save_path = "smb://ecn-techwin.ecn.purdue.edu/Research/PLM/Restricted/Research/Project Folders/Active/ADT - Assembly Digital Thread/FOD/Images/pencils/Imgaug images/"

# path on Luca workstation
# path = "Z:\Restricted\Research\Project Folders\Active\ADT - Assembly Digital Thread\FOD\Images\pencils\Synthetic images\\"
# save_path = "Z:\Restricted\Research\Project Folders\Active\ADT - Assembly Digital Thread\FOD\Images\pencils\Imgaug images\\"

# Example batch of images.
# The array has shape (32, 64, 64, 3) and dtype uint8.

for y in range(2):   #Modifies 2 images from image 18


    img = str(y+18) + ".jpg"    #changes the file name for each loop
    image = make_copies(img, 64) # make 64 copies of img and store it in a np.array


    tree = ET.parse(path + str(y+18) + '.xml') 
    root = tree.getroot() # get root object

    height = int(root.find("size")[0].text)
    width = int(root.find("size")[1].text)
    channels = int(root.find("size")[2].text)
    ref_path = str(root.find('path').text)
    ref_file = str(root.find('filename').text)
    class_name = str(root.find('object')[0].text)


    bbs = create_bbs(root, image[0].shape)
    bbox_coordinates = []
    for member in root.findall('object'):
        
        # bbox coordinates
        xmin = int(member[4][0].text)
        ymin = int(member[4][1].text)
        xmax = int(member[4][2].text)
        ymax = int(member[4][3].text)
        # store data in list
        bbox_coordinates.append([xmin, ymin, xmax, ymax])
    
    bbs = BoundingBoxesOnImage([BoundingBox(x1=arr[0], x2=arr[1], y1 = arr[2], y2 = arr[3]) for arr in bbox_coordinates], shape=image[0].shape)
    
    Allbbs = []   
    for h in range(10):   #make x copies of the original bounding boxes for transformation
        Allbbs.append(bbs)

    freq = 0.5
    sometimes = lambda aug: iaa.Sometimes(freq, aug) # initialize sometimes as a function that runs the augmentation "aug" (freq * 100)% of the time
    seq = iaa.Sequential([  #randomly transforms the image
        iaa.Fliplr(0.5), # horizontal flips
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
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-25, 25),
            shear=(-8, 8)
        )
    ], random_order=True) # apply augmenters in random order


    images_aug , bbs_aug = seq(images=image, bounding_boxes=Allbbs)  #applys the transformation to each image in the array and to the bounding boxes

    #bbs_aug = bbs_aug.remove_out_of_image().clip_out_of_image() #This is an attempt to fix broken bounding boxes Current issue is list has no attribute 'remove_out_of_image()'

    for i in range(10):
        il.imwrite(save_path + str(i+51+(y*10)) + '.jpg', images_aug[i])  #saves each transformed image
        writer = Writer(path + img, height, width)

        for bbox in bbs_aug[i]:
           writer.addObject(class_name,bbox.x2,bbox.y2,bbox.x1,bbox.y1)

        writer.save(save_path + str(i+51+(y*10))+ '.xml')

