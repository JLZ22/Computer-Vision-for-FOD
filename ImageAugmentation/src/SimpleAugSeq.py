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

path = "Z:\Restricted\Research\Project Folders\Active\ADT - Assembly Digital Thread\FOD\Images\pencils\Synthetic images\\"
save_path = "Z:\Restricted\Research\Project Folders\Active\ADT - Assembly Digital Thread\FOD\Images\pencils\Imgaug images\\"

contents = os.listdir(path) 
ia.seed(1)
sometimes = lambda aug: iaa.Sometimes(0.5, aug)


# Example batch of images.
# The array has shape (32, 64, 64, 3) and dtype uint8.

for y in range(2):   #Modifies 4 images from image 17


    img = str(y+17) + ".jpg"    #changes the file name for each loop
    image = np.array(
        [il.imread(path + img) for _ in range(64)],  #opens the designated image file across 64 instances in an array
        dtype=np.uint8
    )


    tree = ET.parse(path + str(y+17) + '.xml') 
    root = tree.getroot() # get root object

    height = int(root.find("size")[0].text)
    width = int(root.find("size")[1].text)
    channels = int(root.find("size")[2].text)
    ref_path = str(root.find('path').text)
    ref_file = str(root.find('filename').text)
    class_name = str(root.find('object')[0].text)


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


    images_aug , bbs_aug = seq(images=image, bounding_boxes=bbs)  #applys the transformation to each image in the array and to the bounding boxes

    for i in range(64):
        il.imwrite(save_path + str(i+51+(y*64)) + '.jpg', images_aug[i])  #saves each transformed image
        writer = Writer(path + img, height, width)

        """for x in bbs_aug[i]:
            writer.addObject(class_name,x[0],x[2],x[3],x[4])

        writer.save(save_path + str(i+50+(y*64))+ '.xml')
"""
#ia.imshow(np.hstack(images_aug))

