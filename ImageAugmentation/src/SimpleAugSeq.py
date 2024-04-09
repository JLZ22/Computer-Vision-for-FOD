#Original code "https://imgaug.readthedocs.io/en/latest/source/examples_basics.html"
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import imageio as il
import os 
import xml.etree.ElementTree as ET
from pascal_voc_writer import Writer
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

class SimpleAugSeq:
    def __init__(self, path: str, save_path: str, seed: int, num_copies: int , names: list) -> None:
        self.path = path
        self.save_path = path
        self.seed = seed
        self.num_copies = num_copies
        self.names = names
        ia.seed(self.seed)

    # Return an array of copies of the image stored at 
    # path/img. The array has num_copies number of copies.
    def make_copies_images(self, img: str, num_copies: int) -> np.array:
        return np.array(
            [il.imread(path + img) for _ in range(int)],
            dtype=np.uint8
        )

    # Make num_copies number of the bbs object and return it 
    # in an array
    def make_copies_bboxes(self, bbs: BoundingBoxesOnImage, num_copies: int) -> np.array:
        return [bbs for _ in range(num_copies)]
    
    def create_sequential(self) -> iaa.Sequential:
        return iaa.Sequential([  #randomly transforms the image
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
            ], 
            random_order=True) # apply augmenters in random order

    # Return a BoundingBoxesOnImage object with the 
    # given root and shape by automatically creating a 
    # new BoundingBox object for every object 
    # in the root
    def create_bbs(self, root, shape: int) -> BoundingBoxesOnImage:
        bboxes = []
        for member in root.findall('object'):
            xmin = int(member[4][0].text)
            ymin = int(member[4][1].text)
            xmax = int(member[4][2].text)
            ymax = int(member[4][3].text)
            bboxes.append(BoundingBox(x1=xmin, y1=ymin, x2=xmax, y2=ymax))
        return BoundingBoxesOnImage(bboxes, shape)
    
    # Save the images and corresponding xml files 
    # 
    # Naming convention: name + '_synth_' + i + '.file_extension'
    # e.g. The third synthetic image of 18.jpg will be called
    #      `18_synth_2.jpg`. Its xml file will have the same name.
    def save_aug_pairs(self, 
                       imgs: np.array, 
                       bbss: np.array, 
                       original_name: str, 
                       height, 
                       width, 
                       class_name) -> None:
        for i in range(imgs.shape[0]):
            curr_path = self.save_path + original_name
            img_path = curr_path + '.jpg'
            xml_path = curr_path + '.xml'
            il.imwrite(img_path, imgs[i])
            writer = Writer(img_path, height, width)
            for box in bbss[i]:
                writer.addObject(class_name, box.x1, box.y1, box.x2, box.y2)

            writer.save(xml_path)

    # The primary function in charge of 
    # augmenting everything
    def augment(self):
        # augments every image in the list given to the constructor 
        for name in self.names:
            # get the tree for the current xml file as well as its root
            tree = ET.parse(path + name + '.xml') 
            root = tree.getroot()

            images = self.make_copies(name+'.jpg', self.num_copies) # make num_copies number of copies of the current image 
            bbs = self.create_bbs(root, images[0].shape) # create the BoundingBoxesOnImage object for the current image
            allbbs = self.make_copies_bboxes(bbs, self.num_copies) # make num_copies number of copies of the current image's corresponding xml file

            seq = self.create_sequential() # create the sequential object in charge of the augmentation

            images_aug, bbs_aug = seq(images=images, bounding_boxes=allbbs)
                
            #bbs_aug = bbs_aug.remove_out_of_image().clip_out_of_image() #This is an attempt to fix broken bounding boxes Current issue is list has no attribute 'remove_out_of_image()'
            height = int(root.find("size")[0].text)
            width = int(root.find("size")[1].text)
            class_name = str(root.find('object')[0].text)
            self.save_aug_pairs(images_aug, bbs_aug, name, height, width, class_name)


if __name__ == '__main__':
    # path on John laptop
    path = "smb://ecn-techwin.ecn.purdue.edu/Research/PLM/Restricted/Research/Project Folders/Active/ADT - Assembly Digital Thread/FOD/Images/pencils/Synthetic images/"
    save_path = "smb://ecn-techwin.ecn.purdue.edu/Research/PLM/Restricted/Research/Project Folders/Active/ADT - Assembly Digital Thread/FOD/Images/pencils/Imgaug images/"

    # path on Luca workstation
    # path = "Z:\Restricted\Research\Project Folders\Active\ADT - Assembly Digital Thread\FOD\Images\pencils\Synthetic images\\"
    # save_path = "Z:\Restricted\Research\Project Folders\Active\ADT - Assembly Digital Thread\FOD\Images\pencils\Imgaug images\\"

    simple_aug = SimpleAugSeq(path, save_path, 1, 64, ['18', '19', '20'])
    simple_aug.augment()