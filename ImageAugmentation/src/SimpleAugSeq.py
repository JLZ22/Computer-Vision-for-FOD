#Original code 'https://imgaug.readthedocs.io/en/latest/source/examples_basics.html'
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import cv2
import xml.etree.ElementTree as ET
import json 
import os
from pascal_voc_writer import Writer
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from halo import Halo

def print_red(text):
    print('\033[91m{}\033[0m'.format(text))

def print_green(text):
    print('\033[92m{}\033[0m'.format(text))

class SimpleAugSeq:
    def __init__(self, path: str, save_path: str, seed: int, num_copies: int , names: list) -> None:
        self.path = path
        self.save_path = save_path
        self.seed = seed
        self.num_copies = num_copies
        self.names = names
        ia.seed(self.seed)

    # Return an array of copies of the image stored at 
    # path/img. The array has num_copies number of copies.
    def make_copies_images(self, img: str) -> np.array:
        path = os.path.abspath(img)
        return np.array(
            [cv2.imread((path)) for _ in range(self.num_copies)],
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
                scale={'x': (0.8, 1.2), 'y': (0.8, 1.2)}, 
                
                # horizontal and vertical shifts
                translate_percent={'x': (-0.2, 0.2), 'y': (-0.2, 0.2)}, # TODO: explore changing the fill color to something that looks like the work station 

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
            bboxes.append(BoundingBox(x1=xmin, y1=ymin, x2=xmax, y2=ymax))
        return BoundingBoxesOnImage(bboxes, shape)
    
    # Save the images and corresponding xml files 
    # 
    # Naming convention: name + '_aug_' + i + '.file_extension'
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
            img_path = curr_path + '_aug_' + str(i) + '.jpg'
            xml_path = curr_path + '_aug_' + str(i) + '.xml'
            cv2.imwrite(img_path, imgs[i])
            writer = Writer(img_path, height, width)
            for box in bbss[i]:
                writer.addObject(class_name, box.x1, box.y1, box.x2, box.y2)

            writer.save(xml_path)

    # The primary function in charge of 
    # augmenting everything
    def augment(self):
        # augments every image in the list given to the constructor 
        read_path_is_dir = os.path.isdir(self.path)
        save_path_is_dir = os.path.isdir(self.save_path)
        print(f'Read Location: \"{self.path}\"')
        if read_path_is_dir:
            print_green(f'Read location is a directory.')
        else:
            print_red(f'Read location is not a directory. Exiting.')
            exit(1)

        print(f'Save Location: \"{self.save_path}\"')
        if save_path_is_dir:
            print_green(f'Save location is a directory.')
        else:
            print_red(f'Save location is not a directory. Exiting')
            exit(1)

        print(f'Num Compies:   {self.num_copies}')
        if self.num_copies < 0:
            print_red('Invalid number of copies.')
            exit(1)

        proceed = input('Type \"y\" to proceed. ')
        print('---------------------------------')
        if (proceed.lower() != 'y'):
            print_red('Failed to confirm augmentation.')
            exit()
        for name in self.names:
            with Halo(text=f'Augmenting \"{name}.jpg/xml\"\'', spinner='dots'):
                img_path = name + '.jpg'
                xml_path = name + '.xml'
                if (not os.path.exists(self.path + img_path)):
                    print_red(f'\n\"{img_path}\" does not exist in the read directory. Continuing to next data point.')
                    print('---------------------------------')
                    continue
                if (not os.path.exists(self.path + xml_path)):
                    print_red(f'\n{xml_path} does not exist in the read directory. Continuing to next data point.')
                    print('---------------------------------')
                    continue

                img_path = self.path + img_path
                xml_path = self.path + xml_path

                # get the tree for the current xml file as well as its root
                tree = ET.parse(xml_path) 
                root = tree.getroot()

                images = self.make_copies_images(img_path) # make num_copies number of copies of the current image 
                bbs = self.create_bbs(root, images[0].shape) # create the BoundingBoxesOnImage object for the current image
                allbbs = self.make_copies_bboxes(bbs) # make num_copies number of copies of the current image's corresponding xml file

                seq = self.create_sequential() # create the sequential object in charge of the augmentation

                images_aug, bbs_aug = seq(images=images, bounding_boxes=allbbs)
                
                #bbs_aug = bbs_aug.remove_out_of_image().clip_out_of_image() #This is an attempt to fix broken bounding boxes Current issue is list has no attribute 'remove_out_of_image()'
                height = int(root.find('size')[0].text)
                width = int(root.find('size')[1].text)
                class_name = str(root.find('object')[0].text)
                # self.save_aug_pairs(images_aug, bbs_aug, name, height, width, class_name)
            print_green(f'Saved {self.num_copies} augmented versions of {name} as {name}_aug_<copy#>.jpg/xml')
            print('---------------------------------')

if __name__ == '__main__':
    read_path = ''
    save_path = ''
    import os
    json_path = os.path.join('..','config.json')
    file_names = []
    for aut in range(3276):
        if aut >= 3250:
            file_names.append(str(aut+1))

    with open(json_path) as f:
        d = json.load(f)
        read_path = d['path']
        save_path = d['save_path']
    
    simple_aug = SimpleAugSeq(path=read_path, 
                              save_path=save_path, 
                              seed=1, 
                              num_copies=30, 
                              names=['18', '19', '333dsffssfsddsifsifjis', '20']) 
    simple_aug.augment()
