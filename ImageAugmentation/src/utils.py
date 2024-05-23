from pathlib import Path
import imgaug as ia
import imgaug.augmenters as iaa
import cv2
import numpy as np
import os
from pascal_voc_writer import Writer
import xml.etree.ElementTree as ET
import  xml.dom.minidom
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

def print_red(text):
    print("\033[91m{}\033[0m".format(text))

def print_green(text):
    print("\033[92m{}\033[0m".format(text))

def get_jpg_names(path):
    return [item for item in path.iterdir() if item.suffix.lower() in {'.jpg', '.jpeg'}]

def rename(path, startIndex):
    files = get_jpg_names(path)
    files.sort()
    print(files)
    for count, filename in enumerate(files, start=startIndex):
        name = filename.stem
        oldJPG = Path(path, name + '.jpg')
        oldXML = Path(path, name + '.xml')
        newJPG = Path(path, str(count) + '.jpg')
        newXML = Path(path, str(count) + '.xml')
        print(f'{oldJPG} -> {newJPG}')
        print(f'{oldXML} -> {newXML}')
        oldJPG.rename(newJPG)
        oldXML.rename(newXML)

# delete all files in the given path
def deleteFiles(path):
    if not path.exists() or not path.is_dir():
        print_red(f"Directory: '{path}' does not exist or is not a directory.")
        return
    for f in path.iterdir():
        if f.is_file():
            f.unlink()
    print_green(f"Deleted all files in the directory: '{path}'")

def subtract_mean(image):
    # calculate per channel mean pixel values
    mean = np.mean(image, axis=(0, 1))
    # subtract the mean from the image
    image = image - mean
    return image

def get_bboxes(path, jpg_files):
    bbss = []
    
    return bbss

def pad_and_resize_all_square(path, save_path, dim, batchsize = 16, bbss = False):
    # check read and write paths
    if not path.exists() or not path.is_dir():
        print_red(f"Directory: '{path}' does not exist or is not a directory.")
        return
    # create save directory if it does not exist
    # track if we created the save directory
    save_exists = True
    if not save_path.exists() or not save_path.is_dir():
        save_exists = False
        os.mkdir(str(save_path))
    
    # augmenters
    aug = iaa.Sequential([
        iaa.PadToSquare(),
        iaa.Resize({'height' : dim, 'width' : dim})
    ])

    # read images, pad and resize
    try:
        image_files = list(path.glob('*.jpg')) + list(path.glob('*.jpeg'))
        xml_files = list(path.glob('*.xml'))
        for i in range(0, len(image_files), batchsize):
            batch = image_files[i:i+batchsize]
            images = [cv2.imread(str(image_path)) for image_path in batch]
            resized = []
            res_bbss = []
            if bbss:
                boxes = get_bboxes(path, batch)
                resized, res_bbss = aug(images=images, bounding_boxes=boxes)
            else:
                resized = aug(images=images)
            for i, img in enumerate(resized):
                cv2.imwrite(str(save_path / str('resized_' + str(batch[i].name))) , img)
            if bbss:
                ...
    except Exception as e:
        print(e)
        # if we created the save directory, delete it
        if not save_exists:
            os.rmdir(str(save_path))
