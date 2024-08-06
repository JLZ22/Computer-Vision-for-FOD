from pathlib import Path
import imgaug.augmenters as iaa
import cv2
import numpy as np
import os
from pascal import annotation_from_xml
from pascal_voc_writer import Writer
import xml.etree.ElementTree as ET
import  xml.dom.minidom as xdm
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import psutil
import traceback
import math
import json
import shutil
from tqdm import tqdm
import random
import torch

def print_red(text):
    print("\033[91m{}\033[0m".format(text))

def print_green(text):
    print("\033[92m{}\033[0m".format(text))

def is_json_valid(json_path: Path):
    '''
    Checks if the given json path points to a valid json file.

    json_path: A pathlib.Path object which represents the path to the 
               json file in question 
    '''
    json_path = Path(json_path)
    if not json_path.exists() or not json_path.is_file() or json_path.suffix != '.json':
        print_red(f"File: '{json_path}' does not exist or is not a json file.")
        return False
    try:
        with open(json_path) as f:
            json.load(f)
    except:
        print_red(f"File: '{json_path}' is not a valid json file.")
        return False
    return True

def get_jpg_paths(dir, range=(-1, -1)):
    '''
    Get the paths to jpg files in the directory. If a range is specifiied,
    only gets the jpg paths that fall within the range inclusive. If a file's 
    name is not an integer, this function will skip it. 

    range: A tuple that specifies the range of jpg paths to retrieve.
    '''
    # check if range is valid
    if range[0] == -1 and range[1] != -1:
        print_red("Invalid range.")
        return 
    if range[0] != -1 and range[1] == -1:
        print_red("Invalid range.")
        return
    if range[0] > range[1]:
        print_red("Invalid range.")
        return
    # check path is valid
    dir = Path(dir)
    if not dir.exists() or not dir.is_dir():
        print_red(f"Directory: '{dir}' does not exist or is not a directory.")
        return
    # get all jpg files in the directory
    jpgs = list(dir.glob('*.jpg')) + list(dir.glob('*.jpeg'))
    if range == (-1, -1):
        return jpgs
    # get all jpg files in the directory within the range
    out = []
    for item in jpgs:
        try:
            index = int(item.stem)
        except ValueError:
            continue
        if index >= range[0] and index <= range[1]:
            out.append(item)
    return out

def rename_in_directory(read_dir: Path, 
                        startIndex=0,
                        prefix = '', 
                        extensions=[]):
    '''
    Rename all the jpg, xml, and txt files in the directory to the
    format prefix + index + '.jpg' and prefix + index + '.xml'. If 
    extensions are specified, it will only rename files with those 
    extensions. This function has no recursive functionality.

    read_dir:   The directory where we are renaming files.\n
    startIndex: The index that the first renamed file will be assigned.
                all following files will be named by incrementing from the 
                startIndex.\n
    prefix:     An optional prefix that will be prepended to the beginning of 
                every renamed file. \n
    extensions: A list of strings that specifies the extension(s) of the files 
                to be renamed.\n
    '''
    read_dir = Path(read_dir)
    files = get_jpg_paths(read_dir)
    for count, filename in enumerate(files, start=startIndex):
        if filename.is_dir():
            continue
        name = filename.stem
        oldTXT = Path(read_dir, name + '.txt')
        oldJPG = Path(read_dir, filename.name)
        oldXML = Path(read_dir, name + '.xml')
        newJPG = Path(read_dir, prefix + str(count) + '.jpg')
        newXML = Path(read_dir, prefix + str(count) + '.xml')
        newTXT = Path(read_dir, prefix + str(count) + '.txt')
        # check if .jpg or .jpeg is in the extensions
        if (not extensions or '.jpg' in extensions or '.jpeg' in extensions):
            if oldJPG.is_file():
                oldJPG.rename(newJPG)
                print(f'{oldJPG} -> {newJPG}')
            else:
                print_red(f"File: '{oldJPG}' does not exist. Skipping...")

        if (not extensions or '.xml' in extensions):
            if oldXML.is_file():
                oldXML.rename(newXML)
                print(f'{oldXML} -> {newXML}')
            else:
                print_red(f"File: '{oldXML}' does not exist. Skipping...")

        if (not extensions or '.txt' in extensions):
            if oldTXT.is_file():
                oldTXT.rename(newTXT)
                print(f'{oldTXT} -> {newTXT}')
            else:
                print_red(f"File: '{oldTXT}' does not exist. Skipping...")

def delete_files(read_dir: Path, 
                 recursive=False,
                 verbose=True):
    '''
    Delete all files in the directory. If it is recursive, removes all files 
    without affecting directory structure.

    read_dir:   The directory where all files will be deleted.\n
    recursive:  A boolean that determines whether or not files will be 
                removed recursively.\n
    verbose:    A boolean that determines whether or not the function will print 
                error/success messages.\n
    '''
    read_dir = Path(read_dir)
    if not read_dir.exists() or not read_dir.is_dir():
        if verbose:
            print_red(f"Directory: '{read_dir}' does not exist or is not a directory.")
        return
    for f in read_dir.iterdir():
        if f.is_file():
            f.unlink()
        if recursive:
            if f.is_dir():
                delete_files(f, recursive, False)
    if verbose:
        if recursive:
            print_green(f"Recursively deleted all files in the directory: '{read_dir}'")
        else:
            print_green(f"Deleted all files in the directory: '{read_dir}'")

def subtract_mean(image: cv2.typing.MatLike):
    '''
    Subtract the mean pixel values from the image and returns the image. 

    image: A MatLike object that represents an image.\n
    '''
    image = np.array(image)
    # calculate per channel mean pixel values
    mean = np.mean(image, axis=(0, 1))
    # subtract the mean from the image
    image = image - mean
    return image

def subtract_mean_in_directory(read_dir: Path, 
                               save_dir=None,
                               progress=True):
    '''
    Subtract the mean pixel values from all the jpg files in the directory.

    read_dir:   The directory where jpgs will be read from.\n
    save_dir:   The directory where modified jpgs will be saved.\n
    progress:   Boolean that determines whether or not a progress bar is shown.\n
    '''
    read_dir = Path(read_dir)
    if save_dir is None:
        save_dir = read_dir
    save_dir = Path(save_dir)
    save_created = False
    if not read_dir.exists() or not read_dir.is_dir():
        print_red(f"Directory: '{read_dir}' does not exist or is not a directory.")
        return
    if not save_dir.exists():
        save_dir.mkdir()
        save_created = True
    try:
        jpgs = get_jpg_paths(read_dir)
        iter = tqdm(jpgs, desc="Subtracting Mean Pixel Value") if progress else jpgs
        for jpg in iter:
            img = cv2.imread(str(jpg))
            if img is None:
                print_red(f"Failed to read image: {jpg}")
                continue
            img = subtract_mean(img)
            cv2.imwrite(str(save_dir / jpg.name), img)
    except:
        traceback.print_exc()
        if save_created:
            save_dir.rmdir()
                
def get_corresponding_bbox(read_dir: Path, jpg_path: Path):
    '''
    Creates the BoundingBoxesOnImage object from the xml file with the same name as the 
    jpg path provided. Returns the BoundingBoxesOnImage object. 

    read_dir: The directory where the xml and jpg files exist.\n
    jpg_path: The path of the jpg file for which we are finding an xml file.\n
    '''
    read_dir = Path(read_dir)
    jpg_path = Path(jpg_path)
    name = jpg_path.stem
    xml = read_dir / (name + '.xml')
    if not xml.exists():
        return None
    tree = ET.parse(str(xml))
    root = tree.getroot()
    bbs = create_bbs(root, cv2.imread(str(jpg_path)).shape)
    return bbs

def get_yolo_label_map(json_path: Path, key_is_id=True):
    '''
    Get the yolo label map from the json file.

    json_path:  The path to the json file that contains the label map.\n
    key_is_id:  A boolean that determines whether or not the key in the label map\n
    '''
    json_path = Path(json_path)
    if not is_json_valid(json_path):
        return
    label_map = {}
    with open(json_path) as f:
        categories = json.load(f)['categories']
        for category in categories:
            if key_is_id:
                label_map[category['id']] = category['name']
            else:
                label_map[category['name']] = category['id']
    return label_map

def visualize_pascalvoc_annotations_in_directory(read_dir: Path, 
                                                 save_dir: Path,
                                                 progress=True):
    '''
    Overlays the bounding boxes from the xml files on the images in the directory and 
    saves new images with the bounding boxes drawn in save_dir. The function is a modified 
    version of code found at https://piyush-kulkarni.medium.com/visualize-the-xml-annotations-in-python-c9696ba9c188. 

    read_dir:   The directory where the xml and jpg files exist.\n
    save_dir:   The directory where the new images with bounding boxes drawn will be saved.\n
    progress:   A boolean that determines whether or not a progress bar is shown.\n
    '''
    read_dir = Path(read_dir)
    save_dir = Path(save_dir)
    save_created = False
    # check read and write paths
    if not read_dir.exists() or not read_dir.is_dir():
        print_red(f"Directory: '{read_dir}' does not exist or is not a directory.")
        return
    if save_dir.exists() and not save_dir.is_dir():
        print_red(f"Directory: '{save_dir}' exists but is not a directory.")
        return
    if not save_dir.exists():
        save_dir.mkdir()
        save_created = True
    try:
        # get images and xml files
        images = list(read_dir.glob('*.jpg')) + list(read_dir.glob('*.jpeg'))
        xml = list(read_dir.glob('*.xml'))
        # assert they are the same length
        assert(len(images) == len(xml))
        iter = tqdm(images, desc="Visualizing PascalVOC Annotations") if progress else images

        for file in iter:
            filename = file.stem
            img_path = file
            xml_path = read_dir / (filename + '.xml')
            img = cv2.imread(str(img_path))
            if img is None:
                pass
            dom = xdm.parse(str(xml_path))
            root = dom.documentElement
            objects=dom.getElementsByTagName("object")

            width =  img.shape[1]
            height = img.shape[0]
            # calculate thickness of bounding boxes and font 
            # with respect to image size (the constants are manually tuned)
            boxThickness = max(width, height) / 500
            boxThickness = round(boxThickness) if round(boxThickness) > 0 else 1
            fontScale = min(width, height) / 1200
            fontThickness = math.ceil(max(width, height) / 1000)

            # get bounding boxes
            for i in range(objects.length):
                
                bndbox = root.getElementsByTagName('bndbox')[i]
                xmin = bndbox.getElementsByTagName('xmin')[0]
                ymin = bndbox.getElementsByTagName('ymin')[0]
                xmax = bndbox.getElementsByTagName('xmax')[0]
                ymax = bndbox.getElementsByTagName('ymax')[0]
                xmin_data=xmin.childNodes[0].data
                ymin_data=ymin.childNodes[0].data
                xmax_data=xmax.childNodes[0].data
                ymax_data=ymax.childNodes[0].data
                
                # draw bounding boxes
                cv2.rectangle(img,(int(float(xmin_data)),int(float(ymin_data))),
                                (int(float(xmax_data)),int(float(ymax_data))),
                                (55,255,155),
                                boxThickness)
                # add label
                label = root.getElementsByTagName('name')[i]
                label_data = label.childNodes[0].data

                cv2.putText(img, 
                            label_data, 
                            [int(float(xmin_data)) + 2, int(float(ymin_data)) - 2], 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            fontScale, 
                            (176, 38, 255), 
                            math.ceil(fontThickness))
                
            # save image with bounding boxes drawn
            cv2.imwrite(str(save_dir / (filename + '.jpg')),img)
    except:
        traceback.print_exc()
        if save_created:
            save_dir.rmdir()

def visualize_yolo_annotations_in_directory(read_dir: Path, 
                                            save_dir: Path, 
                                            json_path: Path,
                                            progress=True):
    '''
    Overlay the bounding boxes from the txt files on the images in the directory and
    save new images with the bounding boxes drawn in save_dir.

    read_dir:   The directory where the txt and jpg files exist.\n
    save_dir:   The directory where the new images with bounding boxes drawn will be saved.\n
    json_path:  The path to the json file that contains the label map.\n
    progress:   A boolean that determines whether or not a progress bar is shown.\n
    '''
    read_dir = Path(read_dir)
    save_dir = Path(save_dir)
    json_path = Path(json_path)
    save_created = False
    # check read and write paths
    if not read_dir.exists() or not read_dir.is_dir():
        print_red(f"Directory: '{read_dir}' does not exist or is not a directory.")
        return
    if save_dir.exists() and not save_dir.is_dir():
        print_red(f"Directory: '{save_dir}' exists but is not a directory.")
        return
    if not save_dir.exists():
        save_dir.mkdir()
        save_created = True
    if not is_json_valid(json_path):
        return
    
    try:
        label_map = get_yolo_label_map(json_path)
        # get images and txt files
        image_paths = get_jpg_paths(read_dir)
        txt = list(read_dir.glob('*.txt'))
        assert(len(image_paths) == len(txt))
        iter = tqdm(image_paths, desc="Visualizing YOLO Annotations") if progress else image_paths
        for img_path in iter:
            txt = Path(read_dir / (img_path.stem + '.txt'))
            img = cv2.imread(str(img_path))
            if not txt.exists() or img is None:
                continue

            with open(txt, 'r') as f:
                lines = f.readlines()
            
            width =  img.shape[1]
            height = img.shape[0]

            for line in lines:
                line = line.split()
                label = line[0]
                x = float(line[1])
                y = float(line[2])
                w = float(line[3])
                h = float(line[4])
                # calculate bounding box coordinates
                x1 = int((x - w / 2) * img.shape[1])
                y1 = int((y - h / 2) * img.shape[0])
                x2 = int((x + w / 2) * img.shape[1])
                y2 = int((y + h / 2) * img.shape[0])


                # calculate thickness of bounding boxes and font 
                # with respect to image size (the constants are manually tuned)
                boxThickness = max(width, height) / 500
                boxThickness = round(boxThickness) if round(boxThickness) > 0 else 1
                fontScale = min(width, height) / 1200
                fontThickness = math.ceil(max(width, height) / 1000)

                # draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), (55,255,155), boxThickness)
                # add label
                cv2.putText(img, 
                            label_map[int(label)], 
                            [x1 + 2, y1 - 2], 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            fontScale, 
                            (176, 38, 255), 
                            fontThickness)
                
            # save image with bounding boxes drawn
            cv2.imwrite(str(save_dir / (img_path.name)), img)

    except Exception as e:
        traceback.print_exc()
        if save_created:
            save_dir.rmdir()

def make_copies_bboxes(bbs: BoundingBoxesOnImage, num_copies: int) -> np.array:
    '''
    Make num_copies number of the bbs object and return it 
    in an array.

    bbs:        The BoundingBoxesOnImage object that will be copied.\n
    num_copies: The number of copies that will be made.\n
    TODO: change to use generator
    '''
    return [bbs for _ in range(num_copies)]

def make_copies_images(name, num_copies: int) -> np.array:
    '''
    Return an array of copies of the image stored at 
    path/img. The array has num_copies number of copies.

    name:       The path to the image.\n
    num_copies: The number of copies that will be made.\n
    TODO: change to use generator
    '''
    return np.array(
        [cv2.imread(name) for _ in range(num_copies)],
        dtype=np.uint8
    )

def create_bbs(root, shape: int) -> BoundingBoxesOnImage:
    '''
    Return a BoundingBoxesOnImage object with the
    given root and shape by automatically creating a
    new BoundingBox object for every object in the root.

    root:   The root of the xml file.\n
    shape:  The shape of the image.\n
    '''
    bboxes = []
    for member in root.findall('object'):
        bbox = member.find('bndbox')
        xmin = int(float(bbox.find('xmin').text))
        ymin = int(float(bbox.find('ymin').text))
        xmax = int(float(bbox.find('xmax').text))
        ymax = int(float(bbox.find('ymax').text))
        bboxes.append(BoundingBox(x1=xmin, y1=ymin, x2=xmax, y2=ymax, label=member.find('name').text))
    return BoundingBoxesOnImage(bboxes, shape)

def get_children_mem_consumption():
    '''
    Return the memory consumption in bytes of all children processes.
    If no children processes are found, return 0.
    '''
    pid = os.getpid()
    children = psutil.Process(pid).children(recursive=True)
    return sum([child.memory_info().rss for child in children])

def lowercase_labels_in_directory(read_dir: Path, 
                                  save_dir=None,
                                  progress=True):
    '''
    Lowercase all labels in the xml files in the directory.
    If save_dir is None, the xml files will be saved in the same directory.

    read_dir:   The directory where the xml files exist.\n
    save_dir:   The directory where the modified xml files will be saved.\n
    progress:   A boolean that determines whether or not a progress bar is shown.\n
    '''
    read_dir = Path(read_dir)
    if save_dir == None:
        save_dir = read_dir
        save_diff = False
    save_dir = Path(save_dir)
    if not read_dir.exists() or not read_dir.is_dir():
        print_red(f"Directory: '{read_dir}' does not exist or is not a directory.")
        return
    
    # get all jpg files in the directory
    jpgPaths = get_jpg_paths(read_dir)
    iter = tqdm(jpgPaths, desc="Lower Casing Labels") if progress else jpgPaths
    for img in iter:
        xml = read_dir / (img.stem + '.xml')
        if not xml.exists():
            print_red(f"XML file: '{xml}' does not exist.")
            continue
        tree = ET.parse(str(xml))
        root = tree.getroot()
        for member in root.findall('object'):
            member.find('name').text = member.find('name').text.lower()
        tree.write(str(save_dir / (img.stem + '.xml')))
        # save corresponding jpg in save_dir
        if save_diff:
            img.copy2(save_dir / img.name)

def update_jpg_path_in_xml(read_dir: Path, 
                           save_dir=None, 
                           new_path=None,
                           progress=True):
    '''
    Update the path in the xml files to new_path.
    If save_dir is None, the xml files will be saved in the same directory.
    If new_path is None, the path in the xml files will be updated to the read path.

    read_dir:   The directory where the xml files exist.\n
    save_dir:   The directory where the modified xml files will be saved.\n
    new_path:   The new path that will be updated in the xml files.\n
    progress:   A boolean that determines whether or not a progress bar is shown.\n
    '''
    read_dir = Path(read_dir)
    if save_dir == None:
        save_dir = read_dir
    if new_path == None:
        new_path = read_dir
    save_dir = Path(save_dir)
    new_path = Path(new_path)
    # check read and write paths
    if not read_dir.exists() or not read_dir.is_dir():
        print_red(f"Directory: '{read_dir}' does not exist or is not a directory.")
        return
    if not save_dir.exists() or not save_dir.is_dir():
        print_red(f"Directory: '{save_dir}' does not exist or is not a directory.")
        return
    if not new_path.exists() or not new_path.is_dir():
        print_red(f"Directory: '{new_path}' does not exist or is not a directory.")
        return
    
    # get all jpg files in the directory
    jpgPaths = get_jpg_paths(read_dir)
    iter = tqdm(jpgPaths, desc="Updating JPG Path in XML") if progress else jpgPaths

    # update the path in the xml files
    for img in iter:
        xml = read_dir / (img.stem + '.xml')
        if not xml.exists():
            print_red(f"XML file: '{xml}' does not exist.")
            continue
        tree = ET.parse(str(xml))
        root = tree.getroot()
        for member in root.findall('path'):
            member.text = str(new_path / img.name)
        tree.write(str(save_dir / (img.stem + '.xml')))
        # save corresponding jpg in save_dir
        cv2.imwrite(str(save_dir / img.name), cv2.imread(str(img)))

def aug_in_directory(read_dir: Path,
                    save_dir: Path, 
                    aug: iaa.Augmenter, 
                    includeXML=True,
                    progress=True):
    '''
    Perform the augmentation on the images in the directory
    and save the augmented images in the save_dir directory.
    If includeXML is True, the bounding boxes will be augmented.

    read_dir:   The directory where the images and xml files exist.\n
    save_dir:   The directory where the augmented images and xml files will be saved.\n
    aug:        The imgaug augmenter that will be used to augment the images.\n
    includeXML: A boolean that determines whether or not the xml files will be augmented.\n
    '''
    # check read and write paths
    read_dir = Path(read_dir)
    save_dir = Path(save_dir)
    if not read_dir.exists() or not read_dir.is_dir():
        print_red(f"Directory: '{read_dir}' does not exist or is not a directory.")
        return
    
    # create save directory if it does not exist
    save_exists = True
    if not save_dir.exists():
        save_dir.mkdir()
        save_exists = False

    # get image paths
    jpgPaths = get_jpg_paths(read_dir)

    try:
        # augment images
        iter = tqdm(jpgPaths, desc="Performing Given Augmentation") if progress else jpgPaths
        for img in iter:
            # read image and check if it is valid
            image = cv2.imread(str(img))
            if image is None:
                print_red(f"Failed to read image: {img}")
                continue

            # get corresponding bounding boxes
            bbs = get_corresponding_bbox(read_dir, img)

            # check if we should augment the bounding boxes
            hasXML = bbs is not None and includeXML

            # augment image and bounding boxes if bounding boxes exists 
            # and the user wants to include the xml files (aka bounding boxes)
            # in the augmentation 
            if hasXML:
                image, bbs = aug.augment(image=image, bounding_boxes=bbs)
                # save augmented bboxes
                height, width, _ = image.shape
                writer = Writer(str(save_dir / img.name), width=width, height=height)
                for box in bbs.bounding_boxes:
                    writer.addObject(box.label, box.x1, box.y1, box.x2, box.y2)
                writer.save(str(save_dir / (img.stem + '.xml')))
            else:
                image = aug.augment(image=image)

            # save augmented image
            cv2.imwrite(str(save_dir / img.name), image)
    except Exception as e:
        traceback.print_exc()
        # if we created the save directory, delete it
        if not save_exists:
            save_dir.rmdir()

def flip_horizontal_in_directory(read_dir, save_dir, includeXML=True):
    '''
    Flip the images in the directory horizontally and save them. 
    If includeXML is True, the bounding boxes will be flipped as well.

    read_dir:   The directory where the images and xml files exist.\n
    save_dir:   The directory where the flipped images and xml files will be saved.\n
    includeXML: A boolean that determines whether or not the xml files will be flipped.\n
    '''
    aug = iaa.Fliplr(1.0)
    aug_in_directory(read_dir, save_dir, aug, includeXML)

def flip_vertical_in_directory(read_dir, save_dir, includeXML=True):
    '''
    Flip the images in the directory vertically and save them.
    If includeXML is True, the bounding boxes will be flipped as well.

    read_dir:   The directory where the images and xml files exist.\n
    save_dir:   The directory where the flipped images and xml files will be saved.\n
    includeXML: A boolean that determines whether or not the xml files will be flipped.\n
    '''
    aug = iaa.Flipud(1.0)
    aug_in_directory(read_dir, save_dir, aug, includeXML)

def rotate_in_directory(read_dir, save_dir, angle, includeXML=True):
    '''
    Rotate the images in the directory by the given angle and save them
    without altering the aspect ratio.
    If includeXML is True, the bounding boxes will be rotated as well.

    read_dir:   The directory where the images and xml files exist.\n
    save_dir:   The directory where the rotated images and xml files will be saved.\n
    angle:      The angle by which the images will be rotated.\n
    '''
    if angle % 90 != 0:
        aug = iaa.Affine(rotate=angle)
        aug_in_directory(read_dir, save_dir, aug, includeXML)
    else:
        rotate_90_in_directory(read_dir, save_dir, angle // 90)

def rotate_90_in_directory(read_dir, save_dir, repetitions=1, includeXML=True):
    '''
    Rotate the images in the directory by 90 degrees and save them and save 
    them without altering the aspect ratio.
    If includeXML is True, the bounding boxes will be rotated as well.

    read_dir:       The directory where the images and xml files exist.\n
    save_dir:       The directory where the rotated images and xml files will be saved.\n
    repetitions:    The number of times the images will be rotated by 90 degrees.\n
    includeXML:     A boolean that determines whether or not the xml files will be rotated.\n
    '''
    aug = iaa.Rot90(repetitions)
    aug_in_directory(read_dir, save_dir, aug, includeXML)

def resize_in_directory(read_dir: Path, save_dir: Path, width=512, height=512, includeXML=True):
    '''
    Resize the images in the directory to the given width and height and save them.
    This does not guarantee that the aspect ratio will remain the same.
    If includeXML is True, the bounding boxes will be resized as well.

    read_dir:   The directory where the images and xml files exist.\n
    save_dir:   The directory where the resized images and xml files will be saved.\n
    width:      The width to which the images will be resized.\n
    height:     The height to which the images will be resized.\n
    includeXML: A boolean that determines whether or not the xml files will be resized.\n
    '''
    # augmenters
    aug = iaa.Resize({"height": height, "width": width})
    aug_in_directory(read_dir, save_dir, aug, includeXML)

def pad_and_resize_square_in_directory(read_dir: Path, 
                                       save_dir: Path, 
                                       dim=512, 
                                       includeXML=True):
    '''
    Pad the images in the directory to make them square and 
    resize them to the given dimension while maintaining aspect ratio.
    If includeXML is True, the bounding boxes will be resized as well.

    read_dir:   The directory where the images and xml files exist.\n
    save_dir:   The directory where the padded and resized images and 
                xml files will be saved.\n
    dim:        The dimension to which the images will be resized.\n
    includeXML: A boolean that determines whether or not the xml files will be resized.\n
    '''
    # augmenters
    aug = iaa.Sequential([
        iaa.PadToSquare(pad_mode="edge"),
        iaa.Resize(dim)
    ])
    aug_in_directory(read_dir, save_dir, aug, includeXML)

def copy_files_in_directory(read_dir: Path, 
                            save_dir: Path, 
                            extensions=[],
                            progress=True):
    '''
    Copy the files in the read_dir directory to the save_dir directory.

    read_dir:   The directory where the files will be copied from.\n
    save_dir:   The directory where the copied files will be saved.\n
    extensions: A list of strings that specifies the extension(s) of the files
                to be copied.\n
    progress:   A boolean that determines whether or not a progress bar is shown.\n
    '''
    read_dir = Path(read_dir)
    save_dir = Path(save_dir)
    if not read_dir.exists() or not read_dir.is_dir():
        print_red(f"Directory: '{read_dir}' does not exist or is not a directory.")
        return
    if not save_dir.exists():
        save_dir.mkdir()

    # get all files in the directory
    if extensions:
        files = [f for f in read_dir.iterdir() if f.suffix in extensions]
    else:
        files = list(read_dir.iterdir())
    
    # copy data to save_dir
    iter = tqdm(files, desc="Copying Files") if progress else files
    for f in iter:
        if f.is_file():
            shutil.copy2(f, save_dir)

def move_files_in_directory(read_dir: Path, 
                            save_dir: Path, 
                            extensions=[],
                            progress=True):
    '''
    Move the files in the read_dir directory to the save_dir directory.

    read_dir:   The directory where the files will be moved from.\n
    save_dir:   The directory where the moved files will be saved.\n
    extensions: A list of strings that specifies the extension(s) of the files
                to be moved.\n
    progress:   A boolean that determines whether or not a progress bar is shown.\n
    '''
    read_dir = Path(read_dir)
    save_dir = Path(save_dir)
    if not read_dir.exists() or not read_dir.is_dir():
        print_red(f"Directory: '{read_dir}' does not exist or is not a directory.")
        return
    if not save_dir.exists():
        save_dir.mkdir()

    # get all files in the directory
    if extensions:
        files = [f for f in read_dir.iterdir() if f.suffix in extensions]
    else:
        files = list(read_dir.iterdir())
    
    # copy data to save_dir
    iter = tqdm(files, desc="Moving Files") if progress else files
    for f in iter:
        if f.is_file():
            shutil.move(f, save_dir)

def rotate_image_and_save(img_read_path: Path, 
                          img_save_dir=None, 
                          rotateCode=cv2.ROTATE_90_CLOCKWISE):
    '''
    Rotate the image at img_read_dir by the given angle and save it in img_save_dir.
    The rotateCode is the cv2 rotate code.

    img_read_path:  The path to the image that will be rotated.\n
    img_save_dir:   The directory where the rotated image will be saved.\n
    rotateCode:     The cv2 rotate code that determines how the image will be rotated.\n
    '''
    img_read_path = Path(img_read_path)
    if img_save_dir == None:
        img_save_dir = img_read_path.parent
    img_save_dir = Path(img_save_dir)
    if not img_read_path.exists() or not img_read_path.is_file():
        print_red(f"Directory: '{img_read_path}' does not exist or is not a file.")
        return
    if not img_save_dir.is_dir():
        print_red(f"Directory: '{img_save_dir}' is not a directory.")
        return
    
    img = cv2.imread(str(img_read_path))
    if img is None:
        print_red(f"Failed to read image: {img_read_path}")
        return
    img = cv2.rotate(img, rotateCode=rotateCode)
    cv2.imwrite(str(img_save_dir / img_read_path.name), img)

def rotate_image_and_save_in_directory(read_dir: Path, 
                                       save_dir=None, 
                                       rotateCode=cv2.ROTATE_90_CLOCKWISE, 
                                       range=(-1, -1),
                                       progress=True):
    '''
    Rotate all the images in the directory by the given rotate code 
    and save them in the save_dir directory if range is (-1, -1).
    If range is not (-1, -1), only rotate the images in the range.

    read_dir:       The directory where the images exist.\n
    save_dir:       The directory where the rotated images will be saved.\n
    rotateCode:     The cv2 rotate code that determines how the images will be rotated.\n
    range:          The range of images that will be rotated. If (-1, -1), all images will be rotated.\n
    progress:       A boolean that determines whether or not a progress bar is shown.\n
    '''
    read_dir = Path(read_dir)
    if save_dir == None:
        save_dir = read_dir
    save_dir = Path(save_dir)

    if not read_dir.exists() or not read_dir.is_dir():
        print_red(f"Directory: '{read_dir}' does not exist or is not a directory.")
        return
    if not save_dir.exists():
        save_dir.mkdir()

    jpgPaths = get_jpg_paths(read_dir, range)
    iter = tqdm(jpgPaths, desc="Rotating Images") if progress else jpgPaths
    for img in iter:
        rotate_image_and_save(img, save_dir, rotateCode)

def delete_all_xml_without_jpg(read_dir: Path, progress=True):
    '''
    Delete all xml files in the directory that do not have a corresponding jpg file.

    read_dir:   The directory where the xml and jpg files exist.\n
    progress:   A boolean that determines whether or not a progress bar is shown.\n
    '''
    read_dir = Path(read_dir)
    if not read_dir.exists() or not read_dir.is_dir():
        print_red(f"Directory: '{read_dir}' does not exist or is not a directory.")
        return
    jpgPaths = get_jpg_paths(read_dir)
    xmlPaths = list(read_dir.glob('*.xml'))
    iter = tqdm(xmlPaths, desc="Deleting Lone XMLs") if progress else xmlPaths
    for xml in iter:
        name = xml.stem
        if not any([jpg.stem == name for jpg in jpgPaths]):
            xml.unlink()

def count_files_in_directory(read_dir: Path, extensions=[]):
    '''
    Count the number of files in the directory.

    read_dir:   The directory where the files exist.\n
    extensions: A list of strings that specifies the extension(s) of the files
    '''
    read_dir = Path(read_dir)
    if not read_dir.exists() or not read_dir.is_dir():
        print_red(f"Directory: '{read_dir}' does not exist or is not a directory.")
        return
    count = 0
    for f in read_dir.iterdir():
        if extensions and f.suffix not in extensions:
            continue
        elif f.is_file():
            count += 1
    return count

def jpeg_to_jpg(read_dir: Path, progress=True):
    '''
    Rename all jpeg files in the directory to jpg.

    read_dir:   The directory where the jpeg files exist.\n
    progress:   A boolean that determines whether or not a progress bar is shown.\n
    '''
    read_dir = Path(read_dir)
    if not read_dir.exists() or not read_dir.is_dir():
        print_red(f"Directory: '{read_dir}' does not exist or is not a directory.")
        return
    jpeg = list(read_dir.glob('*.jpeg'))
    iter = tqdm(jpeg, desc="Converting JPEG to JPG") if progress else jpeg
    for img in iter:
        img.rename(read_dir / (str(img.stem) + '.jpg'))

def cut_off_bbox(xml_pth: Path):
    '''
    Cut off the bounding box in the xml file that is outside the image.

    xml_pth:    The path to the xml file.\n
    '''
    xml_pth = Path(xml_pth)
    if not xml_pth.exists() or not xml_pth.is_file() or xml_pth.suffix != '.xml':
        print_red(f"File: '{xml_pth}' does not exist or is not an xml file.")
        return
    
    tree = ET.parse(str(xml_pth))
    root = tree.getroot()
    width = int(root.find("size").find("width").text)
    height = int(root.find("size").find("height").text)
    for member in root.findall('object'):
        # get bounding box
        bbox = member.find('bndbox')
        xmin = int(float(bbox.find('xmin').text))
        ymin = int(float(bbox.find('ymin').text))
        xmax = int(float(bbox.find('xmax').text))
        ymax = int(float(bbox.find('ymax').text))

        # cut off bounding box if it is outside the image
        if xmin < 0:
            xmin = 0
        elif xmin > width:
            xmin = width

        if ymin < 0:
            ymin = 0
        elif ymin > height:
            ymin = height

        if xmax < 0:
            xmax = 0
        elif xmax > width:
            xmax = width
        
        if ymax < 0:
            ymax = 0
        elif ymax > height:
            ymax = height

        # update bounding box
        bbox.find('xmin').text = str(xmin)
        bbox.find('ymin').text = str(ymin)
        bbox.find('xmax').text = str(xmax)
        bbox.find('ymax').text = str(ymax)
        
    tree.write(str(xml_pth))
    
def cut_off_bboxes_in_directory(read_dir: Path, progress=True):
    '''
    Cut off the bounding boxes in the xml files that are outside the image.

    read_dir:   The directory where the xml files exist.\n
    progress:   A boolean that determines whether or not a progress bar is shown.\n
    '''
    read_dir = Path(read_dir)
    if not read_dir.exists() or not read_dir.is_dir():
        print_red(f"Directory: '{read_dir}' does not exist or is not a directory.")
        return
    xmlPaths = list(read_dir.glob('*.xml'))
    iter = tqdm(xmlPaths, desc="Cutting Off BBoxes") if progress else xmlPaths
    for p in iter:
        try: 
            cut_off_bbox(p)
        except:
            print_red(f"Failed to cut off bbox in file: {p}")

def pascalvoc_to_yolo(xml_path: Path, save_file_path: Path, json_path: Path):
    '''
    Convert pascal voc xml file to yolo txt file.

    xml_path:           The path to the xml file.\n
    save_file_path:     The path to the txt file where the bounding boxes will be saved.\n
    json_path:          The path to the json file that contains the label map.\n
    '''
    xml_path = Path(xml_path)
    save_file_path = Path(save_file_path)
    json_path = Path(json_path)
    save_created = False
    if not xml_path.exists() or not xml_path.is_file() or xml_path.suffix != '.xml':
        print_red(f"File: '{xml_path}' does not exist or is not an xml file.")
        return
    if not save_file_path.suffix == '.txt':
        print_red(f"File: '{save_file_path}' is not a txt file.")
        return
    elif not save_file_path.is_file():
        save_file_path.touch()
        save_created = True
    if not is_json_valid(json_path):
        return
    
    try:
        label_map = get_yolo_label_map(json_path, key_is_id=False)
        # get the bounding boxes from the xml file
        ann = annotation_from_xml(xml_path)
        # write the bounding boxes to the yolo txt file
        with open((save_file_path), 'w') as f:
            f.write(ann.to_yolo(label_map, 5))

        # move the corresponding image
        img_path = xml_path.with_suffix('.jpg')
        if img_path.exists():
            shutil.copy2(img_path, save_file_path.with_suffix('.jpg'))
    except Exception as e:
        traceback.print_exc()
        if save_created:
            save_file_path.unlink()

def pascalvoc_to_yolo_in_directory(read_dir: Path, 
                                   save_dir: Path, 
                                   json: Path,
                                   verbose=False,
                                   progress=True):
    '''
    Convert all pascal_voc xml files in the directory to yolo txt files.

    read_dir:   The directory where the xml files exist.\n
    save_dir:   The directory where the yolo txt files will be saved.\n
    json:       The path to the json file that contains the label map.\n
    verbose:    A boolean that determines whether or not the function prints messages.\n
    progress:   A boolean that determines whether or not a progress bar is shown.\n
    '''
    read_dir = Path(read_dir)
    save_dir = Path(save_dir)
    if not read_dir.exists() or not read_dir.is_dir():
        if verbose:
            print_red(f"Directory: '{read_dir}' does not exist or is not a directory.")
        return
    if not save_dir.exists():
        save_dir.mkdir()
        
    xmlPaths = list(read_dir.glob('*.xml'))
    iter = tqdm(xmlPaths, desc="Converting PascalVOC to YOLO") if progress else xmlPaths
    for xml in iter:
        try:
            pascalvoc_to_yolo(xml, save_dir / (xml.stem + '.txt'), json)
        except:
            if verbose:
                print_red(f"Failed to convert xml file: {xml}")
            continue
    if verbose:
        print_green(f"Successfully converted all xml files in the directory: '{read_dir}' to yolo txt files.")

def move_percent_of_datapoints_in_directory(read_dir: Path,
                                            save_dir: Path, 
                                            percent=0.1,
                                            random_sample=False):
    '''
    Move a percentage of the data points in the read directory 
    to the save directory. One data point consists of a jpg file and 
    a corresponding xml or txt file. 

    read_dir:       The directory where the data points exist.\n
    save_dir:       The directory where the moved data points will be saved.\n
    percent:        The percentage of data points that will be moved.\n
    random_sample:  A boolean that determines whether or not the data points will be randomly sampled.\n
    '''
    num = int(count_files_in_directory(read_dir, ['.jpg']) * percent)
    move_number_of_datapoints_in_directory(read_dir, 
                                           save_dir, 
                                           num, 
                                           random_sample)

def move_number_of_datapoints_in_directory(read_dir: Path, 
                                           save_dir: Path, 
                                           num_files=10,
                                           random_sample=False,
                                           progress=True):
    '''
    Move a number of data points in the read directory to the save directory.

    read_dir:       The directory where the data points exist.\n
    save_dir:       The directory where the moved data points will be saved.\n
    num_files:      The number of data points that will be moved.\n
    random_sample:  A boolean that determines whether or not the data points will be randomly sampled.\n
    progress:       A boolean that determines whether or not a progress bar is shown.\n
    '''
    read_dir = Path(read_dir)
    save_dir = Path(save_dir)
    save_created = False
    if not read_dir.exists() or not read_dir.is_dir():
        print_red(f"Directory: '{read_dir}' does not exist or is not a directory.")
        return
    if not save_dir.exists():
        save_dir.mkdir()
        save_created = True
    try:
        jpgPaths = get_jpg_paths(read_dir)
        if random_sample:
            jpgPaths = random.sample(jpgPaths, num_files)
        else:
            jpgPaths = jpgPaths[:num_files]
        iter = tqdm(jpgPaths, desc="Moving Datapoints") if progress else jpgPaths
        for img in iter:
            xml = read_dir / (img.stem + '.xml')
            txt = read_dir / (img.stem + '.txt')
            if xml.exists() and xml.suffix:
                shutil.move(xml, save_dir)
            if txt.exists() and txt.suffix:
                shutil.move(txt, save_dir)
            if img.exists() and img.suffix:
                shutil.move(img, save_dir)
    except:
        traceback.print_exc()
        if save_created:
            save_dir.rmdir()

def copy_percent_of_datapoints_in_directory(read_dir: Path, 
                                            save_dir: Path, 
                                            percent=0.1,
                                            random_sample=False):
    '''
    Copy a percentage of the data points in the read directory.

    read_dir:       The directory where the data points exist.\n
    save_dir:       The directory where the copied data points will be saved.\n
    percent:        The percentage of data points that will be copied.\n
    random_sample:  A boolean that determines whether or not the data points will be randomly sampled.\n
    '''
    num = int(count_files_in_directory(read_dir, ['.jpg']) * percent)
    copy_number_of_datapoints_in_directory(read_dir, 
                                           save_dir, 
                                           num, 
                                           random_sample, 
                                           )

def copy_number_of_datapoints_in_directory(read_dir: Path, 
                                           save_dir: Path, 
                                           num_files=10,
                                           random_sample=False,
                                           progress=True):
    '''
    Copy a number of data points in the read directory to the save directory.

    read_dir:       The directory where the data points exist.\n
    save_dir:       The directory where the copied data points will be saved.\n
    num_files:      The number of data points that will be copied.\n
    random_sample:  A boolean that determines whether or not the data points will be randomly sampled.\n
    progress:       A boolean that determines whether or not a progress bar is shown.\n
    '''
    read_dir = Path(read_dir)
    save_dir = Path(save_dir)
    save_created = False
    if not read_dir.exists() or not read_dir.is_dir():
        print_red(f"Directory: '{read_dir}' does not exist or is not a directory.")
        return
    if not save_dir.exists():
        save_dir.mkdir()
        save_created = True

    try:
        jpgPaths = get_jpg_paths(read_dir)
        assert(num_files <= len(jpgPaths))
        if random_sample:
            jpgPaths = random.sample(jpgPaths, num_files)
        else:
            jpgPaths = jpgPaths[:num_files]
        iter = tqdm(jpgPaths, desc="Copying Datapoints") if progress else jpgPaths
        for img in iter:
            xml = read_dir / (img.stem + '.xml')
            txt = read_dir / (img.stem + '.txt')
            if xml.exists() and xml.suffix:
                shutil.copy2(xml, save_dir)
            if txt.exists() and txt.suffix:
                shutil.copy2(txt, save_dir)
            if img.exists() and img.suffix:
                shutil.copy2(img, save_dir)
    except:
        traceback.print_exc()
        if save_created:
            save_dir.rmdir()

def split_number_datapoints_in_directory(read_dir: Path, 
                                         img_dir: Path, 
                                         ann_dir: Path,
                                         num: int,
                                         progress=True):
    '''
    Split the images and annotations in the read directory into two separate directories.

    read_dir:   The directory where the images and annotations exist.\n
    img_dir:    The directory where the images will be saved.\n
    ann_dir:    The directory where the annotations will be saved.\n
    num:        The number of images and annotations that will be split.\n
    progress:   A boolean that determines whether or not a progress bar is shown.\n
    '''
    num = int(num)
    assert(num > 0)
    read_dir = Path(read_dir)
    img_dir = Path(img_dir)
    ann_dir = Path(ann_dir)
    img_created = False
    ann_created = False

    if not read_dir.exists() or not read_dir.is_dir():
        print_red(f"Directory: '{read_dir}' does not exist or is not a directory.")
        return
    if not img_dir.exists():
        img_dir.mkdir(parents=True)
        img_created = True
    if not ann_dir.exists():
        ann_dir.mkdir(parents=True)
        ann_created = True
    try:
        jpgPaths = get_jpg_paths(read_dir)
        assert(num <= len(jpgPaths))
        jpgPaths = random.sample(jpgPaths, num)
        iter = tqdm(jpgPaths, desc="Splitting Datapoints") if progress else jpgPaths
        for img in iter:
            xml = read_dir / (img.stem + '.xml')
            txt = read_dir / (img.stem + '.txt')
            if xml.exists():
                shutil.move(xml, ann_dir)
            if txt.exists():
                shutil.move(txt, ann_dir)
            if img.exists():
                shutil.move(img, img_dir)
    except:
        traceback.print_exc()
        if img_created:
            img_dir.rmdir()
        if ann_created:
            ann_dir.rmdir()

def count_data_points_in_directory(read_dir: Path):
    '''
    Count the number of data points in the directory.

    read_dir:   The directory where the data points exist.\n
    '''
    read_dir = Path(read_dir)
    if not read_dir.exists() or not read_dir.is_dir():
        print_red(f"Directory: '{read_dir}' does not exist or is not a directory.")
        return
    return count_files_in_directory(read_dir, ['.jpg'])

def partition_yolo_data_for_training(read_dir: Path, 
                                save_dir: Path, 
                                train_percent=0.8,
                                test_percent=0.1,
                                verbose=False,
                                append=False):
    '''
    Partition the data in the read directory into training, validation, 
    and test sets according to the YOLO file structure and given percentages.
    Typically, allocating test is optional. If you would not like to partition
    a test set, set test_percent to 0.

    read_dir:       The directory where the images and labels exist.\n
    save_dir:       The directory where the partitioned data will be saved.\n
    train_percent:  The percentage of data that will be used for training.\n
    test_percent:   The percentage of data that will be used for testing.\n
    verbose:        A boolean that determines whether or not the function prints messages.\n
    append:         A boolean that determines whether or not the data will be appended to the save directory.\n
    '''
    read_dir = Path(read_dir)
    save_dir = Path(save_dir)
    save_created = False
    if not read_dir.exists() or not read_dir.is_dir():
        if verbose:
            print_red(f"Directory: '{read_dir}' does not exist or is not a directory.")
        return False
    if not save_dir.exists():
        save_dir.mkdir()
        save_created = True
    elif not append:
        delete_files(save_dir, True)

    try:
        # calculate the number of files in each set
        num_files = count_data_points_in_directory(read_dir)
        train_num = int(num_files * train_percent)
        test_num = int(num_files * test_percent)
        val_num = num_files - train_num - test_num

        split_number_datapoints_in_directory(read_dir,
                                             save_dir / 'images/val',
                                             save_dir / 'labels/val',
                                             val_num)
        
        split_number_datapoints_in_directory(read_dir,
                                             save_dir / 'images/test',
                                             save_dir / 'labels/test',
                                             test_num)
        
        split_number_datapoints_in_directory(read_dir,
                                             save_dir / 'images/train',
                                             save_dir / 'labels/train',
                                             train_num)
        
        # verify the yolo file structure
        train_after = count_data_points_in_directory(save_dir / 'images/train')
        val_after = count_data_points_in_directory(save_dir / 'images/val')
        test_after = count_data_points_in_directory(save_dir / 'images/test')
        if ((verify_yolo_file_structure(save_dir, 
                                    test_num > 0, 
                                    True)) and 
            (train_num == train_after) and
            (val_num == val_after) and
            (test_num == test_after)):
            if verbose:
                print_green(f"Successfully partitioned the data. {num_files} data points in total.")
            return True
    except:
        traceback.print_exc()
        if save_created:
            shutil.rmtree(save_dir)
        return False

def verify_yolo_file_structure(read_dir: Path, 
                               test=True,
                               verbose=False):
    '''
    Check if the yolo file structure is correct and checks if 
    the names of the images and labels in the train, val, and test
    are the same.

    read_dir:   The directory where the images and labels exist.\n
    test:       A boolean that determines whether or not the test set will be checked.\n
    verbose:    A boolean that determines whether or not the function prints messages.\n
    '''
    read_dir = Path(read_dir)
    if not read_dir.exists() or not read_dir.is_dir():
        if verbose:
            print_red(f"Directory: '{read_dir}' does not exist or is not a directory.")
        return False, None
    
    images = read_dir / 'images'
    labels = read_dir / 'labels'
    if not images.is_dir():
        if verbose:
            print_red(f"Directory: '{images}' does not exist or is not a directory.")
        return False, None
    if not labels.is_dir():
        if verbose:
            print_red(f"Directory: '{labels}' does not exist or is not a directory.")
        return False, None
    
    img_test = images / 'test'
    img_train = images / 'train'
    img_val = images / 'val'

    lab_test = labels / 'test'
    lab_train = labels / 'train'
    lab_val = labels / 'val'

    if not img_train.is_dir():
        if verbose:
            print_red(f"Directory: '{img_train}' does not exist or is not a directory.")
        return False, None
    if not img_val.is_dir():
        if verbose:
            print_red(f"Directory: '{img_val}' does not exist or is not a directory.")
        return False, None
    
    train_diff = diff_names_between_directories(img_train, lab_train)
    val_diff = diff_names_between_directories(img_val, lab_val)
    if len(train_diff) > 0:
        if verbose:
            print_red(f"Images and labels in train directory do not match. Below are the differences.")
            print(train_diff)
        return False
    if len(val_diff) > 0:
        if verbose:
            print_red(f"Images and labels in val directory do not match. Below are the differences.")
            print(val_diff)
        return False

    if test:
        if not img_test.is_dir():
            if verbose:
                print_red(f"Directory: '{img_test}' does not exist or is not a directory.")
            return False
        if not lab_test.is_dir():
            if verbose:
                print_red(f"Directory: '{lab_test}' does not exist or is not a directory.")
            return False
        
        test_diff = diff_names_between_directories(img_test, lab_test)
        if len(test_diff) > 0:
            if verbose:
                print_red(f"Images and labels in test directory do not match. Below are the differences.")
                print(test_diff)
            return False
    if verbose:
        print_green("The yolo file structure is correct.")
    return True
        
    
def diff_names_between_directories(dir1: Path, dir2: Path):
    '''
    Returns the names of the files in dir1 that are not in dir2 or vice versa.

    dir1:   The first directory.\n
    dir2:   The second directory.\n
    '''
    dir1 = Path(dir1)
    dir2 = Path(dir2)
    if not dir1.exists() or not dir1.is_dir():
        print_red(f"Directory: '{dir1}' does not exist or is not a directory.")
        return
    if not dir2.exists() or not dir2.is_dir():
        print_red(f"Directory: '{dir2}' does not exist or is not a directory.")
        return
    files1 = set([f.stem for f in dir1.iterdir()])
    files2 = set([f.stem for f in dir2.iterdir()])
    return files1 - files2 if len(files1) > len(files2) else files2 - files1

'''
Determine the device to use for training.
'''
def get_device(use_gpu=True, use_mps=True):
    if torch.cuda.is_available() and use_gpu:
        device = 'cuda'
    elif torch.backends.mps.is_available() and use_mps:
        device = 'mps'
    else:
        device = 'cpu'

    return device
