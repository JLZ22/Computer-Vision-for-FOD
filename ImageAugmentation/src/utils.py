from pathlib import Path
import imgaug.augmenters as iaa
import cv2
import numpy as np
import os
from pascal_voc_writer import Writer
import xml.etree.ElementTree as ET
import  xml.dom.minidom as xdm
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import psutil
import traceback
import math

def print_red(text):
    print("\033[91m{}\033[0m".format(text))

def print_green(text):
    print("\033[92m{}\033[0m".format(text))

'''
Get the paths to all jpg files in the directory.
'''
def get_jpg_paths(path, range=(-1, -1)):
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
    path = Path(path)
    if not path.exists() or not path.is_dir():
        print_red(f"Directory: '{path}' does not exist or is not a directory.")
        return
    # get all jpg files in the directory
    jpgs = list(path.glob('*.jpg')) + list(path.glob('*.jpeg'))
    if range == (-1, -1):
        return jpgs
    # get all jpg files in the directory within the range
    out = []
    for item in jpgs:
        index = int(item.stem)
        if index >= range[0] and index <= range[1]:
            out.append(item)
    return out

'''
Rename all the jpg and xml files in the directory to the
format prefix + index + '.jpg' and prefix + index + '.xml'.

This function has no recursive functionality.
'''
def rename(read_path: Path, startIndex=0, prefix = ''):
    read_path = Path(read_path)
    files = get_jpg_paths(read_path)
    for count, filename in enumerate(files, start=startIndex):
        if filename.is_dir():
            continue
        name = filename.stem
        oldJPG = Path(read_path, filename.name)
        oldXML = Path(read_path, name + '.xml')
        newJPG = Path(read_path, prefix + str(count) + '.jpg')
        newXML = Path(read_path, prefix + str(count) + '.xml')
        print(f'{oldJPG} -> {newJPG}')
        print(f'{oldXML} -> {newXML}')
        if oldJPG.is_file():
            oldJPG.rename(newJPG)
        if oldXML.is_file():
            oldXML.rename(newXML)

'''
Delete all files in the directory.
'''
def delete_files(read_path: Path):
    read_path = Path(read_path)
    if not read_path.exists() or not read_path.is_dir():
        print_red(f"Directory: '{read_path}' does not exist or is not a directory.")
        return
    for f in read_path.iterdir():
        if f.is_file():
            f.unlink()
    print_green(f"Deleted all files in the directory: '{read_path}'")

'''
Subtract the mean pixel values from the image.
'''
def subtract_mean(image):
    image = np.array(image)
    # calculate per channel mean pixel values
    mean = np.mean(image, axis=(0, 1))
    # subtract the mean from the image
    image = image - mean
    return image

'''
Get the bounding boxes from the xml file in the read_path
that corresponds to the jpg file. If the xml file does not 
exist, return None.
'''
def get_corresponding_bbox(read_path: Path, jpg_path: Path):
    read_path = Path(read_path)
    jpg_path = Path(jpg_path)
    name = jpg_path.stem
    xml = read_path / (name + '.xml')
    if not xml.exists():
        return None
    tree = ET.parse(str(xml))
    root = tree.getroot()
    bbs = create_bbs(root, cv2.imread(str(jpg_path)).shape)
    return bbs

'''
https://piyush-kulkarni.medium.com/visualize-the-xml-annotations-in-python-c9696ba9c188
'''
def visualize_annotations(read_path, save_path):
    read_path = Path(read_path)
    save_path = Path(save_path)
    # check read and write paths
    if not read_path.exists() or not read_path.is_dir():
        print_red(f"Directory: '{read_path}' does not exist or is not a directory.")
        return
    if save_path.exists() and not save_path.is_dir():
        print_red(f"Directory: '{save_path}' exists but is not a directory.")
        return
    if not save_path.exists():
        os.mkdir(str(save_path))

    # get images and xml files
    images = list(read_path.glob('*.jpg')) + list(read_path.glob('*.jpeg'))
    xml = list(read_path.glob('*.xml'))
    # assert they are the same length
    assert(len(images) == len(xml))
    
    for file in images:
        filename = file.stem
        img_path = file
        xml_path = read_path / (filename + '.xml')
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
        fontScale = min(width, height) / 1200
        fontThickness = max(width, height) / 1000

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
                               round(boxThickness) if round(boxThickness) > 0 else 1)
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
        cv2.imwrite(str(save_path / (filename + '.jpg')),img)

'''
Make num_copies number of the bbs object and return it 
in an array
'''
def make_copies_bboxes(bbs: BoundingBoxesOnImage, num_copies: int) -> np.array:
    return [bbs for _ in range(num_copies)]

'''
Return an array of copies of the image stored at 
path/img. The array has num_copies number of copies.
'''
def make_copies_images(name, num_copies: int) -> np.array:
    return np.array(
        [cv2.imread(name) for _ in range(num_copies)],
        dtype=np.uint8
    )

'''
Return a BoundingBoxesOnImage object with the
given root and shape by automatically creating a
new BoundingBox object for every object in the root.
'''
def create_bbs(root, shape: int) -> BoundingBoxesOnImage:
    bboxes = []
    for member in root.findall('object'):
        bbox = member.find('bndbox')
        xmin = int(float(bbox.find('xmin').text))
        ymin = int(float(bbox.find('ymin').text))
        xmax = int(float(bbox.find('xmax').text))
        ymax = int(float(bbox.find('ymax').text))
        bboxes.append(BoundingBox(x1=xmin, y1=ymin, x2=xmax, y2=ymax, label=member.find('name').text))
    return BoundingBoxesOnImage(bboxes, shape)

'''
Get the memory consumption of all children processes
If no children processes are found, return 0
'''
def get_children_mem_consumption():
    pid = os.getpid()
    children = psutil.Process(pid).children(recursive=True)
    return sum([child.memory_info().rss for child in children])

'''
Lowercase the labels in the xml files in the directory.
If save_path is None, the xml files will be saved in the same directory.
'''
def lowercase_labels_in_directory(read_path: Path, save_path=None):
    read_path = Path(read_path)
    if save_path == None:
        save_path = read_path
    save_path = Path(save_path)
    if not read_path.exists() or not read_path.is_dir():
        print_red(f"Directory: '{read_path}' does not exist or is not a directory.")
        return
    jpgPaths = get_jpg_paths(read_path)
    for img in jpgPaths:
        xml = read_path / (img.stem + '.xml')
        if not xml.exists():
            print_red(f"XML file: '{xml}' does not exist.")
            continue
        tree = ET.parse(str(xml))
        root = tree.getroot()
        for member in root.findall('object'):
            member.find('name').text = member.find('name').text.lower()
        tree.write(str(save_path / (img.stem + '.xml')))
        # save corresponding jpg in save_path
        cv2.imwrite(str(save_path / img.name), cv2.imread(str(img)))

'''
Update the path in the xml files to the new path.
If save_path is None, the xml files will be saved in the same directory.
If new_path is None, the path in the xml files will be updated to the read path.
'''
def update_path(read_path, save_path=None, new_path=None):
    read_path = Path(read_path)
    if save_path == None:
        save_path = read_path
    if new_path == None:
        new_path = read_path
    save_path = Path(save_path)
    new_path = Path(new_path)
    if not read_path.exists() or not read_path.is_dir():
        print_red(f"Directory: '{read_path}' does not exist or is not a directory.")
        return
    if not save_path.exists() or not save_path.is_dir():
        print_red(f"Directory: '{save_path}' does not exist or is not a directory.")
        return
    if not new_path.exists() or not new_path.is_dir():
        print_red(f"Directory: '{new_path}' does not exist or is not a directory.")
        return
    jpgPaths = get_jpg_paths(read_path)
    for i, img in enumerate(jpgPaths):
        xml = read_path / (img.stem + '.xml')
        if not xml.exists():
            print_red(f"XML file: '{xml}' does not exist.")
            continue
        tree = ET.parse(str(xml))
        root = tree.getroot()
        for member in root.findall('path'):
            member.text = str(new_path / img.name)
        tree.write(str(save_path / (img.stem + '.xml')))
        # save corresponding jpg in save_path
        cv2.imwrite(str(save_path / img.name), cv2.imread(str(img)))

'''
Perform the augmentation on the images in the directory
and save the augmented images in the save_path directory.
If includeXML is True, the bounding boxes will be augmented.
'''
def aug_in_directory(read_path, save_path, aug, includeXML=True):
    # check read and write paths
    read_path = Path(read_path)
    save_path = Path(save_path)
    if not read_path.exists() or not read_path.is_dir():
        print_red(f"Directory: '{read_path}' does not exist or is not a directory.")
        return
    
    # create save directory if it does not exist
    save_exists = True
    if not save_path.exists():
        os.mkdir(str(save_path))
        save_exists = False

    # get image paths
    jpgPaths = get_jpg_paths(read_path)

    try:
        # augment images
        for img in jpgPaths:
            # read image and check if it is valid
            image = cv2.imread(str(img))
            if image is None:
                print_red(f"Failed to read image: {img}")
                continue

            # get corresponding bounding boxes
            bbs = get_corresponding_bbox(read_path, img)

            # check if we should augment the bounding boxes
            hasXML = bbs is not None and includeXML

            # augment image and bounding boxes if bounding boxes exists 
            # and the user wants to include the xml files (aka bounding boxes)
            # in the augmentation 
            if hasXML:
                image, bbs = aug.augment(image=image, bounding_boxes=bbs)
                # save augmented bboxes
                height, width, _ = image.shape
                writer = Writer(str(save_path / img.name), width=width, height=height)
                for box in bbs.bounding_boxes:
                    writer.addObject(box.label, box.x1, box.y1, box.x2, box.y2)
                writer.save(str(save_path / (img.stem + '.xml'))
                )
            else:
                image = aug.augment(image=image)

            # save augmented image
            cv2.imwrite(str(save_path / img.name), image)
    except Exception as e:
        traceback.print_exc()
        # if we created the save directory, delete it
        if not save_exists:
            os.rmdir(str(save_path))

'''
Flip the images in the directory horizontally and save them. 
If includeXML is True, the bounding boxes will be flipped as well.
'''
def flip_horizontal_in_directory(read_path, save_path, includeXML=True):
    aug = iaa.Fliplr(1.0)
    aug_in_directory(read_path, save_path, aug, includeXML)

'''
Flip the images in the directory vertically and save them.
If includeXML is True, the bounding boxes will be flipped as well.
'''
def flip_vertical_in_directory(read_path, save_path, includeXML=True):
    aug = iaa.Flipud(1.0)
    aug_in_directory(read_path, save_path, aug, includeXML)

'''
Rotate the images in the directory by the given angle and save them
without altering the aspect ratio.
If includeXML is True, the bounding boxes will be rotated as well.
'''
def rotate_in_directory(read_path, save_path, angle, includeXML=True):
    if angle % 90 != 0:
        aug = iaa.Affine(rotate=angle)
        aug_in_directory(read_path, save_path, aug, includeXML)
    else:
        rotate_90_in_directory(read_path, save_path, angle // 90)

'''
Rotate the images in the directory by 90 degrees and save them and save 
them without altering the aspect ratio.
If includeXML is True, the bounding boxes will be rotated as well.
'''
def rotate_90_in_directory(read_path, save_path, repetitions=1, includeXML=True):
    aug = iaa.Rot90(repetitions)
    aug_in_directory(read_path, save_path, aug, includeXML)

'''
Resize the images in the directory to the given width and height and save them.
This does not guarantee that the aspect ratio will remain the same.
If includeXML is True, the bounding boxes will be resized as well.
'''
def resize_in_directory(read_path: Path, save_path: Path, width=512, height=512, includeXML=True):
    # augmenters
    aug = iaa.Resize({"height": height, "width": width})
    aug_in_directory(read_path, save_path, aug, includeXML)

'''
Pad the images in the directory to make them square and 
resize them to the given dimension while maintaining aspect ratio.
If includeXML is True, the bounding boxes will be resized as well.
'''
def pad_and_resize_square_in_directory(read_path: Path, save_path: Path, dim=512, includeXML=True):
    # augmenters
    aug = iaa.Sequential([
        iaa.PadToSquare(),
        iaa.Resize(dim)
    ])
    aug_in_directory(read_path, save_path, aug, includeXML)

'''
Copy the files in the read_path directory to the save_path directory.
'''
def copy_files_in_directory(read_path: Path, save_path: Path):
    read_path = Path(read_path)
    save_path = Path(save_path)
    if not read_path.exists() or not read_path.is_dir():
        print_red(f"Directory: '{read_path}' does not exist or is not a directory.")
        return
    if not save_path.exists():
        os.mkdir(str(save_path))

    # get all files in the directory
    files = list(read_path.iterdir())
    
    # copy data to save_path
    for f in files:
        if f.is_file():
            os.system(f'cp {f} {save_path}')

'''
Rotate the image at img_read_path by the given angle and save it in img_save_dir.
The rotateCode is the cv2 rotate code.
'''
def rotate_image_and_save(img_read_path: Path, img_save_dir: Path, rotateCode: int):
    img_read_path = Path(img_read_path)
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

'''
Rotate all the images in the directory by the given rotate code 
and save them in the save_path directory if range is (-1, -1).
If range is not (-1, -1), only rotate the images in the range.
'''
def rotate_images_and_save(read_path: Path, save_path=None, rotateCode=cv2.ROTATE_90_CLOCKWISE, range=(-1, -1)):
    read_path = Path(read_path)
    if save_path == None:
        save_path = read_path
    save_path = Path(save_path)

    if not read_path.exists() or not read_path.is_dir():
        print_red(f"Directory: '{read_path}' does not exist or is not a directory.")
        return
    if not save_path.exists():
        os.mkdir(str(save_path))
    jpgPaths = get_jpg_paths(read_path, range)
    for img in jpgPaths:
        rotate_image_and_save(img, save_path, rotateCode)

'''
Delete all xml files in the directory that do not have a corresponding jpg file
'''
def delete_all_xml_without_jpg(read_path: Path):
    read_path = Path(read_path)
    if not read_path.exists() or not read_path.is_dir():
        print_red(f"Directory: '{read_path}' does not exist or is not a directory.")
        return
    jpgPaths = get_jpg_paths(read_path)
    xmlPaths = list(read_path.glob('*.xml'))
    for xml in xmlPaths:
        name = xml.stem
        if not any([jpg.stem == name for jpg in jpgPaths]):
            xml.unlink()

'''
Count the number of files in the directory.
'''
def count_files_in_directory(read_path: Path):
    read_path = Path(read_path)
    if not read_path.exists() or not read_path.is_dir():
        print_red(f"Directory: '{read_path}' does not exist or is not a directory.")
        return
    count = 0
    for f in read_path.iterdir():
        if f.is_file():
            count += 1
    return count

def jpeg_to_jpg(read_path: Path):
    read_path = Path(read_path)
    if not read_path.exists() or not read_path.is_dir():
        print_red(f"Directory: '{read_path}' does not exist or is not a directory.")
        return
    jpg = list(read_path.glob('*.jpeg'))
    for img in jpg:
        img.rename(read_path / (str(img.stem) + '.jpg'))