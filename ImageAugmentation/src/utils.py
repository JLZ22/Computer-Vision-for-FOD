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

def print_red(text):
    print("\033[91m{}\033[0m".format(text))

def print_green(text):
    print("\033[92m{}\033[0m".format(text))

def get_jpg_paths(path):
    return [item for item in path.iterdir() if item.suffix.lower() in {'.jpg', '.jpeg'}]

def rename(path: Path, startIndex=0, prefix = ''):
    path = Path(path)
    files = get_jpg_paths(path)
    files.sort()
    for count, filename in enumerate(files, start=startIndex):
        if filename.is_dir():
            continue
        elif not filename.is_file():
            print_red(f"File: '{filename}' does not exist.")
            continue
        name = filename.stem
        oldJPG = Path(path, name + '.jpg')
        oldXML = Path(path, name + '.xml')
        newJPG = Path(path, prefix + str(count) + '.jpg')
        newXML = Path(path, prefix + str(count) + '.xml')
        print(f'{oldJPG} -> {newJPG}')
        print(f'{oldXML} -> {newXML}')
        if oldJPG.is_file():
            oldJPG.rename(newJPG)
        if oldXML.is_file():
            oldXML.rename(newXML)

# delete all files in the given path
def deleteFiles(path: Path):
    path = Path(path)
    if not path.exists() or not path.is_dir():
        print_red(f"Directory: '{path}' does not exist or is not a directory.")
        return
    for f in path.iterdir():
        if f.is_file():
            f.unlink()
    print_green(f"Deleted all files in the directory: '{path}'")

def subtract_mean(image):
    image = np.array(image)
    # calculate per channel mean pixel values
    mean = np.mean(image, axis=(0, 1))
    # subtract the mean from the image
    image = image - mean
    return image

def get_bbox(path: Path, jpg_path: Path):
    path = Path(path)
    jpg_path = Path(jpg_path)
    name = jpg_path.stem
    xml = path / (name + '.xml')
    if not xml.exists():
        return None
    tree = ET.parse(str(xml))
    root = tree.getroot()
    bbs = create_bbs(root, cv2.imread(str(jpg_path)).shape)
    return bbs

def get_bboxes(path: Path, jpg_paths: list[Path]):
    path = Path(path)
    bbss = []
    for img in jpg_paths:
        bbs = get_bbox(path, img)
        if bbs is not None:
            bbss.append(bbs)
    return bbss

def pad_and_resize_all_square(path: Path, save_path: Path, dim, batchsize = 16):
    path = Path(path)
    save_path = Path(save_path)
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
        bbss = True
        image_files = list(path.glob('*.jpg')) + list(path.glob('*.jpeg'))
        if list(path.glob('*.xml')) != []:
            assert len(image_files) == len(list(path.glob('*.xml')))
        else:
            bbss = False
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
                for i, bbs in enumerate(res_bbss):
                    writer = Writer(str(save_path / str('resized_' + str(batch[i].name))), dim, dim)
                    for box in bbs.bounding_boxes:
                        writer.addObject(box.label, box.x1, box.y1, box.x2, box.y2)
                    writer.save(str(save_path / ('resized_' + str(batch[i].stem + '.xml'))))
    except Exception as e:
        traceback.print_exc()
        # if we created the save directory, delete it
        if not save_exists:
            os.rmdir(str(save_path))

'''
https://piyush-kulkarni.medium.com/visualize-the-xml-annotations-in-python-c9696ba9c188
'''
def visualize_annotations(path, save_path):
    path = Path(path)
    save_path = Path(save_path)
    # check read and write paths
    if not path.exists() or not path.is_dir():
        print_red(f"Directory: '{path}' does not exist or is not a directory.")
        return
    if save_path.exists() and not save_path.is_dir():
        print_red(f"Directory: '{save_path}' exists but is not a directory.")
        return
    if not save_path.exists():
        os.mkdir(str(save_path))

    # get images and xml files
    images = list(path.glob('*.jpg')) + list(path.glob('*.jpeg'))
    xml = list(path.glob('*.xml'))
    # assert they are the same length
    assert(len(images) == len(xml))
    
    for file in images:
        filename = file.stem
        img_path = file
        xml_path = path / (filename + '.xml')
        img = cv2.imread(str(img_path))
        if img is None:
            pass
        dom = xdm.parse(str(xml_path))
        root = dom.documentElement
        objects=dom.getElementsByTagName("object")

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
                               1)
            # add label
            label = root.getElementsByTagName('name')[i]
            label_data = label.childNodes[0].data

            cv2.putText(img, 
                        label_data, 
                        [int(float(xmin_data)) + 2, int(float(ymin_data)) - 2], 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.3, 
                        (0,0,0), 
                        1)
        # save image with bounding boxes drawn
        cv2.imwrite(str(save_path / (filename + '.jpg')),img)

# Make num_copies number of the bbs object and return it 
# in an array
def make_copies_bboxes(bbs: BoundingBoxesOnImage, num_copies: int) -> np.array:
    return [bbs for _ in range(num_copies)]

# Return an array of copies of the image stored at 
# path/img. The array has num_copies number of copies.
def make_copies_images(name, num_copies: int) -> np.array:
    return np.array(
        [cv2.imread(name) for _ in range(num_copies)],
        dtype=np.uint8
    )

# Return a BoundingBoxesOnImage object with the 
# given root and shape by automatically creating a 
# new BoundingBox object for every object 
# in the root
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

# Gets all file names in the directory that end in .jpg
def getJPGFileNames(path: Path):
    path = Path(path)
    jpg = list(path.glob('*.jpg')) + list(path.glob('*.jpeg'))
    return [item.stem for item in jpg]

# Get the memory consumption of all children processes
# If no children processes are found, return 0
def get_children_mem_consumption():
    pid = os.getpid()
    children = psutil.Process(pid).children(recursive=True)
    return sum([child.memory_info().rss for child in children])

def lowerCaseLabels(path: Path, save_path=None):
    path = Path(path)
    if save_path == None:
        save_path = path
    save_path = Path(save_path)
    if not path.exists() or not path.is_dir():
        print_red(f"Directory: '{path}' does not exist or is not a directory.")
        return
    jpgPaths = get_jpg_paths(path)
    for i, img in enumerate(jpgPaths):
        xml = path / (img.stem + '.xml')
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

def updatePath(path, save_path=None, new_path=None):
    path = Path(path)
    if save_path == None:
        save_path = path
    if new_path == None:
        new_path = path
    save_path = Path(save_path)
    new_path = Path(new_path)
    if not path.exists() or not path.is_dir():
        print_red(f"Directory: '{path}' does not exist or is not a directory.")
        return
    if not save_path.exists() or not save_path.is_dir():
        print_red(f"Directory: '{save_path}' does not exist or is not a directory.")
        return
    if not new_path.exists() or not new_path.is_dir():
        print_red(f"Directory: '{new_path}' does not exist or is not a directory.")
        return
    jpgPaths = get_jpg_paths(path)
    for i, img in enumerate(jpgPaths):
        xml = path / (img.stem + '.xml')
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

def flipHorizontalInDirectory(path, save_path):
    aug = iaa.Fliplr(1.0)
    augInDirectory(path, save_path, aug)

def flipVerticalInDirectory(path, save_path):
    aug = iaa.Flipud(1.0)
    augInDirectory(path, save_path, aug)

def rotateInDirectory(path, save_path, angle):
    aug = iaa.Affine(rotate=angle)
    augInDirectory(path, save_path, aug)

def rotate90InDirectory(path, save_path):
    aug = iaa.Rot90(1)
    augInDirectory(path, save_path, aug)

def augInDirectory(path, save_path, aug):
    path = Path(path)
    save_path = Path(save_path)
    if not path.exists() or not path.is_dir():
        print_red(f"Directory: '{path}' does not exist or is not a directory.")
        return
    jpgPaths = get_jpg_paths(path)
    bboxes = get_bboxes(path, jpgPaths)
    for i, img in enumerate(jpgPaths):
        image = cv2.imread(str(img))
        if image is None:
            print_red(f"Failed to read image: {img}")
            continue
        bbs = bboxes[i]
        imgaug, bbsaug = aug.augment(image=image, bounding_boxes=bbs)
        # save augmented image
        cv2.imwrite(str(save_path / img.name), imgaug)

        # save augmented bboxes
        height, width, _ = imgaug.shape
        writer = Writer(str(save_path / img.name), width=width, height=height)
        for box in bbsaug.bounding_boxes:
            writer.addObject(box.label, box.x1, box.y1, box.x2, box.y2)
        writer.save(str(save_path / (img.stem + '.xml')))