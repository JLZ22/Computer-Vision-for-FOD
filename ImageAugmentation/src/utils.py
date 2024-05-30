from pathlib import Path
import imgaug.augmenters as iaa
import cv2
import numpy as np
import os
from pascal_voc_writer import Writer
import xml.etree.ElementTree as ET
import  xml.dom.minidom
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import psutil
import traceback

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
    for img in jpg_files:
        name = img.stem
        xml = path / (name + '.xml')
        tree = ET.parse(str(xml))
        root = tree.getroot()
        bbs = create_bbs(root, cv2.imread(str(img)).shape)
        bbss.append(bbs)
    return bbss

def pad_and_resize_all_square(path, save_path, dim, batchsize = 16):
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
    # get images
    images = [item for item in path.iterdir() if item.suffix.lower() in {'.jpg', '.jpeg'}]
    
    for file in images:
        filename = file.stem
        img_path = file
        xml_path = path / (filename + '.xml')
        img = cv2.imread(str(img_path))
        if img is None:
            pass
        dom = xml.dom.minidom.parse(str(xml_path))
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
                               2)
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

#gets all file names in the directory that end in .jpg
def getFileNames(path: Path):
    return [item.stem for item in path.iterdir()]

# Get the memory consumption of all children processes
# If no children processes are found, return 0
def get_children_mem_consumption():
    pid = os.getpid()
    children = psutil.Process(pid).children(recursive=True)
    return sum([child.memory_info().rss for child in children])