from pathlib import Path
import imgaug as ia
import imgaug.augmenters as iaa
import cv2
import numpy as np
import os

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

def resize_All_JPGs(path, save_path, width, height, batchsize = 16):
    if not path.exists() or not path.is_dir():
        print_red(f"Directory: '{path}' does not exist or is not a directory.")
        return
    save_exists = True
    if not save_path.exists() or not save_path.is_dir():
        save_exists = False
        os.mkdir(str(save_path))
    
    aug = iaa.Resize({'height' : height, 'width' : width})
    img = cv2.imread('../test_data/test_images/3279.jpg')
    res = aug.augment_image(img)
    cv2.imshow('Original', img)
    cv2.imshow('Resized', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # try:
    #     image_files = list(path.glob('*.jpg')) + list(path.glob('*.jpeg'))
    #     for i in range(0, len(image_files), batchsize):
    #         batch = image_files[i:i+batchsize]
    #         images = [cv2.imread(str(image_path)) for image_path in batch]
    #         resized = [cv2.resize(img) for img in images]
    #         for i in resized.shape[0]:
    #             cv2.imwrite(str(save_path / str('resized_' + str(batch[i].name))) , resized[i])
    # except Exception as e:
    #     print(e)
    #     if not save_exists:
    #         os.rmdir(str(save_path))
