# Computer Vision for FOD

Leveraging computer vision to aid in foreign object detection in a manufacturing setting.

Before beginning to work on this project, it is important to be familiar with git. A brief git tutorial can be found [here](https://github.com/JLZ22/Git-Tutorial-for-New-Users).

It is also **STRONGLY RECOMMENDED** that you work in a python virtual environment. One option is to use python3's built in [venv](https://docs.python.org/3/library/venv.html) while another is to use [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) and its [virtual environment manager](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). I (John Zeng, B.S. AI + Computer Science at Purdue, 2026) recommend using python3's built in venv because it is more lightweight than conda. **When using virtual environments, be sure to include those in the `.gitignore`. You can read more about them in the git tutorial above.** This project was developed using `Python3.12`.

## Package List

```
ultralytics
opencv-python
imgaug 
numpy 
pascal_voc
pascal_voc_writer 
pascal-voc-tools
tqdm
matplotlib
clearml
pdoc
```

### Steps: 
1. Run `pip install -r requirements.txt`. This installs everything necessary if you are using **only CPU**
2. If you are using **CUDA** or **RocM**, 
    - uninstall `torch` and `torchvision` with `pip uninstall torch torchvision`. 
    - reinstall the PyTorch packages for your machine specifications by visiting their getting started [website](https://pytorch.org/get-started/locally/). 
3. Optional `clearml` setup:
    - make an [account](https://app.clear.ml/login) and follow setup instructions on the website. 

### Package Notes

- `pytorch`: depending on your system, pytorch may have different arguments for installation. Since the default PyTorch packages that ultralytics installs for you does not include CUDA or RocM support, you may need to reinstall for yourself. 

## Documentation

To find current API documentation of the main branch, please visit this [page](https://jlz22.github.io/Computer-Vision-for-FOD/index.html) which has been generated by [pdoc](https://pdoc.dev/). To use `pdoc` to generate documentation for a different `*.py` file (in the case that you are working off/on a different branch), you can run `pdoc <file1.py> [file2.py ...]` which will start a web server and open a automatically. When creating new `*.py` files, please ensure that they are documented such that `pdoc` formats it well. 

## Usage for Augmentation, Training, and Tuning

### Augmentation

1. ensure that your directory of training data is in the PascalVOC format
    - for each image, there is a corresponding xml file with the same name
    - these images should be jpgs 

2. run the `pad_and_resize_square_in_directory` function from the `Utils` module on your dataset and save it to a new folder
    - this is critical. Otherwise, it is likely that your system will run out of memory because it will be doing a lot of 4k processing. 

3. create a new augmenter object with these arguments

```
SimpleAugSeq(
            <insert correct read path>, 
            <insert correct save path>, 
            num_copies = 64, 
            processes = 12,
            )
```  

4. run the `augment` function

Now you should have a directory of augmented PascalVOC datapoints (jpg/xml pairs)

### Pre-processing for YOLO

Now that you have a directory of PascalVOC datapoints, we must process the data into the file structure and format that is compatible with training YOLO models. 

Note: The reason we augmented in PascalVOC and are converting to YOLO is because of some short-sitedness on my part towards the beginning of the development of these modules. 

1. check that you have a `Computer-Vision-for-FOD/test_data/notes.json` file 
    - this contains important information for the conversion from PascalVOC to YOLO

2. Run the function `pascalvoc_to_yolo_in_directory` which is found in the `Utils` module where the `json` field is the path to the `notes.json` file

3. Run the function `partition_yolo_data_for_training` from the `Utils` module
    - you should only need to modify the `read_dir` and `save_dir` fields
    
Now, you should have a file structure that looks like this. (`<save_dir>` should be replaced with the name of the directory that contains your dataset)

```
<save_dir>
|
---- images
|     |
|     ---- test
|     |
|     ---- train
|     |
|     ---- val
|
---- labels
      |
      ---- test
      |
      ---- train
      |
      ---- val
```

4. Create a `dataset.yaml` file under `<save_dir>` 
    - Ex: 
        ```
        path: /absolute/path/to/<save_dir>
        train: images/train
        val: images/val
        test: images/test

        nc: 5
        names:
            0: allen wrench
            1: pencil
            2: screwdriver
            3: tool bit
            4: wrench
        ```

Now, you should be ready for tuning. 

### Tuning

Tuning improves the accuracy of your model but is very time consuming.

1. create a `config.yaml` file
    - Ex: 
        ```
        model_path: /path/to/model.pt/file # if this does not exist, the tune script 
                                           # should download it for you
        project: FOD # name of project

        train:
            data_path: /path/to/dataset.yaml # path to the dataset you will use for training
            hyp: /path/to/best_hyperparameters.yaml # you may ignore this for this step
            name: train # name of the folder the results will be saved to
            epochs: 10
            batch_size: 32
            imgsz: 640 # size of image 
            patience: 10

        tune:
            iterations: 75
        ```
    - This file provides the necessary arguments to run `tune.py`. `data_path` is the path to the dataset.yaml file that you created in the previous section. `name` is the name of the directory that will be created for each train instance (if the name already exists, then a number will be appended at the end). No need to change the items including and after the `epochs` entry.

2. run `./tune.py ../config.yaml` with Python

The script will run for a few hours (maybe days) and there will be a `detect` directory created that contains the results. In one of the subdirectories of `detect`, you will find a yaml file called `best_hyperparameters.yaml`. Copy and paste the path to this file into the `hyp:` field above for training. 

### Train

The train workflow is exactly the same as tune. Just run `./train.py ../config.yaml` instead. The main variables that affect performance are epochs and batch_size. Tuning should handle the other hyperparameters. To make sure that you use the hyperparameters that you found in the tuning step, set 
```
train:
    hyp: /path/to/best_hyperparameters.yaml
```

## Augmentation Notes

[imgaug](https://github.com/aleju/imgaug?tab=readme-ov-file) is an open source software that aids in image augmentation. We will be using that to augment our training and test images.

Images were subject to the following augmentations in random order for training. 
- vertical/horizontal flips
- random crops between 0% and 10% of the image
- Gaussian Blur (applied 50% of the time)
- strengthenning or weakenning contrast between 75% and 150% the original value
- Gaussian Noise (applied 50% of the time)
- randomly adjust brightness between 80% and 120% the original value for 20% of images
- Randomly zoomed, or translated

After augmentation, for each image in the dataset, the mean pixel value per channel is subtracted from it according to the 7th source in the literature review section.

In a random sample of 128 data points, we found that 4 data points were faulty. Using a one sample t-test, we concluded with 99% confidence that between 0.084% and 7.09% of the dataset of 10,048 images are faulty.

### Note for using augmentation modules:

Be **very careful** when augmenting with **high resolution images** because it can be memory intensive and **crash** your computer. **ALWAYS**, resize your images to sub-1k dimensions before attempting augmentation. 

## Todo

- [x] find solution to memory issue
- [x] finish augmenting rest of the classes
- [x] resize images and corresponding bounding boxes
- [x] transform labels from PascalVOC to YOLO 
- [x] rename labels for consistency
- [x] subtract mean pixel value from all images 
- [x] partition data into training and validation
- [x] separate txt and jpg
- [x] train on augmented data
- [x] train on unaugmented data
- [x] use tracking to highlight objects that are within a roi for a certain duration
- [x] find most effective standard for determining if an object is in the roi or not
- [x] keep objects highlighted if they leave the roi briefly
- [ ] test in assembly space on 1 camera
- [ ] test in assembly space on 3 cameras 

## Info about dataset version 1

- Post augmentation labels per class: 
    - `allen wrench:` 4608 labels
    - `pencil:` 17728 labels
    - `screwdriver:` 5120 labels
    - `tool bit:` 10688 labels
    - `wrench:` 10624 labels
    - `total:` 48768 labels

- Pre augmentation labels per class: 
    - `allen wrench:` 72 labels
    - `pencil:` 277 labels
    - `screwdriver:` 80 labels
    - `tool bit:` 167 labels
    - `wrench:` 166 labels
    - `total: ` 762 labels

- Normalized labels per class: 
    - `allen wrench:` 0.094
    - `pencil:` 0.364 
    - `screwdriver:` 0.105 
    - `tool bit:` 0.219 
    - `wrench:` 0.218 

## Literature Review

1. (March 2024) [Small-Scale Foreign Object Debris Detection Using Deep Learning and Dual Light Modes](https://www.mdpi.com/2076-3417/14/5/2162)
2. (February 2024) [A two-stage deep learning method for foreign object detection and localization](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13069/130690G/A-two-stage-deep-learning-method-for-foreign-object-detection/10.1117/12.3024079.full#_=_)
3. (July 2023) [A doubt–confirmation-based visual detection method for foreign object debris aided by assembly models](https://cdnsciencepub.com/doi/full/10.1139/tcsme-2022-0143)
4. (Sept 2022)[Foreign Object Detection on an Assembly Line](https://link.springer.com/content/pdf/10.1007/978-981-19-2600-6_29.pdf)
5. (June 2022) [Foreign objects detection using deep learning techniques for graphic card assembly line](https://link.springer.com/article/10.1007/s10845-022-01980-7)
6. (April 2020) [Deep Learning Models for Visual Inspection on Automotive Assembling Line](https://arxiv.org/ftp/arxiv/papers/2007/2007.01857.pdf)
7. (July 2019) [Best Practices for Preparing and Augmenting Image Data for CNNs](https://machinelearningmastery.com/best-practices-for-preparing-and-augmenting-image-data-for-convolutional-neural-networks/)
8. (September 2012) [Practical Recommendations for Gradient-Based Training of Deep
Architectures](https://arxiv.org/pdf/1206.5533)