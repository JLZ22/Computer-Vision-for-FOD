# Computer Vision for FOD<br>

Leveraging computer vision to aid in foreign object detection in a manufacturing setting.<br>

Currently in development. <br>

This is a research project under Professor Nathaniel Hartman at Purdue University. It is meant to be kept private. **DO NOT SHARE WITH ANYONE** who is not also working with/for Professor Hartman unless given the go-ahead from Professor Hartman.<br>

Before beginning to work on this project, it is important to be familiar with git. A brief git tutorial can be found [here](https://github.com/JLZ22/Git-Tutorial-for-New-Users).<br>

It is also **STRONGLY RECOMMENDED** that you work in a python virtual environment. One option is to use python3's built in [venv](https://docs.python.org/3/library/venv.html) while another is to use [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) and its [virtual environment manager](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). I (John Zeng, B.S. AI + Computer Science at Purdue, 2026) recommend using python3's built in venv because it is more lightweight than conda. **When using virtual environments, be sure to include those in the `.gitignore`. You can read more about them in the git tutorial above.**<br>

## Package List<br>

```
ultralytics 
opencv-python
imgaug 
numpy 
pascal_voc_writer 
pascal-voc-tools
pascal-voc
tqdm
matplotlib
clearml
```

To install using `pip`, run `pip install -r requirements.txt`. For proper functionality of `clearml`, make an [account](https://app.clear.ml/login) and follow setup instructions.  

## Augmentation Notes<br>

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

In a random sample of 128 data points, we found that 4 data points were faulty. Using a one sample t-test, we concluded with 99% confidence that 0.084% and 7.09% of the dataset of 10,048 images is faulty.

### Current Functionality

- Capable of performing image augmentation on jpg/xml pairs
- Equipped with various utility functions to help with large quantity of file operations
- Can perform a benchmark on your machine to determine the optimal number of proceses to use for multiprocessing for image augmentation
- Can run the default YOLOv8 model which was trained on the COCO dataset. 

### Issues

- img augmentation with multiprocessing runs into errors with excessive memory consumption

### Todo

- [x] find solution to memory issue<br>
- [x] finish augmenting rest of the classes<br>
- [x] resize images and corresponding bounding boxes<br>
- [x] transform labels from PascalVOC to YOLO <br>
- [x] rename labels for consistency<br>
- [x] subtract mean pixel value from all images 
- [ ] partition data into training and validation<br>
- [ ] separate txt and jpg<br>
- [ ] train<br>

## Literature Review<br>

1. (March 2024) [Small-Scale Foreign Object Debris Detection Using Deep Learning and Dual Light Modes](https://www.mdpi.com/2076-3417/14/5/2162)<br>
2. (February 2024) [A two-stage deep learning method for foreign object detection and localization](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13069/130690G/A-two-stage-deep-learning-method-for-foreign-object-detection/10.1117/12.3024079.full#_=_)<br>
3. (July 2023) [A doubt–confirmation-based visual detection method for foreign object debris aided by assembly models](https://cdnsciencepub.com/doi/full/10.1139/tcsme-2022-0143)<br>
4. (Sept 2022)[Foreign Object Detection on an Assembly Line](https://link.springer.com/content/pdf/10.1007/978-981-19-2600-6_29.pdf)<br>
5. (June 2022) [Foreign objects detection using deep learning techniques for graphic card assembly line](https://link.springer.com/article/10.1007/s10845-022-01980-7)<br>
6. (April 2020) [Deep Learning Models for Visual Inspection on Automotive Assembling Line](https://arxiv.org/ftp/arxiv/papers/2007/2007.01857.pdf)<br>
7. (July 2019) [Best Practices for Preparing and Augmenting Image Data for CNNs](https://machinelearningmastery.com/best-practices-for-preparing-and-augmenting-image-data-for-convolutional-neural-networks/)