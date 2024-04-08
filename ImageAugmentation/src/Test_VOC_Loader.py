from voc_tools.utils import VOCDataset
import os 

dataset_path = "Z:\Restricted\Research\Project Folders\Active\ADT - Assembly Digital Thread\FOD\Images"
contents = os.listdir(dataset_path) 
# initialize a dataset
my_dataset = VOCDataset(dataset_path)

# fetch annotation bulk
for annotations, jpeg in my_dataset.train.fetch():
    print(annotations[0].filename, jpeg.image.shape)
# fetch annotation
for anno, jpeg in my_dataset.train.fetch(bulk=False):
    print(anno, jpeg.image.shape)

# parse the annotations into memory for train dataset
my_dataset.train.load()
my_dataset.test.load()

# returns a list of class names in train dataset
my_dataset.train.class_names()
my_dataset.test.class_names()

# save parsed information into csv
#my_dataset.train.load().to_csv("./train_metadata.csv")
#my_dataset.test.load().to_csv("./train_metadata.csv")

# purge the parsed metadata to free memory
my_dataset.train.unload()
my_dataset.test.unload()