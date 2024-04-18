import xml.etree.ElementTree as ET
import os 
from pascal_voc_writer import Writer

#markupsafe version 2.0.1

# parse xml file
tree = ET.parse("Z:\Restricted\Research\Project Folders\Active\ADT - Assembly Digital Thread\FOD\Images\pencils\\1.xml") 
root = tree.getroot() # get root object

height = int(root.find("size")[0].text)
width = int(root.find("size")[1].text)
channels = int(root.find("size")[2].text)
ref_path = str(root.find('path').text)
ref_file = str(root.find('filename').text)

bbox_coordinates = []
for member in root.findall('object'):
    class_name = member[0].text # class name
        
    # bbox coordinates
    xmin = int(member[4][0].text)
    ymin = int(member[4][1].text)
    xmax = int(member[4][2].text)
    ymax = int(member[4][3].text)
    # store data in list
    bbox_coordinates.append([class_name, xmin, ymin, xmax, ymax])

print(bbox_coordinates)
print(ref_file)
print(ref_path)

writer = Writer('Z:\Restricted\Research\Project Folders\Active\ADT - Assembly Digital Thread\FOD\Images\pencils\\1.jpg', height, width)

for x in bbox_coordinates:
    writer.addObject(x[0],x[1],x[2],x[3],x[4])

writer.save('Z:\Restricted\Research\Project Folders\Active\ADT - Assembly Digital Thread\FOD\Images\pencils\\1_test.xml')