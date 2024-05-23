# https://piyush-kulkarni.medium.com/visualize-the-xml-annotations-in-python-c9696ba9c188
import cv2
import  xml.dom.minidom
from pathlib import Path

path=Path('../test_data/test_pairs')
save_path = Path('../test_data/test_pairs_annotated')
images = [item for item in path.iterdir() if item.suffix.lower() in {'.jpg', '.jpeg'}]
for file in images:
	filename = file.stem
	img_path = file
	xml_path = path / (filename + '.xml')
	print(img_path)
	img = cv2.imread(str(img_path))
	if img is None:
		pass
	dom = xml.dom.minidom.parse(str(xml_path))
	root = dom.documentElement
	objects=dom.getElementsByTagName("object")
	print(objects)
	i=0
	for object in objects:
        
		bndbox = root.getElementsByTagName('bndbox')[i]
		xmin = bndbox.getElementsByTagName('xmin')[0]
		ymin = bndbox.getElementsByTagName('ymin')[0]
		xmax = bndbox.getElementsByTagName('xmax')[0]
		ymax = bndbox.getElementsByTagName('ymax')[0]
		xmin_data=xmin.childNodes[0].data
		ymin_data=ymin.childNodes[0].data
		xmax_data=xmax.childNodes[0].data
		ymax_data=ymax.childNodes[0].data
		print(object)        
		print(xmin_data)
		print(ymin_data)
        
		i= i + 1 
		cv2.rectangle(img,(int(xmin_data),int(ymin_data)),(int(xmax_data),int(ymax_data)),(55,255,155),5)
	flag=0
	flag=cv2.imwrite(str(save_path / (filename + '.jpg')),img)
	if flag:
		print(filename,"done")
print("all done ====================================")