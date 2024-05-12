import os

def rename(path, startIndex):
    for count, filename in enumerate(os.listdir(path)):
        print(path + filename)
        print(path + str(count) + '.jpg')
        os.rename(path + filename, path + str(count) + '.xml')
        
path = '/Volumes/Research/PLM/Restricted/Research/Project Folders/Active/ADT - Assembly Digital Thread/FOD/Images/'
path += 'Allen Wrench/'
rename(path, 0)