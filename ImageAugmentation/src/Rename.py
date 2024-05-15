import os

def rename(path, startIndex):
    for count, filename in enumerate(os.listdir(path)):
        # get file extension of filename

        if filename[-4:] != '.jpg':
            continue
        print(path + filename)
        print(path + str(count) + '.jpg')
        os.rename(path + filename, path + str(count) + '.xml')
        
path = '../test_data/test'
path += 'Allen Wrench/'
rename(path, 0)