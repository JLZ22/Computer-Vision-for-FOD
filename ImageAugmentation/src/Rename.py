import os

def rename(path, startIndex):
    files = os.listdir(path)
    files = [f for f in files if f.endswith('.jpg')]
    files.sort()
    for count, filename in enumerate(files, start=startIndex):
        name = filename[:-4]
        oldJPG = os.path.join(path, name + '.jpg')
        oldXML = os.path.join(path, name + '.xml')
        newJPG = os.path.join(path, str(count) + '.jpg')
        newXML = os.path.join(path, str(count) + '.xml')
        print(f'{oldJPG} -> {newJPG}')
        print(f'{oldXML} -> {newXML}')
        # os.rename(oldJPG, newJPG)
        # os.rename(oldXML, newXML)
        
path = '../test_data/test'
rename(path, 0)