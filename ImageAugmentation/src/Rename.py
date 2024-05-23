from pathlib import Path

def rename(path, startIndex):
    files = [item for item in path.iterdir() if item.suffix.lower() in {'.jpg', '.jpeg'}]
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
        
path = Path('..', 'test_data', 'test')
print(f'path: "{path}"')
rename(path, 100)