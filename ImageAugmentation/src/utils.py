from pathlib import Path

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
        print_red(f"Directory: '{path}' does not exist.")
        return
    for f in path.iterdir():
        if f.is_file():
            f.unlink()
    print_green(f"Deleted all files in the directory: '{path}'")