import os
import sys
from pathlib import Path
from PIL import Image


def convert_image(src, dest):
    rgba_image = Image.open(src)
    rgba_image.convert('RGB')
    rgba_image.save(dest)


for folder in Path("heroes").iterdir():
    if not folder.is_file():
        num_files = int(0)
        for path in Path(folder).iterdir():
            if path.is_file():

                old_extension = path.suffix
                directory = path.parent
                new_name = f"{num_files}" + old_extension

                path.rename(Path(directory, new_name))

                num_files += 1
        print(f'{folder} has {num_files} samples')
