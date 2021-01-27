import os
import sys

num_files = int(0)
for path in pathlib.Path("ana").iterdir():
    if path.is_file():

        old_extension = path.suffix
        directory = path.parent
        new_name = f"{num_files}"+ old_extension

        path.rename(pathlib.Path(directory, new_name))
        num_files +=1