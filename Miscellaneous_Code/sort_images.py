import os 
from pathlib import Path
import shutil

iters = [500,1000,2000,5000]

os.chdir("/home/kamau/Dogs")

for num in iters:
    Path(f"Images_{num}").mkdir(exist_ok = True)
    Path(f"Labels{num}").mkdir(exist_ok = True)

    os.chdir("/home/kamau/OIDv4_ToolKit/OID/Dataset/train/Dog")

    for file in os.listdir():
        if file == "Label":
            continue
        else:
            for i in range(iter):
                name,ext = os.path.splitext(file)

                if ext == '.txt':
                    shutil.move(file,"/home/kamau/Dogs")

