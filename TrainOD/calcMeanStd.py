import csv
import os
from PIL import Image
import numpy as np

csv_name = "train.csv"

with open(os.path.join(".data", "pet", csv_name)) as f:
    reader = csv.reader(f)
    nr_lines = 0
    mean = 0
    std = 0
    nr_lines = 0
    for line in reader:
        nr_lines +=1
        (filename, label, startX, startY, endX, endY) = line
        # read image
        image = Image.open(filename).convert('RGB')
        # calculate mean ad std
        im = np.asarray(image)
        img_float = im.astype(np.float32) / 255.0
        mean += np.mean(img_float, axis=(0,1))
        std += np.std(img_float, axis=(0,1))

print("Mean:")
print(mean/nr_lines)
print("Std:")
print(std/nr_lines)

# Mean:
# [0.47964323 0.4472308  0.39698297]
# Std:
# [0.23054191 0.22776432 0.22883151]