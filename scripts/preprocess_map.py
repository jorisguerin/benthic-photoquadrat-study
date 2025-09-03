import sys

sys.path.append('.')  # For running from root directory
sys.path.append('..')  # For running from scripts/ directory

import rasterio
import numpy as np
import cv2

import os
os.makedirs('figures', exist_ok=True)

path = "./data/map.tif"
with rasterio.open(path) as src:
    img = src.read(1)
img = img[:9000, :]

# Cropping and rotating raster
mask = img != 15
coordinates = np.column_stack(np.where(mask))
rect = cv2.minAreaRect(coordinates)
angle = -rect[2]

height, width = img.shape
cos = np.abs(np.cos(np.radians(angle)))
sin = np.abs(np.sin(np.radians(angle)))
new_width = int(width * cos + height * sin)
new_height = int(height * cos + width * sin)

M = cv2.getRotationMatrix2D((width/2, height/2), angle, 1.0)
M[0, 2] += new_width/2 - width/2
M[1, 2] += new_height/2 - height/2

rotated = cv2.warpAffine(img, M, (new_width, new_height),
                         flags=cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_CONSTANT,
                         borderValue=0)

mask = ~((rotated == 0) | (rotated == 15))
kernel = np.ones((5,5), np.uint8)
mask = cv2.erode(mask.astype(np.uint8), kernel)
coordinates = np.column_stack(np.where(mask))
rect = cv2.minAreaRect(coordinates)
box = cv2.boxPoints(rect)

x, y, w, h = cv2.boundingRect(box.astype(np.int32))
cropped = rotated[x:x+w-10, y:y+h-5]
cropped[cropped > 10] = 0

np.save("./data/map.npy", cropped)