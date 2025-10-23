from PIL import Image
import numpy as np
import os

root = "card_images_small/train/"

max_val = 255.
mean = np.array([0.,0.,0.])
std = np.array([0.,0.,0.])
count = 0
for im_name in os.listdir(root):
    im = Image.open(os.path.join(root, im_name))
    im = np.asarray(im, dtype = np.float32) / max_val
    m = np.mean(im, axis = (0, 1))
    s = np.std(im, axis = (0, 1))
    mean += m
    std += s
    count += 1

mean = mean / count
std = std / count

print("mean: {}".format(mean))
print("std: {}".format(std))
