import os
from PIL import Image

input_dir = "card_images_small/train"
output_dir = "card_images_small/invalid_size"
expected_width = 268
expected_height = 391

for im_name in os.listdir(input_dir):
    im = Image.open(os.path.join(input_dir, im_name))
    width, height = im.size
    if (width != expected_width) or (height != expected_height):
        os.rename(os.path.join(input_dir, im_name), os.path.join(output_dir, im_name))
