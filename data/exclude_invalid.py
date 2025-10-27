import os
from PIL import Image

images_dir = "card_images_small"
input_dir = os.path.join(images_dir, "train")
output_dir = os.path.join(images_dir, "invalid_size")
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
expected_width = 268
expected_height = 391

for im_name in os.listdir(input_dir):
    im = Image.open(os.path.join(input_dir, im_name))
    width, height = im.size
    if (width != expected_width) or (height != expected_height):
        os.rename(os.path.join(input_dir, im_name), os.path.join(output_dir, im_name))
