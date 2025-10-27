import requests
import pandas as pd
import json
from PIL import Image
from io import BytesIO
import os

images_dir = 'card_images_small'
train_dir = os.path.join(images_dir, "train")
test_dir = os.path.join(images_dir, "test")
if not os.path.exists(images_dir):
    os.mkdir(images_dir)
if not os.path.exists(train_dir):
    os.mkdir(train_dir)
if not os.path.exists(test_dir):
    os.mkdir(test_dir)

r = requests.get('https://db.ygoprodeck.com/api/v7/cardinfo.php')
df = pd.DataFrame(json.loads(r.text)['data'])

for id in df["id"]:
    r = requests.get('https://images.ygoprodeck.com/images/cards_small/{}.jpg'.format(id))
    im = Image.open(BytesIO(r.content))
    im.save(os.path.join(train_dir, str(id)+'.jpg'))
