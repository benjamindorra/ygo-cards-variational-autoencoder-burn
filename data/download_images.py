import requests
import pandas as pd
import json
from PIL import Image
from io import BytesIO
import os

images_dir = 'card_images_small'
if not os.path.exists(images_dir):
    os.mkdir(images_dir)

r = requests.get('https://db.ygoprodeck.com/api/v7/cardinfo.php')
df = pd.DataFrame(json.loads(r.text)['data'])

for id in df["id"]:
    r = requests.get('https://images.ygoprodeck.com/images/cards_small/{}.jpg'.format(id))
    im = Image.open(BytesIO(r.content))
    im.save(os.path.join(images_dir, str(id)+'.jpg'))
