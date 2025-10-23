import requests
import pandas as pd
import json
from PIL import Image
from io import BytesIO

with open("ygo_cardinfo.php.json") as f:
    buf = json.load(f)
    buf = buf['data']
df = pd.DataFrame(buf)
for id in df["id"]:
    r = requests.get('https://images.ygoprodeck.com/images/cards_small/{}.jpg'.format(id))
    im = Image.open(BytesIO(r.content))
    im.save("card_images_small/{}.jpg".format(id))
