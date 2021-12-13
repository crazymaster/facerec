from PIL import Image
import numpy as np


def read_img(file: str) -> np.array:
    return np.array(Image.open(file))


img = read_img('data/lena.jpg')
print(img.shape)

pil_img = Image.fromarray(img)
