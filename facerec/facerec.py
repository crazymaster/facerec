from PIL import Image
import numpy as np


def read_img(file: str) -> np.array:
    return np.array(Image.open(file))


image = read_img('data/lena.jpg')
