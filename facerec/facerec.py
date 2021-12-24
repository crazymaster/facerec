from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def read_img(file: str) -> np.ndarray:
    return np.array(Image.open(file))


def rgb2hsv(rgb: np.ndarray) -> np.ndarray:
    """RGB画像からHSV画像に変換

    Args:
        rgb (np.ndarray): RGB画像
    Returns:
        np.ndarray: HSV画像

    H: [0, 360], S: [0, 100], V: [0, 100]

    円柱モデルで彩度は S = (max - min) / max
    >>> rgb2hsv(np.array([[[50, 120, 239], [163, 200, 130]]]))
    array([[[217,  79,  93],
            [ 91,  35,  78]]])
    >>> rgb2hsv(np.array([[[215, 79, 93], [90, 35, 78], [230, 133, 116]]]))
    array([[[353,  63,  84],
            [313,  61,  35],
            [  8,  49,  90]]])
    >>> rgb2hsv(np.array([[[0, 0, 0], [100, 100, 100], [255, 255 , 255]]]))
    array([[[  0,   0,   0],
            [  0,   0,  39],
            [  0,   0, 100]]])
    >>> rgb2hsv(np.array([[[192, 96, 100]]]))
    array([[[357,  50,  75]]])
    """

    input_shape = rgb.shape
    rgb = rgb.reshape(-1, 3).astype('float')
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)

    v = 100 * maxc / 255

    deltac = maxc - minc
    s = np.divide(100 * deltac, maxc, where=(maxc != 0))
    s[maxc == 0] = 0

    h = np.empty_like(v)
    deltac[deltac == 0] = 1  # ゼロ除算を避けるために代入。後で当該の要素は上書きされる。
    h[maxc == r] = 60 * (g[maxc == r] - b[maxc == r]) / deltac[maxc == r]
    h[maxc == g] = 60 * (b[maxc == g] - r[maxc == g]) / deltac[maxc == g] + 120
    h[maxc == b] = 60 * (r[maxc == b] - g[maxc == b]) / deltac[maxc == b] + 240
    h[minc == maxc] = 0
    h[h < 0] += 360

    res = np.dstack([h, s, v])
    return res.reshape(input_shape).astype('int')


def hsv2rgb(hsv: np.ndarray) -> np.ndarray:
    """HSV画像からRGB画像に変換

    Args:
        hsv_img (np.ndarray): HSV画像
    Returns:
        np.ndarray: RGB画像
    >>> hsv2rgb(np.array([[[215, 79, 93], [90, 35, 78]]]))
    array([[[ 49, 127, 237],
            [164, 198, 129]]], dtype=uint8)
    >>> hsv2rgb(np.array([[[0, 0, 0], [0, 100, 100], [360, 100, 100]]]))
    array([[[  0,   0,   0],
            [255,   0,   0],
            [255,   0,   0]]], dtype=uint8)
    >>> hsv2rgb(np.array([[[358,  50,  75]]]))
    array([[[191,  95,  98]]], dtype=uint8)
    """

    input_shape = hsv.shape
    hsv = hsv.reshape(-1, 3).astype('float')
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]

    maxc = 255 * v / 100
    minc = maxc - maxc * s / 100

    i = h // 60
    deltac = maxc - minc
    maxc, minc = maxc.reshape(-1, 1), minc.reshape(-1, 1)
    deltac, h = deltac.reshape(-1, 1), h.reshape(-1, 1)
    x1 = deltac * np.abs(120 - h) / 60 + minc
    x2 = deltac * np.abs(240 - h) / 60 + minc

    rgb = np.zeros_like(hsv)
    rgb[i == 0] = np.hstack([maxc, deltac * h / 60 + minc, minc])[i == 0]
    rgb[i == 1] = np.hstack([x1, maxc, minc])[i == 1]
    rgb[i == 2] = np.hstack([minc, maxc, x1])[i == 2]
    rgb[i == 3] = np.hstack([minc, x2, maxc])[i == 3]
    rgb[i == 4] = np.hstack([x2, minc, maxc])[i == 4]
    rgb[(i == 5) | (i == 6)] = np.hstack([maxc, minc, deltac * (360 - h) / 60 + minc])[(i == 5) | (i == 6)]
    rgb[s == 0] = np.hstack([maxc, maxc, maxc])[s == 0]

    return rgb.reshape(input_shape).astype(np.uint8)


def trim(array, x, y, width, height):
    return array[y:y + height, x:x+width]


def region_hist(img: np.ndarray, num_split: int = 16) -> np.ndarray:
    """RGB画像をn*nに分割し、各領域についてヒストグラムを求める

    Args:
        img (np.ndarray): RGB画像
        num_split (int): 分割数
    """

    h, w, _ = img.shape
    # print(img.shape)
    region_hists = np.empty((num_split, num_split, 360))

    for h_count in range(num_split):
        pos_h = h // num_split * h_count
        for w_count in range(num_split):
            pos_w = w // num_split * w_count
            im_trim = trim(img, pos_w, pos_h, w // num_split, h // num_split)
            # print(pos_w, pos_h, w // num_split, h // num_split)
            hist, _ = np.histogram(rgb2hsv(im_trim)[:, :, 0].ravel(), range=(0, 360), bins=360, density=True)
            region_hists[h_count, w_count, :] = hist
            # plt.imshow(im_trim)
            # plt.show()

    return region_hists


def cos_sim_matrix(A_mat, B_mat):
    """
    item-vevtorの行列(またはベクトル)が与えられた際にitem-vevtor間のコサイン類似度行列を求める
    """
    d = A_mat * B_mat.T  # 各ベクトル同士の内積を要素とする行列

    # 各ベクトルの大きさの平方根
    A_norm = (A_mat ** 2).sum(axis=2, keepdims=True) ** .5
    B_norm = (B_mat ** 2).sum(axis=0, keepdims=True) ** .5

    # それぞれのベクトルの大きさの平方根で割っている
    return d / A_norm / B_norm.T


if __name__ == "__main__":
    img = read_img('./data/lena.jpg')
    print(img.shape)
    hist = region_hist(img, 2)

    # hsv_img = rgb2hsv(img)
    # rgb_img = hsv2rgb(hsv_img)
    # pil_img = Image.fromarray(img)
    # pil_img.save('../data/temp/lena_save_pillow.jpg')

    # hsv2rgb(rgb2hsv(np.array([[[192, 96, 100]]])))
