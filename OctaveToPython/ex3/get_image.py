import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.499, 0.387, 0.114])


def get_my_images(path):
    path = np.array(path)
    greys = np.ones((path.size, 400))
    print(path.size)
    for i in range(path.size):
        img = mpimg.imread(path[i])
        grey = rgb2gray(img)
        greys[i, :] = grey.swapaxes(0, 1).ravel()
    return greys
