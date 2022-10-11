import cv2
import os

import numpy as np


def see_sizes(folder):
    sizes = []
    images = os.listdir(folder)
    max_x = 0
    max_y = 0
    for image in images:
        if ".jpg" in image or ".png" in image:
            im = cv2.imread(folder + image)
            (y, x, _) = im.shape
            sizes.append((y, x))
            if x > max_x:
                max_x = x
            if y > max_y:
                max_y = y

    return sizes, max_x, max_y


def set_size(image, x, y, image_name, write=False, folder=None):
    dim = (x, y)
    image_resize = cv2.resize(image, dim)
    if write:
        cv2.imwrite(folder + "resized_" + image_name, image_resize)


def set_sizes_real_images():
    sizes, max_x, max_y = see_sizes("dataset/real/")
    print(sizes)
    print(max_x, max_y)
    images = os.listdir("dataset/real/")
    for image in images:
        if ".jpg" in image:
            img = cv2.imread("dataset/real/" + image)
            set_size(
                img, max_x, max_y, image, write=True, folder="dataset/real/resized/"
            )


def set_sizes_fake_images():
    # sizes, max_x, max_y = see_sizes("dataset_letters/real/")
    x = 66
    y = 240
    images = os.listdir("dataset_letters/fake/")
    for image in images:
        if ".png" in image:
            img = cv2.imread("dataset_letters/fake/" + image)
            set_size(
                img, x, y, image, write=True, folder="dataset_letters/fake/resized/"
            )


set_sizes_fake_images()

"""
img_1 = "dataset/real/resized/resized_2+2.jpg"
img_2 = "dataset/resized/resized_2.png"

img1 = cv2.imread(img_1)
img2 = cv2.imread(img_2)

hori = np.hstack((img1, img2))
cv2.imshow("Horizontal", hori)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
