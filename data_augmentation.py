import random
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os


def image_rotation(image):
    degrees = random.randint(-20, 20)
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), degrees, 1.0)
    result = cv2.warpAffine(image, M, (w, h))

    return result


def image_crop(image):
    scale = 0.8 + 0.2 * random.random()
    height, width = int(image.shape[0] * scale), int(image.shape[1] * scale)
    x = random.randint(0, image.shape[1] - int(width))
    y = random.randint(0, image.shape[0] - int(height))
    cropped = image[y : y + height, x : x + width]

    return cv2.resize(cropped, (image.shape[1], image.shape[0]))


def image_dilate(img, kernel=np.ones((3, 3), np.uint8), iterations=1):
    return cv2.erode(img, kernel, iterations=iterations)


def image_translation(image):
    shift_X = random.randint(-10, 10)
    shift_Y = random.randint(-10, 10)
    M = np.float32([[1, 0, shift_X], [0, 1, shift_Y]])
    translated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    return translated
