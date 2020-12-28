import os
import cv2 as cv
import matplotlib.pyplot as plt
from pprint import pprint


def file_name(file_dir):
    for files in os.walk(file_dir):
        pass
    return files


if __name__ == '__main__':
    # resize
    files = file_name("./data/rgb")
    for file in files[2]:
        rgb = cv.imread("./data/rgb/%s" % file)
        rgb_resize = cv.resize(rgb, (256, 256))
        cv.imwrite("./data/rgb_resize/%s" % file, rgb_resize)
        print("imwrite %s" % file)

    # rgb2gray
    files = file_name("./data/rgb_resize")
    for file in files[2]:
        rgb = cv.imread("./data/rgb_resize/%s" % file)
        gray = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)
        cv.imwrite("./data/gray/%s" % file, gray)
        print("imwrite %s" % file)
