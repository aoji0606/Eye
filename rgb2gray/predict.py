import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image

img = cv.imread("./data/temp/2007_006449.jpg")
img = cv.resize(img, (256, 256))
X = (np.array(img) / 255.0).reshape([1, 256, 256, 3])

model = tf.keras.models.load_model("./model.h5")
gray = model.predict(X)
gray = np.array(gray)
gray = gray.reshape(256, 256)

gray = Image.fromarray(gray * 255.0)
gray = gray.convert('L')
gray.save("./gray.png")
gray.show()
