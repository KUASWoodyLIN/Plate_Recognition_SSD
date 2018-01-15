import os
import cv2
import numpy as np


path = os.path.join(os.path.dirname(__file__), 'test.jpg')

image = cv2.imread(path)
image = np.array(image, dtype=np.float32)/255
print(1)

