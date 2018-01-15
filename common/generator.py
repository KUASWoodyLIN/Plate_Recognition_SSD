import numpy as np
import cv2


def train_generator(x_data, y_data, batch_size=64):
  while True:
    for start in range(0, len(x_data), batch_size):
      x_batch = []
      end = min(start + batch_size, len(x_data))
      i_train_batch = x_data[start:end]
      y_batch = y_data[start:end]
      for i in i_train_batch:
        x_batch.append(np.array(cv2.imread(i), dtype=np.float32) / 255)
      yield x_batch, y_batch


def valid_generator(x_data, y_data, batch_size=64):
  while True:
    for start in range(0, len(x_data), batch_size):
      x_batch = []
      end = min(start + batch_size, len(x_data))
      i_train_batch = x_data[start:end]
      y_batch = y_data[start:end]
      for i in i_train_batch:
        x_batch.append(np.array(cv2.imread(i), dtype=np.float32) / 255)
      yield x_batch, y_batch

