import numpy as np
import skimage.io as io
import skimage.color as color

def load_raw_image(path):
  raw_image = io.imread(path)
  return raw_image

def load_image(path):
  raw_image = io.imread(path)
  gray_image = color.rgb2gray(raw_image)
  return gray_image

def crop_image(gray_image, crop_size):
  h, w = gray_image.shape
  new_h = (h // crop_size) * crop_size
  new_w = (w // crop_size) * crop_size
  cropped_image = gray_image[:new_h, :new_w]
  return cropped_image

def split_to_bloks(cropped_image, block_size):
  h, w = cropped_image.shape
  split_image = np.reshape(cropped_image, (h // block_size, block_size, w // block_size, block_size))
  split_image = np.transpose(split_image, (0, 2, 1, 3))
  return split_image