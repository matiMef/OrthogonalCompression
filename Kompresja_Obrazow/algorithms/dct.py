import numpy as np
from scipy.fftpack import dct, idct

def calculate_image_dimensions(image):
  img_h = np.shape(image)[0]
  img_w = np.shape(image)[1]
  return [img_h, img_w]

def merge_blocks_into_image(image_combined, block_size):
  h, w = calculate_image_dimensions(image_combined)
  temp = image_combined.transpose(0, 2, 1, 3)
  image_combined = np.reshape(temp, (h * block_size, w * block_size))
  return image_combined

def calculate_compression_mask(M, N):
  mask_size = M // 2
  mask = np.zeros((M, N))
  for i in range (0, mask_size):
    for j in range(mask_size - i):
      mask[i, j] = 1
  return mask


def apply_dct_to_all_blocks(split_image):
  img_h, img_w = calculate_image_dimensions(split_image)
  reconstructed_image = np.zeros_like(split_image)
  for i in range(img_h):
    for j in range(img_w):
      image_block = dct_compression(split_image, i, j)
      reconstructed_image[i, j] = image_block
  return reconstructed_image

def dct_compression(split_image, img_h, img_w):
  img_block = split_image[img_h, img_w]
  M, N = calculate_image_dimensions(img_block)
  PI = np.pi

  m = np.arange(M)
  p = np.arange(M)
  m = np.reshape(m, (1, -1))
  p = np.reshape(p, (-1, 1))
  
  T = np.cos((PI * (2*m+1) * p) / (2*M))
  T[0, :] /= np.sqrt(M)         
  T[1:, :] /= np.sqrt(M / 2)
  B = (T @ img_block @ T.T)

  # with np.printoptions(edgeitems=8, precision=1, linewidth=1000):
  #   if(img_h == 0 and img_w == 0):
  #     print(B)
  
  mask = calculate_compression_mask(M,N)
  B_masked = B * mask
  A_compressed = T.T @ B_masked @ T
  return A_compressed


def apply_scipy_dct_to_all_blocks(split_image):
  img_h, img_w = calculate_image_dimensions(split_image)
  reconstructed_image = np.zeros_like(split_image)
  for i in range(img_h):
    for j in range(img_w):
      image_block = scipy_dct(split_image[i,j], i, j)
      reconstructed_image[i, j] = image_block
  return reconstructed_image

def scipy_dct(split_image, img_h, img_w):
  B = dct(dct(split_image, axis=0, norm='ortho'), axis=1, norm='ortho')
  # with np.printoptions(edgeitems=8, precision=1, linewidth=1000):
  #   if(img_h == 0 and img_w == 0):
  #     print(B)
  M, N = calculate_image_dimensions(split_image)
  mask = calculate_compression_mask(M, N)
  B_compressed = B * mask
  A_compressed = idct(idct(B_compressed, axis=0, norm='ortho'), axis=1, norm='ortho')
  return A_compressed