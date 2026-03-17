import matplotlib
import matplotlib.pyplot as plt
import skimage.io as io
import numpy as np
import skimage.color as color
from scipy.fftpack import dct, idct

matrix_size = 8
path = "E:\\remciov3\\politechnika\\Magisterka\\Identyfikacja i modelowanie statystyczne\\orthogonal_compression\\Kompresja_Obrazow\\test_model.jpg"

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
  splitted_image = np.reshape(cropped_image, (h // block_size, block_size, w // block_size, block_size))
  splitted_image = np.transpose(splitted_image, (0, 2, 1, 3))
  return splitted_image

def show_blocks_grid(splited_image):
  for i in range(15):
    for j in range(15):
      plt.subplot(15, 15, i*15 + j + 1)
      plt.imshow(splited_image[i, j], cmap = 'gray')
      plt.axis('off')
  matplotlib.pyplot.imshow(splited_image[15, 0], cmap = 'gray')
  matplotlib.pyplot.show()
  print(type(splited_image))
  print(np.shape(splited_image))

def calculate_img_dimensions(img):
  img_h = np.shape(img)[0]
  img_w = np.shape(img)[1]
  return [img_h, img_w]

def calculate_compression_mask(M, N):
  mask_size = M // 2
  mask = np.zeros((M, N))
  for i in range (0, mask_size):
    for j in range(mask_size - i):
      mask[i, j] = 1
  return mask

def dct_compression(splited_image, img_h, img_w):
  img_block = splited_image[img_h, img_w]
  M, N = calculate_img_dimensions(img_block)
  PI = np.pi

  m = np.arange(M)
  p = np.arange(M)
  m = np.reshape(m, (1, -1))
  p = np.reshape(p, (-1, 1))
  
  T = np.cos((PI * (2*m+1) * p) / (2*M))
  T[0, :] /= np.sqrt(M)         
  T[1:, :] /= np.sqrt(M / 2)
  B = (T @ img_block @ T.T)

  with np.printoptions(edgeitems=8, precision=1, linewidth=1000):
    if(img_h == 0 and img_w == 0):
      print(B)
  
  mask = calculate_compression_mask(M,N)
  B_masked = B * mask
  A_compressed = T.T @ B_masked @ T
  return A_compressed

def compress_dct_image(splitted_image):
  img_h, img_w = calculate_img_dimensions(splitted_image)
  image_compressed = np.zeros_like(splitted_image)
  for i in range(img_h):
    for j in range(img_w):
      image_block = dct_compression(splitted_image, i, j)
      image_compressed[i, j] = image_block
  return image_compressed

def scipy_dct(splitted_image):
 with np.printoptions(edgeitems = 8, precision = 2, linewidth = 1000):
  B = dct(dct(splitted_image.T, norm='ortho').T, norm='ortho')
  M, N = calculate_img_dimensions(splitted_image)
  mask = calculate_compression_mask(M, N)
  B_compressed = B * mask
  A_compressed = idct(idct(B_compressed, norm='ortho').T, norm='ortho').T
  return A_compressed
 
def compress_scipy_dct_image(splitted_image):
  img_h, img_w = calculate_img_dimensions(splitted_image)
  image_compressed = np.zeros_like(splitted_image)
  for i in range(img_h):
    for j in range(img_w):
      image_block = scipy_dct(splitted_image[i,j])
      image_compressed[i, j] = image_block
  return image_compressed

def splited_image_to_image(image_compressed, block_size):
  h, w = calculate_img_dimensions(image_compressed)
  temp = image_compressed.transpose(0, 2, 1, 3)
  image_combined = np.reshape(temp, (h * block_size, w * block_size))
  return image_combined

def calculate_snr(original, reconstructed):
  og = np.sum(original ** 2)
  dif = np.sum((original - reconstructed) ** 2 )
  SNR = 10 * np.log10 (og/dif)
  return SNR

def cosinuse_approximation(block,m ):
 with np.printoptions(edgeitems = 8, precision = 2, linewidth = 1000):
  B = dct(dct(block.T, norm='ortho').T, norm='ortho')
  M, N = calculate_img_dimensions(block)
  B_compressed = B * (np.abs(B) >= m  )
  A_compressed = idct(idct(B_compressed, norm='ortho').T, norm='ortho').T
  return A_compressed
 
def compress_cosine_image(cosinus_img, m):
  img_h, img_w = calculate_img_dimensions(cosinus_img)
  image_cosinus_compresed = np.zeros_like(cosinus_img)
  for i in range(img_h):
    for j in range(img_w):
      image_cosinus_compresed[i, j] = cosinuse_approximation(cosinus_img[i,j], m)
  return image_cosinus_compresed

def noise_calculate(original, compresed):
  return original - compresed
def dct_coefficients(image):
  FC = dct(dct(image.T, norm='ortho').T, norm='ortho')
  return FC
 
def show_decompression_efect(img, img_dct, img_scipy_dct, img_cosinus, snr, noise, FC):
  plt.figure(figsize=(32, 32))
  plt.subplot(1, 3, 1)
  plt.title("Przed kompresją")
  plt.imshow(img, cmap='gray')
  plt.subplot(1, 3, 2)
  plt.title("Po kompresji")
  plt.imshow(img_dct, cmap ='gray')
  plt.subplot(1, 3, 3)
  plt.title("Po kompresji 2")
  plt.imshow(img_scipy_dct, cmap='gray')
  plt.subplot(2, 3, 1)
  plt.title(f"Po cosinus\nSNR = {snr:.2f} dB")
  plt.imshow(img_cosinus, cmap='gray')
  plt.subplot(2, 3, 2)
  plt.title("Szum kompresji")
  plt.imshow(noise, cmap='gray')
  plt.subplot(2, 3, 3)
  plt.title("Współczynniki DCT")
  plt.imshow(np.log(1e-5 + np.abs(FC)), cmap='gray')
  plt.show()

def main():
  gray_image = load_image(path)
  cropped_image = crop_image(gray_image, matrix_size)
  splitted_image = split_to_bloks(cropped_image, matrix_size)
  image_compressed_dct = compress_dct_image(splitted_image)
  image_compressed_scipy_dct = compress_scipy_dct_image(splitted_image)
  dct_image = splited_image_to_image(image_compressed_dct, matrix_size)
  scipy_dct_image = splited_image_to_image(image_compressed_scipy_dct, matrix_size)
  img_compressed_cosinus = compress_cosine_image(splitted_image, 0.1)
  img_cosinus_compressed = splited_image_to_image(img_compressed_cosinus, matrix_size)
  snr = calculate_snr(gray_image, dct_image)
  noise = noise_calculate(gray_image, dct_image)
  FC = dct_coefficients(gray_image)
  show_decompression_efect(gray_image, dct_image, scipy_dct_image, img_cosinus_compressed, snr, noise, FC)


  

main()