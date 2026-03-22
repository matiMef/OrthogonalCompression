import timeit
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import skimage.io as io
import skimage.color as color
from scipy.fftpack import dct, idct

matrix_size = 8
path = "test_model2.jpg"

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

def show_blocks_grid(split_image):
  plt.title("Obrazek po podziale na bloki")
  for i in range(15):
    for j in range(15):
      plt.subplot(15, 15, i*15 + j + 1)
      plt.imshow(split_image[i, j], cmap = 'gray')
      plt.axis('off')
  plt.show()
  # plt.imshow(split_image[15, 0], cmap = 'gray')
  # print(type(split_image))
  # print(np.shape(split_image))

# === Funkcje pomocnicze ===

def load_raw_image(path):
  raw_image = io.imread(path)
  return raw_image

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

def start_time_measure():
    start = timeit.default_timer()
    return start

def end_time_measure(start):
    end = timeit.default_timer()
    time_elapsed = end - start
    time_elapsed = (float(f'{time_elapsed:.4f}'))
    return time_elapsed

# === Wizualizacje ===

# Logika
def calculate_snr(cropped_image, combined_image):
  og = np.sum(cropped_image ** 2)
  dif = np.sum((cropped_image - combined_image) ** 2 )
  SNR = 10 * np.log10 (og/dif)
  return SNR

def calculate_noise(gray_image, combined_image):
  return gray_image - combined_image

def calculate_dct_coefficients(gray_image):
  FC = dct(dct(gray_image, axis = 0, norm='ortho'), axis=1, norm='ortho')
  return FC

def calculate_fft_coefficients(gray_image):
  FF = np.fft.fft2(gray_image)
  return FF

def caculate_aproximation_error(gray_image):
  FC = calculate_dct_coefficients(gray_image)
  FF = calculate_fft_coefficients(gray_image)
  sorted_dct = np.sort(np.abs(FC).flatten())[::-1]
  sorted_fft = np.sort(np.abs(FF).flatten())[::-1]
  total_energy_dct = np.sum(FC ** 2)
  total_energy_dft = np.sum(np.abs(FF) ** 2)
    
  errors_dct = []
  errors_fft = []
    
  M_max = len(sorted_dct)
  if M_max > 50000:
    M_max = 50000
    
  for m in range(1, M_max):
      energy_kept_dct = np.sum(sorted_dct[:m] ** 2)
      energy_kept_dft = np.sum(sorted_fft[:m] ** 2)
      errors_dct.append((total_energy_dct - energy_kept_dct) / total_energy_dct)
      errors_fft.append((total_energy_dft - energy_kept_dft) / total_energy_dft)
    
  return errors_dct, errors_fft

# Renderowanie
def total_time_chart(times):
    width = 0.2 
    
    type_label = ['Total time']
    time_means = {
        'DCT': np.sum(times[0]),
        'Scipy-DCT': np.sum(times[1]),
        'Numpy-FFT': np.sum(times[2])
    }
    
    x = np.arange(len(type_label))  
    fig, ax = plt.subplots(figsize=(6, 5)) 

    num_items = len(time_means)
    total_group_width = num_items * width
    start_pos = x - (total_group_width / 2) + (width / 2)

    for i, (attribute, measurement) in enumerate(time_means.items()):
        pos = start_pos + (i * width)
        rects = ax.bar(pos, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3, fmt='%.4f') 

    ax.set_ylabel('Execution time (s)')
    ax.set_title('Total execution time comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(type_label)
    ax.set_xlim(x[0] - 0.5, x[0] + 0.5) 
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

def show_decompression_efect(gray_image, dct_image, scipy_dct_image):
  plt.figure(figsize=(12, 6))
  plt.subplot(1, 3, 1)
  plt.title("Przed kompresją")
  plt.imshow(gray_image, cmap='gray')
  plt.subplot(1, 3, 2)
  plt.title("Po kompresji DCT")
  plt.imshow(dct_image, cmap ='gray')
  plt.subplot(1, 3, 3)
  plt.title("Po kompresji Scipy DCT")
  plt.imshow(scipy_dct_image, cmap='gray')
  plt.show()

def show_SNR(cropped_image, dct_image, scipy_dct_image, fft_image):
  plt.figure(figsize=(12, 6))
  SNR_dct = calculate_snr(cropped_image, dct_image)
  SNR_scipy_dct = calculate_snr(cropped_image, scipy_dct_image)
  SNR_numpy_fft = calculate_snr(cropped_image, fft_image)
  plt.subplot(1, 3, 1)
  plt.title(f"DCT\nSNR = {SNR_dct:.2f} dB")
  plt.imshow(dct_image, cmap='gray')
  plt.subplot(1, 3, 2)
  plt.title(f"Scipy DCT\nSNR = {SNR_scipy_dct:.2f} dB")
  plt.imshow(scipy_dct_image, cmap ='gray')
  plt.subplot(1, 3, 3)
  plt.title(f"Numpy FFT\nSNR = {SNR_numpy_fft:.2f} dB")
  plt.imshow(fft_image, cmap='gray')
  plt.show()

def show_correlation(cropped_image):
  errors_dct, errors_fft = caculate_aproximation_error(cropped_image)
  
  plt.title("Błąd aproksymacji")
  plt.plot(np.log10(errors_dct), color='red', label='DCT')
  plt.plot(np.log10(errors_fft), color='blue', label='Fourier')
  plt.title("log10(epsilon[M]^2)")
  plt.xlabel("M - liczba zachowanych współczynników")
  plt.ylabel("log10(błąd)")
  plt.show()

def show_coeffcients(cropped_image):
  FC = calculate_dct_coefficients(cropped_image)
  FF = calculate_fft_coefficients(cropped_image)
  
  plt.figure(figsize=(12, 6))
  plt.subplot(1, 2, 1)
  plt.title("Scipy DCT")
  plt.imshow(np.log(1e-5 + np.abs(FC)), cmap='gray')
  plt.subplot(1, 2, 2)
  plt.title("Numpy FFT")
  plt.imshow(np.log(1e-5 + np.abs(FF)), cmap='gray')
  plt.show()

def dct_compression(split_image, img_h, img_w):
  img_block = split_image[img_h, img_w]
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

def combine_dct_image(split_image):
  img_h, img_w = calculate_img_dimensions(split_image)
  image_combined = np.zeros_like(split_image)
  for i in range(img_h):
    for j in range(img_w):
      image_block = dct_compression(split_image, i, j)
      image_combined [i, j] = image_block
  return image_combined 

def scipy_dct(split_image, img_h, img_w):
 with np.printoptions(edgeitems = 8, precision = 2, linewidth = 1000):
  B = dct(dct(split_image, axis=0, norm='ortho'), axis=1, norm='ortho')
  with np.printoptions(edgeitems=8, precision=1, linewidth=1000):
    if(img_h == 0 and img_w == 0):
      print(B)
  M, N = calculate_img_dimensions(split_image)
  mask = calculate_compression_mask(M, N)
  B_compressed = B * mask
  A_compressed = idct(idct(B_compressed, axis=0, norm='ortho'), axis=1, norm='ortho')
  return A_compressed

def combine_scipy_dct_image(split_image):
  img_h, img_w = calculate_img_dimensions(split_image)
  image_combined = np.zeros_like(split_image)
  for i in range(img_h):
    for j in range(img_w):
      image_block = scipy_dct(split_image[i,j], i, j)
      image_combined[i, j] = image_block
  return image_combined

def reshape_combined_image(image_combined, block_size):
  h, w = calculate_img_dimensions(image_combined)
  temp = image_combined.transpose(0, 2, 1, 3)
  image_combined = np.reshape(temp, (h * block_size, w * block_size))
  return image_combined

def numpy_fft(cropped_image, M):
  Transformation = np.fft.fft2(cropped_image)
  mask = np.abs(Transformation) >= M
  Transformation_compressed = Transformation * mask
  fft_image = np.real(np.fft.ifft2(Transformation_compressed))
  return fft_image

def dct_compress_image(split_image, img_h, img_w):
  img_block = split_image[img_h, img_w]
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

  mask = calculate_compression_mask(M,M)
  B_masked = B * mask

  return B_masked

def transform_B_into_row(B_masked):
  B_masked = B_masked[B_masked != 0]
  return B_masked
  
def compress_B(split_image):
    img_h, img_w, M, N = np.shape(split_image)
    all_data = [] 
    
    all_data.extend([img_h])
    all_data.extend([img_w])
    all_data.extend([M])

    for i in range(img_h):
        for j in range(img_w):
          B_masked = dct_compress_image(split_image, i, j)
          compressed_values = transform_B_into_row(B_masked)
          all_data.extend(compressed_values)
    
    final_data = np.array(all_data, dtype=np.float16)
    final_data2 = np.round(final_data, 2)
    np.savez_compressed('kompresja.cwelpeg.npz', data=final_data2)

def decompress_B(filename):
    archive = np.load(filename)
    final_data = archive['data']
    img_h, img_w, M = int(final_data[0]), int(final_data[1]), int(final_data[2])
    final_data = np.delete(final_data, 0, 0)
    final_data = np.delete(final_data, 0, 0)
    final_data = np.delete(final_data, 0, 0)
    mask_size = M // 2
    count_per_block = 0
    for i in range(mask_size):
        for j in range(mask_size - i):
            count_per_block += 1
            
    full_image = np.zeros((img_h * M, img_w * M)) 
    
    PI = np.pi
    m = np.arange(M).reshape(1, -1)
    p = np.arange(M).reshape(-1, 1)
    T = np.cos((PI * (2*m+1) * p) / (2*M))
    T[0, :] /= np.sqrt(M)
    T[1:, :] /= np.sqrt(M / 2)
    
    ptr = 0
    for i in range(img_h):
        for j in range(img_w):
            block_flat = final_data[ptr : ptr + count_per_block]
            ptr += count_per_block
            
            B_reconstructed = np.zeros((M, M))
            v_idx = 0
            for row in range(mask_size):
                for col in range(mask_size - row):
                    if v_idx < len(block_flat):
                        B_reconstructed[row, col] = block_flat[v_idx]
                        v_idx += 1
            
            img_block = T.T @ B_reconstructed @ T
            full_image[i*M : (i+1)*M, j*M : (j+1)*M] = img_block
    return full_image

def compress_image_to_file(split_image):
  compress_B(split_image)
  full_image = decompress_B('kompresja.cwelpeg.npz')
  plt.imshow(full_image, cmap='gray')
  plt.title("CwELpeg")
  plt.axis('off')
  plt.show()

def main():
  gray_image = load_image(path)
  cropped_image = crop_image(gray_image, matrix_size)
  split_image = split_to_bloks(cropped_image, matrix_size)
  show_blocks_grid(split_image)
  
  times = np.zeros((3,1))
  start = start_time_measure()
  image_compressed_dct = combine_dct_image(split_image)
  time = end_time_measure(start)
  times[0] = time
  
  start = start_time_measure()
  image_compressed_scipy_dct = combine_scipy_dct_image(split_image)
  time = end_time_measure(start)
  times[1] = time
  
  start = start_time_measure()
  fft_image = numpy_fft(cropped_image, 50)
  time = end_time_measure(start)
  times[2] = time

  total_time_chart(times)

  dct_image = reshape_combined_image(image_compressed_dct, matrix_size)
  scipy_dct_image = reshape_combined_image(image_compressed_scipy_dct, matrix_size)
  show_decompression_efect(gray_image, dct_image, scipy_dct_image)

  show_SNR(cropped_image, dct_image, scipy_dct_image, fft_image)
  show_correlation(cropped_image)
  show_coeffcients(cropped_image)

  compress_image_to_file(split_image)

main()