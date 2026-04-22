import numpy as np
import matplotlib.pyplot as plt

def calculate_image_dimensions(image):
  img_h = np.shape(image)[0]
  img_w = np.shape(image)[1]
  return [img_h, img_w]

def calculate_compression_mask(M, N):
  mask_size = M // 2
  mask = np.zeros((M, N))
  for i in range (0, mask_size):
    for j in range(mask_size - i):
      mask[i, j] = 1
  return mask

def dct_compress_image(split_image, img_h, img_w ):
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
    np.savez_compressed('compressions/kompresja.celpeg.npz', data=final_data2)

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

def save_dct_image_to_file(split_image):
  compress_B(split_image)
  full_image = decompress_B('compressions/kompresja.celpeg.npz')
  plt.imshow(full_image, cmap='gray')
  plt.title("CELpeg")
  plt.axis('off')
  plt.show()