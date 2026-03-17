import matplotlib
import matplotlib.pyplot as plt
import skimage.io as io
import numpy as np
import skimage.color as color

def crop_image(image, crop_size):
  h, w = image.shape
  new_h = (h // crop_size) * crop_size
  new_w = (w // crop_size) * crop_size
  return image[:new_h, :new_w]

def splint_to_bloks(image, block_size):
  h, w = image.shape
  splited = np.reshape(image, (h // block_size, block_size, w // block_size, block_size))
  splited = np.transpose(splited,(0,2,1,3))
  return splited

path = "test_model.jpg"
image = io.imread(path)
image = color.rgb2gray(image)
image = crop_image(image, 8)

img = splint_to_bloks(image, 8)

img_h = np.shape(img)[0]
img_w = np.shape(img)[1]
# print(img_h, img_w)

def dct_compression(img, img_h, img_w):
 with np.printoptions(edgeitems=8, precision=2, linewidth=1000):
  # print("blok",img[img_h, img_w])
  
  img_block = img[img_h, img_w]

  M = img_block.shape[0]
  N = img_block.shape[1]
  PI = np.pi

  alfa_p = np.full((M, 1), np.sqrt(2/M))
  alfa_q = np.full((1, N), np.sqrt(2/N))
  alfa_p[0, 0] = 1/np.sqrt(M)
  alfa_q[0, 0] = 1/np.sqrt(N)
  # alfa_pq = alfa_p * alfa_q
  # print(alfa_pq)

  m=np.arange(M)
  p=np.arange(M)
  m=np.reshape(m, (1, -1))
  p=np.reshape(p, (-1, 1))
  T = np.cos((PI * (2*m+1) * p) / (2*M))

  T[0, :] /= np.sqrt(M)         
  T[1:, :] /= np.sqrt(M / 2)
  # test = T @ T.T
  # print(np.round(test))

  B = (T @ img_block @ T.T)
  # print("Po transformacji",B)  
  
  mask_size = M//2
  mask = np.zeros((M,N))
  for i in range (0, mask_size):
    for j in range(mask_size-i):
      mask[i, j] = 1
  
  B_masked = B * mask

  A_compressed = T.T @ B_masked @ T
  return A_compressed
  # print("Po kompresji", A_compressed)

img_compressed = np.zeros_like(img)

for i in range(img_h):
   for j in range(img_w):
    # print(img.shape)
    block = dct_compression(img, i, j)
    img_compressed[i, j] = block 

print(img_compressed)
print(img_compressed.shape)


def to_image(image, block_size):
  h = image.shape[0]
  w = image.shape[1]
  print(h, w)
  temp = image.transpose(0, 2, 1, 3)
  combined = np.reshape(temp, (h * block_size, w * block_size))
  
  return combined

new_image = to_image(img_compressed, 8)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Przed kompresją")
plt.imshow(image, cmap='gray')
plt.subplot(1, 2, 2)
plt.title("Po kompresji")
plt.imshow(new_image, cmap='gray')
plt.show()

# print(img.shape)
# print(img)

#wyświetlenie fragmnetu obrazu z podziałem na bloki
# for i in range(30):
#   for j in range(30):
#     plt.subplot(30, 30, i*30 + j + 1)
#     plt.imshow(img[i,j], cmap='gray')
#     plt.axis('off')
# matplotlib.pyplot.imshow(img[30,0], cmap='gray')
# matplotlib.pyplot.show()
# print(type(image))
# print(image.shape)