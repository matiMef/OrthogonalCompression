import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct

def show_time_chart(times):
    labels = ['DCT', 'Scipy-DCT', 'Numpy-FFT', 'SFT']
    values = [np.sum(t) for t in times]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=['tab:blue', 'tab:orange', 'tab:green', 'tab:red'])
    
    plt.bar_label(bars, padding=3, fmt='%.4f')
    plt.ylabel('Czas (s)')
    plt.title('Czas pracy algorytmów')
    plt.tight_layout()
    plt.show()

def show_blocks_grid(split_image):
  plt.title("Obraz po podziale na bloki")
  for i in range(15):
    for j in range(15):
      plt.subplot(15, 15, i*15 + j + 1)
      plt.imshow(split_image[i, j], cmap = 'gray')
      plt.axis('off')
  plt.show()

def show_decompression_efect(cropped_image, dct_image, fft_image, sft_image):
  plt.figure(figsize=(16, 5))
  plt.subplot(1, 4, 1)
  plt.title("Przed kompresją")
  plt.imshow(cropped_image, cmap='gray')
  plt.axis('off')
  
  plt.subplot(1, 4, 2)
  plt.title("Po kompresji DCT")
  plt.imshow(dct_image, cmap ='gray')
  plt.axis('off')
  
  plt.subplot(1, 4, 3)
  plt.title("Po kompresji Scipy FFT")
  plt.imshow(fft_image, cmap='gray')
  plt.axis('off')
  
  plt.subplot(1, 4, 4)
  plt.title("Po kompresji SFT")
  plt.imshow(sft_image, cmap='gray')
  plt.axis('off')
  plt.show()


def calculate_snr(cropped_image, combined_image):
  og = np.sum(cropped_image ** 2)
  dif = np.sum((cropped_image - combined_image) ** 2 )
  SNR = 10 * np.log10 (og/dif)
  return SNR

def show_snr(cropped_image, dct_image, scipy_dct_image, fft_image, sft_image):
  SNR_dct = calculate_snr(cropped_image, dct_image)
  SNR_scipy_dct = calculate_snr(cropped_image, scipy_dct_image)
  SNR_numpy_fft = calculate_snr(cropped_image, fft_image)
  SNR_numpy_sft = calculate_snr(cropped_image, sft_image)
  
  plt.figure(figsize=(12, 10)) 
  plt.subplot(2, 2, 1)
  plt.title(f"DCT\nSNR = {SNR_dct:.2f} dB")
  plt.imshow(dct_image, cmap='gray')
  
  plt.subplot(2, 2, 2)
  plt.title(f"Scipy DCT\nSNR = {SNR_scipy_dct:.2f} dB")
  plt.imshow(scipy_dct_image, cmap ='gray')
  
  plt.subplot(2, 2, 3)
  plt.title(f"Numpy FFT\nSNR = {SNR_numpy_fft:.2f} dB")
  plt.imshow(fft_image, cmap='gray')
 
  plt.subplot(2, 2, 4)
  plt.title(f"SFT\nSNR = {SNR_numpy_sft:.2f} dB")
  plt.imshow(sft_image, cmap='gray')
  plt.tight_layout()
  plt.show()

def show_phase_comparison(cropped_image, threshold):
  FF = calculate_fft_coefficients(cropped_image)

  ff_row = FF[0, :]
  freqs = np.fft.fftfreq(ff_row.size)
  freqs_shifted = np.fft.fftshift(freqs)
  ff_row_shifted = np.fft.fftshift(ff_row)

  phase_before = np.angle(ff_row_shifted)

  mask = np.abs(ff_row_shifted) >= threshold
  phase_after = np.where(mask, phase_before, np.nan)

  kept_pct = 100 * np.sum(mask) / mask.size

  fig, axes = plt.subplots(3, 1, figsize=(12, 10))
  fig.suptitle(f"Faza FFT — porównanie przed i po kompresji (próg = {threshold})")

  axes[0].plot(freqs_shifted, np.abs(ff_row_shifted), color='steelblue', linewidth=1)
  axes[0].axhline(y=threshold, color='red', linestyle='--', linewidth=1, label=f'próg = {threshold}')
  axes[0].set_title("Amplituda widma FFT")
  axes[0].set_ylabel("|F(k)|")
  axes[0].set_xlabel("Częstotliwość")
  axes[0].legend()
  axes[0].grid(True, alpha=0.3)

  axes[1].plot(freqs_shifted, phase_before, color='steelblue', linewidth=0.8, label='przed kompresją')
  axes[1].set_title("Faza przed kompresją")
  axes[1].set_ylabel("Faza (rad)")
  axes[1].set_xlabel("Częstotliwość")
  axes[1].set_ylim(-np.pi - 0.2, np.pi + 0.2)
  axes[1].set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
  axes[1].set_yticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
  axes[1].grid(True, alpha=0.3)
  axes[1].legend()

  phase_kept_line = np.where(mask, phase_before, np.nan)
  axes[2].plot(freqs_shifted, phase_before, color='steelblue', linewidth=0.8, alpha=0.25, label='przed kompresją')
  axes[2].plot(freqs_shifted, phase_kept_line, color='tomato', linewidth=1.2, label=f'po kompresji ({kept_pct:.1f}% zachowanych)')
  axes[2].set_title("Faza po kompresji — zachowane współczynniki (przerwy = wyzerowane)")
  axes[2].set_ylabel("Faza (rad)")
  axes[2].set_xlabel("Częstotliwość")
  axes[2].set_ylim(-np.pi - 0.2, np.pi + 0.2)
  axes[2].set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
  axes[2].set_yticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
  axes[2].grid(True, alpha=0.3)
  axes[2].legend()

  plt.tight_layout()
  plt.show()


def calculate_dct_coefficients(cropped_image):
  FC = dct(dct(cropped_image, axis = 0, norm='ortho'), axis=1, norm='ortho')
  return FC

def calculate_fft_coefficients(cropped_image):
  FF = np.fft.fft2(cropped_image)
  return FF

def caculate_aproximation_error(cropped_image):
  FC = calculate_dct_coefficients(cropped_image)
  FF = calculate_fft_coefficients(cropped_image)
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

def show_correlation(cropped_image):
  errors_dct, errors_fft = caculate_aproximation_error(cropped_image)
  
  plt.title("Błąd aproksymacji")
  plt.plot(np.log10(errors_dct), color='red', label='DCT')
  plt.plot(np.log10(errors_fft), color='blue', label='Fourier')
  plt.xlabel("M - liczba zachowanych współczynników")
  plt.ylabel("log10(błąd)")
  plt.show()

def show_coeffcients(cropped_image):
  FC = calculate_dct_coefficients(cropped_image)
  FF = calculate_fft_coefficients(cropped_image)
  
  plt.figure(figsize=(18, 6))
  plt.subplot(1, 2, 1)
  plt.title("Scipy DCT")
  plt.imshow(np.log(1e-5 + np.abs(FC)), cmap='gray')
  
  plt.subplot(1, 2, 2)
  plt.title("Numpy FFT")
  plt.imshow(np.log(1e-5 + np.abs(FF)), cmap='gray')
  plt.show()






