import sys
import numpy as np
from utils import time_measure, image, visualizations, sft_visualisation
from algorithms import dct, dct_format, fft, sft

dct_block_size = 64
path = "test_models/test_model2.jpg"
files_path = "test_models/"

def main():
    gray_image = image.load_image(path)
    cropped_image = image.crop_image(gray_image, dct_block_size)
    split_image = image.split_to_bloks(cropped_image, dct_block_size)
    
    dct_compressed_image = dct.apply_dct_to_all_blocks(split_image)
    dct_image = dct.merge_blocks_into_image(dct_compressed_image, dct_block_size)
    scipy_dct_compressed_image = dct.apply_scipy_dct_to_all_blocks(split_image)
    scipy_dct_image = dct.merge_blocks_into_image(scipy_dct_compressed_image, dct_block_size)
    fft_image = fft.numpy_fft(cropped_image, 50)
    sft_image = sft.sft(cropped_image, keep_fraction=0.01)
    
    visualizations.show_blocks_grid(split_image)
    visualizations.show_decompression_efect(cropped_image, dct_image, fft_image, sft_image)
    visualizations.show_snr(cropped_image, dct_image, scipy_dct_image, fft_image, sft_image)
    visualizations.show_metrics_comparison(cropped_image, dct_image, scipy_dct_image, fft_image, sft_image)
    visualizations.show_phase_comparison(cropped_image, 50)
    visualizations.show_correlation(cropped_image)
    visualizations.show_coeffcients(cropped_image)

    sft_visualisation.show_sft_steps(cropped_image, keep_fraction=0.01)

    times = np.zeros((4,1))
    times[0] = time_measure.time_dct(split_image, dct_block_size)
    times[1] = time_measure.time_scipy_dct(split_image, dct_block_size)
    times[2] = time_measure.time_numpy_fft(cropped_image)
    times[3] = time_measure.time_sft(cropped_image)
    visualizations.show_time_chart(times)

    dct_format.save_dct_image_to_file(split_image)

    benchmark_results = time_measure.time_benchmark(files_path, 1, dct_block_size)
    visualizations.show_benchmark_chart(benchmark_results)

    dct_benchmark_results = time_measure.dct_time_benchmark(files_path)
    visualizations.show_dct_benchmark_chart(dct_benchmark_results)
    
if __name__ == "__main__":
  sys.exit(main())