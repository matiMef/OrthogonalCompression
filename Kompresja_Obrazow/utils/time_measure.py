import timeit
import os
import statistics

def start_time_measure():
    start = timeit.default_timer()
    return start

def end_time_measure(start):
    end = timeit.default_timer()
    time_elapsed = end - start
    time_elapsed = (float(f'{time_elapsed:.4f}'))
    return time_elapsed

def time_dct(split_image, dct_block_size):
    from algorithms.dct import apply_dct_to_all_blocks, merge_blocks_into_image
    start = start_time_measure()
    compressed_image = apply_dct_to_all_blocks(split_image)
    merge_blocks_into_image(compressed_image, dct_block_size)
    time = end_time_measure(start)
    return time 

def time_scipy_dct(split_image, dct_block_size):
    from algorithms.dct import apply_scipy_dct_to_all_blocks, merge_blocks_into_image
    start = start_time_measure()
    compressed_image = apply_scipy_dct_to_all_blocks(split_image)
    merge_blocks_into_image(compressed_image, dct_block_size)
    time = end_time_measure(start)
    return time

def time_numpy_fft(cropped_image):
    from algorithms.fft import numpy_fft
    start = start_time_measure()
    numpy_fft(cropped_image, 50)
    time = end_time_measure(start)
    return time

def time_sft(cropped_image):
    from algorithms.sft import sft
    start = start_time_measure()
    sft(cropped_image, keep_fraction=0.0001)
    time = end_time_measure(start)
    return time

def load_and_prepare_image(path, dct_block_size):
    from utils.image import load_image, crop_image
    gray_image = load_image(path)
    cropped_image = crop_image(gray_image, dct_block_size)
    return cropped_image

def time_benchmark(folder_path, count, dct_block_size):
    from utils.image import split_to_bloks
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    total_files = len(files)
    
    benchmark_results = {}

    for index, file in enumerate(files):
        path = os.path.join(folder_path, file)
        cropped_image = load_and_prepare_image(path, dct_block_size)
        split_image = split_to_bloks(cropped_image, dct_block_size)
        print(f"--- Benchmark dla: {file} ({count} powtórzeń) ---")

        times = {
            "dct": [],
            "scipy_dct": [],
            "fft": [],
            "sft": []
        }

        for i in range(count):
            times["dct"].append(time_dct(split_image, dct_block_size))
            times["scipy_dct"].append(time_scipy_dct(split_image, dct_block_size))
            times["fft"].append(time_numpy_fft(cropped_image))
            times["sft"].append(time_sft(cropped_image))

        avg_results = {k: statistics.mean(v) for k, v in times.items()}
        benchmark_results[file] = avg_results

        processed_count = index + 1
        percentage = (processed_count / total_files) * 100
        print(f"Postęp: {percentage:3.0f}% | Przetworzono {processed_count}/{total_files} plików ({file})")

    return benchmark_results

def dct_time_benchmark(folder_path):
    from utils.image import split_to_bloks

    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    total_files = len(files)
    
    benchmark_results = {}
    dct_block_sizes = [8, 16, 32, 64, 96]

    for index, file in enumerate(files):
        path = os.path.join(folder_path, file)
        print(f"--- Benchmark dla: {file} ---")

        file_performance = {}

        for size in dct_block_sizes:
            cropped_image = load_and_prepare_image(path, size)
            split_image = split_to_bloks(cropped_image, size)
            elapsed_time = time_dct(split_image, size)
            file_performance[f"DCT_{size}"] = elapsed_time

        benchmark_results[file] = file_performance
        processed_count = index + 1
        percentage = (processed_count / total_files) * 100
        print(f"Postęp: {percentage:3.0f}% | Przetworzono {processed_count}/{total_files} plików")

    return benchmark_results