import timeit

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