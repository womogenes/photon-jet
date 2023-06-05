# Utility functions and whatnot
import math
import tensorflow as tf

def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])

def print_gpu_memory():
    memory_usage = tf.config.experimental.get_memory_info("GPU:0")
    print("GPU memory usage")
    print(f"  current: {convert_size(memory_usage['current'])}")
    print(f"  peak:    {convert_size(memory_usage['peak'])}")
