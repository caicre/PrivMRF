import numpy as np

# cupy is optional: without it (or without a GPU) PrivMRF runs on numpy
try:
    import cupy as cp
except ImportError:
    cp = None


def gpu_available():
    if cp is None:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def get_xp(gpu):
    return cp if gpu else np


def array_module(array):
    if cp is None:
        return np
    return cp.get_array_module(array)


def to_numpy(array):
    if cp is not None and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return array
