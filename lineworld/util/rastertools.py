import numpy as np


def normalize_to_uint8(data):
    return (np.iinfo(np.uint8).max * ((data - np.min(data)) / np.ptp(data))).astype(np.uint8)
