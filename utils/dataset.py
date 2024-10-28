import tensorflow as tf
from pathlib import Path
from tifffile import TiffFile
import gc
import numpy as np
import csv


def make_dataset(filename) -> tf.data.Dataset:
    with TiffFile(filename) as tif:
        tmp_numpy = tif.asarray()

    tmp_tensor = tf.convert_to_tensor(tmp_numpy, dtype=tf.float32)

    dataset = tf.data.Dataset.from_tensor_slices(tmp_tensor)

    del tmp_numpy
    del tmp_tensor
    gc.collect()

    return dataset


def load_runrest(filename) -> np.ndarray:
    with open(filename) as f:
        reader = csv.reader(f)
        labels = [row for row in reader]

    return np.array(labels)
