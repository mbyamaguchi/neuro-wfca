import tensorflow as tf
from pathlib import Path
from tifffile import TiffFile
import gc
import numpy as np
import csv


def load_tiffseq(filename) -> np.ndarray:
    with TiffFile(filename) as tif:
        return tif.asarray()


def min_max(tiffnumpy) -> np.ndarray:
    minvalue = tiffnumpy.min(axis=0)
    maxvalue = tiffnumpy.max(axis=0)

    return np.array(
        [(x - minvalue) / (maxvalue - minvalue) for x in tiffnumpy]
    )


def make_dataset(tiffnumpy) -> tf.data.Dataset:
    tmp_tensor = tf.convert_to_tensor(tiffnumpy, dtype=tf.float32)

    dataset = tf.data.Dataset.from_tensor_slices(tmp_tensor)
    del tmp_tensor
    gc.collect()

    return dataset


def load_runrest(filename) -> np.ndarray:
    with open(filename) as f:
        reader = csv.reader(f)
        labels = [row for row in reader]

    return np.array(labels, dtype=np.uint8)