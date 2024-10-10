import os
import numpy as np
import matplotlib as plt
import tensorflow as tf
from cats_and_dogs_utils import *

## count sample size
data_set_size = {}
DATA_PATH = "data"
data_set_size["train"] = count_files_in_dir(os.path.join(DATA_PATH, "train"))
data_set_size["test"] = count_files_in_dir(os.path.join(DATA_PATH, "test"))
data_set_size["validation"] = count_files_in_dir(os.path.join(DATA_PATH, "validation"))
print(data_set_size)


## get max image size
files_path = "data/train/cats"


## create dataset
## for each directory
tf.keras.preprocessing.image_dataset_from_directory(
    "data/train",
    batch_size = 32,
    image_size = (256,256),
    shuffle = True,
    pad_to_aspect_ratio = True



)
