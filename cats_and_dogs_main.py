import os
import numpy as np
import matplotlib as plt
import tensorflow as tf
from cats_and_dogs_utils import *

data_set_size = {}
DATA_PATH = "./cats_and_dogs"
data_set_size["train"] = count_files_in_dir(os.path.join(DATA_PATH, "train"))
data_set_size["test"] = count_files_in_dir(os.path.join(DATA_PATH, "test"))
data_set_size["validation"] = count_files_in_dir(os.path.join(DATA_PATH, "validation"))
print(data_set_size)