import os
from PIL import Image, UnidentifiedImageError
import tensorflow as tf
import numpy as np

def count_files_in_dir(dir_path):
    count = 0
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            count += 1
    return count

def find_max_image_size(dir_path):
    max_width = 0
    max_height = 0
    width, height = 0,0
    for root, dirs,files in os.walk(dir_path):
        for file in files:
            if not (file.startswith('.') or file.startswith('..')):
                try:
                    file_path = os.path.join(root,file)
                    width, height = Image.open(file_path).size
                except UnidentifiedImageError:
                    for dir in dirs:
                        try: 
                            file_path = os.path.join(dir,file)
                            width, height = Image.open(file_path).size
                        except UnidentifiedImageError:
                            pass
                max_height = max(max_height, height)
                max_width = max(max_width, width)
    return max_width, max_height

def create_ds(dir_path, image_size):
    dataset = tf.keras.utils.image_dataset_from_directory(
        dir_path,
        batch_size = 32,
        image_size = image_size,
        shuffle = True,
        pad_to_aspect_ratio = True,
        labels = 'inferred',
        label_mode = 'binary',
        class_names = ['cats','dogs'],
        interpolation = 'bilinear')
    return dataset


