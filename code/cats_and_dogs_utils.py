import os
from PIL import Image, UnidentifiedImageError
import tensorflow as tf
import numpy as np
from shutil import rmtree

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

def shrink_and_save_image(old_image_path, new_image_path, max_size = (150,150)):
    image = Image.open(old_image_path)
    image.thumbnail(max_size)
    image.save(new_image_path)


def clean_temp_image_path(dir_path):
    if not os.path.exists("temp"):
        os.mkdir("temp")
    temp_image_path = os.path.join("temp",dir_path)
    if os.path.exists(temp_image_path):
        rmtree(temp_image_path)
    # os.mkdir(temp_image_path)

def shrink_all_images(dir_path, max_size = (150,150),target_parent_path = "temp"):
    valid_image_suffix = [
        ".jpeg",
        ".png",
        ".jpg"]
    items_in_dir = os.listdir(dir_path)
    if not items_in_dir:
        return 
    for item in items_in_dir:
        if item.startswith(".") or item.startswith(".."):
            pass
        else:
            for suffix in valid_image_suffix:
                if os.path.join(dir_path,item).endswith(suffix):
                    shrink_and_save_image(
                        old_image_path=os.path.join(dir_path,item),
                        new_image_path=os.path.join(target_parent_path,dir_path,item),
                        max_size=max_size)
            if os.path.isdir(os.path.join(dir_path,item)):
                if not os.path.exists(os.path.join(target_parent_path,dir_path)):
                    os.mkdir(os.path.join(target_parent_path,dir_path))
                os.mkdir(os.path.join(target_parent_path,dir_path,item))
                shrink_all_images(
                    dir_path=os.path.join(dir_path,item),
                    max_size=max_size,
                    target_parent_path=target_parent_path)