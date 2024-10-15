import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from cats_and_dogs_utils import *

## count sample size
data_set_size = {}
DATA_PATH = "data"
data_set_size["train"] = count_files_in_dir(os.path.join(DATA_PATH, "train"))
data_set_size["test"] = count_files_in_dir(os.path.join(DATA_PATH, "test"))
data_set_size["validation"] = count_files_in_dir(os.path.join(DATA_PATH, "validation"))
print(data_set_size)


## create tensorflow datasets
tf_ds = {
    'train':create_ds("data/train",image_size=find_max_image_size("data/train")),
    'test':None,
    'validation':create_ds("data/validation",image_size=find_max_image_size("data/validation"))
    }

# for k in tf_ds:
#     dir_path = os.path.join('data',k)
#     print(tf_ds)
#     print(dir_path)

#     tf_ds[k] = create_ds(dir_path,image_size=find_max_image_size(dir_path))

## create training dataset

## for each directory
# dataset = tf.keras.utils.image_dataset_from_directory(
#     files_path,
#     batch_size = 32,
#     image_size = find_max_image_size(files_path),
#     shuffle = True,
#     pad_to_aspect_ratio = True,
#     seed = 123,
#     labels = 'inferred',
#     label_mode = 'int',
#     interpolation = 'bilinear')

print(tf_ds)



# ## visualising the dataset
# class_names = tf_ds["train"].class_names

# plt.figure(figsize=(10, 10))
# for images, labels in tf_ds["train"].take(1):
#   for i in range(2):
#     ax = plt.subplot(2, 1, i + 1)
#     plt.imshow(images[i].numpy().astype("uint8"))
#     plt.title(class_names[labels[i]])
#     plt.axis("off")
# plt.savefig('temp/ds_sample_plot.png')

normalization_layer = tf.keras.layers.Rescaling(1./255)

num_classes = len(tf_ds["train"].class_names)

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape = (
        find_max_image_size("data/train")[0],
        find_max_image_size("data/train")[1],
        3)), 
    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dense(2)
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

history = model.fit(
    tf_ds["train"],
    validation_data = tf_ds['validation'],
    epochs = 10)

print(model.summary())