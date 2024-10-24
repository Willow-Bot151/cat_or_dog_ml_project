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

max_image_size = find_max_image_size("data")
capped_max_size=(min(max_image_size[0], 250),min(max_image_size[1],250))
clean_temp_image_path(dir_path="data")
shrink_all_images(
    dir_path="data",
    max_size=(capped_max_size))


## create tensorflow datasets
tf_ds = {
    'train':create_ds("data/train",image_size=capped_max_size),
    'test':None,
    'validation':create_ds("data/validation",image_size=capped_max_size)
    }

print(tf_ds)

num_classes = len(tf_ds["train"].class_names)

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape = (
        capped_max_size[0],
        capped_max_size[1],
        3)), 
    tf.keras.layers.Conv2D(16,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(2)
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

epochs = 10

history = model.fit(
    tf_ds["train"],
    validation_data = tf_ds['validation'],
    epochs = epochs)

print(model.summary())

## Visualising training results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1,2,2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.savefig(fname='temp/TrainingAndValidation')