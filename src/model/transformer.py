import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import warnings
warnings.filterwarnings("ignore")
num_classes = 100
input_shape = (32, 32, 3)

#data_dir = tf.keras.utils.image_dataset_from_directory(
#"train",
#labels='inferred',
#label_mode='int',
#class_names=None,
#color_mode='rgb',
#batch_size=32,
#image_size=(96, 96),
#shuffle=True,
#seed=None,
#validation_split=None,
#subset=None,
#interpolation='bilinear',
#follow_links=False,
#crop_to_aspect_ratio=False
#)


#(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()

#print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
#print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")