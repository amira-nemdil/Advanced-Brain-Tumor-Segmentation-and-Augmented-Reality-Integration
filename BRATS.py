import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import nibabel as nib
from glob import glob
import matplotlib.pyplot as plt

# =====================================================
# CONFIGURATION
# =====================================================

DATASET_DIR = r"D:\Khedir-meriem-ESI-SBElAbes\data"  # Change this to your BraTS dataset folder
IMAGE_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 50

# Check GPU availability
device_name = tf.test.gpu_device_name()
if device_name:
    print(f"GPU detected: {device_name}")
else:
    print("No GPU found. Training will be slow!")

# =====================================================
# DATA LOADER
# =====================================================

def load_nifti_image(path):
    """ Load a NIfTI file and return numpy array """
    img = nib.load(path)
    data = img.get_fdata()
    return data

def preprocess_image(image, mask):
    """ Preprocess images: normalize and resize """
    image = tf.image.resize(image[..., np.newaxis], (IMAGE_SIZE, IMAGE_SIZE))
    mask = tf.image.resize(mask[..., np.newaxis], (IMAGE_SIZE, IMAGE_SIZE))

    image = tf.cast(image, tf.float32) / tf.reduce_max(image)
    mask = tf.cast(mask > 0, tf.float32)  # Binary mask

    return image, mask

def data_generator(image_paths, mask_paths):
    """ Generator for loading and preprocessing data """
    while True:
        for img_path, mask_path in zip(image_paths, mask_paths):
            img = load_nifti_image(img_path)
            mask = load_nifti_image(mask_path)

            # Take middle slice for simplicity
            mid_slice = img.shape[2] // 2
            img_slice = img[:, :, mid_slice]
            mask_slice = mask[:, :, mid_slice]

            img_tensor, mask_tensor = preprocess_image(img_slice, mask_slice)

            yield tf.expand_dims(img_tensor, 0), tf.expand_dims(mask_tensor, 0)

# =====================================================
# MODEL (U-Net)
# =====================================================

def unet_model(input_size=(IMAGE_SIZE, IMAGE_SIZE, 1)):
    inputs = layers.Input(input_size)

    # Encoder
    c1 = layers.Conv2D(16, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(16, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D()(c1)

    c2 = layers.Conv2D(32, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(32, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D()(c2)

    c3 = layers.Conv2D(64, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(64, 3, activation='relu', padding='same')(c3)

    # Decoder
    u1 = layers.UpSampling2D()(c3)
    u1 = layers.Concatenate()([u1, c2])
    c4 = layers.Conv2D(32, 3, activation='relu', padding='same')(u1)
    c4 = layers.Conv2D(32, 3, activation='relu', padding='same')(c4)

    u2 = layers.UpSampling2D()(c4)
    u2 = layers.Concatenate()([u2, c1])
    c5 = layers.Conv2D(16, 3, activation='relu', padding='same')(u2)
    c5 = layers.Conv2D(16, 3, activation='relu', padding='same')(c5)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c5)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

# =====================================================
# COMPILE AND TRAIN
# =====================================================

# Example dataset (you should create your list of image/mask paths)
image_files = sorted(glob(os.path.join(DATASET_DIR, "imagesTr", "*.nii.gz")))
mask_files = sorted(glob(os.path.join(DATASET_DIR, "labelsTr", "*.nii.gz")))

train_gen = data_generator(image_files, mask_files)

model = unet_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

steps_per_epoch = len(image_files) // BATCH_SIZE

model.fit(train_gen, steps_per_epoch=steps_per_epoch, epochs=EPOCHS)

# =====================================================
# SAVE MODEL
# =====================================================

model.save("brain_tumor_unet.h5")
print("Model saved!")
