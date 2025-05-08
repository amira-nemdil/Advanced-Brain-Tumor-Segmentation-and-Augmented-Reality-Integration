import os
import numpy as np
import tensorflow as tf
from keras import layers, models
import nibabel as nib
from glob import glob
import matplotlib.pyplot as plt

# =====================================================
# CONFIGURATION
# =====================================================

DATASET_DIR = r"your_dataset_path_here"  # Path to the BRATS dataset
IMAGE_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 50

# Check GPU
device_name = tf.test.gpu_device_name()
if device_name:
    print(f"GPU detected: {device_name}")
else:
    print("No GPU found.")

# =====================================================
# DATA LOADER
# =====================================================

def load_nifti_image(path):
    img = nib.load(path)
    data = img.get_fdata()
    return data

def preprocess_image(image, mask):
    image = tf.image.resize(image[..., np.newaxis], (IMAGE_SIZE, IMAGE_SIZE))
    mask = tf.image.resize(mask[..., np.newaxis], (IMAGE_SIZE, IMAGE_SIZE))

    image = tf.cast(image, tf.float32) / tf.reduce_max(image)
    mask = tf.cast(mask > 0, tf.float32)
    return image, mask

def data_generator(image_paths, mask_paths):
    while True:
        for img_path, mask_path in zip(image_paths, mask_paths):
            img = load_nifti_image(img_path)
            mask = load_nifti_image(mask_path)

            mid_slice = img.shape[2] // 2
            img_slice = img[:, :, mid_slice]
            mask_slice = mask[:, :, mid_slice]

            img_tensor, mask_tensor = preprocess_image(img_slice, mask_slice)

            yield tf.expand_dims(img_tensor, 0), tf.expand_dims(mask_tensor, 0)

# =====================================================
# MODELS
# =====================================================

# --- Basic U-Net ---
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

    return models.Model(inputs, outputs)

# --- Attention Block ---
def attention_block(x, g, inter_channel):
    theta_x = layers.Conv2D(inter_channel, 1)(x)
    phi_g = layers.Conv2D(inter_channel, 1)(g)
    add = layers.Add()([theta_x, phi_g])
    act = layers.Activation('relu')(add)
    psi = layers.Conv2D(1, 1, activation='sigmoid')(act)
    return layers.Multiply()([x, psi])

# --- Attention U-Net ---
def attention_unet(input_size=(IMAGE_SIZE, IMAGE_SIZE, 1)):
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

    # Decoder with Attention
    g1 = layers.UpSampling2D()(c3)
    att1 = attention_block(c2, g1, 32)
    u1 = layers.Concatenate()([g1, att1])
    c4 = layers.Conv2D(32, 3, activation='relu', padding='same')(u1)

    g2 = layers.UpSampling2D()(c4)
    att2 = attention_block(c1, g2, 16)
    u2 = layers.Concatenate()([g2, att2])
    c5 = layers.Conv2D(16, 3, activation='relu', padding='same')(u2)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c5)

    return models.Model(inputs, outputs)

# =====================================================
# TRAINING
# =====================================================

image_files = sorted(glob(os.path.join(DATASET_DIR, "imagesTr", "*.nii.gz")))
mask_files = sorted(glob(os.path.join(DATASET_DIR, "labelsTr", "*.nii.gz")))

train_gen = data_generator(image_files, mask_files)

# ---- Choose model ----
print("Choose model type: 1) U-Net  2) Attention U-Net")
model_choice = input("Enter 1 or 2: ")

if model_choice == "2":
    model = attention_unet()
    print("Using Attention U-Net.")
else:
    model = unet_model()
    print("Using Basic U-Net.")

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

steps_per_epoch = len(image_files) // BATCH_SIZE

model.fit(train_gen, steps_per_epoch=steps_per_epoch, epochs=EPOCHS)

# SAVE
model.save("brain_tumor_segmentation_model.h5")
print("Model saved.")
