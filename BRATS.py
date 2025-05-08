import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import nibabel as nib
from glob import glob
import matplotlib.pyplot as plt

# =====================================================
# CONFIGURATION
# =====================================================

DATASET_DIR = r"D:\Khedir-meriem-ESI-SBElAbes\data"
IMAGE_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 30

# Seed for reproducibility
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

# GPU check
device_name = tf.test.gpu_device_name()
if device_name:
    print(f"✅ GPU detected: {device_name}")
else:
    print("❌ No GPU found. Training may be slow.")

# =====================================================
# DATA LOADER + AUGMENTATION
# =====================================================

def load_nifti_image(path):
    img = nib.load(path)
    return img.get_fdata()

def preprocess_image(image, mask):
    image = tf.image.resize(image[..., np.newaxis], (IMAGE_SIZE, IMAGE_SIZE))
    mask = tf.image.resize(mask[..., np.newaxis], (IMAGE_SIZE, IMAGE_SIZE))
    
    image = tf.cast(image, tf.float32) / tf.reduce_max(image)
    mask = tf.cast(mask > 0, tf.float32)
    
    return image, mask

def augment_image(image, mask):
    # Random flip
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)

    # Random rotation
    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k)
    mask = tf.image.rot90(mask, k)

    # Random brightness
    image = tf.image.random_brightness(image, max_delta=0.1)
    
    return image, mask

def data_generator(image_paths, mask_paths, augment=False):
    while True:
        for img_path, mask_path in zip(image_paths, mask_paths):
            img = load_nifti_image(img_path)
            mask = load_nifti_image(mask_path)

            mid_slice = img.shape[2] // 2
            img_slice = img[:, :, mid_slice]
            mask_slice = mask[:, :, mid_slice]

            img_tensor, mask_tensor = preprocess_image(img_slice, mask_slice)

            if augment:
                img_tensor, mask_tensor = augment_image(img_tensor, mask_tensor)

            yield tf.expand_dims(img_tensor, 0), tf.expand_dims(mask_tensor, 0)

# =====================================================
# MODELS (U-Net + Attention U-Net)
# =====================================================

def unet_model(input_size=(IMAGE_SIZE, IMAGE_SIZE, 1)):
    inputs = layers.Input(input_size)
    c1 = layers.Conv2D(16, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(16, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D()(c1)

    c2 = layers.Conv2D(32, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(32, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D()(c2)

    c3 = layers.Conv2D(64, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(64, 3, activation='relu', padding='same')(c3)

    u1 = layers.UpSampling2D()(c3)
    u1 = layers.Concatenate()([u1, c2])
    c4 = layers.Conv2D(32, 3, activation='relu', padding='same')(u1)

    u2 = layers.UpSampling2D()(c4)
    u2 = layers.Concatenate()([u2, c1])
    c5 = layers.Conv2D(16, 3, activation='relu', padding='same')(u2)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c5)
    return models.Model(inputs, outputs)

def attention_block(x, g, inter_channel):
    theta_x = layers.Conv2D(inter_channel, 1)(x)
    phi_g = layers.Conv2D(inter_channel, 1)(g)
    add = layers.Add()([theta_x, phi_g])
    act = layers.Activation('relu')(add)
    psi = layers.Conv2D(1, 1, activation='sigmoid')(act)
    return layers.Multiply()([x, psi])

def attention_unet(input_size=(IMAGE_SIZE, IMAGE_SIZE, 1)):
    inputs = layers.Input(input_size)
    c1 = layers.Conv2D(16, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(16, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D()(c1)

    c2 = layers.Conv2D(32, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(32, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D()(c2)

    c3 = layers.Conv2D(64, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(64, 3, activation='relu', padding='same')(c3)

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
# TRAINING FUNCTION
# =====================================================

def train_model(model, model_name, train_gen, val_gen, steps_per_epoch, val_steps):
    print(f"\nTraining {model_name}...\n")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(f"{model_name}_best.h5", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.TensorBoard(log_dir=f"./logs/{model_name}")
    ]

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(train_gen, validation_data=val_gen, steps_per_epoch=steps_per_epoch,
                        validation_steps=val_steps, epochs=EPOCHS, callbacks=callbacks)

    model.save(f"{model_name}.h5")
    print(f"{model_name} saved.")

    return history

# =====================================================
# EXECUTION
# =====================================================

image_files = sorted(glob(os.path.join(DATASET_DIR, "imagesTr", "*.nii.gz")))
mask_files = sorted(glob(os.path.join(DATASET_DIR, "labelsTr", "*.nii.gz")))

# Split into training and validation
split_index = int(len(image_files) * 0.8)
train_images, val_images = image_files[:split_index], image_files[split_index:]
train_masks, val_masks = mask_files[:split_index], mask_files[split_index:]

steps_per_epoch = len(train_images) // BATCH_SIZE
val_steps = len(val_images) // BATCH_SIZE

# U-Net
train_gen = data_generator(train_images, train_masks, augment=True)
val_gen = data_generator(val_images, val_masks)
unet = unet_model()
history_unet = train_model(unet, "unet_model", train_gen, val_gen, steps_per_epoch, val_steps)

# Attention U-Net
train_gen = data_generator(train_images, train_masks, augment=True)
val_gen = data_generator(val_images, val_masks)
attention_unet_model = attention_unet()
history_attention = train_model(attention_unet_model, "attention_unet_model", train_gen, val_gen, steps_per_epoch, val_steps)

# =====================================================
# POST TRAINING PLOTS
# =====================================================

def plot_history(history, title):
    plt.figure(figsize=(10,5))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

plot_history(history_unet, "U-Net Training History")
plot_history(history_attention, "Attention U-Net Training History")

print("\n\n=== All Training Finished ===")
