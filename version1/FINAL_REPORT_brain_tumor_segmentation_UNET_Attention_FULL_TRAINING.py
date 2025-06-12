# # 🧠 Brain Tumor Segmentation Using 3D U-Net and Attention U-Net (PyTorch)
# ### Advanced Deep Learning Pipeline for MRI Segmentation
# 
# This notebook presents a complete deep learning pipeline for 3D brain tumor segmentation from MRI images using U-Net and Attention U-Net architectures.
# 
# **Features:**
# - Data preprocessing and loading for multi-modal MRI
# - Custom PyTorch dataset and data loader
# - Implementation of U-Net and Attention U-Net
# - Advanced training loop with early stopping and learning rate scheduling
# - Quantitative evaluation (Dice, loss curves) and qualitative visualizations
# - Easily adaptable for research, internship, and clinical exploration
# 
# ----


# ## 🚀 Training and Evaluation Instructions
# 
# This section explains the steps to train and validate both U-Net and Attention U-Net models:
# 
# 1. **Run all cells sequentially** from top to bottom.
# 2. If you encounter GPU memory issues, **reduce the batch size** in the DataLoader (try 1 or 2).
# 3. **Training**: 
#    ```python
#    train_model(UNet3D(), "unet3d")
#    train_model(AttentionUNet3D(), "att_unet3d")
#    ```
# 4. **Saved models:**
#    - `unet3d_best.pth` (U-Net)
#    - `att_unet3d_best.pth` (Attention U-Net)
# 5. **Results:** Dice scores, plots, and model logs are automatically saved for each run.
# ---


# # 🧠 Brain Tumor Segmentation Using 3D U-Net and Attention U-Net (PyTorch)
# ### Advanced Deep Learning Pipeline for MRI Segmentation
# 
# This notebook presents a complete deep learning pipeline for 3D brain tumor segmentation from MRI images using U-Net and Attention U-Net architectures.
# 
# **Features:**
# - Data preprocessing and loading for multi-modal MRI
# - Custom PyTorch dataset and data loader
# - Implementation of U-Net and Attention U-Net
# - Advanced training loop with early stopping and learning rate scheduling
# - Quantitative evaluation (Dice, loss curves) and qualitative visualizations
# - Easily adaptable for research, internship, and clinical exploration
# 
# ----


# ## ⚙️ Environment Setup and Dependencies
# 
# We use PyTorch for model development, along with utility libraries such as NumPy, matplotlib, and tqdm for progress visualization. Ensure you have a CUDA-compatible GPU for best performance. This notebook is designed to run efficiently on a modern workstation with at least 8 GB GPU memory.
# 


import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from glob import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('✅ Using device:', DEVICE)



import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")




# Confirm GPU
print(torch.cuda.get_device_name(0))

# Confirm one batch
for images, masks in train_loader:
    print(images.shape, masks.shape)
    break

# Confirm model forward
model = UNet().to(DEVICE)
print(model(torch.randn(1, 1, 256, 256).to(DEVICE)).shape)




# ## 📂 Dataset Preparation: Multi-modal MRI Loader
# 
# For this project, we utilize multi-modal MRI brain images stored as `.npy` files. Each image contains 4 modalities (FLAIR, T1, T1ce, T2) stacked along the last axis. The corresponding segmentation mask is a 2D array with integer values indicating tissue type (0 = background, 1 = tumor). This organization enables effective multi-modal learning for robust tumor segmentation.
# 


# ## 📂 Dataset Preparation: Multi-modal MRI Loader
# 
# For this project, we utilize multi-modal MRI brain images stored as `.npy` files. Each image contains 4 modalities (FLAIR, T1, T1ce, T2) stacked along the last axis. The corresponding segmentation mask is a 2D array with integer values indicating tissue type (0 = background, 1 = tumor). This organization enables effective multi-modal learning for robust tumor segmentation.
# 


from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import torch
import os

class BrainTumorDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        if not os.path.isdir(mask_dir):
            raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

        self.image_files = sorted([
            f for f in os.listdir(image_dir) if f.endswith(".npy")
        ])
        self.mask_files = sorted([
            f.replace("image_", "mask_") for f in self.image_files
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path  = os.path.join(self.mask_dir,  self.mask_files[idx])

        image = np.load(image_path)  # shape: (H, W, 4)
        mask = np.load(mask_path)    # shape: (H, W)

        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # (4, H, W)
        mask = torch.tensor(mask, dtype=torch.long)  # (H, W)

        return image, mask

# Initialize dataset
dataset = BrainTumorDataset(
    r"D:\Khedir-meriem-ESI-SBElAbes\data\input_data_4channels_z_score\train\images",
    r"D:\Khedir-meriem-ESI-SBElAbes\data\input_data_4channels_z_score\train\masks"
)



# Split into train/validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

print(f"✅ Dataset loaded — Train: {len(train_set)} | Validation: {len(val_set)}")




# ## 🧠 Model Architectures: U-Net & Attention U-Net
# 
# **U-Net:** The U-Net architecture is a widely used encoder-decoder convolutional neural network for biomedical image segmentation. It features skip connections that preserve spatial detail, enabling precise localization.
# 
# **Attention U-Net:** The Attention U-Net augments the standard U-Net with attention gates that allow the network to focus on relevant image regions, further improving segmentation accuracy, especially in challenging cases.
# 


class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=2):
        super(UNet, self).__init__()
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(),
                nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU())
        self.enc1 = conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = conv_block(128, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = conv_block(128, 64)
        self.output = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.output(d1)



class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        return x * self.psi(self.relu(self.W_g(g) + self.W_x(x)))



class AttentionUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=2):
        super(AttentionUNet, self).__init__()
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(),
                nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU())
        self.enc1 = conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = conv_block(128, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.att2 = AttentionBlock(128, 128, 64)
        self.dec2 = conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.att1 = AttentionBlock(64, 64, 32)
        self.dec1 = conv_block(128, 64)
        self.output = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.up2(b); d2 = self.att2(g=d2, x=e2); d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2); d1 = self.att1(g=d1, x=e1); d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.output(d1)



# ## 📂 Dataset Preparation: Multi-modal MRI Loader
# 
# For this project, we utilize multi-modal MRI brain images stored as `.npy` files. Each image contains 4 modalities (FLAIR, T1, T1ce, T2) stacked along the last axis. The corresponding segmentation mask is a 2D array with integer values indicating tissue type (0 = background, 1 = tumor). This organization enables effective multi-modal learning for robust tumor segmentation.
# 


dataset = BrainTumorDataset(
    r"D:\Khedir-meriem-ESI-SBElAbes\data\input_data_4channels_z_score\train\images",
    r"D:\Khedir-meriem-ESI-SBElAbes\data\input_data_4channels_z_score\train\masks"
)

print("Total samples:", len(dataset))
img, mask = dataset[0]
print("Image shape:", img.shape)
print("Mask shape:", mask.shape)
# Split dataset into train and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
model = UNet().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
# Training loop
num_epochs = 20
train_losses = []
val_losses = []




# ## 📐 Dice Coefficient Metric
# Used to evaluate how well the predicted segmentation matches the ground truth.


# 
# ## 📐 Dice Coefficient Metric (For Hard Labels)
# 
# **Why this version?**
# - Correct for "hard" class masks, e.g., after `torch.argmax(model(x), dim=1)`
# - Works for multi-class or binary segmentation where each pixel/voxel is an integer label
# 
# > For classic "soft" Dice (e.g. sigmoid/thresholded masks), see the note in comments.
# 
# **Expected shapes:**
# - Prediction: `[B, D, H, W]` (or `[B, H, W]`) — *from `argmax` over model output*
# - Target:     `[B, D, H, W]` (or `[B, H, W]`) — *ground truth mask*
# 


def dice_coeff(pred, target, eps=1e-6):
    """
    Dice coefficient for hard class-labeled masks (multi-class or binary).
    Args:
        pred (Tensor): predicted mask, e.g., shape [B, D, H, W]
        target (Tensor): ground truth mask, same shape
    Returns:
        Dice score (float)
    """
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred == target).float().sum()
    dice = (2. * intersection) / (pred.numel() + target.numel() + eps)
    return dice

# Note: For "soft" Dice (with sigmoid/thresholded masks, binary only):
# intersection = (pred * target).sum()
# dice = (2. * intersection + eps) / (pred.sum() + target.sum() + eps)




# ## 🧪 Training Function (AMP + EarlyStopping + Scheduler)


# 
# ## 🏋️ Training Function: AMP, Early Stopping, Scheduler
# - Uses **Automatic Mixed Precision (AMP)** for faster training
# - Early stopping after no improvement for 10 epochs
# - CSV log for every run; saves best weights and Dice curve plot
# 



from torch.cuda.amp import autocast, GradScaler
import csv

def train_model(model, name):
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    criterion = nn.CrossEntropyLoss()

    scaler = GradScaler()
    best_dice = 0
    patience = 10
    trigger = 0
    train_dice, val_dice = [], []

    # Prepare CSV log file
    log_path = f"{name}_log.csv"
    with open(log_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "TrainLoss", "ValDice"])

    for epoch in range(1, 101):
        model.train()
        total_loss = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()

            with autocast():
                out = model(x)
                loss = criterion(out, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        # Validation
        model.eval()
        dice = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                with autocast():
                    pred = torch.argmax(model(x), dim=1)
                dice += dice_coeff(pred.cpu(), y.cpu())
        avg_dice = dice / len(val_loader)
        val_dice.append(avg_dice.item())

        # Log to CSV
        with open(log_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([epoch, total_loss, avg_dice.item()])

        print(f"[{name}] Epoch {epoch}, Loss: {total_loss:.4f}, Val Dice: {avg_dice:.4f}")
        scheduler.step(total_loss)

        if avg_dice > best_dice:
            best_dice = avg_dice
            torch.save(model.state_dict(), f"{name}_best.pth")
            print(f"✅ Saved best model with Dice: {best_dice:.4f}")
            trigger = 0
        else:
            trigger += 1
            if trigger >= patience:
                print("⏹️ Early stopping triggered.")
                break

    # After training: plot Dice scores
    plt.figure(figsize=(8, 5))
    plt.plot(val_dice, label='Validation Dice')
    plt.xlabel("Epoch")
    plt.ylabel("Dice Coefficient")
    plt.title(f"Validation Dice Curve ({name})")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{name}_dice_plot.png")
    plt.show()




# ## 🧠 Train Both Models (100 Epochs Max + Early Stopping)


train_model(UNet(), "unet")
train_model(AttentionUNet(), "attention_unet")



# ## 📊 Plot Dice Score Over Epochs


# If you store dice per epoch above, you can plot:
# plt.plot(train_dice, label='Train Dice')
# plt.plot(val_dice, label='Val Dice')
# plt.legend(); plt.xlabel('Epoch'); plt.ylabel('Dice'); plt.title('Training Curve'); plt.show()



# ## 🔍 Visualize Predictions


def visualize_predictions(model_path, model_class):
    model = model_class().to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    x, y = next(iter(val_loader))
    x, y = x.to(DEVICE), y.to(DEVICE)
    with torch.no_grad():
        pred = torch.argmax(model(x), dim=1)

    plt.figure(figsize=(12, 4))
    for i in range(3):
        plt.subplot(3, 3, i*3+1); plt.imshow(x[i][0].cpu(), cmap='gray'); plt.title('Input')
        plt.subplot(3, 3, i*3+2); plt.imshow(y[i].cpu(), cmap='gray'); plt.title('GT')
        plt.subplot(3, 3, i*3+3); plt.imshow(pred[i].cpu(), cmap='gray'); plt.title('Prediction')
    plt.tight_layout(); plt.show()

visualize_predictions("unet_best.pth", UNet)
visualize_predictions("attention_unet_best.pth", AttentionUNet)



# ## 💾 Export Models: TorchScript + ONNX


dummy = torch.randn(1, 4, 256, 256).to(DEVICE)
model = AttentionUNet().to(DEVICE)
model.load_state_dict(torch.load("attention_unet_best.pth"))
torch.jit.trace(model, dummy).save("attention_unet.pt")
torch.onnx.export(model, dummy, "attention_unet.onnx", input_names=["input"], output_names=["output"], opset_version=11)
print("✅ Exported to .pt and .onnx")



# ## ✅ Summary & Deployment Tips
# - Best Dice model saved as `.pth`
# - Deployment-ready formats: `.pt` (TorchScript), `.onnx`
# - Use ONNX for ONNX Runtime, TensorRT, or OpenVINO.
# - Ideal for mobile, embedded, or web deployment.
# 
# **Next step:** Try converting the ONNX model into TensorRT or integrating in a simple Flask demo.


# ## 🧪 Advanced Data Augmentation with Albumentations
# We use Albumentations to apply more realistic and effective augmentations during training.


import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

class AugmentedBrainTumorDataset(Dataset):
    def __init__(self, image_dir, mask_dir, augment=False):
        self.image_paths = sorted(glob(os.path.join(image_dir, '*.npy')))
        self.mask_paths = sorted(glob(os.path.join(mask_dir, '*.npy')))
        self.augment = augment
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.load(self.image_paths[idx]).astype(np.float32)
        mask = np.load(self.mask_paths[idx]).astype(np.uint8)
        image = image.transpose(2, 0, 1)  # [C, H, W]
        if self.augment:
            augmented = self.transform(image=image.transpose(1, 2, 0), mask=mask)
            image, mask = augmented['image'], augmented['mask']
        else:
            image = torch.tensor(image, dtype=torch.float32)
            mask = torch.tensor(mask, dtype=torch.long)
        return image, mask



# Replace old dataset and reload
dataset = AugmentedBrainTumorDataset(
    r"D:\Khedir-meriem-ESI-SBElAbes\data\input_data_4channels_z_score\train\images",
    r"D:\Khedir-meriem-ESI-SBElAbes\data\input_data_4channels_z_score\train\masks"
    augment=True
)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
val_loader = DataLoader(val_set, batch_size=4, shuffle=False)



# ## 🔗 Ensemble Prediction: U-Net + Attention U-Net
# Combines predictions from both models to potentially improve segmentation accuracy.


def ensemble_predict(x):
    model1 = UNet().to(DEVICE)
    model1.load_state_dict(torch.load("unet_best.pth"))
    model2 = AttentionUNet().to(DEVICE)
    model2.load_state_dict(torch.load("attention_unet_best.pth"))
    model1.eval(); model2.eval()

    with torch.no_grad():
        out1 = model1(x)
        out2 = model2(x)
        avg_out = (out1 + out2) / 2
        pred = torch.argmax(avg_out, dim=1)
    return pred



# Visualize ensemble prediction
x, y = next(iter(val_loader))
x, y = x.to(DEVICE), y.to(DEVICE)
pred = ensemble_predict(x)

plt.figure(figsize=(12, 4))
for i in range(3):
    plt.subplot(3, 3, i*3+1); plt.imshow(x[i][0].cpu(), cmap='gray'); plt.title('Input')
    plt.subplot(3, 3, i*3+2); plt.imshow(y[i].cpu(), cmap='gray'); plt.title('Ground Truth')
    plt.subplot(3, 3, i*3+3); plt.imshow(pred[i].cpu(), cmap='gray'); plt.title('Ensemble Pred')
plt.tight_layout(); plt.show()


