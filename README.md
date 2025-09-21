# Advanced Brain Tumor Segmentation with AR Integration

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-orange.svg)](https://www.tensorflow.org/)

## Overview
This repository implements advanced deep learning models for 3D brain tumor segmentation and integrates the results with Augmented Reality (AR) visualization. Designed for medical imaging researchers and developers, it combines state-of-the-art neural networks with immersive visualization techniques.

## Key Features
- 🧠 **Dual Architecture Support**: Implementation of both Attention UNet 3D and UNet 3D models
- 🕶️ **AR Integration**: Real-time 3D visualization of segmentation results using AR headsets
- 🚀 **Pre-trained Models**: Ready-to-use models trained on BraTS datasets
- 📊 **Interactive Notebooks**: Jupyter notebooks for training, validation, and AR visualization
- 🧩 **Multi-modal Support**: Compatible with MRI (T1, T1ce, T2, FLAIR) sequences

## Installation
```bash
git clone https://github.com/amira-nemdil/Advanced-Brain-Tumor-Segmentation-and-Augmented-Reality-Integration.git
cd Advanced-Brain-Tumor-Segmentation-and-Augmented-Reality-Integration

# Create virtual environment
python -m venv arseg
source arseg/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download pre-trained models
wget https://example.com/pretrained_models.zip && unzip pretrained_models.zip
```

## Usage
### 1. Brain Tumor Segmentation
```python
from models import AttentionUNet3D

model = AttentionUNet3D(in_channels=4, out_channels=3)
predictions = model.predict(mri_volume)
```

### 2. AR Visualization
```bash
python ar_visualization.py \
  --input_path samples/patient_001.nii.gz \
  --output_path results/ar_visualization/
```

### 3. Training New Models
See included Jupyter notebooks for:
- Data preprocessing pipelines
- Model training configurations
- Performance evaluation metrics

## Dataset Preparation
Structure your dataset as:
```
data/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

## AR Requirements
- Microsoft HoloLens 2 or compatible AR headset
- Unity 2022.3+ (for custom visualization development)
- ARCore/ARKit for mobile device support

## Results
| Model           | Dice Score | Sensitivity | Specificity |
|-----------------|------------|-------------|-------------|
| UNet 3D         | 0.89       | 0.91        | 0.99        |
| Attention UNet3D| 0.92       | 0.93        | 0.99        |

## Contributing
Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License
Distributed under MIT License. See `LICENSE` for details.

## Acknowledgments
- BraTS dataset providers
- MONAI for medical imaging preprocessing
- PyTorch Lightning team
