# Deep Learning Models for Emotion Recognition

Emotion recognition is a key area in human-computer interaction, enabling machines to interpret and respond to human emotions. This project explores state-of-the-art deep learning techniques, particularly Convolutional Neural Networks (CNNs), to detect facial emotions using the FER-2013 dataset.

##  Project Overview

This repository presents a deep learning-based emotion recognition system trained on the FER-2013 dataset. It highlights:

- The use of CNNs for spatial feature extraction from facial images.
- Transfer learning from pre-trained models (e.g., VGGFace, InceptionResNetV2).
- Data augmentation to improve model generalization.
- Evaluation of performance using standard classification metrics.

##  Goals

- Develop and benchmark a CNN architecture for facial emotion recognition.
- Apply transfer learning and data augmentation to boost performance.
- Compare with baseline CNN models.
- Enable reproducibility and future extensibility.

##  Dataset

**FER-2013** — Facial Expression Recognition dataset  
- **Images**: 35,887 grayscale images (48x48 pixels)  
- **Emotions**: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral  
- **Train/Test Split**: ~28,000 training / ~7,000 test  
- Dataset available via [Kaggle FER-2013](https://www.kaggle.com/datasets/msambare/fer2013)

##  Model Architecture

- CNN-based deep learning model
- Pre-trained model support (e.g., VGGFace, InceptionResNetV2)
- Transfer Learning with fine-tuning
- Dropout & Batch Normalization for regularization
- Data augmentation techniques:
  - Horizontal flip
  - Rotation
  - Zoom
  - Width/height shift

##  Experimental Setup

| Metric         | Result (Proposed CNN) |
|----------------|------------------------|
| Accuracy       | 87.86% (train), 97.30% (val) |
| Loss           | 0.3313 (train), 0.1617 (val) |
| Epochs         | 10                     |
| Learning Rate  | 1e-4                   |

> Note: Results may vary slightly based on training conditions.

