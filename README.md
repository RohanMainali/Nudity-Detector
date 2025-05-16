# 🔞 Nudity Detection in Images using Deep Learning

This project implements an image classification model to detect nudity in images using transfer learning with **MobileNetV2**. It's optimized for performance on GPU environments and is built using TensorFlow and Keras.

## 📌 Key Features

* 🔍 **Nudity Detection**: Binary classification between nude and non-nude images.
* ⚙️ **Transfer Learning**: Uses pre-trained MobileNetV2 for feature extraction.
* 🧪 **Data Augmentation**: Improves generalization using `ImageDataGenerator`.
* 🧠 **Custom Fine-tuning**: Adds dense layers on top of MobileNet for classification.
* 📈 **Training Callbacks**: Early stopping and model checkpointing integrated.

## 🖥️ Technology Stack

* Python
* TensorFlow / Keras
* scikit-learn
* NumPy / Pandas
* MobileNetV2 (pre-trained on ImageNet)

## 🗃️ Dataset

* Dataset used: **safe vs. NSFW images** (Web Scraped and manually verified).
* Preprocessing includes image resizing, normalization, and augmentation.

## 📁 File Structure

```
.
├── nuditydetection.ipynb       # Main training and evaluation notebook
├── data/                       # Folder containing training and validation images
├── models/                     # Saved model checkpoints
├── outputs/                    # Metrics, logs, or visualizations
└── README.md
```


## 📊 Evaluation Metrics

* Accuracy
* Confusion Matrix
* Precision / Recall
* ROC AUC (optional)

## 📌 Model Summary

* Base: `MobileNetV2` with ImageNet weights
* Head: Global Average Pooling + Dense layers
* Optimizer: Adam
* Loss: Binary Crossentropy

