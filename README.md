🍎 Apple Disease Detection — Deep Learning Project

Automated classification of apple foliar diseases using transfer learning (MobileNetV2), data augmentation, and multi-class evaluation on high-variance leaf images.

📌 Overview

Apple plants suffer from multiple foliar diseases that significantly reduce yield quality and quantity. Manual disease identification is slow, labor-intensive, and prone to human error.

This project builds a deep learning model, based on MobileNetV2 + custom classification layers, to classify apple leaf diseases into 6 categories with high accuracy. The project was evaluated using real-world leaf images from an FGVC challenge dataset.

🎯 Objectives

Classify apple leaf images into 6 disease categories

Use transfer learning for efficient training

Apply data augmentation for better generalization

Monitor training/validation performance using metrics & visualizations

Save trained model (.h5) for deployment or inference

🗂 Dataset

Source:
Plant Pathology 2021 — FGVC8 (Kaggle Competition)
https://www.kaggle.com/competitions/plant-pathology-2021-fgvc8/overview

Final Classes Used (after regrouping):

Scab

Powdery Mildew

Rust

Frog Eye Leaf Spot

Complex (multiple/combination diseases merged)

Healthy

Data preprocessing included:

Resizing

Normalization

Dataset splitting

Label merging for “Complex” categories

🔧 Technologies Used

Python

TensorFlow / Keras

MobileNetV2 Transfer Learning

OpenCV

Matplotlib / Seaborn

NumPy / Pandas

🧠 Model Architecture

The model uses MobileNetV2 pretrained on ImageNet as the frozen base.

Custom classification layers:

GlobalAveragePooling2D

Dense(128), ReLU

Dropout

Dense(6), Softmax

This architecture balances efficiency and accuracy, ideal for leaf disease classification tasks.

🏋️ Training Process
1. Preprocessing

Image resizing

Data normalization

Label standardization

Augmentation:

Horizontal flips

Random rotations

Shifts & brightness changes

2. Training Settings

Optimizer: Adam

Loss: Categorical Crossentropy

Callbacks:

EarlyStopping

ModelCheckpoint

3. Final Outputs

Your collected metrics:

Metric	Score
Training Accuracy	91.59%
Validation Accuracy	82.72%

These metrics show good generalization for a multi-class agricultural image dataset with high visual variability.

📊 Results & Visualizations
📈 Accuracy Curves

accuracy_curves.png

Shows training vs validation accuracy trends

📉 Loss Curves

loss_curves.png

Monitors training/validation loss

🧩 Confusion Matrices

You have individual confusion matrices for each disease class:

confusion_matrix_scab.png

confusion_matrix_rust.png

confusion_matrix_powdery_mildew.png

confusion_matrix_frog_eye_leaf_spot.png

confusion_matrix_complex.png

confusion_matrix_healthy.png

These help identify which classes the model handles well and where misclassifications occur.

🖼 Sample Predictions

Images included:

prediction_image_1.png

prediction_image_2.png

…

prediction_image_10.png

These show the model’s real inference output on unseen test images.

🗂 Project Structure
apple-disease-detection/
│── model.py                     # Model architecture & training pipeline
│── apple_disease_model.h5       # Saved trained model
│── accuracy_curves.png
│── loss_curves.png
│── accuracy_metrics.txt
│── confusion_matrix_scab.png
│── confusion_matrix_rust.png
│── confusion_matrix_powdery_mildew.png
│── confusion_matrix_frog_eye_leaf_spot.png
│── confusion_matrix_complex.png
│── confusion_matrix_healthy.png
│── prediction_image_1.png
│── prediction_image_2.png
│── ... (others)
│── test_images/                 # Folder of unseen test images
│── train.csv
│── submission.csv
│── sample_submission.csv
└── README.md

🚀 Future Improvements

Use EfficientNet-B0/B2 for improved feature extraction

Add Grad-CAM visualization for explainability

Hyperparameter search using Optuna

Deploy model via FastAPI

Convert model to TFLite for edge devices

📬 Contact

Arnav Saxena
LinkedIn: https://www.linkedin.com/in/arnav-saxena-a9a217367

Email: arnav12saxena@gmail.com
