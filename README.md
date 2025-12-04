# 🍎 Apple Disease Detection — Deep Learning Project

Automated classification of apple foliar diseases using transfer learning (**MobileNetV2**), data augmentation, and multi-class evaluation on high-variance leaf images.

---

## 📌 Overview

Apple plants suffer from multiple foliar diseases that significantly reduce yield quality and quantity.  
Manual disease identification is slow, labor-intensive, and prone to human error.

This project builds a **deep learning model**, based on **MobileNetV2 + custom classification layers**, to classify apple leaf diseases into **6 categories** with high accuracy.  
The model was trained and evaluated on real-world leaf images from an **FGVC challenge dataset**.

---

## 🎯 Objectives

- Classify apple leaf images into **6 disease categories**
- Use **transfer learning** for efficient training
- Apply **data augmentation** for better generalization
- Track training/validation performance using visualizations
- Save the trained model (`.h5`) for inference or deployment

---

## 🗂 Dataset

**Source:**  
Plant Pathology 2021 — FGVC8 (Kaggle Competition)  
https://www.kaggle.com/competitions/plant-pathology-2021-fgvc8/overview

**Final Classes Used (after regrouping):**
- Scab  
- Powdery Mildew  
- Rust  
- Frog Eye Leaf Spot  
- Complex *(combined disease categories)*  
- Healthy  

**Data preprocessing included:**
- Resizing  
- Normalization  
- Dataset splitting  
- Label merging for “Complex” diseases  

---

## 🔧 Technologies Used

- Python  
- TensorFlow / Keras  
- **MobileNetV2** (Transfer Learning)  
- OpenCV  
- Matplotlib / Seaborn  
- NumPy / Pandas  

---

## 🧠 Model Architecture

The model uses **MobileNetV2** pretrained on ImageNet as the frozen base.

**Custom classification head:**
- `GlobalAveragePooling2D`
- `Dense(128, activation='relu')`
- `Dropout`
- `Dense(6, activation='softmax')`

This architecture balances **efficiency** and **accuracy**, making it ideal for agricultural disease classification.

---

## 🏋️ Training Process

### **1. Preprocessing**
- Image resizing  
- Normalization  
- Label standardization  
- Data augmentation:
  - Horizontal flips  
  - Random rotations  
  - Shifts  
  - Brightness adjustments  

### **2. Training Settings**
- **Optimizer:** Adam  
- **Loss:** Categorical Crossentropy  
- **Callbacks:**  
  - `EarlyStopping`  
  - `ModelCheckpoint`

### **3. Final Outputs**

| Metric | Score |
|--------|--------|
| **Training Accuracy** | **91.59%** |
| **Validation Accuracy** | **82.72%** |

These metrics demonstrate strong generalization on a complex multi-class dataset.

---

## 📊 Results & Visualizations

### 📈 Accuracy Curves
`accuracy_curves.png`  
Displays training vs validation accuracy trends.

### 📉 Loss Curves
`loss_curves.png`  
Shows convergence behavior during training.

### 🧩 Confusion Matrices
Individual confusion matrices for each disease class:

- `confusion_matrix_scab.png`  
- `confusion_matrix_rust.png`  
- `confusion_matrix_powdery_mildew.png`  
- `confusion_matrix_frog_eye_leaf_spot.png`  
- `confusion_matrix_complex.png`  
- `confusion_matrix_healthy.png`  

These visualizations highlight model strengths and misclassification patterns.

### 🖼 Sample Predictions  
Example prediction outputs on unseen test images:

- `prediction_image_1.png`  
- `prediction_image_2.png`  
- ...  
- `prediction_image_10.png`  

---

## 🗂 Project Structure

```
apple-disease-detection/
│── model.py
│── train.csv
│── submission.csv
│── sample_submission.csv
│── accuracy_metrics.txt
│
├── results/
│ ├── accuracy_curves.png
│ ├── loss_curves.png
│ ├── confusion_matrix_scab.png
│ ├── confusion_matrix_rust.png
│ ├── confusion_matrix_powdery_mildew.png
│ ├── confusion_matrix_frog_eye_leaf_spot.png
│ ├── confusion_matrix_complex.png
│ └── confusion_matrix_healthy.png
│
├── predictions/
│ ├── prediction_image_1.png
│ ├── prediction_image_2.png
│ ├── prediction_image_3.png
│ ├── prediction_image_4.png
│ ├── prediction_image_5.png
│ ├── prediction_image_6.png
│ ├── prediction_image_7.png
│ ├── prediction_image_8.png
│ ├── prediction_image_9.png
│ └── prediction_image_10.png
│
└── test_images/
│ ├── test1.jpg
│ ├── test2.jpg
│ └── test3.jpg
│
└── README.md
```

---

## 🚀 Future Improvements

- Use **EfficientNet-B0/B2** for better feature extraction  
- Add **Grad-CAM** for explainable AI  
- Perform **hyperparameter tuning** with Optuna  
- Deploy using **FastAPI / Flask**  
- Convert to **TensorFlow Lite** for mobile or edge devices  

---

## 📬 Contact

**Arnav Saxena**  
🔗 LinkedIn: https://www.linkedin.com/in/arnav-saxena-a9a217367  
📧 Email: **arnav12saxena@gmail.com**

---

