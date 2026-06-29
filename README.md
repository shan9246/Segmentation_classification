# Transformer-Based Multi-Task Breast Ultrasound Analysis

A transformer-based multi-task deep learning framework for **breast ultrasound lesion segmentation** and **tumor classification** using **SegFormer**. The model leverages a shared transformer encoder with task-specific heads to simultaneously perform pixel-wise lesion segmentation and classify ultrasound images into **Benign**, **Malignant**, and **Normal** categories.

---

## 📌 Overview

Breast ultrasound imaging is widely used for early breast cancer diagnosis. This project implements a **multi-task learning (MTL)** framework that jointly learns:

- **Lesion Segmentation** – Generates accurate pixel-wise lesion masks.
- **Tumor Classification** – Predicts whether an image is **Benign**, **Malignant**, or **Normal**.

By sharing feature representations between both tasks, the model improves learning efficiency while reducing computational redundancy.

---

## ✨ Features

- Transformer-based **SegFormer** architecture
- Shared encoder with dual-task learning
- Dedicated segmentation and classification heads
- Fine-tuning of pretrained transformer weights
- Multi-task optimization using combined loss functions
- Automatic evaluation using segmentation and classification metrics
- Modular PyTorch implementation

---

## 🏗️ Model Architecture

```text
                  Input Ultrasound Image
                           │
                           ▼
                SegFormer Transformer Encoder
                           │
            ┌──────────────┴──────────────┐
            ▼                             ▼
    Segmentation Head            Classification Head
            │                             │
            ▼                             ▼
     Lesion Segmentation          Benign / Malignant / Normal
             Mask
```

---

## 📂 Dataset

The model is trained on the **BUSI (Breast Ultrasound Images)** dataset.

### Classification Classes

- Benign
- Malignant
- Normal

### Segmentation Target

- Pixel-wise breast lesion masks

---

## ⚙️ Training

### Loss Functions

| Task | Loss |
|------|------|
| Segmentation | Dice Loss |
| Classification | Weighted Cross-Entropy Loss |

### Optimizer

- AdamW

### Learning Strategy

- Learning Rate Scheduler
- Early Stopping
- Best Model Checkpointing

---

## 📊 Evaluation Metrics

### Segmentation

- Dice Score
- Intersection over Union (IoU)

### Classification

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

## 🛠️ Technologies Used

- Python
- PyTorch
- SegFormer
- Hugging Face Transformers
- OpenCV
- NumPy
- Matplotlib
- scikit-learn



## 📈 Results

The proposed multi-task framework simultaneously performs:

- Breast lesion segmentation
- Breast tumor classification

using a shared transformer encoder, demonstrating the effectiveness of transformer-based multi-task learning for breast ultrasound analysis.

---

## 🔮 Future Work

- Incorporate attention-guided feature fusion
- Evaluate on additional breast ultrasound datasets
- Deploy the model for real-time clinical inference
- Extend to multi-class lesion segmentation



## 👨‍💻 Author

**Shantanu Sharma**

M.Tech in Robotics and Mobility Systems  
Indian Institute of Technology Jodhpur
