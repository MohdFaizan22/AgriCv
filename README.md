# 🌿 AgriVision AI — Plant Disease Detection using Deep Learning

AgriVision AI is an end-to-end Deep Learning + Computer Vision project that detects plant diseases from leaf images using Transfer Learning with EfficientNetB0.

The system predicts 38 different plant disease categories with high accuracy and provides:

* Disease Prediction
* Confidence Scores
* Grad-CAM Visualization
* Disease Description
* Symptoms
* Prevention Methods
* Cure Suggestions

Built using TensorFlow, OpenCV, EfficientNet, and Gradio.

---

# 🚀 Features

✅ Plant Disease Detection using AI
✅ 38 Disease Categories
✅ Transfer Learning with EfficientNetB0
✅ Fine-Tuned Deep Learning Model
✅ Leaf Segmentation for Better Predictions
✅ Grad-CAM Explainability Visualization
✅ Top-3 Predictions with Confidence Bars
✅ Disease Information & Cure Suggestions
✅ Interactive Gradio Web Application
✅ Real-Time Image Prediction System

---

# 🧠 Problem Statement

Plant diseases significantly reduce agricultural productivity and crop quality.

Traditional disease identification:

* requires expert knowledge
* is time-consuming
* may delay treatment

AgriVision AI helps farmers and researchers instantly identify plant diseases using leaf images.

---

# 🌱 Why This Project Matters

This project combines:

* Artificial Intelligence
* Agriculture
* Deep Learning
* Computer Vision
* Explainable AI

Applications:

* Smart Farming
* Precision Agriculture
* AI-Based Crop Monitoring
* Agricultural Decision Support Systems

---

# 🏗️ Project Workflow

```text
Leaf Image
   ↓
Leaf Segmentation
   ↓
Image Preprocessing
   ↓
EfficientNetB0 Model
   ↓
Disease Prediction
   ↓
Grad-CAM Visualization
   ↓
Disease Information & Cure Suggestions
```

---

# 🛠️ Tech Stack

## Programming Language

* Python

## Deep Learning Frameworks

* TensorFlow
* Keras

## Computer Vision

* OpenCV
* Grad-CAM

## Deployment

* Gradio

## Image Processing

* NumPy
* Pillow
* rembg

---

# 📂 Dataset

## PlantVillage Dataset

* 54,000+ Images
* 38 Classes
* Multiple Crops & Diseases

Dataset Source:
[https://www.kaggle.com/datasets/emmarex/plantdisease](https://www.kaggle.com/datasets/emmarex/plantdisease)

---

# 📊 Disease Categories

The model supports 38 classes including:

* Apple Diseases
* Corn Diseases
* Tomato Diseases
* Potato Diseases
* Grape Diseases
* Strawberry Diseases
* Peach Diseases
* Pepper Diseases
* Soybean Diseases
* Healthy Leaf Detection

---

# 🧪 Model Development Phases

## Phase 1 — Dataset Preparation

Performed:

* Data Loading
* Train/Validation Split
* Image Augmentation

Techniques:

* Rotation
* Zoom
* Horizontal Flip
* Rescaling

---

## Phase 2 — Baseline CNN Model

Built a custom CNN using:

* Conv2D
* MaxPooling
* BatchNormalization
* Dropout
* Dense Layers

Purpose:

* Establish baseline performance
* Understand CNN workflow

---

## Phase 3 — Transfer Learning

Used:

### EfficientNetB0

Advantages:

* Better Feature Extraction
* Higher Accuracy
* Fewer Parameters
* Faster Training

Initially froze pretrained layers.

---

## Phase 4 — Fine Tuning

Unfroze upper EfficientNet layers and retrained using:

* Low Learning Rate
* Additional Epochs

Result:
✅ Significant performance improvement

---

# 📈 Final Results

| Metric              | Value            |
| ------------------- | ---------------- |
| Validation Accuracy | 96%              |
| Model               | EfficientNetB0   |
| Classes             | 38               |
| Dataset Size        | 54K+ Images      |
| Framework           | TensorFlow/Keras |

---

# 🔍 Explainable AI with Grad-CAM

Grad-CAM highlights regions of the image influencing model predictions.

Benefits:

* Improves transparency
* Helps visualize model focus
* Makes predictions explainable

---

# ✂️ Leaf Segmentation

Implemented background removal using:

### rembg

Benefits:

* Reduces background noise
* Improves internet image prediction
* Better real-world generalization

---

# 🌐 Web Application

Built using:

### Gradio

Features:

* Upload Leaf Image
* View Top Predictions
* Confidence Bars
* Grad-CAM Heatmap
* Disease Information
* Cure Suggestions

---

# 📁 Project Structure

```bash
Plant_Disease/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── utils/
│   ├── predict.py
│   ├── gradcam.py
│   ├── segmentation.py
│   └── disease_info.py
│
├── sample_images/
│
└── notebooks/
```

---

# ⚙️ Installation

## 1. Clone Repository

```bash
git clone YOUR_GITHUB_REPO_LINK
cd Plant_Disease
```

---

## 2. Create Virtual Environment

```bash
python -m venv venv
```

---

## 3. Activate Environment

### Windows

```bash
venv\Scripts\activate
```

### Mac/Linux

```bash
source venv/bin/activate
```

---

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 📥 Download Trained Model

Due to GitHub file size limitations, the trained model is hosted externally.

Download Model Here:
[PASTE_YOUR_GOOGLE_DRIVE_LINK_HERE](https://drive.google.com/file/d/1gPvOvKM3Ri8UEPHdqoFIQmN0MqZbX_cf/view?usp=sharing)

After downloading, place the model inside:

```bash
Plant_Disease/
│
├── final_agrivision_model.keras
├── app.py
└── utils/
```

---

# ▶️ Run Application

```bash
python app.py
```

Application runs at:

```bash
http://127.0.0.1:7860
```

---

# 📦 Required Libraries

```txt
tensorflow
opencv-python
gradio
numpy
pillow
matplotlib
rembg
onnxruntime
```

---

# 🧠 Key Learnings

Through this project I learned:

* Transfer Learning
* EfficientNet Architecture
* CNN Fundamentals
* Fine Tuning
* Grad-CAM Explainability
* Leaf Segmentation
* Deep Learning Deployment
* Real-World Image Challenges
* Domain Shift Problems
* Model Generalization

---

# ⚠️ Real-World Challenges

The model performs strongly on PlantVillage-style images.

Challenges with internet images include:

* Complex backgrounds
* Different lighting conditions
* Blurry images
* Domain shift
* Real-world variability

Future improvements:

* Real farm dataset training
* Lesion segmentation
* Higher resolution models
* Advanced explainability techniques

---

# 🚀 Future Improvements

* Mobile App Deployment
* Multi-Language Support
* Real-Time Webcam Detection
* Cloud Deployment
* Disease Severity Estimation
* PDF Report Generation
* Advanced Explainability Methods
* Farmer Advisory System

---

# 👨‍💻 Author

Mohd Faizanullah

AI/ML Enthusiast | Deep Learning | Computer Vision | Generative AI

---

# ⭐ Support

If you like this project, give it a star ⭐ on GitHub.
