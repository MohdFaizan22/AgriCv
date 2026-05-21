# 🌿 AgriVision AI — Plant Disease Detection using Deep Learning

AgriVision AI is an end-to-end Computer Vision + Deep Learning project that detects plant diseases from leaf images using Transfer Learning with EfficientNetB0.

The system predicts 38 different plant disease categories with high accuracy and provides:
- Disease prediction
- Confidence scores
- Grad-CAM visualization
- Disease information
- Prevention methods
- Cure suggestions

Built using TensorFlow, OpenCV, Gradio, and EfficientNet.

---

# 🚀 Features

✅ Plant Disease Detection from Leaf Images  
✅ 38 Disease Categories  
✅ Transfer Learning with EfficientNetB0  
✅ Grad-CAM Explainability Visualization  
✅ Leaf Segmentation for Better Prediction  
✅ Top-3 Predictions with Confidence Bars  
✅ Disease Description + Symptoms + Cure Suggestions  
✅ Interactive Gradio Web App  
✅ Real-Time AI Prediction System  

---

# 🧠 Problem Statement

Farmers often struggle to identify crop diseases early, leading to:
- reduced crop yield
- financial loss
- excessive pesticide usage

AgriVision AI helps detect diseases instantly using smartphone leaf images.

---

# 🌱 Why This Project Matters

This project combines:
- Computer Vision
- Agriculture
- Deep Learning
- Explainable AI

Plant disease detection is widely used in modern AgriTech platforms like:
- Plantix
- CropIn
- Nuru

This makes the project highly relevant for:
- AI Engineer roles
- ML Engineer roles
- Computer Vision roles
- Agricultural AI startups

---

# 🏗️ Project Architecture

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
Disease Information + Cure Suggestion

---

# 🛠️ Tech Stack

## Languages
- Python

## Deep Learning
- TensorFlow
- Keras

## Computer Vision
- OpenCV
- Grad-CAM
- Transfer Learning

## Deployment
- Gradio

## Image Processing
- NumPy
- PIL
- rembg

---

# 📂 Dataset

## PlantVillage Dataset
- 54,000+ images
- 38 classes
- Multiple crops and diseases

Dataset Source:
https://www.kaggle.com/datasets/emmarex/plantdisease

---

# 🧪 Model Development Phases

# Phase 1 — Dataset Preparation
- Loaded PlantVillage dataset
- Applied train-validation split
- Image augmentation using ImageDataGenerator

Techniques:
- Rotation
- Zoom
- Horizontal Flip
- Rescaling

---

# Phase 2 — Baseline CNN Model

Built custom CNN:
- Conv2D
- MaxPooling
- BatchNormalization
- Dropout
- Dense Layers

Achieved strong baseline performance.

---

# Phase 3 — Transfer Learning

Used:
## EfficientNetB0

Advantages:
- Better feature extraction
- Faster convergence
- Higher accuracy
- Fewer parameters

Frozen pretrained layers initially.

---

# Phase 4 — Fine Tuning

Unfroze last layers of EfficientNetB0.

Fine-tuned with:
- Low Learning Rate (1e-5)
- Additional epochs

Result:
✅ Significant accuracy improvement

---

# 📈 Final Results

| Metric | Value |
|---|---|
| Validation Accuracy | 96% |
| Classes | 38 |
| Dataset Size | 54K+ Images |
| Model | EfficientNetB0 |
| Framework | TensorFlow/Keras |

---

# 🔍 Explainable AI with Grad-CAM

Grad-CAM highlights image regions influencing predictions.

Benefits:
- Improves model transparency
- Visualizes disease focus regions
- Helps explain AI decisions

---

# ✂️ Leaf Segmentation

Added background removal using:
## rembg

Benefits:
- Better internet image predictions
- Reduced background noise
- Improved real-world performance

---

# 🌐 Web Application

Built using:
## Gradio

Features:
- Upload leaf image
- View predictions
- Confidence bars
- Grad-CAM heatmap
- Disease information
- Cure suggestions

---

# 📁 Project Structure

```bash
Plant_Disease/
│
├── app.py
├── disease_info.py
├── requirements.txt
├── README.md
├── final_agrivision_model.keras
│
├── utils/
    ├── predict.py
    ├── gradcam.py
    └── segmentation.py

---
# ⚙️ InstallationL:

Clone Repository
git clone <your-github-repo-link>
cd Plant_Disease
Create Virtual Environment
python -m venv venv
Activate Environment
Windows
venv\Scripts\activate
Mac/Linux
source venv/bin/activate
Install Dependencies
pip install -r requirements.txt
▶️ Run Application
python app.py

Application runs at:

http://127.0.0.1:7860

---
📦 Required Libraries
tensorflow
opencv-python
gradio
numpy
pillow
matplotlib
rembg
onnxruntime

🧠 Key Learnings

Through this project I learned:

Transfer Learning
CNN Architectures
EfficientNet
Fine Tuning
Grad-CAM Explainability
Image Segmentation
Deep Learning Deployment
Real-World Dataset Challenges
Domain Shift Problems
Model Generalization.

⚠️ Real-World Challenges

The model performs strongly on PlantVillage-style images.

Challenges with internet images:

Different lighting
Complex backgrounds
Domain shift
Low-quality images

Future improvements:

Real farm dataset training
Lesion segmentation
Higher resolution models
Advanced explainability methods
🚀 Future Improvements
Mobile App Deployment
Multi-language Support
Webcam Detection
Real-Time Farm Monitoring
Cloud Deployment
Disease Severity Estimation
PDF Report Generation
Advanced Explainability
Farmer Advisory System

👨‍💻 Author:
Mohd Faizanullah
