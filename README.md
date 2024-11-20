# Women Safety Analytics: Protecting Women from Safety Threats

## Overview
Women Safety Analytics is a Flask-based AI solution designed to enhance women's safety. The project uses advanced analytics to detect safety threats in real-time and send alerts to authorities when required.

### Key Features:
- **Emotion Detection:** Detects emotions such as fear, anger, or disgust in real-time.
- **Gender Detection:** Identifies the presence of women in video frames or images.
- **SOS Alert System:** Automatically sends an SOS email to authorities if a woman exhibits distressing emotions for 30 seconds.
- **AI Models:** Two machine learning models are integrated:
  - Emotion Detection Model (Accuracy: 72%)
  - Gender Detection Model (Accuracy: 95%)
- **Web Application:** A user-friendly Flask web app to run the system efficiently.

---

## Project Workflow
1. **Video/Image Capture:** The system processes video or image frames to detect gender and emotions.
2. **Emotion Analysis:** If a woman is detected, the system analyzes her emotions.
3. **Alert Mechanism:** Sends an SOS email using SMTP if distress is detected for a prolonged duration.

---

## System Requirements
- **OS:** Windows 10
- **Hardware:** A PC with a webcam or a Laptop with an integrated camera
- **Software Dependencies:**
  - Python 3.10.7
  - Flask
  - TensorFlow
  - Keras
  - OpenCV
  - Pandas, NumPy
  - scikit-learn
  - tqdm
  - SMTP library

---

   ## Jupyter Notebooks
The repository contains the following Jupyter notebooks:
1. **Emotion_Detection.ipynb**: Training and evaluating the emotion detection model.
2. **Gender_Detection.ipynb**: Training and evaluating the gender detection model.

You can run these notebooks using Jupyter Notebook or JupyterLab:
```bash
jupyter notebook

