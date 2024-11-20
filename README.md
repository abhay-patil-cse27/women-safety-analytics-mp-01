# Women Safety Analytics: Protecting Women from Safety Threats

## Overview
Women Safety Analytics is a Flask-based AI solution designed to enhance women's safety. The project uses advanced analytics to detect safety threats in real-time and send alerts to authorities when required.

### Key Features:
- **Emotion Detection:** Detects emotions such as fear, anger, or disgust in real-time.
- **Gender Detection:** Identifies the presence of women in video frames or images.
- **SOS Alert System:** Automatically sends an SOS email to authorities if a woman exhibits distressing emotions for 30 seconds.
- **AI Models:** Two machine learning models are integrated:
  - Emotion Detection Model (Accuracy: 72%)
  - Gender Detection Model
- **Web Application:** A user-friendly Flask web app to run the system efficiently.

---

## Project Workflow
1. **Video/Image Capture:** The system processes video or image frames to detect gender and emotions.
2. **Emotion Analysis:** If a woman is detected, the system analyzes her emotions.
3. **Alert Mechanism:** Sends an SOS email using SMTP if distress is detected for a prolonged duration.

---

## System Requirements
- **OS:** Windows 10
- **Hardware:** HP Laptop with an integrated camera
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

## Installation and Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/women-safety-analytics.git
   cd women-safety-analytics
