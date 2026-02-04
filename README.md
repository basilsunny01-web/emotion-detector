# Emotion Detector

An emotion detection application built using machine learning.  
This project includes model training, inference, and a simple web-based user interface.

---

## Features
- Train an emotion classification model
- Predict emotions from input text
- Simple and clean web UI
- Separation of training and inference logic
- Trained model files excluded from Git for size efficiency

---

## Project Structure
emotion-detector/
├── app.py # Application entry point
├── train_model.py # Model training script
├── templates/ # HTML templates
├── screenshots/ # UI screenshots
├── emotion_model/ # Trained model files (ignored by Git)
├── .gitignore
└── README.md

---

## Model Files
Trained model files are **not included in this repository** due to their large size.

After training, place the generated model files inside:

---

## Requirements
- Python 3.8+
- Required Python libraries (install via pip)

```bash
pip install -r requirements.txt
