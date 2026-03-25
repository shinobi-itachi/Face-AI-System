# Face AI System 🚀  
### Mask Detection + Emotion Recognition (Real-Time)

---

## 📌 Overview
Face AI System is an end-to-end Deep Learning application that performs:

- Face Mask Detection  
- Facial Emotion Recognition  
- Real-time Webcam Detection  
- Image Upload Prediction via Web Interface  

The system is designed as a **multi-stage AI pipeline**, where mask detection is performed first, followed by emotion detection only when the face is clearly visible.

---

## 🧠 Key Features

- 🔹 Mask Detection using Transfer Learning (MobileNetV2)  
- 🔹 Emotion Detection (7 classes – FER2013 dataset)  
- 🔹 Face Detection using OpenCV Haar Cascade  
- 🔹 Real-time Webcam Inference  
- 🔹 Image Upload Prediction (Flask API)  
- 🔹 Confidence-based Filtering (handles uncertain predictions)  
- 🔹 Alert System (e.g., No Mask + Angry → Risk)  
- 🔹 Latency Tracking  

---

## ⚙️ Tech Stack

- Python  
- TensorFlow / Keras  
- OpenCV  
- Flask  
- NumPy  
- Scikit-learn  

---

## 🏗️ Project Structure
# Face AI System 🚀  
### Mask Detection + Emotion Recognition (Real-Time)

---

## 📌 Overview
Face AI System is an end-to-end Deep Learning application that performs:

- Face Mask Detection  
- Facial Emotion Recognition  
- Real-time Webcam Detection  
- Image Upload Prediction via Web Interface  

The system is designed as a **multi-stage AI pipeline**, where mask detection is performed first, followed by emotion detection only when the face is clearly visible.

---

## 🧠 Key Features

- 🔹 Mask Detection using Transfer Learning (MobileNetV2)  
- 🔹 Emotion Detection (7 classes – FER2013 dataset)  
- 🔹 Face Detection using OpenCV Haar Cascade  
- 🔹 Real-time Webcam Inference  
- 🔹 Image Upload Prediction (Flask API)  
- 🔹 Confidence-based Filtering (handles uncertain predictions)  
- 🔹 Alert System (e.g., No Mask + Angry → Risk)  
- 🔹 Latency Tracking  

---

## ⚙️ Tech Stack

- Python  
- TensorFlow / Keras  
- OpenCV  
- Flask  
- NumPy  
- Scikit-learn  

---

## 🏗️ Project Structure
Face-AI-System/
│
├── app.py
├── requirements.txt
│
├── models/
│ ├── mask_model.h5
│ ├── emotion_model.h5
│ └── emotion_classes.json
│
├── training/
│ ├── train_mask_tl.py
│ ├── train_emotion_tl.py
│ ├── evaluate_mask.py
│ └── evaluate_emotion.py
│
├── utils.py
│
├── templates/
│ └── index.html

Face-AI-System/
│
├── app.py
├── requirements.txt
│
├── models/
│ ├── mask_model.h5
│ ├── emotion_model.h5
│ └── emotion_classes.json
│
├── training/
│ ├── train_mask_tl.py
│ ├── train_emotion_tl.py
│ ├── evaluate_mask.py
│ └── evaluate_emotion.py
│
├── utils.py
│
├── templates/
│ └── index.html

---

## 🚀 How to Run

### 1. Clone the repository

git clone https://github.com/shinobi-itachi/Face-AI-System.git

cd Face-AI-System


### 2. Install dependencies

pip install -r requirements.txt


### 3. Run the application

python app.py


### 4. Open in browser

http://127.0.0.1:5000/


---

## 📊 Model Details

### 🟢 Mask Detection
- Transfer Learning using MobileNetV2  
- Hyperparameter tuning using Keras Tuner  
- Binary classification: Mask / No Mask  

---

### 🔵 Emotion Detection
- Transfer Learning using MobileNetV2  
- Dataset: FER2013  
- Classes:
  - Angry  
  - Disgust  
  - Fear  
  - Happy  
  - Sad  
  - Surprise  
  - Neutral  

---

## ⚠️ Limitations

- Emotion detection accuracy is limited due to:
  - Low-resolution dataset (FER2013)  
  - High similarity between emotion classes  
  - Sensitivity to lighting and face angles  

- Real-time performance may vary on CPU systems  

---

## 🧠 Key Learnings

- Built an end-to-end Deep Learning system  
- Designed multi-stage inference pipeline  
- Implemented real-time webcam streaming  
- Handled uncertainty using confidence thresholds  
- Deployed model using Flask  

---

## 🚀 Future Improvements

- Fine-tuning of emotion model  
- Better dataset (higher resolution)  
- GPU-based real-time optimization  
- Multi-face detection support  
- Advanced face detection (MTCNN / RetinaFace)  

---

## 👤 Author

**Rohit Kamble**

---

## ⭐ If you like this project
Give it a ⭐ on GitHub!
