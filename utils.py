import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    if len(faces) == 0:
        return None

    # pick the largest face
    largest_face = max(faces, key=lambda box: box[2] * box[3])
    x, y, w, h = largest_face
    face = img[y:y+h, x:x+w]
    return face

def preprocess_mask(img):
    face = detect_face(img)
    if face is None:
        return None

    face = cv2.resize(face, (224, 224))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=0)
    return face

def preprocess_emotion(img):
    face = detect_face(img)
    if face is None:
        return None

    face = cv2.resize(face, (224, 224))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=0)
    return face