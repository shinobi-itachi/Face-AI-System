from flask import Flask, render_template, request, jsonify, Response
import numpy as np
import cv2
import tensorflow as tf
import json
import time
from utils import preprocess_mask, preprocess_emotion

app = Flask(__name__)

# Load models
mask_model = tf.keras.models.load_model("models/mask_model.h5")
emotion_model = tf.keras.models.load_model("models/emotion_model.h5")

# Load classes
with open("models/emotion_classes.json") as f:
    emotion_classes = json.load(f)

emotion_classes = {int(k): v for k, v in emotion_classes.items()}


# ----------------------------
# HOME
# ----------------------------
@app.route('/')
def home():
    return render_template('index.html')


# ----------------------------
# IMAGE UPLOAD PREDICTION
# ----------------------------
@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()

    file = request.files['image']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    mask_input = preprocess_mask(img)

    if mask_input is None:
        return jsonify({"error": "No face detected"})

    mask_pred = mask_model.predict(mask_input, verbose=0)[0][0]

    if mask_pred > 0.5:
        mask_label = "No Mask"
        mask_conf = float(mask_pred)
    else:
        mask_label = "Mask"
        mask_conf = float(1 - mask_pred)

    # Emotion logic
    if mask_label == "Mask":
        emotion = "Not reliable (Mask detected)"
        emotion_conf = 0.0
        reason = "Face covered"
    else:
        emo_input = preprocess_emotion(img)

        if emo_input is None:
            emotion = "No face detected"
            emotion_conf = 0.0
            reason = "Face not detected"
        else:
            emo_pred = emotion_model.predict(emo_input, verbose=0)[0]
            emotion_conf = float(np.max(emo_pred))
            emotion_idx = int(np.argmax(emo_pred))

            if emotion_conf < 0.6:
                emotion = "Uncertain"
                reason = "Low confidence"
            else:
                emotion = emotion_classes[emotion_idx]
                reason = "High confidence"

    # Alert logic
    if mask_label == "No Mask" and emotion == "Angry":
        alert = "⚠️ Risk"
    elif mask_label == "Mask":
        alert = "Emotion blocked"
    else:
        alert = "Normal"

    latency = round(time.time() - start_time, 3)

    return jsonify({
        "mask": mask_label,
        "emotion": emotion,
        "alert": alert,
        "latency": latency
    })


# ----------------------------
# WEBCAM STREAM
# ----------------------------
def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Mask prediction
        mask_input = preprocess_mask(frame)

        if mask_input is not None:
            mask_pred = mask_model.predict(mask_input, verbose=0)[0][0]

            if mask_pred > 0.5:
                mask_label = "No Mask"
            else:
                mask_label = "Mask"

            # Emotion only if no mask
            if mask_label == "Mask":
                emotion = "Blocked"
                alert = "Mask"
            else:
                emo_input = preprocess_emotion(frame)

                if emo_input is not None:
                    emo_pred = emotion_model.predict(emo_input, verbose=0)[0]
                    emotion_idx = int(np.argmax(emo_pred))
                    emotion = emotion_classes[emotion_idx]

                    if emotion == "Angry":
                        alert = "⚠️ Risk"
                    else:
                        alert = "Normal"
                else:
                    emotion = "No Face"
                    alert = "Error"

            # Display text
            cv2.putText(frame, f"Mask: {mask_label}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            cv2.putText(frame, f"Emotion: {emotion}", (10,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

            cv2.putText(frame, f"Alert: {alert}", (10,90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ----------------------------
# RUN
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)