import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)
DATASET_PATH = "dataset/emotion/test"

datagen = ImageDataGenerator(rescale=1./255)

val = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

model = tf.keras.models.load_model("models/emotion_model.h5")

val_preds = model.predict(val, verbose=0)
y_pred = np.argmax(val_preds, axis=1)
y_true = val.classes

class_names = list(val.class_indices.keys())

print("\nClassification Report:\n")
print(classification_report(
    y_true,
    y_pred,
    target_names=class_names
))

cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:\n")
print(cm)

print("\nClass-wise Accuracy Summary:\n")
for i, class_name in enumerate(class_names):
    total = cm[i].sum()
    correct = cm[i][i]
    acc = correct / total if total > 0 else 0
    print(f"{class_name}: {correct}/{total} correct -> {acc:.2%}")

print("\nError Analysis Notes:")
print("- Check which emotions have lower recall.")
print("- If visually similar classes are confused, mention that in interview.")
print("- Common confusion usually happens between sad/neutral/fear.")