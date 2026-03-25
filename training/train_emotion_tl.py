import tensorflow as tf
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

IMG_SIZE = (224, 224)

# ----------------------------
# DATA (TRAIN + TEST)
# ----------------------------
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train = datagen.flow_from_directory(
    "dataset/emotion/train",
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode='categorical'
)

val = datagen.flow_from_directory(
    "dataset/emotion/test",
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode='categorical'
)

# ----------------------------
# MODEL
# ----------------------------
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
output = Dense(train.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ----------------------------
# TRAIN
# ----------------------------
model.fit(
    train,
    validation_data=val,
    epochs=10
)

# ----------------------------
# SAVE
# ----------------------------
model.save("models/emotion_model.h5")

mapping = {v:k for k,v in train.class_indices.items()}
with open("models/emotion_classes.json","w") as f:
    json.dump(mapping,f)

print(" Emotion Model Saved!")