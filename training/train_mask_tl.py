import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)
DATASET_PATH = "dataset/mask"

# ----------------------------
# DATA
# ----------------------------
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# ----------------------------
# MODEL BUILDER (TUNER)
# ----------------------------
def build_model(hp):

    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224,224,3)
    )

    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # 🔥 Dropout tuning
    dropout = hp.Float("dropout", 0.3, 0.7, step=0.1)
    x = Dropout(dropout)(x)

    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=output)

    # 🔥 Optimizer tuning
    lr = hp.Float("lr", 1e-4, 1e-2, sampling='log')
    opt_choice = hp.Choice("optimizer", ["adam", "sgd", "rmsprop"])

    if opt_choice == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    elif opt_choice == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    else:
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

# ----------------------------
# TUNER
# ----------------------------
tuner = kt.RandomSearch(
    build_model,
    objective="val_accuracy",
    max_trials=5,
    directory="tuner_logs",
    project_name="mask_detection"
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# ----------------------------
# SEARCH
# ----------------------------
tuner.search(
    train,
    validation_data=val,
    epochs=5,
    callbacks=[early_stop]
)

# ----------------------------
# FINAL MODEL
# ----------------------------
best_hp = tuner.get_best_hyperparameters(1)[0]
model = tuner.hypermodel.build(best_hp)

model.fit(
    train,
    validation_data=val,
    epochs=10
)

model.save("models/mask_model.h5")

print("🔥 Tuned Mask Model Saved!")