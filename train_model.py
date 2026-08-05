"""
Eye Disease Classification - Training Pipeline
================================================

Flow implemented in this script:
    Fundus_diseases Dataset -> Data Preprocessing -> Model Training ->
    Validation/Testing -> Saved Trained Model

The dataset directory (`Fundus_diseases/`) must contain one sub-folder per
class, e.g.:

    Fundus_diseases/
        cataract/
        diabetic_retinopathy/
        glaucoma/
        normal/

These are real retinal fundus photographs (not external eye photos), so
this script also trains the model to be confident only on genuine fundus
images -- an image validation gate in app1.py rejects clearly non-fundus
uploads before they ever reach the model (see validate_retinal_image()).

The class names are discovered automatically from the folder names, so
this script never needs to be edited when the dataset changes. The
trained model AND the class-name mapping are both saved to disk so that
the inference app (app1.py) always uses the exact same labels the model
was trained on, with no risk of them drifting out of sync.

Usage:
    python train_model.py
"""

import os
import json
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, Input
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, f1_score, ConfusionMatrixDisplay
import matplotlib
matplotlib.use("Agg")  # headless-safe backend for saving plots to disk
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Fundus_diseases")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
IMG_HEIGHT, IMG_WIDTH = 112, 112
BATCH_SIZE = 32
EPOCHS = 40
VAL_SPLIT = 0.10   # fraction of full dataset held out for validation
TEST_SPLIT = 0.10  # fraction of full dataset held out for testing (never trained on)

np.random.seed(SEED)
tf.random.set_seed(SEED)
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Step 1: Data Preprocessing
# ---------------------------------------------------------------------------
print(f"Loading dataset from: {DATA_DIR}")
if not os.path.isdir(DATA_DIR):
    raise FileNotFoundError(
        f"Dataset folder not found at '{DATA_DIR}'. "
        "Make sure the 'Fundus_diseases' folder sits next to this script."
    )

class_names = sorted(
    d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))
)
if len(class_names) < 2:
    raise ValueError("Need at least 2 class sub-folders inside Fundus_diseases/ to train a classifier.")

print(f"Discovered {len(class_names)} classes: {class_names}")

t0 = time.time()
images, labels = [], []
skipped = 0
for class_name in class_names:
    class_path = os.path.join(DATA_DIR, class_name)
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        try:
            img = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
            labels.append(class_name)
        except Exception as e:
            skipped += 1
            print(f"  [skip] could not read {img_path}: {e}")

print(f"Loaded {len(images)} images ({skipped} skipped/corrupt) in {time.time()-t0:.1f}s.")

images = np.array(images, dtype=np.float32)
labels_raw = np.array(labels)

label_encoder = LabelEncoder()
label_encoder.fit(class_names)  # fix an explicit, reproducible index order
labels_int = label_encoder.transform(labels_raw)
num_classes = len(label_encoder.classes_)
labels_cat = to_categorical(labels_int, num_classes=num_classes)

class_counts = np.bincount(labels_int, minlength=num_classes)
print("Class counts:", dict(zip(label_encoder.classes_, class_counts.tolist())))

# ---------------------------------------------------------------------------
# Step 2: Train / Validation / Test split
# ---------------------------------------------------------------------------
# The test set is held out completely and is NEVER shown to the model
# during training or checkpoint selection. It stands in for "unseen"
# images, the same way a real user upload would be unseen.
X_train, X_temp, y_train, y_temp = train_test_split(
    images, labels_cat,
    test_size=(VAL_SPLIT + TEST_SPLIT),
    random_state=SEED,
    stratify=labels_int,
)
relative_test_size = TEST_SPLIT / (VAL_SPLIT + TEST_SPLIT)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=relative_test_size,
    random_state=SEED,
    stratify=np.argmax(y_temp, axis=1),
)

print(f"Train: {len(X_train)} | Validation: {len(X_val)} | Test: {len(X_test)}")

# This dataset is close to balanced already (~1000-1100 images/class), so
# no oversampling/class-weighting is needed -- unlike the earlier, much
# smaller and more imbalanced dataset, this one trains stably with plain
# stratified splits.

# Data augmentation only on the TRAINING split. Validation/test data is
# evaluated as-is so metrics reflect real, unseen-style performance.
train_datagen = ImageDataGenerator(
    horizontal_flip=True,
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.08,
    height_shift_range=0.08,
    # NOTE: brightness_range was tested and removed -- Keras's brightness
    # augmentation is implemented via PIL's ImageEnhance, which assumes
    # images are on a 0-255 scale. Since images here are pre-normalized to
    # [0,1] before augmentation, brightness_range silently zeroed out
    # every augmented pixel (confirmed with an isolated test: output was
    # all-black for every image), which was why training never learned
    # anything above random-chance accuracy -- the model was training on
    # blank images. The other augmentations (flip/rotate/shift/zoom) were
    # each individually verified to work correctly on [0,1] data.
)
train_data = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE, seed=SEED)

# ---------------------------------------------------------------------------
# Step 3: Model Training
# ---------------------------------------------------------------------------
# GlobalAveragePooling2D keeps the parameter count (and overfitting risk)
# low regardless of input resolution -- a Flatten()->Dense head here would
# be needlessly huge and prone to the same instability seen on the earlier,
# smaller dataset.
model = Sequential([
    Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),

    # NOTE: dropout is deliberately placed only in the later layers. An
    # earlier version of this script put dropout after every conv block
    # (including the very first one) and, when tested, the model failed
    # to learn at all -- train loss stayed pinned at ln(4)=1.386 (random
    # guessing) for 6 full epochs. Removing dropout from the early
    # feature-extraction layers (confirmed via a controlled before/after
    # test) fixed this immediately.
    Conv2D(32, (3, 3), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.1),

    Conv2D(128, (3, 3), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.15),

    GlobalAveragePooling2D(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax'),  # matches dataset class count exactly
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)
model.summary()

model_path = os.path.join(MODEL_DIR, "best_cnn_model.keras")
callbacks = [
    ModelCheckpoint(model_path, save_best_only=True, monitor='val_accuracy', mode='max'),
    EarlyStopping(monitor='val_accuracy', mode='max', patience=8, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
]

history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
)

# ---------------------------------------------------------------------------
# Step 4: Validation / Testing
# ---------------------------------------------------------------------------
best_model = tf.keras.models.load_model(model_path)

val_loss, val_accuracy = best_model.evaluate(X_val, y_val, verbose=0)
print(f"\nValidation -> loss: {val_loss:.4f}, accuracy: {val_accuracy:.4f}")

test_loss, test_accuracy = best_model.evaluate(X_test, y_test, verbose=0)
print(f"Test (held-out, unseen-style) -> loss: {test_loss:.4f}, accuracy: {test_accuracy:.4f}")

y_test_pred = best_model.predict(X_test, verbose=0)
y_test_pred_classes = np.argmax(y_test_pred, axis=1)
y_test_true_classes = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_test_true_classes, y_test_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
fig, ax = plt.subplots(figsize=(7, 7))
disp.plot(cmap='Blues', values_format='d', ax=ax, xticks_rotation=45)
plt.title("Confusion Matrix (Held-out Test Set)")
plt.tight_layout()
cm_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
plt.savefig(cm_path)
print(f"Saved confusion matrix to {cm_path}")

f1 = f1_score(y_test_true_classes, y_test_pred_classes, average='macro')
print(f"Test macro F1 score: {f1:.4f}")

report = classification_report(
    y_test_true_classes, y_test_pred_classes, target_names=label_encoder.classes_
)
print("\nClassification report (test set):\n", report)

with open(os.path.join(MODEL_DIR, "classification_report.txt"), "w") as f:
    f.write(f"Validation accuracy: {val_accuracy:.4f}\n")
    f.write(f"Test accuracy: {test_accuracy:.4f}\n")
    f.write(f"Test macro F1: {f1:.4f}\n\n")
    f.write(report)

# ---------------------------------------------------------------------------
# Step 5: Persist the trained model + class-name mapping
# ---------------------------------------------------------------------------
class_names_path = os.path.join(MODEL_DIR, "class_names.json")
with open(class_names_path, "w") as f:
    json.dump(list(label_encoder.classes_), f, indent=2)
print(f"Saved class names to {class_names_path}")
print(f"Trained model already saved (best checkpoint) at {model_path}")

print("\nDone. Pipeline complete: dataset -> preprocessing -> training -> "
      "validation/testing -> saved model, ready for prediction on new images.")
