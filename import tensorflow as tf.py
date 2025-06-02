import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, f1_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

data_dir = "C:/Users/pmls/Downloads/archive (1)/dataset"  
img_height, img_width = 128, 128
images = []
labels = []

for class_name in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_name)
    if os.path.isdir(class_path):  
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            try:
                
                img = load_img(img_path, target_size=(img_height, img_width))
                
                img_array = img_to_array(img) / 255.0
                images.append(img_array)
                labels.append(class_name)  
            except Exception as e:
                print(f"not found: {img_path}, Hata: {e}")

images = np.array(images)
labels = np.array(labels)


label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)


labels = to_categorical(labels, num_classes=4)

X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

train_datagen = ImageDataGenerator(horizontal_flip=True, rotation_range=10)
val_datagen = ImageDataGenerator()

train_data = train_datagen.flow(X_train, y_train, batch_size=32)
val_data = val_datagen.flow(X_val, y_val, batch_size=32)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25), 

    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(128, activation='relu'),  
    Dropout(0.5), 
    Dense(4, activation='softmax') 
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

os.makedirs("models", exist_ok=True)

model_checkpoint = ModelCheckpoint("models/best_cnn_model.keras", save_best_only=True, monitor='val_loss')

history = model.fit(
    train_data, 
    epochs=20,  
    validation_data=val_data, 
    callbacks=[model_checkpoint]  
)


val_loss, val_accuracy = model.evaluate(val_data)
print(f"val loss: {val_loss:.4f}, val accuracy: {val_accuracy:.4f}")

y_val_pred = model.predict(X_val)
y_val_pred_classes = np.argmax(y_val_pred, axis=1) 
y_val_true_classes = np.argmax(y_val, axis=1)  

cm = confusion_matrix(y_val_true_classes, y_val_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap='Blues', values_format='d')
plt.title("confusion matrix")
plt.show()

# F1 Score
f1 = f1_score(y_val_true_classes, y_val_pred_classes, average='macro')
print(f"f1 score: {f1:.4f}")

report = classification_report(y_val_true_classes, y_val_pred_classes, target_names=label_encoder.classes_)
print("\nclassification report:\n", report)

best_model = tf.keras.models.load_model("models/best_cnn_model.keras")

