import os
import random
import numpy as np

# Set Seed
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  

random.seed(SEED)
np.random.seed(SEED)

import tensorflow as tf
tf.random.set_seed(SEED)

tf.config.experimental.enable_op_determinism()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, InputLayer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

# Load Dataset
DATASET_PATH = 'datasetNorma'
gestures = ['Maju', 'Kanan', 'Kiri', 'Stop', 'Netral']
num_sequences = 80
sequence_length = 20
num_classes = len(gestures)

def load_dataset(dataset_path, gestures, num_sequences, sequence_length):
    X, y = [], []
    for label, gesture in enumerate(gestures):
        for sequence_num in range(num_sequences):
            sequence = []
            for frame_num in range(sequence_length):
                npy_path = os.path.join(dataset_path, gesture, f'sequence_{sequence_num}', f'frame_{frame_num}.npy')
                if os.path.exists(npy_path):
                    landmarks = np.load(npy_path)
                    sequence.append(landmarks)
            if len(sequence) == sequence_length:
                X.append(sequence)
                y.append(label)
    return np.array(X), np.array(y)

print("Memuat dataset...")
X, y = load_dataset(DATASET_PATH, gestures, num_sequences, sequence_length)
print(f"Dataset dimuat: X shape: {X.shape}, y shape: {y.shape}")

y = to_categorical(y, num_classes=num_classes)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=SEED, shuffle=True
)
print(f"Training data: {X_train.shape}, Testing data: {X_test.shape}")


#  Arsitetur LSTM

model = Sequential()
model.add(InputLayer(input_shape=(sequence_length, X.shape[2])))
model.add(LSTM(48, return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(24))
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

print("Training dimulai...")
history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping],
    shuffle=True 
)

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

model.save('Model1.keras')
print("Model berhasil disimpan")

# Grafik
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Confusion Matrik
y_train_pred = np.argmax(model.predict(X_train), axis=1)
y_test_pred = np.argmax(model.predict(X_test), axis=1)
y_train_true = np.argmax(y_train, axis=1)
y_test_true = np.argmax(y_test, axis=1)

classes = gestures

cm_train = confusion_matrix(y_train_true, y_train_pred)
disp_train = ConfusionMatrixDisplay(cm_train, display_labels=classes)
disp_train.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix - Training Set')
plt.show()

cm_test = confusion_matrix(y_test_true, y_test_pred)
disp_test = ConfusionMatrixDisplay(cm_test, display_labels=classes)
disp_test.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix - Validation Set')
plt.show()

print("Classification Report - Training Set")
print(classification_report(y_train_true, y_train_pred, target_names=classes))

print("Classification Report - Testing Set")
print(classification_report(y_test_true, y_test_pred, target_names=classes))

def calculate_tp_tn_fp_fn(cm):
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (tp + fp + fn)
    return tp, tn, fp, fn

tp_test, tn_test, fp_test, fn_test = calculate_tp_tn_fp_fn(cm_test)

print("TP, TN, FP, FN per class (Testing Set):")
for i, gesture in enumerate(classes):
    print(f"{gesture}: TP={tp_test[i]}, TN={tn_test[i]}, FP={fp_test[i]}, FN={fn_test[i]}")
