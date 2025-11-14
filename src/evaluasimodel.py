# Evaluasi Model dengan dataset testing

import os
import random
import numpy as np

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)  
np.random.seed(SEED)

import tensorflow as tf
tf.random.set_seed(SEED)

import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

# ======== KONFIGURASI =========
MODEL_PATH = 'Model1.keras' 
DATASET_PATH = 'datasettestN_2'     
gestures = ['Maju', 'Kanan', 'Kiri', 'Stop', 'Netral']
num_classes = len(gestures)
sequence_length = 20
num_sequences = 30 

def load_dataset(dataset_path, gestures, num_sequences, sequence_length):
    X, y = [], []
    for label, gesture in enumerate(gestures):
        for seq in range(num_sequences):
            sequence = []
            for frame in range(sequence_length):
                npy_path = os.path.join(dataset_path, gesture, f'sequence_{seq}', f'frame_{frame}.npy')
                if os.path.exists(npy_path):
                    sequence.append(np.load(npy_path))
            if len(sequence) == sequence_length:
                X.append(sequence)
                y.append(label)
    return np.array(X), np.array(y)

print("Memuat model dan dataset...")
model = load_model(MODEL_PATH)
X_test, y_test = load_dataset(DATASET_PATH, gestures, num_sequences, sequence_length)
y_test_cat = to_categorical(y_test, num_classes=num_classes)


loss, acc = model.evaluate(X_test, y_test_cat)
print(f"Akurasi: {acc:.4f} | Loss: {loss:.4f}")


y_pred = np.argmax(model.predict(X_test), axis=1)

folder_name = datetime.now().strftime("EvaluasiTestingBaru_%Y%m%d_%H%M%S")
os.makedirs(folder_name, exist_ok=True)

# ======== CONFUSION MATRIX =========
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=gestures)
fig_cm, ax = plt.subplots(figsize=(6, 6))
disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
plt.title("Confusion Matrix - Dataset Testing")
plt.savefig(os.path.join(folder_name, "confusion_matrix_testing_baru.png"))
plt.close(fig_cm)

# ======== CLASSIFICATION REPORT =========
report = classification_report(y_test, y_pred, target_names=gestures, output_dict=True)
pd.DataFrame(report).transpose().to_csv(os.path.join(folder_name, "classification_report_testing_baru.csv"))

# ======== TP, TN, FP, FN =========
def calculate_confusion_elements(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    metrics = []
    for i, class_name in enumerate(classes):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - (TP + FP + FN)
        metrics.append({'Class': class_name, 'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN})
    return pd.DataFrame(metrics)

conf_matrix_df = calculate_confusion_elements(y_test, y_pred, gestures)
conf_matrix_df.to_csv(os.path.join(folder_name, "TP_TN_FP_FN_testing_baru.csv"), index=False)

print(f"Hasil evaluasi disimpan dalam folder: {folder_name}")
