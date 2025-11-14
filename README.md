# 🎯 Wheelchair Movement Control Based on Head Gesture using LSTM

This project implements a **wheelchair control system** using **head gestures** detected via webcam and classified using a **Long Short-Term Memory (LSTM)** model.  
Developed as part of my final project at **Institut Teknologi Sepuluh Nopember (ITS)**.

---

## 🧠 Overview

Individuals with **tetraplegia** face difficulty operating conventional wheelchairs using their hands.  
This system allows **hands-free wheelchair control** by interpreting **head movements** captured by a camera in real time.

The model detects and classifies five gestures:
- 🟢 Forward  
- 🔵 Backward  
- 🟡 Left  
- 🟠 Right  
- 🔴 Stop  

---

## 🧩 Technologies Used

| Component | Description |
|------------|--------------|
| **MediaPipe FaceMesh** | Extracts 468 facial landmarks |
| **TensorFlow / Keras** | LSTM-based sequence classification |
| **ESP32** | Wireless control of wheelchair motors |
| **Intel NUC / Laptop** | Edge computing for real-time inference |

---

## ⚙️ System Architecture

![System Diagram](docs/system_diagram.png)

1. **Camera** captures head movements.  
2. **MediaPipe** extracts facial landmarks (x, y, z).  
3. **Data normalization** aligns features using interocular distance.  
4. **LSTM model** classifies gestures (Forward, Backward, Left, Right, Stop).  
5. **ESP32** receives predicted command via Wi-Fi to move the wheelchair.

---

## 🧪 Model Training

**Dataset:**
- 400 samples total (5 classes × 80 samples)
- Each sample = 20 frames
- Features extracted from MediaPipe FaceMesh

**Model 1 (LSTM):**
- 2 stacked LSTM layers (48 & 24 units)
- Dropout 0.4 and 0.3  
- Dense(64, ReLU) → Dense(5, Softmax)
- Validation Accuracy: **100%**

**Model 2 (BiLSTM):**
- Bidirectional LSTM layers (32 & 16 units)
- Dropout (0.2, 0.3)
- Validation Accuracy: **98%**

---

## 📊 Evaluation Results

| Condition | Accuracy | Notes |
|------------|-----------|-------|
| 50 cm camera distance | **98%** | Optimal |
| 70 cm camera distance | 91% | Less accurate |
| 100 lux lighting | **97.3%** | Best performance |
| New user test | >80% | Good generalization |





