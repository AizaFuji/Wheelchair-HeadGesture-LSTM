# Program untuk skenario Pengujian 

import cv2
import time
import datetime
import os
import csv
import numpy as np
import mediapipe as mp
from collections import deque
from tensorflow.keras.models import load_model

MODEL_PATH       = 'Model1.keras'
SEQUENCE_LENGTH  = 20
GESTURE_CLASSES  = ['Maju', 'Kiri', 'Kanan', 'Stop', 'Netral']
nod_time_thresh  = 1.0 
KEY_MAP = {
    ord('1'): 'Maju',
    ord('2'): 'Kiri',
    ord('3'): 'Kanan',
    ord('4'): 'Stop',
    ord('5'): 'Mundur',
    ord('0'): 'NA',
}



# Setup output folder & filenames
ts          = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR  = f'Testing_{ts}'
os.makedirs(OUTPUT_DIR, exist_ok=True)
VIDEO_PATH  = os.path.join(OUTPUT_DIR, f'rekaman_{ts}.avi')
FPS_CSV     = os.path.join(OUTPUT_DIR, f'fps_{ts}.csv')
LOG_CSV     = os.path.join(OUTPUT_DIR, f'log_{ts}.csv')
EVENT_CSV   = os.path.join(OUTPUT_DIR, f'events_{ts}.csv')


mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1,
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)
mp_draw   = mp.solutions.drawing_utils
blue_spec = mp_draw.DrawingSpec(color=(255,0,0), thickness=1, circle_radius=1)

# Model LSTM
model = load_model(MODEL_PATH)

seq_buffer      = deque(maxlen=SEQUENCE_LENGTH)
prev_base_pred  = None
nod_times       = deque(maxlen=2)
previous_class  = 'NA'
frame_times     = []

# Ground-truth & frame counter
current_gt = 'NA'
frame_idx  = 0

# Camera & writer setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Gagal membuka kamera")
w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_cam = cap.get(cv2.CAP_PROP_FPS) or 20.0
fourcc  = cv2.VideoWriter_fourcc(*'XVID')
writer  = cv2.VideoWriter(VIDEO_PATH, fourcc, fps_cam, (w, h))

# Make file CSV
log_file   = open(LOG_CSV,  'w', newline='')
log_writer = csv.writer(log_file)
log_writer.writerow(['timestamp','frame','gt_label','detected_class'])

fps_file    = open(FPS_CSV,  'w', newline='')
fps_writer  = csv.writer(fps_file)
fps_writer.writerow(['frame','fps'])

event_file   = open(EVENT_CSV,  'w', newline='')
event_writer = csv.writer(event_file)
event_writer.writerow(['timestamp','frame','gt_label','detected_class'])

try:
    while True:
        t_start = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res   = mp_face_mesh.process(rgb)

        detected_class = previous_class

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0]
            mp_draw.draw_landmarks(
                frame, lm, mp.solutions.face_mesh.FACEMESH_TESSELATION,
                blue_spec, blue_spec
            )
            pts = np.array([[p.x, p.y, p.z] for p in lm.landmark]).flatten()
            kp  = pts.reshape(-1,3)
            le, re = kp[33,:2], kp[263,:2]
            center = (le + re) / 2
            scale  = np.linalg.norm(le - re) or 1.0
            kp[:,:2] = (kp[:,:2] - center) / scale
            kp[:,2]  /= scale
            seq_buffer.append(kp.flatten())

            if len(seq_buffer) == SEQUENCE_LENGTH:
                inp       = np.expand_dims(np.array(seq_buffer), 0)
                probs     = model.predict(inp, verbose=0)[0]
                base_pred = GESTURE_CLASSES[np.argmax(probs)]

                if base_pred != 'Netral':
                    if base_pred == 'Stop' and prev_base_pred != 'Stop':
                        now = time.time()
                        nod_times.append(now)
                        if len(nod_times) == 2 and nod_times[1] - nod_times[0] <= nod_time_thresh:
                            detected_class = 'Mundur'
                            nod_times.clear()
                        else:
                            detected_class = 'Stop'
                    elif base_pred in ['Maju','Kiri','Kanan']:
                        detected_class = base_pred
                        nod_times.clear()
                prev_base_pred = base_pred
        else:
            detected_class = 'NA'
            prev_base_pred  = None
            nod_times.clear()
            seq_buffer.clear()

        if detected_class == 'Netral':
            detected_class = previous_class

        cv2.putText(frame, f'GT: {current_gt}', (10,110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,200,255), 2)
        cv2.putText(frame, f'Class: {detected_class}', (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        # FPS
        fps_now = 1.0 / (time.time() - t_start)
        frame_times.append(fps_now)
        fps_writer.writerow([frame_idx, f'{fps_now:.2f}'])
        cv2.putText(frame, f'FPS: {fps_now:.1f}', (10,70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

        writer.write(frame)
        cv2.imshow('Real-time Gesture Control', frame)
        log_writer.writerow([time.time(), frame_idx, current_gt, detected_class])

        key = cv2.waitKey(1) & 0xFF
        if key == 27:      

            break
        elif key in KEY_MAP:
            current_gt = KEY_MAP[key]
            event_writer.writerow([time.time(), frame_idx, current_gt, detected_class])

        previous_class = detected_class
        frame_idx += 1

finally:
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    log_file.close()
    fps_file.close()
    event_file.close()

    avg_fps = sum(frame_times) / len(frame_times) if frame_times else 0.0
    print(f'Video saved:    {VIDEO_PATH}')
    print(f'FPS report:    {FPS_CSV}')
    print(f'Frame log:     {LOG_CSV}')
    print(f'Event log:     {EVENT_CSV}')
    print(f'Average FPS:   {avg_fps:.2f}')
