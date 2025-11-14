import cv2
import threading
import time
import socket
import numpy as np
import mediapipe as mp
from collections import deque
from tensorflow.keras.models import load_model


MODEL_PATH        = 'Model1.keras'
SEQUENCE_LENGTH   = 20
GESTURE_CLASSES   = ['Maju', 'Kiri', 'Kanan', 'Stop', 'Netral']
nod_time_thresh   = 1.0  # detik maksimal antar-nod untuk Mundur
ESP_HOST          = "192.168.137.220"
ESP_PORT          = 80

COMMAND_MAP = {
    'Maju':   'D\n',
    'Kiri':   'A\n',
    'Kanan':  'E\n',
    'Stop':   'C\n',
    'Mundur': 'B\n'
}

def init_camera_fast():
    """
    Fast camera initialization with multiple strategies
    """
    print("[INFO] Initializing camera...")
    
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if cap.isOpened():
            print("[INFO] Camera opened with DirectShow backend")
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            return cap
    except Exception as e:
        print(f"[WARNING] DirectShow failed: {e}")
    raise RuntimeError("No working camera found")

class SocketCommunicator:
    def __init__(self, host, port):
        self.host = host
        self.port = port 
        self.socket = None
        self.lock = threading.Lock()
        self.keep_running = True
        self.reconnect_thread = threading.Thread(target=self.reconnect_loop, daemon=True)
        self.reconnect_thread.start()

    def connect(self):
        with self.lock:
            if self.socket:  
                return
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect((self.host, self.port))
                self.socket = s
                print(f"[INFO] Terkoneksi dengan ESP32 di {self.host}:{self.port}")
            except socket.error as e:
                self.socket = None
                print(f"[ERROR] Gagal koneksi ke ESP32: {e}")

    def reconnect_loop(self):
        while self.keep_running:
            if not self.socket:
                print("[INFO] Mencoba reconnect ke ESP32...")
                self.connect()
            time.sleep(3)

    def send(self, data):
        with self.lock:
            if self.socket:
                try:
                    self.socket.sendall(data)
                    print(f"[DEBUG] Data terkirim: {data.decode().strip()}")
                except socket.error as e:
                    print(f"[ERROR] Gagal mengirim data: {e}")
                    self.socket.close()
                    self.socket = None
            else:
                print("[WARNING] Socket belum terkoneksi. Tidak bisa mengirim data.")

    def close(self):
        self.keep_running = False
        with self.lock:
            if self.socket:
                self.socket.close()
                self.socket = None


print("[INFO] Loading model...")
model = load_model(MODEL_PATH)

print("[INFO] Setting up MediaPipe...")
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)
mp_draw = mp.solutions.drawing_utils
blue_spec = mp_draw.DrawingSpec(color=(255,0,0), thickness=1, circle_radius=1)

print("[INFO] Connecting to ESP32...")
comm = SocketCommunicator(ESP_HOST, ESP_PORT)

# Sequence state
seq_buffer      = deque(maxlen=SEQUENCE_LENGTH)
prev_base_pred  = None
nod_times       = deque(maxlen=2)
previous_class  = 'NA'


cap = init_camera_fast()


print("[INFO] Applying camera optimizations...")
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
cap.set(cv2.CAP_PROP_EXPOSURE, -6)
cap.set(cv2.CAP_PROP_AUTO_WB, 0)

print("[INFO] Warming up camera...")
for i in range(5):
    ret, frame = cap.read()
    if not ret:
        break
    time.sleep(0.1)

print("[INFO] Camera ready! Starting gesture recognition...")

try:
    while True:
        t_start = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mp_face_mesh.process(rgb)

        current_class = previous_class

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0]
            mp_draw.draw_landmarks(
                frame, lm, mp.solutions.face_mesh.FACEMESH_TESSELATION,
                blue_spec, blue_spec
            )
            pts = np.array([[p.x, p.y, p.z] for p in lm.landmark]).flatten()
            kp = pts.reshape(-1, 3)
            le, re = kp[33, :2], kp[263, :2]
            center = (le + re) / 2
            scale = np.linalg.norm(le - re) or 1.0
            kp[:, :2] = (kp[:, :2] - center) / scale
            kp[:, 2] /= scale
            seq_buffer.append(kp.flatten())

            if len(seq_buffer) == SEQUENCE_LENGTH:
                inp = np.expand_dims(np.array(seq_buffer), 0)
                probs = model.predict(inp, verbose=0)[0]
                base_pred = GESTURE_CLASSES[np.argmax(probs)]

                if base_pred != 'Netral':
                    if base_pred == 'Stop' and prev_base_pred != 'Stop':
                        now = time.time()
                        nod_times.append(now)
                        if len(nod_times) == 2 and nod_times[1] - nod_times[0] <= nod_time_thresh:
                            current_class = 'Mundur'
                            nod_times.clear()
                        else:
                            current_class = 'Stop'
                    elif base_pred in ['Maju', 'Kiri', 'Kanan']:
                        current_class = base_pred
                        nod_times.clear()

                if current_class in COMMAND_MAP:
                    comm.send(COMMAND_MAP[current_class].encode('utf-8'))

                prev_base_pred = base_pred
        else:
            current_class = 'NA'
            prev_base_pred = None
            nod_times.clear()
            seq_buffer.clear()

        if current_class == 'Netral':
            current_class = previous_class

        fps_now = 1.0 / (time.time() - t_start)
        cv2.putText(frame, f'Class: {current_class}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(frame, f'FPS: {fps_now:.1f}', (10,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

        cv2.imshow('Real-time Gesture Control', frame)
        previous_class = current_class

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

finally:
    cap.release()
    cv2.destroyAllWindows()