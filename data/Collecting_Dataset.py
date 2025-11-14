import cv2
import mediapipe as mp
import numpy as np
import os

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

DATA_PATH = 'dataset' 

gestures = ['Maju', 'Kanan', 'Kiri', 'Stop', 'Netral']
num_sequences = 80
sequence_length = 20  

def create_gesture_folders(data_path, gestures, num_sequences):
    for gesture in gestures:             
        gesture_path = os.path.join(data_path, gesture)
        if not os.path.exists(gesture_path):
            os.makedirs(gesture_path)
            print(f"Folder '{gesture}' created!")
        
        for sequence_num in range(num_sequences):
            sequence_path = os.path.join(gesture_path, f'sequence_{sequence_num}')
            if not os.path.exists(sequence_path):
                os.makedirs(sequence_path)
                print(f"Folder for sequence {sequence_num} in '{gesture}' created!")

def save_landmarks_and_images(face_landmarks, image, gesture, sequence_num, frame_num):
    sequence_path = os.path.join(DATA_PATH, gesture, f'sequence_{sequence_num}')
    
    # file npy
    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark]).flatten()
    npy_file_path = os.path.join(sequence_path, f'frame_{frame_num}.npy')
    np.save(npy_file_path, landmarks)

    # citra asli
    jpg_file_path = os.path.join(sequence_path, f'frame_{frame_num}.jpg')
    cv2.imwrite(jpg_file_path, image)

    # facemesh
    black_image = np.zeros_like(image)  
    draw_face_landmarks(black_image, face_landmarks, only_landmarks=False)  
    black_file_path = os.path.join(sequence_path, f'frame_{frame_num}-black.jpg')
    cv2.imwrite(black_file_path, black_image)

def draw_face_landmarks(image, face_landmarks, only_landmarks=False): 
    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_TESSELATION,  
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=2),  
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1) 
    )

def collect_data_interactive(gesture, num_sequences, sequence_length):
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Kamera tidak dapat dibuka.")
        return

    for sequence_num in range(num_sequences):
        print(f"Siap untuk mengumpulkan data untuk gesture: {gesture}, sequence: {sequence_num}. Tekan 'f' untuk memulai sequence.")

        frame_count = 0
        collecting = False

        while frame_count < sequence_length:
            ret, frame = cap.read()
            if not ret:
                print("Gagal membaca frame dari kamera.")
                break

            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    draw_face_landmarks(frame, face_landmarks)

                    if collecting:
                        save_landmarks_and_images(face_landmarks, frame, gesture, sequence_num, frame_count)
                        print(f"Frame {frame_count} captured for sequence {sequence_num} of {gesture}")
                        frame_count += 1

            
            display_frame = frame.copy()  
            cv2.putText(display_frame, f'{gesture} - Seq {sequence_num} Frame {frame_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if not collecting:
                cv2.putText(display_frame, 'Press "f" to start sequence', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(display_frame, 'Press "ESC" to exit', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Head Gesture Collection', display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('f') and not collecting:  # f untuk ambil data
                collecting = True  
            elif key == 27: 
                cap.release()
                cv2.destroyAllWindows()
                return

            
            if collecting and frame_count >= sequence_length:
                break

    cap.release()
    cv2.destroyAllWindows()

create_gesture_folders(DATA_PATH, gestures, num_sequences)

for gesture in gestures:
    collect_data_interactive(gesture, num_sequences, sequence_length)

print("Pengumpulan data selesai!")
