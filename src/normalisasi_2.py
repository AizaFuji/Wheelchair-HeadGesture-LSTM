import numpy as np
import os
import shutil

# Folder dataset asli dan tujuan normalisasi
DATA_PATH      = 'dataset20'
NORM_PATH      = 'datasetNorma'
gestures       = ['Maju', 'Kanan', 'Kiri', 'Stop', 'Netral']
num_sequences  = 200 # jumlah sequence per gesture
sequence_length = 20  # jumlah frame per sequence

def create_normalized_folders(base_path, gestures, num_sequences):
    """
    Hapus folder lama (jika ada) lalu buat ulang:
    base_path/gesture/sequence_i/
    """
    if os.path.exists(base_path):
        shutil.rmtree(base_path)
    for g in gestures:
        for seq in range(num_sequences):
            path = os.path.join(base_path, g, f'sequence_{seq}')
            os.makedirs(path, exist_ok=True)

def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Normalisasi landmarks wajah berdasarkan interocular distance.
    Input: 
      landmarks shape (N_landmarks*3,) atau (N_landmarks,3)
    Output:
      flattened array shape (N_landmarks*3,)
    """
    kp = landmarks.reshape(-1, 3)   # jadi (num_landmarks, 3)
    # Landmark mata kiri & kanan (FaceMesh indices)
    left_eye  = kp[33, :2]
    right_eye = kp[263, :2]
    # Hitung midpoint & skala
    center = (left_eye + right_eye) / 2
    scale  = np.linalg.norm(left_eye - right_eye)
    if scale < 1e-6:
        scale = 1.0
    # Normalisasi X,Y relatif center lalu dibagi scale
    kp[:, :2] = (kp[:, :2] - center) / scale
    # Normalisasi Z relatif scale agar proporsional
    kp[:, 2] /= scale
    return kp.flatten()

def normalize_and_save(orig_root, norm_root, gestures, num_seq, seq_len):
    """
    Loop folder gesture → sequence_i → frame_j.npy,
    normalize, lalu simpan ke norm_root dengan nama file sama.
    """
    for g in gestures:
        for seq in range(num_seq):
            for frame in range(seq_len):
                src = os.path.join(orig_root, g,
                                   f'sequence_{seq}',
                                   f'frame_{frame}.npy')
                dst = os.path.join(norm_root, g,
                                   f'sequence_{seq}',
                                   f'frame_{frame}.npy')
                if not os.path.isfile(src):
                    continue
                # muat dan normalisasi
                lm  = np.load(src)
                nlm = normalize_landmarks(lm)
                # simpan
                np.save(dst, nlm)
                print(f"[OK] {g}/sequence_{seq}/frame_{frame}.npy")

if __name__ == '__main__':
    # 1) Buat folder tujuan dengan struktur yang sama
    create_normalized_folders(NORM_PATH, gestures, num_sequences)
    # 2) Jalankan normalisasi & simpan
    normalize_and_save(DATA_PATH, NORM_PATH,
                       gestures, num_sequences, sequence_length)
    print("✅ Semua frame telah dinormalisasi dan disimpan.")
