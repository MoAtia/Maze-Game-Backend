import numpy as np

def preprocess_landmarks(landmarks_df):
    coords = landmarks_df.to_numpy().reshape(-1, 3)
    wrist = coords[0][:2]
    mid_tip = coords[12][:2]
    scale = np.linalg.norm(mid_tip - wrist)
    coords[:, :2] = (coords[:, :2] - wrist) / (scale + 1e-6)
    coords = coords[:, :2].flatten()
    return coords[2:].reshape(1, -1)
