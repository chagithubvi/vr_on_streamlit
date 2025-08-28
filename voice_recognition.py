# voice_recognition.py

import sqlite3
import torch
import torchaudio
import numpy as np
from scipy.spatial.distance import cosine
from scipy.signal import butter, sosfilt
from speechbrain.inference import SpeakerRecognition
import tempfile
import io
import warnings
import os

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Constants
DB_PATH = "enrollment_log3.db"
SAMPLE_RATE = 16000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-xvect-voxceleb",
    savedir="pretrained_models/spkrec-xvect-voxceleb"
)
model = model.to(device)

def apply_noise_reduction(audio, sr):
    sos = butter(10, 100, 'hp', fs=sr, output='sos')
    return sosfilt(sos, audio)

def extract_embedding(audio_bytes):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    signal, fs = torchaudio.load(tmp_path, normalize=True)
    signal = signal.to(device)
    if fs != SAMPLE_RATE:
        resample = torchaudio.transforms.Resample(fs, SAMPLE_RATE).to(device)
        signal = resample(signal)
    with torch.no_grad():
        emb = model.encode_batch(signal).squeeze().cpu().numpy()
    emb = emb.flatten()
    emb = np.pad(emb, (0, 512 - emb.shape[0]), 'constant') if emb.shape[0] < 512 else emb[:512]
    emb /= np.linalg.norm(emb)
    os.remove(tmp_path)
    return emb

def load_enrolled_embeddings():
    enrolled = {}
    thresholds = {}
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.execute("SELECT name, embedding, threshold FROM enrollment")
            for name, blob, threshold in cursor.fetchall():
                emb_array = np.frombuffer(blob, dtype=np.float32).reshape(-1, 512)
                enrolled[name] = emb_array
                thresholds[name] = float(threshold)
    except sqlite3.Error:
        pass  # Return empty if DB doesn't exist
    return enrolled, thresholds

def compute_speaker_thresholds(known_speakers):
    speaker_thresholds = {}
    for speaker, embeddings in known_speakers.items():
        distances = []
        for i, emb1 in enumerate(embeddings):
            for j, emb2 in enumerate(embeddings):
                if i != j:
                    distances.append(cosine(emb1, emb2))
        speaker_thresholds[speaker] = min(np.max(distances) * 1.1, 0.70) if distances else 0.70
    return speaker_thresholds

def recognize(test_emb, known_speakers, speaker_thresholds):
    best_distance = float("inf")
    best_speaker = "Unknown"
    for speaker, embeddings in known_speakers.items():
        for emb in embeddings:
            distance = cosine(test_emb, emb)
            if distance < best_distance:
                best_distance = distance
                best_speaker = speaker
    threshold = speaker_thresholds.get(best_speaker, 0.70)
    return best_speaker, best_distance if best_distance <= threshold else "Unknown", best_distance