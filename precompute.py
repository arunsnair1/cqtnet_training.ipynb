import os
import json
import time
import numpy as np
import librosa
from multiprocessing import Pool, cpu_count

SAMPLE_RATE = 22050
DURATION = 30
N_BINS = 84
HOP_LENGTH = 512
DATA_DIR = './data'
OUTPUT_DIR = './cqt_features'

def get_song_id(filename):
    first_part = os.path.basename(filename).split("-")[0]
    return int(first_part) if first_part.isdigit() else None

def process_file(args):
    mp3_path, npy_path = args
    if os.path.exists(npy_path):
        return "skipped"
    try:
        y, _ = librosa.load(mp3_path, sr=SAMPLE_RATE, duration=DURATION, mono=True)
        target_len = SAMPLE_RATE * DURATION
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
            
        cqt = librosa.cqt(y, sr=SAMPLE_RATE, n_bins=N_BINS, bins_per_octave=12, hop_length=HOP_LENGTH)
        cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
        
        os.makedirs(os.path.dirname(npy_path), exist_ok=True)
        np.save(npy_path, cqt_db)
        return "ok"
    except Exception:
        return "error"

def process_split(split):
    mp3_dir = os.path.join(DATA_DIR, split, "indian")
    npy_dir = os.path.join(OUTPUT_DIR, split)
    os.makedirs(npy_dir, exist_ok=True)
    
    mp3_files = sorted([f for f in os.listdir(mp3_dir) if f.endswith(".mp3")])
    tasks, labels, song_ids_set = [], {}, set()
    for fname in mp3_files:
        sid = get_song_id(fname)
        if sid is None: continue
        tasks.append((os.path.join(mp3_dir, fname), os.path.join(npy_dir, fname.replace(".mp3", ".npy"))))
        labels[fname.replace(".mp3", ".npy")] = sid
        song_ids_set.add(sid)
        
    workers = min(cpu_count(), 8)
    done, skip, err = 0, 0, 0
    with Pool(workers) as pool:
        for i, status in enumerate(pool.imap_unordered(process_file, tasks)):
            if status == "ok": done += 1
            elif status == "skipped": skip += 1
            else: err += 1
                
    sid_to_label = {s: i for i, s in enumerate(sorted(song_ids_set))}
    label_data = {
        "song_id_to_label": sid_to_label, 
        "num_classes": len(sid_to_label),
        "files": {fname: sid_to_label[sid] for fname, sid in labels.items()}
    }
    
    with open(os.path.join(npy_dir, "labels.json"), "w") as f: 
        json.dump(label_data, f, indent=2)
        
    return len(sid_to_label)

if __name__ == "__main__":
    print("Starting feature extraction...")
    train_c = process_split("train")
    test_c = process_split("test")
    print(f"Feature extraction complete. Train classes: {train_c}, Test classes: {test_c}")
