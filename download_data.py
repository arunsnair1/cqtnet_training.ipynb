import os
import time
from huggingface_hub import list_repo_tree, hf_hub_download

REPO_ID = "rojin254/PP4-indian-music-csi"
DATA_DIR = "./data"

def download_split(split_name):
    path = f"{split_name}/indian"
    print(f"Listing {split_name} files...")
    
    all_files = list(list_repo_tree(REPO_ID, path_in_repo=path, repo_type="dataset"))
    mp3_files = [f for f in all_files if f.path.endswith(".mp3")]
    
    print(f"Found {len(mp3_files)} mp3 files in {split_name} split.")
    downloaded, skipped = 0, 0
    start_time = time.time()
    
    for i, f in enumerate(mp3_files):
        local_path = os.path.join(DATA_DIR, f.path)
        if os.path.exists(local_path) and os.path.getsize(local_path) == f.size:
            skipped += 1
            continue
            
        hf_hub_download(repo_id=REPO_ID, filename=f.path, repo_type="dataset", local_dir=DATA_DIR)
        downloaded += 1
        
        if (i + 1) % 50 == 0 or (i + 1) == len(mp3_files):
            print(f"Progress: [{i+1}/{len(mp3_files)}] Downloaded: {downloaded}, Skipped: {skipped}")
            
    print(f"Completed {split_name} download in {time.time()-start_time:.0f}s.")
    return len(mp3_files)

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    print("Initiating dataset download...")
    train_count = download_split("train")
    test_count = download_split("test")
    print(f"Dataset preparation complete. Train files: {train_count}, Test files: {test_count}")
