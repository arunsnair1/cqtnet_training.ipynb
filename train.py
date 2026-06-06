import os
import json
import time
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from model import CQTNet

BATCH_SIZE = 64
EPOCHS = 200
LR = 0.0005
FEATURES_DIR = './cqt_features'
CHECKPOINT_DIR = './checkpoints'

class CQTDataset(Dataset):
    def __init__(self, split):
        self.split_dir = os.path.join(FEATURES_DIR, split)
        with open(os.path.join(self.split_dir, "labels.json"), "r") as f:
            self.labels_info = json.load(f)
        self.files = list(self.labels_info["files"].keys())
        
    def __len__(self): 
        return len(self.files)
        
    def __getitem__(self, idx):
        fname = self.files[idx]
        label = self.labels_info["files"][fname]
        feat = np.load(os.path.join(self.split_dir, fname))
        
        # Fixed 400 frames
        TARGET_FRAMES = 400
        if feat.shape[1] > TARGET_FRAMES:
            start = np.random.randint(0, feat.shape[1] - TARGET_FRAMES)
            feat = feat[:, start:start+TARGET_FRAMES]
        else:
            feat = np.pad(feat, ((0,0), (0, TARGET_FRAMES - feat.shape[1])))
            
        return torch.FloatTensor(np.expand_dims(feat, 0)), label

def train_model():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print("Initializing training pipeline...")
    
    train_data = CQTDataset("train")
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    num_classes = train_data.labels_info["num_classes"]
    model = CQTNet(num_classes).cuda()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = GradScaler()
    
    best_loss = float('inf')
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, correct, total = 0, 0, 0
        start_time = time.time()
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            
            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        print(f"Epoch {epoch:03d}/{EPOCHS} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pth"))

if __name__ == "__main__":
    train_model()
    print("Training complete.")
