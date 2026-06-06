# CQTNet for Indian Music Cover Song Identification

This repository contains the training code for a custom CNN (`CQTNet`) trained to identify Indian music cover songs from MP3 audio files. 

## Workflow
The pipeline processes raw `.mp3` files into Constant-Q Transform (CQT) spectrograms using `librosa`, and then trains a deep CNN on an NVIDIA A100 GPU to classify the songs.

## Files
- `download_data.py`: Downloads the PP4-indian-music dataset from HuggingFace Hub.
- `precompute.py`: Extracts CQT features from raw audio and stores them as `.npy` arrays.
- `model.py`: PyTorch implementation of the `CQTNet` architecture.
- `train.py`: Main training loop with Automatic Mixed Precision (AMP) support for fast training.

## Usage
1. `python download_data.py`
2. `python precompute.py`
3. `python train.py`

## Pre-trained Model
The final model weights achieve >94% accuracy and are hosted on HuggingFace at `aruncodeshere/cqtnet-indian-music`.
