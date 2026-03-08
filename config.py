import torch
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FOLDS = [f"fold_{i}" for i in range(1, 6)]

CLASS_NAMES = ["glass", "metal", "paper", "plastic"]
NUM_CLASSES = len(CLASS_NAMES) + 1  # background dahil

BATCH_SIZE = 2
NUM_EPOCHS = 10
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 5e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)