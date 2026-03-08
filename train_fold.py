import os
import torch
from torch.utils.data import DataLoader

from config import (
    BASE_DIR,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    DEVICE,
    CHECKPOINT_DIR
)
from dataset import YOLODetectionDataset
from model import get_model
from utils import collate_fn


def get_fold_paths(fold_name):
    train_images = os.path.join(BASE_DIR, fold_name, "train", "images")
    train_labels = os.path.join(BASE_DIR, fold_name, "train", "labels")
    valid_images = os.path.join(BASE_DIR, fold_name, "valid", "images")
    valid_labels = os.path.join(BASE_DIR, fold_name, "valid", "labels")

    return train_images, train_labels, valid_images, valid_labels


def train_one_epoch(model, dataloader, optimizer, device, epoch, fold_name):
    model.train()
    total_loss = 0.0

    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

    avg_loss = total_loss / len(dataloader)
    print(f"{fold_name} | Epoch [{epoch+1}] Train Loss: {avg_loss:.4f}")
    return avg_loss


@torch.no_grad()
def validate_one_epoch(model, dataloader, device, epoch, fold_name):
    model.train()
    total_loss = 0.0

    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()

    avg_loss = total_loss / len(dataloader)
    print(f"{fold_name} | Epoch [{epoch+1}] Valid Loss: {avg_loss:.4f}")
    return avg_loss


def train_fold(fold_name):
    print(f"\n===== {fold_name} eğitimi başlıyor =====")

    train_images, train_labels, valid_images, valid_labels = get_fold_paths(fold_name)

    train_dataset = YOLODetectionDataset(train_images, train_labels)
    valid_dataset = YOLODetectionDataset(valid_images, valid_labels)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    model = get_model().to(DEVICE)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_valid_loss = float("inf")
    save_path = os.path.join(CHECKPOINT_DIR, f"{fold_name}_best.pth")

    history = {
        "train_losses": [],
        "valid_losses": []
    }

    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE, epoch, fold_name)
        valid_loss = validate_one_epoch(model, valid_loader, DEVICE, epoch, fold_name)

        history["train_losses"].append(train_loss)
        history["valid_losses"].append(valid_loss)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), save_path)
            print(f"{fold_name} için en iyi model kaydedildi: {save_path}")

    print(f"===== {fold_name} eğitimi bitti =====")
    print(f"{fold_name} best valid loss: {best_valid_loss:.4f}")

    return {
        "fold_name": fold_name,
        "best_valid_loss": best_valid_loss,
        "model_path": save_path,
        "history": history
    }