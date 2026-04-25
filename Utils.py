import os
import random

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from Losses import binary_dice_score, binary_iou_score


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def denormalize_image(image):
    image = (image * 0.5) + 0.5
    return torch.clamp(image, 0.0, 1.0)


def save_prediction_overlay(image, mask_gt, mask_pred, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    image_np = denormalize_image(image).permute(1, 2, 0).cpu().numpy()
    gt_np = (mask_gt.squeeze(0).cpu().numpy() > 0.5).astype(np.uint8)
    pred_np = (mask_pred.squeeze(0).cpu().numpy() > 0.5).astype(np.uint8)

    overlay = image_np.copy()

    overlay[gt_np == 1, 1] = np.clip(overlay[gt_np == 1, 1] + 0.5, 0, 1)
    overlay[pred_np == 1, 0] = np.clip(overlay[pred_np == 1, 0] + 0.5, 0, 1)

    out = (overlay * 255).astype(np.uint8)

    Image.fromarray(out).save(output_path)


def evaluate(model, loader, criterion, device):
    model.eval()

    losses = []
    dices = []
    ious = []

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            logits = model(images)

            loss = criterion(logits, masks)
            dice = binary_dice_score(logits, masks)
            iou = binary_iou_score(logits, masks)

            losses.append(loss.item())
            dices.append(dice.item())
            ious.append(iou.item())

    mean_loss = sum(losses) / max(1, len(losses))
    mean_dice = sum(dices) / max(1, len(dices))
    mean_iou = sum(ious) / max(1, len(ious))

    return mean_loss, mean_dice, mean_iou


def plot_training_curves(history, output_path=None):
    epochs_range = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(14, 4))

    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, history["train_loss"], label="Train loss")
    plt.plot(epochs_range, history["val_loss"], label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss curves")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, history["val_dice"], label="Val Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.title("Validation Dice")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, history["val_iou"], label="Val IoU")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.title("Validation IoU")
    plt.legend()

    plt.tight_layout()

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150)

    plt.show()


def visualise_dataset_samples(dataset, n=10):
    n = min(n, len(dataset))

    fig, axes = plt.subplots(2, n, figsize=(12, 5))

    if n == 1:
        axes = np.expand_dims(axes, axis=1)

    for i in range(n):
        image, mask = dataset[i]

        axes[0, i].imshow(denormalize_image(image).permute(1, 2, 0))
        axes[0, i].set_title("Image")
        axes[0, i].axis("off")

        axes[1, i].imshow(mask.squeeze(0), cmap="gray")
        axes[1, i].set_title("Mask")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.show()