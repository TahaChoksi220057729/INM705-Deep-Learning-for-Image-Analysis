import os
import argparse

import torch

from Dataset import build_brisc_dataloaders
from Models import UNet
from Losses import BCEDiceLoss
from Utils import (
    set_seed,
    evaluate,
    save_prediction_overlay,
    plot_training_curves
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train U-Net for BRISC segmentation")

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/segmentation_task"
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints"
    )

    parser.add_argument(
        "--samples_dir",
        type=str,
        default="samples"
    )

    parser.add_argument(
        "--image_size",
        type=int,
        default=256
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=2
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=25
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5
    )

    parser.add_argument(
        "--base_channels",
        type=int,
        default=64
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--use_augmentation",
        action="store_true"
    )

    parser.add_argument(
        "--use_wandb",
        action="store_true"
    )

    parser.add_argument(
        "--wandb_project",
        type=str,
        default="inm705-medical-segmentation"
    )

    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="brisc-unet-script"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(args.seed)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.samples_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    train_loader, val_loader, train_dataset, val_dataset = build_brisc_dataloaders(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_augmentation=args.use_augmentation
    )

    print("Train samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))

    model = UNet(
        in_channels=3,
        out_channels=1,
        base_channels=args.base_channels
    ).to(device)

    criterion = BCEDiceLoss(bce_weight=0.5)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=4
    )

    wandb = None
    use_wandb = True

    if args.use_wandb:
        try:
            import wandb as wandb_module

            wandb = wandb_module
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args)
            )

            use_wandb = True

        except ImportError:
            print("wandb is not installed. Continuing without wandb logging.")

    best_dice = -1.0

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_dice": [],
        "val_iou": []
    }

    for epoch in range(1, args.epochs + 1):
        model.train()

        epoch_losses = []

        for images, masks in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, masks)

            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        train_loss = sum(epoch_losses) / max(1, len(epoch_losses))

        val_loss, val_dice, val_iou = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        scheduler.step(val_dice)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)

        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "val/dice": val_dice,
                "val/iou": val_iou,
                "lr": optimizer.param_groups[0]["lr"],
            })

        if val_dice > best_dice:
            best_dice = val_dice

            best_checkpoint_path = os.path.join(
                args.checkpoint_dir,
                "unet_best.pt"
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "image_size": args.image_size,
                    "base_channels": args.base_channels,
                    "best_dice": best_dice,
                },
                best_checkpoint_path
            )

        with torch.no_grad():
            sample_images, sample_masks = next(iter(val_loader))

            sample_images = sample_images.to(device)
            logits = model(sample_images)
            preds = (torch.sigmoid(logits) > 0.5).float().cpu()

            sample_path = os.path.join(
                args.samples_dir,
                f"epoch_{epoch:03d}_overlay.png"
            )

            save_prediction_overlay(
                sample_images[0].cpu(),
                sample_masks[0],
                preds[0],
                sample_path
            )

            if use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "val/overlay": wandb.Image(sample_path)
                })

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_dice={val_dice:.4f} | "
            f"val_iou={val_iou:.4f}"
        )

    final_checkpoint_path = os.path.join(
        args.checkpoint_dir,
        "unet_final.pt"
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "image_size": args.image_size,
            "base_channels": args.base_channels,
            "best_dice": best_dice,
        },
        final_checkpoint_path
    )

    plot_training_curves(
        history,
        output_path=os.path.join(args.samples_dir, "training_curves.png")
    )

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()