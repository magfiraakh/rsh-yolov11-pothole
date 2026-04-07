from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.adaptive_scale.datasets.pothole_dataset import PotholeDataset, collate_fn
from src.adaptive_scale.models.yolo_scale import YOLOv11Scale
from src.adaptive_scale.training.losses import MultiTaskYOLOScaleLoss
from src.adaptive_scale.training.trainer import Trainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="data")
    p.add_argument("--train-ann", type=str, default="data/train.json")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--num-classes", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save-dir", type=str, default="runs/train")
    return p.parse_args()


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset = PotholeDataset(
        root=args.data_root,
        annotation_file=args.train_ann,
        image_size=640,
        use_segmentation=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
    )

    model = YOLOv11Scale(
        num_classes=args.num_classes,
        num_masks=1,
        predict_log_var=True,
    )

    criterion = MultiTaskYOLOScaleLoss(
        lambda_det=0.0,
        lambda_seg=1.0,
        lambda_gsd=1.0,
        lambda_area=0.0,   # tetap 0 dulu
        use_uncertainty=True,
        use_log_gsd=False,
    )

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=args.device,
        save_dir=save_dir,
        grad_clip=10.0,
        use_amp=True,
        use_area_loss=False,
    )

    trainer.fit(
        train_loader=loader,
        epochs=args.epochs,
        val_loader=None,
        scheduler=None,
    )


if __name__ == "__main__":
    main()