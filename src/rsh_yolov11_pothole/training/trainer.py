from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_

from ..models.yolo_scale import (
    DJI_MAVIC3_FOCAL_MM,
    DJI_MAVIC3_RES_PX,
    DJI_MAVIC3_SENSOR_MM,
)
from .losses import MultiTaskYOLOScaleLoss


class Trainer:
    """
    Trainer untuk YOLOv11Scale + Regression Scale Head (RSH).

    Fungsi utama trainer ini:
    1. Menjembatani batch dataset lama yang masih punya `mpp`
       menjadi `metadata` yang dibutuhkan model.
    2. Menggabungkan instance masks per-image menjadi single binary mask
       agar cocok untuk segmentation loss biner.
    3. Menjalankan training loop dan optional validation loop.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        criterion: MultiTaskYOLOScaleLoss,
        optimizer: torch.optim.Optimizer,
        device: str | torch.device = "cuda",
        save_dir: str | Path = "outputs/train",
        grad_clip: Optional[float] = 10.0,
        use_amp: bool = True,
        use_area_loss: bool = False,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.grad_clip = grad_clip
        self.use_amp = use_amp and self.device.type == "cuda" and torch.cuda.is_available()
        self.use_area_loss = use_area_loss

        self.scaler = GradScaler(enabled=self.use_amp)

        self.model.to(self.device)

    @staticmethod
    def _to_batch_tensor(
        value: Any,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Ubah scalar / list / tensor menjadi tensor shape [B].
        """
        if torch.is_tensor(value):
            t = value.to(device=device, dtype=dtype).flatten()
            if t.numel() == 1:
                return t.repeat(batch_size)
            if t.numel() != batch_size:
                raise ValueError(f"Expected {batch_size} values, got {t.numel()}")
            return t

        if isinstance(value, (int, float)):
            return torch.full((batch_size,), float(value), dtype=dtype, device=device)

        if isinstance(value, (list, tuple)):
            if len(value) == 1:
                return torch.full((batch_size,), float(value[0]), dtype=dtype, device=device)
            if len(value) != batch_size:
                raise ValueError(f"Expected {batch_size} values, got {len(value)}")
            return torch.tensor(value, dtype=dtype, device=device)

        raise TypeError(f"Unsupported value type: {type(value)}")

    def _build_metadata(
        self,
        batch: Dict[str, Any],
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Prioritas:
        1. Jika batch sudah punya `metadata`, pakai langsung.
        2. Jika belum, rekonstruksi metadata dari `mpp` dengan asumsi kamera default.

        Output:
            metadata: [B, 4] = [altitude_m, focal_mm, sensor_mm, resolution_px]
            gsd_gt:   [B]
        """
        if "metadata" in batch and batch["metadata"] is not None:
            metadata = batch["metadata"].to(self.device).float()
            if metadata.ndim != 2 or metadata.shape[1] != 4:
                raise ValueError("metadata harus berbentuk [B, 4]")
            gsd_gt = batch.get("mpp", None)
            if gsd_gt is None:
                gsd_gt = metadata[:, 0] * metadata[:, 2] / (metadata[:, 1] * metadata[:, 3])
            else:
                gsd_gt = self._to_batch_tensor(gsd_gt, batch_size, self.device)
            return metadata, gsd_gt

        if "mpp" not in batch:
            raise KeyError(
                "Batch harus punya `metadata` atau minimal `mpp`. "
                "Dataset Anda saat ini baru punya `mpp`, jadi trainer ini merekonstruksi metadata dari sana."
            )

        gsd_gt = self._to_batch_tensor(batch["mpp"], batch_size, self.device)

        focal_mm = self._to_batch_tensor(
            batch.get("focal_length_mm", DJI_MAVIC3_FOCAL_MM),
            batch_size,
            self.device,
        )
        sensor_mm = self._to_batch_tensor(
            batch.get("sensor_width_mm", DJI_MAVIC3_SENSOR_MM),
            batch_size,
            self.device,
        )
        resolution_px = self._to_batch_tensor(
            batch.get("resolution_px", DJI_MAVIC3_RES_PX),
            batch_size,
            self.device,
        )

        # Dari formula:
        # gsd = (altitude * sensor) / (focal * resolution)
        # => altitude = gsd * focal * resolution / sensor
        altitude_m = gsd_gt * focal_mm * resolution_px / sensor_mm

        metadata = torch.stack(
            [altitude_m, focal_mm, sensor_mm, resolution_px],
            dim=1,
        )
        return metadata, gsd_gt

    def _merge_instance_masks(
        self,
        mask_list: list[Any],
        image_h: int,
        image_w: int,
    ) -> torch.Tensor:
        """
        Dataset saat ini mengembalikan mask per objek.
        Untuk segmentation loss biner, kita gabungkan menjadi 1 mask per-image.

        Output:
            masks: [B, 1, H, W]
        """
        merged_masks = []

        for masks in mask_list:
            if masks is None:
                merged = torch.zeros((image_h, image_w), dtype=torch.float32, device=self.device)
            else:
                masks = masks.to(self.device).float()

                if masks.ndim == 2:
                    merged = masks
                elif masks.ndim == 3:
                    # [N, H, W] -> OR semua object mask
                    merged = (masks.max(dim=0).values > 0.5).float()
                elif masks.ndim == 4 and masks.shape[1] == 1:
                    # [N, 1, H, W]
                    merged = (masks[:, 0].max(dim=0).values > 0.5).float()
                else:
                    raise ValueError(f"Unexpected mask shape: {tuple(masks.shape)}")

                if merged.shape != (image_h, image_w):
                    merged = F.interpolate(
                        merged.unsqueeze(0).unsqueeze(0),
                        size=(image_h, image_w),
                        mode="nearest",
                    ).squeeze(0).squeeze(0)

            merged_masks.append(merged)

        return torch.stack(merged_masks, dim=0).unsqueeze(1)  # [B,1,H,W]

    def _sum_area_targets(self, gt_area_list: list[Any], batch_size: int) -> torch.Tensor:
        """
        Dataset saat ini menyimpan gt_area_m2 per objek.
        Kita jumlahkan menjadi area total per image -> [B].
        """
        area_targets = []

        for area in gt_area_list:
            if area is None:
                area_targets.append(torch.tensor(0.0, device=self.device))
                continue

            if not torch.is_tensor(area):
                area = torch.tensor(area, dtype=torch.float32, device=self.device)
            else:
                area = area.to(self.device).float()

            area = area.flatten()

            if area.numel() == 0:
                area_targets.append(torch.tensor(0.0, device=self.device))
            else:
                valid = area[area >= 0.0]
                if valid.numel() == 0:
                    area_targets.append(torch.tensor(0.0, device=self.device))
                else:
                    area_targets.append(valid.sum())

        if len(area_targets) != batch_size:
            raise ValueError("Jumlah gt_area_m2 tidak cocok dengan batch size")

        return torch.stack(area_targets, dim=0)  # [B]

    def _prepare_batch(
        self,
        batch: Dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Return:
            images   : [B,3,H,W]
            metadata : [B,4]
            targets  : dict untuk criterion
        """
        images = batch["images"].to(self.device).float()
        batch_size = images.shape[0]
        image_h, image_w = images.shape[-2:]

        metadata, gsd_gt = self._build_metadata(batch, batch_size)

        masks = self._merge_instance_masks(batch["masks"], image_h, image_w)

        targets: Dict[str, torch.Tensor] = {
            "masks": masks,
            "gsd_gt": gsd_gt,
            "gsd_valid": (gsd_gt > 0).float(),
        }

        # Area loss default dimatikan.
        # Alasannya: dataset saat ini resize mask ke image_size,
        # sehingga pixel count mask bukan lagi pixel count resolusi asli.
        # Kalau nanti dataset sudah mengeluarkan mask/original-size target yang konsisten,
        # baru aktifkan area loss.
        if self.use_area_loss and "gt_area_m2" in batch:
            area_gt = self._sum_area_targets(batch["gt_area_m2"], batch_size)
            targets["area_gt"] = area_gt
            targets["area_valid"] = (area_gt > 0).float()
        else:
            targets["area_gt"] = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            targets["area_valid"] = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

        return images, metadata, targets

    def _run_step(
        self,
        batch: Dict[str, Any],
        training: bool,
    ) -> Dict[str, float]:
        images, metadata, targets = self._prepare_batch(batch)

        if training:
            self.optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=self.use_amp):
            output = self.model(
                images=images,
                metadata=metadata,
                seg_output_size=(images.shape[-2], images.shape[-1]),  # tetap di resolusi train saat ini
                return_gsd_gt=False,
            )
            loss_out = self.criterion(output=output, targets=targets)
            loss = loss_out.total

        if training:
            self.scaler.scale(loss).backward()

            if self.grad_clip is not None:
                self.scaler.unscale_(self.optimizer)
                clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

        gsd_mae = (output.gsd_pred.detach() - targets["gsd_gt"]).abs().mean().item()

        metrics = dict(loss_out.metrics)
        metrics["gsd_mae"] = float(gsd_mae)
        metrics["lr"] = float(self.optimizer.param_groups[0]["lr"])
        return metrics

    @staticmethod
    def _average_metrics(metric_list: list[Dict[str, float]]) -> Dict[str, float]:
        if len(metric_list) == 0:
            return {}

        keys = metric_list[0].keys()
        avg = {}
        for k in keys:
            avg[k] = float(sum(m[k] for m in metric_list) / len(metric_list))
        return avg

    def train_one_epoch(
        self,
        loader,
        epoch: int,
    ) -> Dict[str, float]:
        self.model.train()
        epoch_metrics = []

        for batch in loader:
            metrics = self._run_step(batch, training=True)
            epoch_metrics.append(metrics)

        avg = self._average_metrics(epoch_metrics)
        print(
            f"[train] epoch={epoch} "
            f"loss={avg.get('loss_total', 0.0):.4f} "
            f"seg={avg.get('loss_seg', 0.0):.4f} "
            f"gsd={avg.get('loss_gsd', 0.0):.4f} "
            f"area={avg.get('loss_area', 0.0):.4f} "
            f"gsd_mae={avg.get('gsd_mae', 0.0):.6f}"
        )
        return avg

    @torch.no_grad()
    def validate_one_epoch(
        self,
        loader,
        epoch: int,
    ) -> Dict[str, float]:
        self.model.eval()
        epoch_metrics = []

        for batch in loader:
            metrics = self._run_step(batch, training=False)
            epoch_metrics.append(metrics)

        avg = self._average_metrics(epoch_metrics)
        print(
            f"[valid] epoch={epoch} "
            f"loss={avg.get('loss_total', 0.0):.4f} "
            f"seg={avg.get('loss_seg', 0.0):.4f} "
            f"gsd={avg.get('loss_gsd', 0.0):.4f} "
            f"area={avg.get('loss_area', 0.0):.4f} "
            f"gsd_mae={avg.get('gsd_mae', 0.0):.6f}"
        )
        return avg

    def save_checkpoint(
        self,
        filename: str,
        epoch: int,
        metrics: Optional[Dict[str, float]] = None,
    ) -> Path:
        ckpt_path = self.save_dir / filename
        torch.save(
            {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "metrics": metrics or {},
            },
            ckpt_path,
        )
        return ckpt_path

    def fit(
        self,
        train_loader,
        epochs: int,
        val_loader=None,
        scheduler=None,
    ) -> None:
        best_score = float("inf")

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_one_epoch(train_loader, epoch)

            if val_loader is not None:
                val_metrics = self.validate_one_epoch(val_loader, epoch)
                monitor = val_metrics["loss_total"]
            else:
                val_metrics = None
                monitor = train_metrics["loss_total"]

            if scheduler is not None:
                try:
                    scheduler.step(monitor)
                except TypeError:
                    scheduler.step()

            self.save_checkpoint("last.pt", epoch, val_metrics or train_metrics)

            if monitor < best_score:
                best_score = monitor
                best_path = self.save_checkpoint("best.pt", epoch, val_metrics or train_metrics)
                print(f"best checkpoint updated: {best_path}")