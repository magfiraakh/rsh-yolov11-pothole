# src/adaptive_scale/losses.py
"""
Loss functions untuk YOLOv11Scale + Regression Scale Head.

Fokus file ini:
    - segmentation loss
    - gsd regression loss
    - optional uncertainty-aware loss
    - optional physical area consistency loss
    - detection loss sementara dibuat placeholder (0.0)

Agar mudah dipakai di thesis workflow, file ini dibuat modular.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptive_scale.models.yolo_scale import (
    ModelOutput,
    compute_physical_area_from_mask,
)


@dataclass
class LossOutput:
    total: torch.Tensor
    det_loss: torch.Tensor
    seg_loss: torch.Tensor
    gsd_loss: torch.Tensor
    area_loss: torch.Tensor
    metrics: Dict[str, float]


def dice_loss_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Dice loss biner dari logits.

    logits:  [B,1,H,W] atau [B,H,W]
    targets: [B,1,H,W] atau [B,H,W]
    """
    if logits.ndim == 4 and logits.shape[1] == 1:
        logits = logits[:, 0]
    if targets.ndim == 4 and targets.shape[1] == 1:
        targets = targets[:, 0]

    probs = torch.sigmoid(logits)
    targets = targets.float()

    intersection = (probs * targets).flatten(1).sum(dim=1)
    union = probs.flatten(1).sum(dim=1) + targets.flatten(1).sum(dim=1)

    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def bce_dice_seg_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
) -> torch.Tensor:
    """
    Kombinasi BCE + Dice untuk segmentasi biner.
    """
    targets = targets.float()
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = dice_loss_from_logits(logits, targets)
    return bce_weight * bce + dice_weight * dice


def masked_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if valid_mask is None:
        return F.l1_loss(pred, target)

    valid_mask = valid_mask.float()
    diff = torch.abs(pred - target) * valid_mask
    denom = valid_mask.sum().clamp_min(1.0)
    return diff.sum() / denom


def heteroscedastic_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    log_var: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Loss robust dengan learned uncertainty:
        exp(-log_var) * |e| + log_var
    """
    err = torch.abs(pred - target)
    loss = torch.exp(-log_var) * err + log_var

    if valid_mask is not None:
        valid_mask = valid_mask.float()
        loss = loss * valid_mask
        denom = valid_mask.sum().clamp_min(1.0)
        return loss.sum() / denom

    return loss.mean()


def area_target_from_mask_and_gsd(
    mask_target: torch.Tensor,
    gsd_gt: torch.Tensor,
) -> torch.Tensor:
    """
    Hitung ground-truth area fisik dari mask GT dan gsd_gt.
    """
    if mask_target.ndim == 4 and mask_target.shape[1] == 1:
        mask_target = mask_target[:, 0]
    elif mask_target.ndim != 3:
        raise ValueError("mask_target harus [B,1,H,W] atau [B,H,W]")

    pixel_count = mask_target.float().flatten(1).sum(dim=1)
    area_gt = pixel_count * (gsd_gt ** 2)
    return area_gt


class MultiTaskYOLOScaleLoss(nn.Module):
    """
    Loss utama untuk model YOLOv11Scale.

    Expected targets:
        targets = {
            "masks": Tensor [B,1,H,W] atau [B,H,W],
            "gsd_gt": Tensor [B],                  # opsional jika output.gsd_gt ada
            "area_gt": Tensor [B],                # opsional, jika ingin area langsung
            "gsd_valid": Tensor [B],              # opsional, 1 valid / 0 ignore
            "area_valid": Tensor [B],             # opsional, 1 valid / 0 ignore
        }

    Catatan:
        - detection loss saat ini placeholder = 0
        - area loss opsional
        - jika log_var tersedia dan use_uncertainty=True, dipakai heteroscedastic loss
    """

    def __init__(
        self,
        lambda_det: float = 0.0,
        lambda_seg: float = 1.0,
        lambda_gsd: float = 1.0,
        lambda_area: float = 0.0,
        use_uncertainty: bool = True,
        use_log_gsd: bool = False,
        seg_bce_weight: float = 0.5,
        seg_dice_weight: float = 0.5,
        area_from_pred_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.lambda_det = lambda_det
        self.lambda_seg = lambda_seg
        self.lambda_gsd = lambda_gsd
        self.lambda_area = lambda_area
        self.use_uncertainty = use_uncertainty
        self.use_log_gsd = use_log_gsd
        self.seg_bce_weight = seg_bce_weight
        self.seg_dice_weight = seg_dice_weight
        self.area_from_pred_threshold = area_from_pred_threshold

    def detection_loss_placeholder(self, output: ModelOutput) -> torch.Tensor:
        device = output.mask_logits.device
        return torch.zeros((), device=device)

    def segmentation_loss(
        self,
        mask_logits: torch.Tensor,
        mask_targets: torch.Tensor,
    ) -> torch.Tensor:
        if mask_targets.ndim == 3:
            mask_targets = mask_targets.unsqueeze(1)
        return bce_dice_seg_loss(
            mask_logits,
            mask_targets,
            bce_weight=self.seg_bce_weight,
            dice_weight=self.seg_dice_weight,
        )

    def gsd_regression_loss(
        self,
        gsd_pred: torch.Tensor,
        gsd_gt: torch.Tensor,
        log_var: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_log_gsd:
            gsd_pred = torch.log(gsd_pred.clamp_min(1e-8))
            gsd_gt = torch.log(gsd_gt.clamp_min(1e-8))

        if self.use_uncertainty and log_var is not None:
            return heteroscedastic_l1_loss(
                pred=gsd_pred,
                target=gsd_gt,
                log_var=log_var,
                valid_mask=valid_mask,
            )

        return masked_l1_loss(
            pred=gsd_pred,
            target=gsd_gt,
            valid_mask=valid_mask,
        )

    def physical_area_loss(
        self,
        mask_logits: torch.Tensor,
        gsd_pred: torch.Tensor,
        area_gt: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        area_pred = compute_physical_area_from_mask(
            mask_logits=mask_logits,
            gsd=gsd_pred,
            threshold=self.area_from_pred_threshold,
            from_logits=True,
        )

        return masked_l1_loss(
            pred=area_pred,
            target=area_gt,
            valid_mask=valid_mask,
        )

    def forward(
        self,
        output: ModelOutput,
        targets: Dict[str, torch.Tensor],
    ) -> LossOutput:
        device = output.mask_logits.device

        det_loss = self.detection_loss_placeholder(output)

        if "masks" not in targets:
            raise KeyError("targets wajib memiliki key 'masks'")

        mask_targets = targets["masks"].to(device).float()
        seg_loss = self.segmentation_loss(output.mask_logits, mask_targets)

        if "gsd_gt" in targets:
            gsd_gt = targets["gsd_gt"].to(device).float()
        elif output.gsd_gt is not None:
            gsd_gt = output.gsd_gt
        else:
            raise KeyError("gsd_gt tidak ditemukan di targets maupun output.gsd_gt")

        gsd_valid = targets.get("gsd_valid", None)
        if gsd_valid is not None:
            gsd_valid = gsd_valid.to(device)

        gsd_loss = self.gsd_regression_loss(
            gsd_pred=output.gsd_pred,
            gsd_gt=gsd_gt,
            log_var=output.log_var,
            valid_mask=gsd_valid,
        )

        area_loss = torch.zeros((), device=device)
        if self.lambda_area > 0.0:
            if "area_gt" in targets:
                area_gt = targets["area_gt"].to(device).float()
            else:
                area_gt = area_target_from_mask_and_gsd(mask_targets, gsd_gt)

            area_valid = targets.get("area_valid", None)
            if area_valid is not None:
                area_valid = area_valid.to(device)

            area_loss = self.physical_area_loss(
                mask_logits=output.mask_logits,
                gsd_pred=output.gsd_pred,
                area_gt=area_gt,
                valid_mask=area_valid,
            )

        total = (
            self.lambda_det * det_loss
            + self.lambda_seg * seg_loss
            + self.lambda_gsd * gsd_loss
            + self.lambda_area * area_loss
        )

        metrics = {
            "loss_total": float(total.detach().cpu().item()),
            "loss_det": float(det_loss.detach().cpu().item()),
            "loss_seg": float(seg_loss.detach().cpu().item()),
            "loss_gsd": float(gsd_loss.detach().cpu().item()),
            "loss_area": float(area_loss.detach().cpu().item()),
            "gsd_pred_mean": float(output.gsd_pred.detach().mean().cpu().item()),
            "gsd_gt_mean": float(gsd_gt.detach().mean().cpu().item()),
        }

        return LossOutput(
            total=total,
            det_loss=det_loss,
            seg_loss=seg_loss,
            gsd_loss=gsd_loss,
            area_loss=area_loss,
            metrics=metrics,
        )