import numpy as np
import torch
import torch.nn as nn
from torchmetrics import Metric

class ARKitLoss(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("lip_bs_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("non_lip_bs_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, lip_blendshape_loss, non_lip_blendshape_loss, total_loss):
        self.lip_bs_loss += lip_blendshape_loss
        self.non_lip_bs_loss += non_lip_blendshape_loss
        self.total_loss += total_loss
        self.count += 1

    def compute(self, split):
        return { loss: getattr(self, loss) / self.count for loss in ["lip_bs_loss", "non_lip_bs_loss", "total_loss"]}

    def __repr__(self):
        return self.loss.__repr__()
    
    def loss2logname(self, loss: str, split: str):
        if loss == "lip_bs_loss":
            return f"lip_bs_loss_{split}"
        elif loss == "non_lip_bs_loss":
            return f"non_lip_bs_loss_{split}"
        elif loss == "total_loss":
            return f"total_loss_{split}"

class MaskedBlendshapeConsistency:
    def __init__(self, lip_weighting=1, non_lip_weighting=1) -> None:
        self.loss = nn.MSELoss(reduction="mean")
        self.lip_blendshape_indices = range(30, 49)
        self.non_lip_blendshape_indices = list(set(range(0, 55)) - set(self.lip_blendshape_indices))
        self.lip_weighting = lip_weighting
        self.non_lip_weighting = non_lip_weighting
    
    def __call__(self, pred, gt, mask):
        pred_lip_blendshapes = pred[:, :, self.lip_blendshape_indices]
        gt_lip_blendshapes = gt[:, :, self.lip_blendshape_indices]
        
        pred_non_lip_blendshapes = pred[:, :, self.non_lip_blendshape_indices]
        gt_non_lip_blendshapes = gt[:, :, self.non_lip_blendshape_indices]
        
        lip_blendshape_loss = self.lip_weighting * self.loss(mask * pred_lip_blendshapes, mask * gt_lip_blendshapes)
        non_lip_blendshape_loss = self.non_lip_weighting * self.loss(mask * pred_non_lip_blendshapes, mask * gt_non_lip_blendshapes)
        
        total_loss = lip_blendshape_loss + non_lip_blendshape_loss

        return lip_blendshape_loss, non_lip_blendshape_loss, total_loss

    def __repr__(self):
        return self.loss.__repr__()