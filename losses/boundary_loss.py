import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt


def compute_distance_map(mask):
    """
    mask: numpy array (D,H,W) binary
    returns signed distance map
    """
    posmask = mask.astype(np.bool_)

    if posmask.any():
        negmask = ~posmask
        dist_out = distance_transform_edt(negmask)
        dist_in  = distance_transform_edt(posmask)
        return dist_out - dist_in
    else:
        return np.zeros_like(mask)


class BoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):
        """
        logits: (B, C, D, H, W)
        targets: (B, D, H, W)
        """

        probs = torch.softmax(logits, dim=1)

        total_loss = 0.0

        B, C = probs.shape[:2]

        for b in range(B):
            for c in range(1, C):  # skip background

                pred = probs[b, c]
                gt   = (targets[b] == c).float()

                if gt.sum() == 0:
                    continue

                # compute distance map (CPU)
                dist_map = compute_distance_map(gt.cpu().numpy())
                dist_map = torch.tensor(dist_map, dtype=torch.float32).to(logits.device)

                # boundary loss
                total_loss += torch.mean(pred * dist_map)

        return total_loss / B