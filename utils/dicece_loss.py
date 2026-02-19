import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceCELoss(nn.Module):
    def __init__(self, weight=None, dice_weight=1.0, ce_weight=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)

        probs = torch.softmax(logits, dim=1)
        targets_onehot = F.one_hot(targets, num_classes=probs.shape[1])
        targets_onehot = targets_onehot.permute(0,4,1,2,3).float()

        dims = (0,2,3,4)
        intersection = torch.sum(probs * targets_onehot, dims)
        union = torch.sum(probs + targets_onehot, dims)

        dice = (2*intersection + 1e-5) / (union + 1e-5)
        # ignore background channel (class 0)
        dice_loss = 1 - dice[1:].mean()
        
        return self.ce_weight*ce_loss + self.dice_weight*dice_loss
