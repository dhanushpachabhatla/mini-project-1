import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class InverseDistanceBoundaryDiceLoss(nn.Module):
    def __init__(self, epsilon=1e-5, lambda_weight=0.6,class_weights=None):
        """
        epsilon: Small constant to prevent division by zero in the weight map.
        lambda_weight: Balances Dice and Boundary loss. 0.5 means equal weight.
        """
        super().__init__()
        self.epsilon = epsilon
        self.lambda_weight = lambda_weight
        self.class_weights = class_weights

    

    def forward(self, logits, targets,D_c_map):
        """
        logits: (B, C, D, H, W)
        targets: (B, D, H, W) - integer class indices
        """
        B, C, D, H, W = logits.shape
        
        probs = torch.softmax(logits, dim=1)
        # Create one-hot targets: shape becomes (B, C, D, H, W)
        targets_onehot = F.one_hot(targets, num_classes=C).permute(0, 4, 1, 2, 3).float()

        # Creating the Weight Map 
        W_map = 1.0 / (torch.abs(D_c_map) + self.epsilon)
        W_map = torch.clamp(W_map, max=10.0)   # or 5.0
        W_map = W_map / (W_map.mean() + 1e-8)
        

        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
            reduction='none'
        ) 
        
        # Multiply by our spatial inverse distance weights and take the mean
        boundary_ce_loss = (ce_loss * W_map).mean()
        
        # --- Eq 6: Standard Dice Loss ---
        dims = (0, 2, 3, 4)
        intersection = torch.sum(probs * targets_onehot, dims)
        union = torch.sum(probs + targets_onehot, dims)
        
        dice = (2.0 * intersection + 1e-5) / (union + 1e-5)
        # Ignore the background class (index 0) when computing Dice
        valid = targets_onehot.sum(dim=(0,2,3,4)) > 0
        dice_loss = 1.0 - dice[1:][valid[1:]].mean()
        
        # --- Eq 7: The Total Loss Function ---
        total_loss = (self.lambda_weight * dice_loss) + ((1.0 - self.lambda_weight) * boundary_ce_loss)
        
        return total_loss