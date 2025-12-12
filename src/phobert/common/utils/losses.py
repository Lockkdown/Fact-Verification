"""Custom loss functions cho fake news classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss để xử lý class imbalance.
    
    Focal Loss = -alpha * (1 - p_t)^gamma * log(p_t)
    
    Reference:
        Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)
    """
    
    def __init__(self, alpha=None, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Args:
            alpha: Class weights (tensor hoặc None). Shape: (num_classes,)
            gamma: Focusing parameter (mặc định 2.0)
            reduction: 'mean' hoặc 'sum'
        """
        super(FocalLoss, self).__init__()
        # Register alpha as buffer để tự động di chuyển với model
        if alpha is not None:
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits (batch_size, num_classes)
            targets: Ground truth labels (batch_size,)
        
        Returns:
            Focal loss scalar
        """
        # Tính cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Tính p_t
        p = torch.exp(-ce_loss)
        
        # Focal term: (1 - p_t)^gamma
        focal_weight = (1 - p) ** self.gamma
        
        # Focal loss
        loss = focal_weight * ce_loss
        
        # Apply alpha weights nếu có (alpha đã ở đúng device nhờ register_buffer)
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss
        
        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def compute_class_weights(labels: list, num_classes: int = 2) -> torch.Tensor:
    """Tính class weights từ distribution.
    
    Công thức: weight[c] = n_samples / (n_classes * n_samples_c)
    
    Args:
        labels: List các labels
        num_classes: Số classes
    
    Returns:
        Tensor weights shape (num_classes,)
    """
    from collections import Counter
    
    label_counts = Counter(labels)
    total = len(labels)
    
    weights = torch.zeros(num_classes)
    for c in range(num_classes):
        count = label_counts.get(c, 1)  # Tránh chia 0
        weights[c] = total / (num_classes * count)
    
    return weights
