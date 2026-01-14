"""Losses including a differentiable Kernel Mean Embedding penalty."""

import torch
import torch.nn.functional as F


def pairwise_rbf(x: torch.Tensor, y: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """Compute RBF kernel matrix between rows of x and y."""
    x_norm = (x ** 2).sum(dim=1).view(-1, 1)
    y_norm = (y ** 2).sum(dim=1).view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y.t())
    return torch.exp(-gamma * dist)


def kme_penalty(source: torch.Tensor, target: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """Kernel Mean Embedding penalty between source and target embeddings."""
    k_ss = pairwise_rbf(source, source, gamma)
    k_tt = pairwise_rbf(target, target, gamma)
    k_st = pairwise_rbf(source, target, gamma)
    return k_ss.mean() + k_tt.mean() - 2.0 * k_st.mean()


def classification_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Standard cross-entropy classification loss."""
    return F.cross_entropy(logits, labels)
