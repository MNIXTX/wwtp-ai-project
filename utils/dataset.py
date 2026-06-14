import torch
from torch.utils.data import Dataset

class TFTDataset(Dataset):
    """简单的 Dataset 包装器，用于 TFT 训练"""
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]