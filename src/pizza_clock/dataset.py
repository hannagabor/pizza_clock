from torch.utils.data import Dataset
import torch as t


class AdditionDataset(Dataset):
    def __init__(self, p: int):
        self.p = p

    def __len__(self):
        return self.p * self.p

    def __getitem__(self, idx: int) -> tuple[t.Tensor, t.Tensor]:
        a = idx // self.p
        b = idx % self.p
        c = (a + b) % self.p
        x = t.Tensor([a, b])
        y = t.Tensor([c])
        return x, y
