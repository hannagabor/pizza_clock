from torch.utils.data import Dataset
import torch as t


class AdditionDataset(Dataset):
    def __init__(self, p: int):
        self.p = p

    def __len__(self):
        return self.p * self.p

    def __getitem__(self, idx: int):
        a = idx // self.p
        b = idx % self.p
        c = (a + b) % self.p
        x = t.zeros((2, self.p))
        x[0, a] = 1.0
        x[1, b] = 1.0
        y = t.zeros((self.p,))
        y[c] = 1.0
        return x, y
