from torch.utils.data import Dataset, DataLoader
import torch as t
from torch.utils.data import random_split
from jaxtyping import Int
from torch import Tensor


class AdditionDataset(Dataset):
    def __init__(self, p: int):
        self.p = p

    def __len__(self):
        return self.p * self.p

    def __getitem__(self, idx: int) -> tuple[Int[Tensor, "2"], Int[Tensor, "1"]]:
        a = idx // self.p
        b = idx % self.p
        c = (a + b) % self.p
        x = t.tensor([a, b], dtype=t.int32)
        y = t.tensor([c], dtype=t.long)
        return x, y


def get_train_val_data(p: int):
    dataset = AdditionDataset(p)
    generator = t.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [0.8, 0.2], generator=generator)
    train_dataloader = DataLoader(
        train_dataset, batch_size=len(train_dataset), shuffle=True
    )
    val_dataloader = DataLoader(val_dataset, batch_size=len(val_dataset))
    return train_dataloader, val_dataloader
