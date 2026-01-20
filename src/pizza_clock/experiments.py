import torch as t
from pizza_clock.dataset import AdditionDataset
from pizza_clock.training import ModularAdditionModelTrainer
from torch.utils.data import DataLoader, random_split
from torch import Tensor, nn
from jaxtyping import Float
import einops

trainer = ModularAdditionModelTrainer(5)
trainer.train(epochs=10)
