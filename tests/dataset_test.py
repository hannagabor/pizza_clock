import pytest
import pizza_clock
from pizza_clock.dataset import AdditionDataset


def test_addition_dataset():
    p = 5
    dataset = AdditionDataset(p)
    assert len(dataset) == p * p
    idx = 0
    for a in range(p):
        for b in range(p):
            c = (a + b) % p
            input, label = dataset[idx]
            print(a, b, c, input, label)
            # Check input tensor
            assert input.shape == (2, p)
            assert input[0].sum().item() == 1.0
            assert input[1].sum().item() == 1.0
            assert input[0, a].item() == 1.0
            assert input[1, b].item() == 1.0

            # Check output tensor
            assert label.shape == (p,)
            assert label.sum().item() == 1.0
            assert label[c].item() == 1.0

            idx += 1
