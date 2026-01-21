import pytest
from pizza_clock.dataset import AdditionDataset, get_train_val_data
from pizza_clock.config import Config


@pytest.mark.parametrize("p", [5, 7, 11])
def test_addition_dataset(p):
    dataset = AdditionDataset(p)
    assert len(dataset) == p * p
    idx = 0
    for a in range(p):
        for b in range(p):
            c = (a + b) % p
            input, label = dataset[idx]
            # Check input tensor
            assert input.shape == (2,)
            assert input[0].item() == a
            assert input[1].item() == b

            # Check output tensor
            assert label.shape == (1,)
            assert label.item() == c

            idx += 1


def test_train_val_split():
    config = Config(p=5, use_wandb=False)
    train_length = 20
    val_length = 5

    train_dataloader, val_dataloader = get_train_val_data(config)

    assert train_length - 1 <= len(train_dataloader.dataset) <= train_length + 1
    assert val_length - 1 <= len(val_dataloader.dataset) <= val_length + 1


if __name__ == "__main__":
    pytest.main([__file__])
