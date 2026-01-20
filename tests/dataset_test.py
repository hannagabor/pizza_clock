import pytest
from pizza_clock.dataset import AdditionDataset, get_train_val_datasets


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
    p = 5
    train_length = 20
    val_length = 5

    train_dataset, val_dataset = get_train_val_datasets(p)

    assert len(train_dataset) == train_length
    assert len(val_dataset) == val_length


if __name__ == "__main__":
    pytest.main([__file__])
