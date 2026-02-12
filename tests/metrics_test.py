import torch as t
from pizza_clock.config import Config
from pizza_clock.models import Model
from pizza_clock.metrics import get_logit_matrix


def test_get_logit_matrix_shape():
    config = Config(p=7, seed=42)
    model = Model(config)

    logit_matrix = get_logit_matrix(model)

    assert logit_matrix.shape == (config.p, config.p)


def test_get_logit_matrix_arrangement():
    """Test that logit_matrix[a, b] contains the logit for the correct answer (a + b) % p."""
    config = Config(p=7, seed=42)
    model = Model(config)

    logit_matrix = get_logit_matrix(model)

    for a in range(config.p):
        for b in range(config.p):
            input_tensor = t.tensor([[a, b]])
            logits = model(input_tensor)[0, -1, :]
            correct_answer = (a + b) % config.p
            expected_logit = logits[correct_answer].item()

            assert t.isclose(logit_matrix[a, b], t.tensor(expected_logit)), (
                f"Mismatch at ({a}, {b}): expected {expected_logit}, got {logit_matrix[a, b].item()}"
            )
