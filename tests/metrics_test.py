import torch as t
from pizza_clock.config import Config
from pizza_clock.models import Model
from pizza_clock.metrics import (
    get_logit_matrix,
    compute_gradient_symmetry,
    compute_distance_irrelevance,
)


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


def test_compute_gradient_symmetry_returns_float():
    """Test that compute_gradient_symmetry returns a float value."""
    config = Config(p=7, seed=42)
    model = Model(config)

    result = compute_gradient_symmetry(model)

    assert isinstance(result, float)


def test_compute_gradient_symmetry_range():
    """Test that gradient symmetry is within expected cosine similarity range [-1, 1]."""
    config = Config(p=7, seed=42)
    model = Model(config)

    result = compute_gradient_symmetry(model)

    assert -1.0 <= result <= 1.0, (
        f"Gradient symmetry {result} is outside valid range [-1, 1]"
    )


def test_compute_gradient_symmetry_deterministic():
    """Test that compute_gradient_symmetry gives consistent results for same seed."""
    config = Config(p=7, seed=42)
    model = Model(config)

    t.manual_seed(0)
    result1 = compute_gradient_symmetry(model)
    t.manual_seed(0)
    result2 = compute_gradient_symmetry(model)

    assert result1 == result2, f"Results differ: {result1} vs {result2}"


def test_compute_distance_irrelevance_returns_float():
    """Test that compute_distance_irrelevance returns a float value."""
    config = Config(p=7, seed=42)
    model = Model(config)

    result = compute_distance_irrelevance(model)

    assert isinstance(result, float)


def test_compute_distance_irrelevance_positive():
    """Test that distance irrelevance is a positive value."""
    config = Config(p=7, seed=42)
    model = Model(config)

    result = compute_distance_irrelevance(model)

    assert result > 0, f"Distance irrelevance {result} should be positive"


def test_compute_distance_irrelevance_deterministic():
    """Test that compute_distance_irrelevance gives consistent results for same model."""
    config = Config(p=7, seed=42)
    model = Model(config)

    result1 = compute_distance_irrelevance(model)
    result2 = compute_distance_irrelevance(model)

    assert result1 == result2, f"Results differ: {result1} vs {result2}"
