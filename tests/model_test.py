from pizza_clock.models import Model
import pytest
import torch as t
from pizza_clock.config import Config


class TestModel:
    def test_model_init(self):
        """Test that Model initializes with correct parameters."""
        config = Config(
            p=10,
            attention_rate=0.5,
            residual_dim=128,
            use_wandb=False,
        )

        model = Model(config)

        assert model.token_embedding_table.num_embeddings == config.p
        assert model.token_embedding_table.embedding_dim == config.residual_dim
        assert model.position_embedding_table.num_embeddings == 2
        assert model.position_embedding_table.embedding_dim == config.residual_dim
        assert model.attention_rate == config.attention_rate
        assert model.num_attention_heads == 4
        assert model.head_dim == config.residual_dim // 4
        assert model.num_mlp_hidden_units == 4 * config.residual_dim

    def test_model_forward(self):
        """Test that model forward pass runs without errors and returns correct shape."""
        config = Config(
            p=11,
            attention_rate=1.0,
            residual_dim=128,
            use_wandb=False,
        )

        model = Model(config)
        batch_size = 3
        sequence_length = 2

        model = Model(config)

        # Create dummy input tensor with token indices
        x = t.randint(0, config.p, (batch_size, sequence_length))

        output = model.forward(x)

        expected_shape = (batch_size, sequence_length, config.p)
        assert output.shape == expected_shape

    def test_model_gradient_flow(self):
        """Test that gradients flow through the model."""
        config = Config(
            p=7,
            attention_rate=0.0,
            residual_dim=128,
            use_wandb=False,
        )

        model = Model(config)
        x = t.randint(0, config.p, (2, 2))

        output = model.forward(x)
        loss = output[:, -1, :].mean()  # Dummy loss for testing
        loss.backward()

        # Check that at least some parameters have gradients
        has_gradients = any(param.grad is not None for param in model.parameters())
        assert has_gradients

    @pytest.mark.parametrize(
        "vocab_size,attention_rate",
        [
            (53, 0.0),
            (113, 0.5),
            (117, 1.0),
        ],
    )
    def test_model_with_different_parameters(self, vocab_size, attention_rate):
        """Test model with various parameter combinations."""
        config = Config(
            p=vocab_size,
            attention_rate=attention_rate,
            residual_dim=128,
            use_wandb=False,
        )
        model = Model(config)
        x = t.randint(0, config.p, (1, 2))

        output = model.forward(x)
        assert output.shape == (1, 2, config.p)

    def test_model_reproducibility(self):
        """Test that model gives same output with same seed."""
        config = Config(
            p=8,
            attention_rate=0.5,
            residual_dim=128,
            use_wandb=False,
        )

        # Set seed and create model
        t.manual_seed(42)
        model1 = Model(config)
        x = t.randint(0, config.p, (2, 2))
        output1 = model1.forward(x)

        # Reset seed and create another model
        t.manual_seed(42)
        model2 = Model(config)
        output2 = model2.forward(x)

        # Outputs should be identical
        assert t.allclose(output1, output2, atol=1e-6)
