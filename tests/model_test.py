from pizza_clock.models import Model
import pytest
import torch as t


class TestModel:
    def test_model_init(self):
        """Test that Model initializes with correct parameters."""
        vocab_size = 10
        attention_rate = 0.5
        residual_dim = 128

        model = Model(vocab_size, attention_rate, residual_dim)

        assert model.token_embedding_table.num_embeddings == vocab_size
        assert model.token_embedding_table.embedding_dim == residual_dim
        assert model.position_embedding_table.num_embeddings == 2
        assert model.position_embedding_table.embedding_dim == residual_dim
        assert model.attention_rate == attention_rate
        assert model.num_attention_heads == 4
        assert model.head_dim == residual_dim // 4
        assert model.num_mlp_hidden_units == 4 * residual_dim

    def test_model_forward(self):
        """Test that model forward pass runs without errors and returns correct shape."""
        vocab_size = 11
        attention_rate = 1.0
        batch_size = 3
        sequence_length = 2

        model = Model(vocab_size, attention_rate)

        # Create dummy input tensor with token indices
        x = t.randint(0, vocab_size, (batch_size, sequence_length))

        output = model.forward(x)

        expected_shape = (batch_size, sequence_length, vocab_size)
        assert output.shape == expected_shape

    def test_model_gradient_flow(self):
        """Test that gradients flow through the model."""
        vocab_size = 7
        attention_rate = 0.0

        model = Model(vocab_size, attention_rate)
        x = t.randint(0, vocab_size, (2, 2))

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
        model = Model(vocab_size, attention_rate)
        x = t.randint(0, vocab_size, (1, 2))

        output = model.forward(x)
        assert output.shape == (1, 2, vocab_size)

    def test_model_reproducibility(self):
        """Test that model gives same output with same seed."""
        vocab_size = 8
        attention_rate = 0.5

        # Set seed and create model
        t.manual_seed(42)
        model1 = Model(vocab_size, attention_rate)
        x = t.randint(0, vocab_size, (2, 2))
        output1 = model1.forward(x)

        # Reset seed and create another model
        t.manual_seed(42)
        model2 = Model(vocab_size, attention_rate)
        output2 = model2.forward(x)

        # Outputs should be identical
        assert t.allclose(output1, output2, atol=1e-6)
