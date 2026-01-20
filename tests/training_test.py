import pytest
import torch as t
from pizza_clock.training import ModularAdditionModelTrainer
from pizza_clock.config import Config
import tempfile
import os
from pizza_clock.models import Model


@pytest.fixture
def small_config():
    return Config(
        p=7,
        residual_dim=32,  # Smaller for faster tests
        device="cpu",  # Force CPU for reproducible tests
        use_wandb=False,
        seed=4,  # Fixed seed for reproducible tests
    )


@pytest.fixture
def trainer(small_config):
    return ModularAdditionModelTrainer(small_config)


class TestModularAdditionModelTrainer:
    def test_trainer_initialization(self, small_config, trainer):
        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.loss_fn is not None
        assert trainer.train_loader is not None
        assert trainer.val_loader is not None
        assert trainer.config == small_config

        assert trainer.model.token_embedding_table.num_embeddings == small_config.p
        assert trainer.model.attention_rate == small_config.attention_rate

    def test_training_step(self, trainer):
        x, y = next(iter(trainer.train_loader))

        initial_loss = trainer.training_step(x, y)

        assert isinstance(initial_loss, float)
        assert initial_loss > 0

        second_loss = trainer.training_step(x, y)
        assert isinstance(second_loss, float)
        assert second_loss >= 0

    def test_evaluation(self, trainer):
        val_loss, val_accuracy = trainer.evaluate()

        assert isinstance(val_loss, float)
        assert isinstance(val_accuracy, float)
        assert val_loss >= 0
        assert 0 <= val_accuracy <= 1

    def test_model_gradient_flow(self, trainer):
        x, y = next(iter(trainer.train_loader))

        trainer.optimizer.zero_grad()

        has_gradients_before = any(
            param.grad is not None for param in trainer.model.parameters()
        )
        assert not has_gradients_before

        trainer.training_step(x, y)

        has_gradients_after = any(
            param.grad is not None and param.grad.abs().sum() > 0
            for param in trainer.model.parameters()
        )
        assert has_gradients_after

    def test_train_short_run(self, trainer):
        x, y = next(iter(trainer.train_loader))
        trained_model = trainer.train(epochs=3, log_every_n_steps=1)

        from pizza_clock.models import Model

        assert isinstance(trained_model, Model)

    def test_model_predictions_change(self, trainer):
        x, y = next(iter(trainer.val_loader))

        trainer.model.eval()
        with t.no_grad():
            initial_output = trainer.model(x)
            initial_predictions = initial_output[:, -1, :].argmax(dim=-1)

        for _ in range(10):
            for train_x, train_y in trainer.train_loader:
                trainer.training_step(train_x, train_y)
                break

        trainer.model.eval()
        with t.no_grad():
            final_output = trainer.model(x)
            final_predictions = final_output[:, -1, :].argmax(dim=-1)

        predictions_changed = not t.equal(initial_predictions, final_predictions)
        assert predictions_changed, "Model predictions should change after training"

    def test_loss_decreases_with_training(self, trainer):
        losses = []

        for epoch in range(5):
            for x, y in trainer.train_loader:
                loss = trainer.training_step(x, y)
                losses.append(loss)

        assert losses[-1] < losses[0], (
            f"Loss should decrease: {losses[0]} -> {losses[-1]}"
        )

    def test_evaluation_reproducibility(self, trainer):
        val_loss_1, val_acc_1 = trainer.evaluate()
        val_loss_2, val_acc_2 = trainer.evaluate()

        assert abs(val_loss_1 - val_loss_2) < 1e-6
        assert abs(val_acc_1 - val_acc_2) < 1e-6

    def test_model_save_and_load(self, trainer):
        for _ in range(5):
            for x, y in trainer.train_loader:
                trainer.training_step(x, y)
                break
        for x, y in trainer.val_loader:
            break

        trainer.model.eval()
        with t.no_grad():
            initial_output = trainer.model(x)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp_file:
            temp_path = tmp_file.name

        try:
            t.save(trainer.model, temp_path)
            new_model = t.load(temp_path, weights_only=False)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

        new_model.eval()
        with t.no_grad():
            loaded_output = new_model(x)
        assert t.allclose(initial_output, loaded_output, atol=1e-6), (
            "Loaded model should produce identical output to original model"
        )

        assert loaded_output.shape == initial_output.shape
        assert loaded_output.shape == (x.shape[0], 2, trainer.config.p)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
