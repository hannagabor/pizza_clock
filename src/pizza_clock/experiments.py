from pizza_clock.training import ModularAdditionModelTrainer
from pizza_clock.config import Config, get_device
import torch as t

for attention_rate in [0.0, 1.0]:
    for i in range(10):
        config = Config(
            p=59,
            attention_rate=attention_rate,
            residual_dim=128,
            device=get_device(),
            use_wandb=True,
            seed=i,
            wandb_name=f"test_model_{i}_attention_{attention_rate}",
            wandb_project_name="modular-addition-attention-scale_embedding",
        )

        trainer = ModularAdditionModelTrainer(config)
        model = trainer.train(epochs=20000, log_every_n_steps=200)
        t.save(
            model, f"saved_models/test_model_{i}_attention_{config.attention_rate}.pt"
        )

# new_model = t.load("saved_models/test_model.pt", weights_only=False)
# for name, param in new_model.named_parameters():
#     print(name)
