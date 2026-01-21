from pizza_clock.training import ModularAdditionModelTrainer
from pizza_clock.config import Config, get_device
import torch as t
from multiprocessing import Pool


def train_model(i: int, attention_rate: float):
    config = Config(
        p=113,
        attention_rate=attention_rate,
        residual_dim=128,
        device=get_device(),
        use_wandb=True,
        seed=i,
        wandb_name=f"test_model_{i}_attention_{attention_rate}",
        wandb_project_name="modular-addition-attention-113-0.3",
        weight_decay=1.0,
        train_fraction=0.3,
    )

    trainer = ModularAdditionModelTrainer(config)
    model = trainer.train(epochs=2000, log_every_n_steps=10)
    t.save(model, f"saved_models/test_model_{i}_attention_{config.attention_rate}.pt")


if __name__ == "__main__":
    args = []
    for i in range(5):
        for attention_rate in [0.0, 1.0]:
            args.append((i, attention_rate))

    with Pool(5) as p:
        p.starmap(train_model, args)

# new_model = t.load("saved_models/test_model.pt", weights_only=False)
# for name, param in new_model.named_parameters():
#     print(name)
