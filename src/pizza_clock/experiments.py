from pizza_clock.training import ModularAdditionModelTrainer
from pizza_clock.config import Config, get_device
import torch as t
from torch.multiprocessing import Pool
from datetime import date


def train_model(config: Config, epochs: int = 20000):
    trainer = ModularAdditionModelTrainer(config)
    model = trainer.train(epochs=epochs, log_every_n_steps=10, save_checkpoints=10)


paper_config = Config(device=get_device())

if __name__ == "__main__":
    args = []
    for seed in range(6, 8):
        for attention_rate in [0.0, 1.0]:
            name = f"{date.today().isoformat()}/attn{attention_rate}_seed{seed}"
            args.append(
                (
                    Config(
                        attention_rate=attention_rate,
                        wandb_project_name="modular-addition-5",
                        model_name=name,
                        use_wandb=True,
                        device=get_device(),
                        seed=seed,
                    ),
                    1000,
                ),
            )
    with Pool(5) as p:
        p.starmap(train_model, args)
