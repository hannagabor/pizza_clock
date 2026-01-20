from pizza_clock.training import ModularAdditionModelTrainer
from pizza_clock.config import Config, get_device

config = Config(
    p=59,
    attention_rate=0.5,
    residual_dim=128,
    device=get_device(),
    use_wandb=False,
)

trainer = ModularAdditionModelTrainer(config)
trainer.train(epochs=2000)
