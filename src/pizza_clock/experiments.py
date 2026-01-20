from pizza_clock.training import ModularAdditionModelTrainer
from pizza_clock.config import Config, get_device
import torch as t

config = Config(
    p=59,
    attention_rate=0.5,
    residual_dim=128,
    device=get_device(),
    use_wandb=True,
)

trainer = ModularAdditionModelTrainer(config)
model = trainer.train(epochs=2000)
t.save(model, "test_model.pt")
new_model = t.load("test_model.pt", weights_only=False)
for name, param in new_model.named_parameters():
    print(name)
