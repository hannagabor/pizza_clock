import torch as t
from torch import nn, Tensor
from pizza_clock.models import Model
from pizza_clock.dataset import get_train_val_data
from jaxtyping import Float
from tqdm import tqdm
import wandb
from pizza_clock.config import Config


class ModularAdditionModelTrainer:
    def __init__(
        self,
        config: Config,
    ):
        self.train_loader, self.val_loader = get_train_val_data(config)
        self.model = Model(config).to(config.device)
        self.optimizer = t.optim.AdamW(
            self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.step = 0
        self.config = config

    def training_step(
        self, x: Float[Tensor, "batch seq_len"], y: Float[Tensor, "batch 1"]
    ) -> float:
        x = x.to(self.config.device)
        y = y.to(self.config.device)
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(x)
        logits = out[:, -1, :]
        loss = self.loss_fn(logits, y.squeeze())
        loss.backward()
        self.optimizer.step()
        if self.config.use_wandb:
            wandb.log({"train loss": loss.item()}, step=self.step)
        return loss.item()

    def train(self, epochs: int = 20000, log_every_n_steps: int = 100) -> Model:
        if self.config.use_wandb:
            wandb.init(
                project=self.config.wandb_project_name, name=self.config.wandb_name
            )
            wandb.watch(self.model)

        pbar = tqdm(range(epochs))
        for epoch in pbar:
            for x, y in self.train_loader:
                train_loss = self.training_step(x, y)
                if epoch % log_every_n_steps == 0:
                    val_loss, val_accuracy = self.evaluate()
                    pbar.set_postfix(
                        loss=f"{train_loss:.3f}, val_loss={val_loss:.3f}, val_acc={val_accuracy:.3f}",
                    )
                self.step += 1

        if self.config.use_wandb:
            wandb.finish()

        return self.model

    def evaluate(self) -> tuple[float, float]:
        self.model.eval()
        with t.no_grad():
            for x, y in self.val_loader:
                x = x.to(self.config.device)
                y = y.to(self.config.device)
                out = self.model(x)
                logits = out[:, -1, :]
                loss = self.loss_fn(logits, y.squeeze())
                accuracy = (logits.argmax(dim=-1) == y.squeeze()).float().mean().item()
                if self.config.use_wandb:
                    wandb.log(
                        {"val loss": loss.item(), "val accuracy": accuracy},
                        step=self.step,
                    )
        return loss.item(), accuracy
