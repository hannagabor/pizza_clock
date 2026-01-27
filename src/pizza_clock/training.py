import torch as t
from torch import nn, Tensor
from pizza_clock.models import Model
from pizza_clock.dataset import get_train_val_data
from jaxtyping import Float
from tqdm import tqdm
import wandb
from pizza_clock.config import Config
from copy import deepcopy
from os import makedirs
import pandas as pd
import json
from pathlib import Path


class ModularAdditionModelTrainer:
    def __init__(
        self,
        config: Config,
    ):
        self.train_loader, self.val_loader = get_train_val_data(config)
        self.model = Model(config).to(config.device)
        self.optimizer = t.optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.98),
        )
        self.scheduler = t.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda step: min(step / 10, 1)
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.step = 0
        self.config = config
        self.loss_data = []
        self.all_models = []
        self.save_model_dir = f"saved_models/{self.config.wandb_name}"
        if Path(self.save_model_dir).exists():
            raise ValueError(
                f"Model directory {self.save_model_dir} already exists. Please choose a different wandb_name to avoid overwriting."
            )

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
        self.scheduler.step()
        if self.config.use_wandb:
            wandb.log({"train loss": loss.item()}, step=self.step)
        return loss.item()

    def train(
        self,
        epochs: int = 20000,
        log_every_n_steps: int = 100,
        save_checkpoints: int = 100,
    ) -> Model:
        if self.config.use_wandb:
            wandb.init(
                project=self.config.wandb_project_name, name=self.config.wandb_name
            )
            self.wandb_run_id = wandb.run.id
            wandb.watch(self.model)

        pbar = tqdm(range(epochs))
        for epoch in pbar:
            for x, y in self.train_loader:
                train_loss = self.training_step(x, y)
                if epoch % log_every_n_steps == 0:
                    val_loss, val_accuracy = self.evaluate()
                    self.loss_data.append(
                        {
                            "step": self.step,
                            "train_loss": train_loss,
                            "val_loss": val_loss,
                            "val_accuracy": val_accuracy,
                        }
                    )
                    pbar.set_postfix(
                        loss=f"{train_loss:.3f}, val_loss={val_loss:.3f}, val_acc={val_accuracy:.3f}",
                    )
                if save_checkpoints > 0 and epoch % save_checkpoints == 0:
                    self.all_models.append(deepcopy(self.model))
                self.step += 1

        if self.config.use_wandb:
            wandb.finish()

        self.save_models()

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

    def save_models(self):
        makedirs(self.save_model_dir, exist_ok=True)
        for i, model in enumerate(self.all_models):
            t.save(
                model,
                f"{self.save_model_dir}/model_{i}.pt",
            )
        t.save(
            self.model,
            f"{self.save_model_dir}/final_model.pt",
        )
        pd.DataFrame(self.loss_data).to_csv(
            f"{self.save_model_dir}/loss_data.csv", index=False
        )
        json.dump(
            self.config.__dict__,
            open(f"{self.save_model_dir}/config.json", "w"),
        )
