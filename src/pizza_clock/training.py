import torch as t
from torch import nn, Tensor
from pizza_clock.models import Model
from pizza_clock.dataset import get_train_val_data
from jaxtyping import Float
from tqdm import tqdm
import wandb
from pizza_clock.config import Config


def get_device() -> str:
    if t.backends.mps.is_available():
        return "mps"
    elif t.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


class ModularAdditionModelTrainer:
    def __init__(
        self,
        config: Config,
    ):
        self.train_loader, self.val_loader = get_train_val_data(config.p)
        self.model = Model(vocab_size=config.p, attention_rate=config.attention_rate)
        self.optimizer = t.optim.AdamW(
            self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def training_step(
        self, x: Float[Tensor, "batch seq_len"], y: Float[Tensor, "batch 1"]
    ) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(x)
        logits = out[:, -1, :]
        loss = self.loss_fn(logits, y.squeeze())
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self, epochs: int = 20000, log_every_n_steps: int = 100):
        wandb.init(project=self.args.wandb_project, name=self.args.wandb_name)
        wandb.watch(self.model)
        pbar = tqdm(range(epochs))
        for epoch in pbar:
            for x, y in self.train_loader:
                train_loss = self.training_step(x, y)
                if epoch % log_every_n_steps == 0:
                    val_loss, val_accuracy = self.evaluate()
                pbar.set_postfix(
                    epoch=f"{epoch + 1}/{epochs}",
                    loss=f"{train_loss:.3f}, val_loss={val_loss:.3f}, val_acc={val_accuracy:.3f}",
                )

    def evaluate(self) -> tuple[float, float]:
        self.model.eval()
        with t.no_grad():
            for x, y in self.val_loader:
                out = self.model(x)
                logits = out[:, -1, :]
                loss = self.loss_fn(logits, y.squeeze())
                accuracy = (logits.argmax(dim=-1) == y.squeeze()).float().mean().item()
        return loss, accuracy
