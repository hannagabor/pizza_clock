import torch as t
from torch import nn, Tensor
from torch.utils.data import DataLoader
from pizza_clock.models import Model
from pizza_clock.dataset import get_train_val_data
from jaxtyping import Float, Int
from tqdm import tqdm


class ModularAdditionModelTrainer:
    def __init__(
        self,
        p,
        lr=1e-3,
        weight_decay=1e-2,
    ):
        self.train_loader, self.val_loader = get_train_val_data(p)
        self.model = Model(vocab_size=p, attention_rate=0.0)
        self.optimizer = t.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
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
