from pizza_clock.training import ModularAdditionModelTrainer
from pizza_clock.config import Config, get_device
import torch as t
from multiprocessing import Pool


def train_model(
    i: int, attention_rate: float, p: int, train_fraction: float, weight_decay: float
):
    name = f"p{p}_attn{attention_rate}_td{train_fraction}_wd{weight_decay}_seed{i}"
    config = Config(
        p=p,
        attention_rate=attention_rate,
        residual_dim=128,
        device=get_device(),
        use_wandb=True,
        seed=i,
        wandb_name=name,
        wandb_project_name="modular-addition-2",
        weight_decay=weight_decay,
        train_fraction=train_fraction,
    )

    trainer = ModularAdditionModelTrainer(config)
    model = trainer.train(epochs=20000, log_every_n_steps=10)

    t.save(
        model,
        f"saved_models/{name}.pt",
    )


if __name__ == "__main__":
    # args = []
    # p_tf_wds = [(113, 0.5, 1.5), (59, 0.5, 2.0), (59, 0.5, 1.0)]
    # for i in range(5):
    #     for attention_rate in [0.0, 1.0]:
    #         # for p in [113, 59]:
    #         #     for train_fraction in [0.3, 0.5, 0.8]:
    #         #         for weight_decay in [1.0, 1.5, 2.0]:
    #         #             args.append(
    #         #                 (i, attention_rate, p, train_fraction, weight_decay)
    #         #             )
    #         for p, train_fraction, weight_decay in p_tf_wds:
    #             args.append((i, attention_rate, p, train_fraction, weight_decay))
    p = 59
    train_fraction = 0.8
    weight_decay = 2.0
    args = []
    for i in range(5):
        for attention_rate in [0.0, 1.0]:
            args.append((i, attention_rate, p, train_fraction, weight_decay))
    with Pool(5) as p:
        p.starmap(train_model, args)
    # train_model(0, 0.0, p, train_fraction, weight_decay)


# new_model = t.load("saved_models/test_model.pt", weights_only=False)
# for name, param in new_model.named_parameters():
#     print(name)


# TODO: read in the wandb runs and do plots for all parameter combinations (merging seeds).
#     Single out one parameter setup and do model checkpoint runs (deep copying the model at every checkpoint. Then run the llc analyzer to find good llc estimator params. Then run the good estimator using all the checkpoints and plot.
#     )
