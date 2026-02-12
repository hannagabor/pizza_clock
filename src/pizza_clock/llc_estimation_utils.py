import json
import os
import typing
from functools import partial
from pathlib import Path
from typing import Type
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch as t
from devinterp.optim.sgld import SGLD
from devinterp.slt.sampler import estimate_learning_coeff_with_summary
from devinterp.utils import plot_trace
from devinterp.vis_utils import EpsilonBetaAnalyzer
from torch.nn import functional as F


from pizza_clock.config import Config, get_device
from pizza_clock.dataset import get_train_val_data
from pizza_clock.metrics import compute_gradient_symmetry
from pizza_clock.metrics import compute_distance_irrelevance


def evaluate_last_position(criterion, model, data):
    x, y = data
    out = model(x)
    logits = out[:, -1, :]  # Get the last position's logits: [batch, vocab]
    return criterion(logits, y), {"output": logits}


evaluate_last_position_ce = partial(evaluate_last_position, F.cross_entropy)


def estimate_llc_given_model(
    model: t.nn.Module,
    loader: t.utils.data.DataLoader,
    evaluate: typing.Callable,
    epsilon: float,
    beta: float,
    sampling_method: Type[t.optim.Optimizer] = SGLD,
    localization: float = 5.0,
    num_chains: int = 2,
    num_draws: int = 500,
    num_burnin_steps: int = 0,
    num_steps_bw_draws: int = 1,
    device: t.device = get_device(),
    online: bool = True,
    verbose: bool = False,
):
    # Copied from devinterp grokking notebook https://github.com/timaeus-research/devinterp/blob/main/examples/grokking.ipynb
    sweep_stats = estimate_learning_coeff_with_summary(
        model,
        loader=loader,
        evaluate=evaluate,
        sampling_method=sampling_method,
        optimizer_kwargs=dict(lr=epsilon, localization=localization, nbeta=beta),
        num_chains=num_chains,  # How many independent chains to run
        num_draws=num_draws,  # How many samples to draw per chain
        num_burnin_steps=num_burnin_steps,  # How many samples to discard at the beginning of each chain
        num_steps_bw_draws=num_steps_bw_draws,  # How many steps to take between each sample
        device=device,
        online=online,
        verbose=verbose,
    )

    sweep_stats["llc/trace"] = np.array(sweep_stats["llc/trace"])
    return sweep_stats


def load_model_and_config(
    dir_path: str,
) -> tuple[t.nn.Module, Config, list[t.nn.Module]]:
    config_json = json.load(open(f"{dir_path}/config.json", "r"))
    config = Config(**config_json)
    final_model = t.load(f"{dir_path}/final_model.pt", map_location=get_device(), weights_only=False)
    all_models = [
        t.load(f"{dir_path}/model_{i}.pt", map_location=get_device(), weights_only=False)
        for i in range(len(list(Path(dir_path).glob("model_*.pt"))))
    ]
    return final_model, config, all_models


def sweep(dir_path: str, min_epsilon=3e-7, max_epsilon=3e-4):
    final_model, config, all_models = load_model_and_config(dir_path)
    train_loader, _ = get_train_val_data(config, squeeze_targets=True)

    evaluate_last_position_ce = partial(evaluate_last_position, F.cross_entropy)
    analyzer = EpsilonBetaAnalyzer()
    analyzer.configure_sweep(
        llc_estimator=estimate_llc_given_model,
        llc_estimator_kwargs=dict(
            model=final_model,
            evaluate=evaluate_last_position_ce,
            device=get_device(),
            loader=train_loader,
        ),
        min_epsilon=min_epsilon,
        max_epsilon=max_epsilon,
        epsilon_samples=5,
        min_beta=None,
        max_beta=None,
        beta_samples=5,
        dataloader=train_loader,
    )
    analyzer.sweep()
    analyzer.plot(div_out_beta=True)
    return analyzer


def estimate_and_plot_llc_for_final_model(
    dir_path, lr=1e-5, nbeta=10.0, localization=10.0, num_chains=3, num_draws=1500
):
    final_model, config, all_models = load_model_and_config(dir_path)
    train_loader, _ = get_train_val_data(config, squeeze_targets=True)

    learning_coeff_stats = estimate_learning_coeff_with_summary(
        final_model,
        loader=train_loader,
        evaluate=evaluate_last_position_ce,
        sampling_method=SGLD,
        optimizer_kwargs=dict(lr=lr, nbeta=nbeta, localization=localization),
        num_chains=num_chains,
        num_draws=num_draws,
        device=get_device(),
        online=True,
    )
    trace = learning_coeff_stats["loss/trace"]
    avg_llc = sum(learning_coeff_stats["llc/means"]) / len(learning_coeff_stats["llc/means"])
    print(dir_path)
    plot_trace(
        trace,
        "Loss",
        x_axis="Step",
        title=f"Loss Trace, avg LLC = {avg_llc:.2f}",
        plot_mean=False,
        plot_std=False,
        fig_size=(12, 9),
        true_lc=None,
    )
    return avg_llc


def estimate_and_plot_llc_for_all_models(
    dir_path: str,
    lr: float = 1e-5,
    nbeta: float = 10.0,
    localization: float = 10.0,
    num_chains: int = 3,
    num_draws: int = 1500,
):
    _, config, all_models = load_model_and_config(dir_path)
    train_loader, _ = get_train_val_data(config, squeeze_targets=True)
    df = pd.read_csv(os.path.join(dir_path, "loss_data.csv"))

    llcs = [
        estimate_learning_coeff_with_summary(
            model,
            loader=train_loader,
            evaluate=evaluate_last_position_ce,
            sampling_method=SGLD,
            optimizer_kwargs=dict(lr=lr, nbeta=nbeta, localization=localization),
            num_chains=num_chains,
            num_draws=num_draws,
            device=get_device(),
            online=True,
        )
        for model in all_models
    ]

    gradient_similarities = [compute_gradient_symmetry(model) for model in all_models]
    distance_irrelevance = [compute_distance_irrelevance(model) for model in all_models]

    fig, ax1 = plt.subplots()
    plt.title(
        f"Lambdahat vs loss for modular addition, p={config.p}, attr_rate={config.attention_rate}, train_frac={config.train_fraction}, nβ={nbeta:.1f}, ε={lr}, num_draws={num_draws}, num_chains={num_chains}"
    )
    ax2 = ax1.twinx()
    print(
        len(df["train_loss"]),
        len(df["val_loss"]),
        len([sum(llc["llc/means"]) / len(llc["llc/means"]) for llc in llcs]),
    )
    ax1.plot(df["val_loss"], label="test loss")
    ax1.plot(df["train_loss"], label="train loss")
    ax1.plot(gradient_similarities, label="Gradient Similarity")
    ax1.plot(distance_irrelevance, label="Distance Irrelevance")

    avg_llc = [sum(llc["llc/means"]) / len(llc["llc/means"]) for llc in llcs]
    ax2.plot(avg_llc, color="g", label="Lambdahat")

    ax1.set_xlabel(f"Checkpoint no. (Every {config.log_every_n_steps} epochs)")
    ax1.set_ylabel("Loss / Gradient Similarity")
    ax2.set_ylabel("Lambdahat", color="g")
    ax2.tick_params(axis="y", labelcolor="g")
    fig.legend(loc="center right")
    plt.savefig(Path(dir_path) / "plot.png")
