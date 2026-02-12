from pizza_clock.training import ModularAdditionModelTrainer
from pizza_clock.config import Config, get_device
from pizza_clock.llc_estimation_utils import (
    estimate_and_plot_llc_for_all_models,
    LLCEstimationConfig,
)
from torch.multiprocessing import Pool
from datetime import date


def train_model(config: Config, epochs: int = 20000) -> None:
    trainer = ModularAdditionModelTrainer(config)
    trainer.train(epochs=epochs, log_every_n_steps=10)


if __name__ == "__main__":
    attention_rates = [0.0, 1.0]
    seeds = range(5)

    training_args = []
    llc_args = []

    for attention_rate in attention_rates:
        for seed in seeds:
            name = f"{date.today().isoformat()}/attn{attention_rate}_seed{seed}"
            training_args.append(
                (
                    Config(
                        attention_rate=attention_rate,
                        wandb_project_name="modular-addition-6",
                        model_name=name,
                        use_wandb=True,
                        device=get_device(),
                        seed=seed,
                    ),
                ),
            )

            llc_args.append(
                (
                    LLCEstimationConfig(
                        dir_path=f"saved_models/{name}",
                        log_every_n_steps=10,
                        lr=1e-4,
                        localization=100,
                        num_draws=2000,
                    ),
                ),
            )

    with Pool(5) as p:
        p.starmap(train_model, training_args)
        # p.starmap(estimate_and_plot_llc_for_final_model, llc_args)
        p.starmap(estimate_and_plot_llc_for_all_models, llc_args)
