import torch
from datasets.mnist import MNIST
from systems.runtime import Runtime

from pytorch_lightning import Trainer
import wandb

import hydra
from omegaconf import DictConfig

@hydra.main(config_path="configs/config.yaml")
def main(cfg: DictConfig) -> None:
    print(cfg.pretty())

    # initialize wandb for logging and tracking
    wandb.init(project=cfg.wandb.project_name, resume=cfg.wandb.resume)

    # create runtime and trainer with args
    runtime = Runtime(cfg)
    trainer = Trainer(**cfg.trainer)

    # start training
    trainer.fit(runtime)

    print("done.")

if __name__ == "__main__":
    main()
