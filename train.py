import torch
from datasets.mnist import MNIST
from systems.runtime import Runtime

from pytorch_lightning import Trainer
from argparse import ArgumentParser

def main():

    parser = ArgumentParser()
    parser = Runtime.add_args(parser)
    parser = Trainer.add_argparse_args(parser)

    hparams = parser.parse_args()

    runtime = Runtime(hparams)

    trainer = Trainer.from_argparse_args(hparams, gpu=2)

    trainer.fit(runtime)

    print("done.")

if __name__ == "__main__":
    main()
