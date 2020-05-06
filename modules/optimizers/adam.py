import torch
from argparse import ArgumentParser

class Adam(torch.optim.Adam):
    def __init__(self, params, hparams):
        self.hparams = hparams
        super().__init__(params,lr=self.hparams.learning_rate)

    @staticmethod
    def add_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.001)
        return parser