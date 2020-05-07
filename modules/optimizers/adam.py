import torch
from argparse import ArgumentParser

class Adam(torch.optim.Adam):

    LEARNING_RATE = 1e-3

    def __init__(self, params, learning_rate=LEARNING_RATE):
        super().__init__(params,lr=learning_rate)

    @staticmethod
    def add_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=Adam.LEARNING_RATE)
        return parser