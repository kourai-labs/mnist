import os

import torch
import pytorch_lightning as pl

from datasets.mnist import MNIST

from modules.networks.mnist_classifier import MNISTClassifier
from modules.losses.cross_entropy import CrossEntropyLoss
from modules.optimizers.adam import Adam
from pytorch_lightning import Trainer

from argparse import ArgumentParser


class Runtime(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.model = MNISTClassifier(input_size=self.hparams.input_size)
        self.training_dataset   = MNIST(os.getcwd() + "/data/datasymlink")
        self.validation_dataset = MNIST(os.getcwd() + "/data/datasymlink",train=False)
        self.loss_function  = CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), self.hparams.learning_rate)


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function.get_loss(y_hat,y)
        return {'loss': loss}
    
    def configure_optimizers(self):
        return self.optimizer
    
    def train_dataloader(self):
        return self.training_dataset.get_dataloader()

    def get_trainer(self):
        return self.trainer

    @staticmethod
    def add_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = MNISTClassifier.add_args(parser)
        parser = Adam.add_args(parser)
        return parser



