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
        self.loss_function  = CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)
    
    ## DATA ##
    def prepare_data(self):
        self.training_dataset   = MNIST(os.getcwd() + "/data/datasymlink")
        self.validation_dataset = MNIST(os.getcwd() + "/data/datasymlink",train=False)

    def train_dataloader(self):
        return self.training_dataset.get_dataloader()
    def val_dataloader(self):
        return self.validation_dataset.get_dataloader()

    ## TRAINING ##

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function.get_loss(y_hat,y)
        return {'loss': loss}

    def configure_optimizers(self):
        self.optimizer = Adam(self.model.parameters(), self.hparams.learning_rate)
        return self.optimizer
    
    ## VALIDATION ##

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function.get_loss(y_hat,y)
        return {'val_loss': loss}
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'val_loss': avg_loss}

    @staticmethod
    def add_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = MNISTClassifier.add_args(parser)
        parser = Adam.add_args(parser)
        return parser



