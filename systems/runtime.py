import os
from argparse import ArgumentParser

import torch
import pytorch_lightning as pl

from datasets.mnist import MNIST
from modules.networks.mnist_classifier import MNISTClassifier
from modules.losses.cross_entropy import CrossEntropyLoss
from modules.optimizers.adam import Adam

from pytorch_lightning import Trainer
from torch.utils.data import random_split, DataLoader


class Runtime(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.model = MNISTClassifier(input_size=self.hparams.input_size)
        self.cross_entropy_loss = CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    ##########    
    ## DATA ##
    ##########

    def prepare_data(self):
        mnist_dataset = MNIST(os.getcwd() + "/data/datasymlink")
        train_samples_n = len(mnist_dataset) * self.hparams.train_val_split
        val_samples_n   = len(mnist_dataset) * ( 1 - self.hparams.train_val_split )
        self.training_dataset, self.validation_dataset = random_split(mnist_dataset,lengths=[len(self.training_dataset)*.9,len(self.training_dataset)*0.1])
        self.test_dataset= MNIST(os.getcwd() + "/data/datasymlink",train=False)

    def train_dataloader(self):
        return DataLoader(self.training_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)
    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)

    ##############
    ## TRAINING ##
    ##############

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.cross_entropy_loss.get_loss(y_hat,y)
        return {'loss': loss}

    def configure_optimizers(self):
        self.optimizer = Adam(self.model.parameters(), self.hparams.learning_rate)
        return self.optimizer
    
    ################
    ## VALIDATION ##
    ################

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.cross_entropy_loss.get_loss(y_hat,y)
        return {'val_loss': loss}
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'val_loss': avg_loss}

    ############# 
    ## TESTING ##
    #############

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return {'val_loss': loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'val_loss': avg_loss }

    ################# 
    ## HYPERPARAMS ##
    ################# 

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        ### add model specific args ###
        parser = MNISTClassifier.add_args(parser)

        ### add optimizer specific args ###
        parser = Adam.add_args(parser)

        ### add scheduler specific args ###
        pass

        ### add loss specific args ###
        pass

        ### add training hyperparameters
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--num_workers", type=int, default=16)
        parser.add_argument("--train_val_split", type=float, default=0.9)

        return parser
