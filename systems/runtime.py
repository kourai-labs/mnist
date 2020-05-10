import os
from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
import hydra

from datasets.mnist import MNIST
from modules.networks.mnist_classifier import MNISTClassifier

from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import random_split, DataLoader
from pytorch_lightning import Trainer


class Runtime(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.runtime

        self.model = MNISTClassifier(**cfg.mnist_classifier)

    def forward(self, x):
        return self.model(x)

    ##########    
    ## DATA ##
    ##########

    def prepare_data(self):
        project_home = hydra.utils.get_original_cwd()
        mnist_dataset = MNIST(root=project_home + "/data/datasymlink")
        train_samples_n = int(len(mnist_dataset) * self.cfg.train_val_split)
        val_samples_n   = len(mnist_dataset) - train_samples_n
        self.training_dataset, self.validation_dataset = random_split(mnist_dataset,lengths=[train_samples_n, val_samples_n])
        self.test_dataset= MNIST(root=project_home + "/data/datasymlink",train=False)

    def train_dataloader(self):
        return DataLoader(self.training_dataset, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers)
    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers)
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers)

    ##############
    ## TRAINING ##
    ##############

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat,y)
        return {'loss': loss}

    def configure_optimizers(self):
        self.optimizer = Adam(self.model.parameters())
        return self.optimizer
    
    ################
    ## VALIDATION ##
    ################

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat,y)
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
