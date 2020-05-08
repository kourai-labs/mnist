import torch
from datasets.mnist import MNIST
from systems.runtime import Runtime

from pytorch_lightning import Trainer
from argparse import ArgumentParser

def main(parser):

    ### Add to parser ###
    parser = Runtime.add_argparse_args(parser)
    parser = Trainer.add_argparse_args(parser)

    #temp_args, _ = parser.parse_known_args()
    
    hparams = parser.parse_args()

    # create runtime and trainer with args
    runtime = Runtime(hparams)
    trainer = Trainer.from_argparse_args(hparams)

    # start training
    trainer.fit(runtime)

    print("done.")

if __name__ == "__main__":
    main(ArgumentParser())
