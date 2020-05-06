from torch.nn import functional as F
from argparse import ArgumentParser

class CrossEntropyLoss():

    @staticmethod
    def add_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # add args here
        return parser

    def get_loss(self,y_hat,y):
        return F.cross_entropy(y_hat,y)

    
