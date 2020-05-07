import torch

from argparse import ArgumentParser

class MNISTClassifier(torch.nn.Module):

    INPUT_SIZE = 28

    def __init__(self, input_size=INPUT_SIZE):
        super(MNISTClassifier, self).__init__()

        self.input_size = input_size

        #TODO remove hardcode for input size
        self.fc1 = torch.nn.Linear(self.input_size*self.input_size, 128)
        self.fc2 = torch.nn.Linear(128, 10)

        self.log_sm = torch.nn.LogSoftmax(dim=0)


    def forward(self, x):
        b, c, w, h = x.size()

        x = x.view(b, -1)
        # Layer 1
        x = self.fc1(x)
        x = torch.relu(x)

        # Layer 2
        x = self.fc2(x)
        x = torch.relu(x)

        # Output 
        return self.log_sm(x)
    
    @staticmethod
    def add_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--input_size', type=int, default=MNISTClassifier.INPUT_SIZE)
        return parser

if __name__ == "__main__":
    model = MNISTClassifier()
    x = torch.randn((1,1,28,28))
    y = model(x)
    print(y)
