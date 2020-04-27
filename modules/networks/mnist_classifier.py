import torch

class MNISTClassifier(torch.nn.Module):

    def __init__(self):
        super(MNISTClassifier, self).__init__()

        #TODO remove hardcode for input size
        self.fc1 = torch.nn.Linear(28*28, 128)
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

if __name__ == "__main__":
    model = MNISTClassifier()
    x = torch.randn((1,1,28,28))
    y = model(x)
    print(y)
