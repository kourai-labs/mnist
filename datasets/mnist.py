import torch
from torchvision import datasets, transforms


class MNIST(datasets.MNIST):
    def __init__(self,root, train=True, transform=None, download=True):
        super(MNIST, self).__init__(root=root,train=train,transform=self._get_transform(),
            target_transform=None,download=download)
    
    def _get_transform(self):
        tforms = []
        tforms.append(transforms.ToTensor())
        tforms.append(transforms.Normalize((0.1307,),(0.3081,)))

        return transforms.Compose(tforms)


    #TODO default batch size, TODO default shuffle
    # move these defaults to Hydra
    def get_dataloader(self, batch_size=64, shuffle=True):
        return torch.utils.data.DataLoader(self, batch_size=batch_size, shuffle=shuffle)
