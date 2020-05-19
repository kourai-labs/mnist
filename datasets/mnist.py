import torch
from torchvision import datasets, transforms


class MNIST(datasets.MNIST):

    def __init__(self, root, image_size=28, transform=None, train=True, download=True):

        self.image_size = image_size 
        self.train = train
        if transform is None:
            transform = self._get_transform()

        super(MNIST, self).__init__(root=root,train=train,transform=transform,
            target_transform=None,download=download)
    
    def _get_transform(self):
        tforms = []
        tforms.append(transforms.ToTensor())
        tforms.append(transforms.Normalize((0.1307,),(0.3081,)))
        return transforms.Compose(tforms)
