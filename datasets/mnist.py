import torch
from torchvision import datasets, transforms


class MNIST(datasets.MNIST):

    DEFAULT_IMAGE_SIZE = 28

    def __init__(self,root, image_size=DEFAULT_IMAGE_SIZE, train=True, transform=None, download=True):

        self.image_size = image_size 
        self.train = train
        super(MNIST, self).__init__(root=root,train=train,transform=self._get_transform(),
            target_transform=None,download=download)
    
    def _get_transform(self):
        tforms = []
        tforms.append(transforms.ToTensor())
        tforms.append(transforms.Normalize((0.1307,),(0.3081,)))
        return transforms.Compose(tforms)
