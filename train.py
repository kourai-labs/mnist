import torch
from datasets.mnist import MNIST


def main():

    # create and load MNIST training dataset
    train_dataset = MNIST(root="data/datasymlink",train=True)
    train_dataloader = train_dataset.get_dataloader()

    # create and load MNIST validation set
    val_dataset = MNIST(root="data/datasymlink",train=False)
    val_dataloader = train_dataset.get_dataloader()

    print("done.")

if __name__ == "__main__":
    main()
