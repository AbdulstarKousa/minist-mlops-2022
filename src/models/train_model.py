# from data import mnist
from model import MyAwesomeModel

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
from dotenv import find_dotenv, load_dotenv

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset


class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """

    def __init__(self):
        getattr(self, "train")()

    def train(self):
        """
        Function to train. Takes a data and create a trained model.
                Parameters:
                    Void
                Returns:
                    Trained model and training-curve.png
        """
        print("Training day and night")
        parser = argparse.ArgumentParser(description="Training arguments")
        parser.add_argument("--lr", default=0.003)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)

        # TODO-Done: Implement training loop here
        model = MyAwesomeModel()
        train_imgs = torch.load(os.path.join("data", "processed", "train_images.pt"))
        train_labs = torch.load(os.path.join("data", "processed", "train_labels.pt"))

        train_ds = TensorDataset(train_imgs, train_labs)
        train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)

        criterion = nn.NLLLoss(reduction="sum")
        optimizer = optim.Adam(model.parameters(), lr=0.003)

        # loop over epochs
        epochs = 10
        train_losses = []
        for e in range(epochs):
            tot_train_loss = 0
            for images, labels in train_dl:
                # Flatten MNIST images into a 784 long vector
                images = images.view(images.shape[0], -1)

                # TODO: Training pass
                optimizer.zero_grad()

                log_ps = model(images)
                loss = criterion(log_ps, labels)
                tot_train_loss += loss.item()

                loss.backward()
                optimizer.step()
            else:
                train_loss = tot_train_loss / len(train_dl.dataset)
                train_losses.append(train_loss)
                print(
                    "Epoch: {}/{}.. ".format(e + 1, epochs),
                    "Training Loss: {:.3f} .. ".format(train_loss),
                )

                # early stopping
                if len(train_losses) >= 2:
                    if train_losses[-1] < train_losses[-2]:
                        torch.save(
                            model, os.path.join("models", f"my_trained_model.pt")
                        )

        # Plot
        plt.plot(np.arange(epochs), np.array(train_losses))
        plt.title("Training Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Training loss")
        plt.savefig(os.path.join("reports", "figures", "training-curve.png"))


if __name__ == "__main__":
    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    TrainOREvaluate()

# to run
# python src/models/train_model.py
