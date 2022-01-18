from data import mnist
from model import MyAwesomeModel

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse

import torch
from torch import nn, optim



class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
    
    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.1)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        # print(args)
        
        # TODO-Done: Implement training loop here
        model = MyAwesomeModel()
        train_dl, _ = mnist()

        criterion = nn.NLLLoss(reduction='sum')
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
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                    "Training Loss: {:.3f} .. ".format(train_loss))

                # early stopping
                if len(train_losses)>=2:
                    if train_losses[-1]< train_losses[-2]:
                        pths = [file for file in os.listdir() if file.endswith(".pth")]
                        for file in pths:
                            os.remove(file)
                        torch.save(model, f'checkpoint-{e+1}-{np.round(train_loss,2)}.pth')




        # Plot
        plt.plot(np.arange(epochs), np.array(train_losses))
        plt.xlabel('Epochs')
        plt.ylabel('Training loss')
        plt.show()


    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('load_model_from', default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO-Done: Implement evaluation logic here
        model = torch.load(args.load_model_from)
        _, test_dl = mnist()

        test_correct = 0  # Number of correct predictions on the test set
        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            for images, labels in test_dl:
                log_ps = model(images)
                ps = torch.exp(log_ps)
                _, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                test_correct += equals.sum().item()
        print("Accuracy: {:.3f}".format(test_correct / len(test_dl.dataset)))


if __name__ == '__main__':
    TrainOREvaluate()