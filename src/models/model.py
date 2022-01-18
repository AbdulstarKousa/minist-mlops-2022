from torch import nn
import torch.nn.functional as F


class MyAwesomeModel(nn.Module):
    def __init__(self):
        """ initilize network """
        super().__init__()
        # Defining the layers, 512, 256, 128, 64, 10 units each
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        # Output layer, 10 units - one for each digit
        self.fc5 = nn.Linear(64, 10)

    def forward(self, x):
        """
        Forward pass through the network, returns the output logits

                Parameters:
                    Batch of tensor MINST images  

                Returns:
                    Tensor of logits
        """

        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.log_softmax(self.fc5(x), dim=1)

        return x
