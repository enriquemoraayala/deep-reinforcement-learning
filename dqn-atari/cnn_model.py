import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, action_size, seed, fc1_units=512):
        """Initialize parameters and build model.
        Params
        ======
            image_size (int): Dimension of each image
            num_channels (int): Image chanels
            action_size (int): number of final actions
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        # 1 input image channel (RGB), 32 output channels
        # 8x8 square convolution kernel
        # stride = 4
        # output size = (W-F)/S +1 = (84-8)/4 +1 = 20
        # the output Tensor for one image, will have the dimensions:
        # (32, 20, 20)

        self.conv1 = nn.Conv2d(1, 32, 8, stride=4)

        # second conv layer: 32 inputs, 64 outputs, 4x4 conv, stride 2
        # output size = (W-F)/S +1 = (20-4)/2 +1 = 9
        # the output tensor will have dimensions: (64, 9, 9)

        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)

        # third conv layer: 64 inputs, 64 outputs, 3x3 conv, stride 1
        # output size = (W-F)/S +1 = (9-3)/1 +1 = 7
        # the output tensor will have dimensions: (64, 7, 7)

        self.conv3 = nn.Conv2d(64, 64, 3)

        # 64 outputs * the 7*7 filtered/pooled map size
        self.fc1 = nn.Linear(64*7*7, fc1_units)

        # dropout with p=0.4
        self.fc1_drop = nn.Dropout(p=0.4)

        # finally, create action_size output channels
        # (for the action_size classes)
        self.fc2 = nn.Linear(fc1_units, action_size)

    # define the feedforward behavior
    def forward(self, x):
        # two conv/relu + pool layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # prep for linear layer
        # this line of code is the equivalent of Flatten in Keras
        # x = x.view(x.size(0), -1)
        x = torch.flatten(x, 1)

        # two linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = self.fc2(x)

        # final output
        return x
