# define the model architecture
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DenoisingCNN(nn.Module):
    def __init__(self,  num_layers, channels=1):
        super(DenoisingCNN, self).__init__()
        # define the layers in this function. taken from https://arxiv.org/pdf/1608.03981
        self.num_layers = num_layers
        self.channels = channels
        self.features = 64
        
        # first layer
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=self.features, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        # middle layers
        self.middle_layers = []
        for i in range(self.num_layers - 2):
            middle_layer = nn.Sequential(
                nn.Conv2d(in_channels=self.features, out_channels=self.features, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(self.features), # batch normalization for training stability
                nn.ReLU()
            )

            # add to the network
            self.middle_layers.append(middle_layer)

        # last layers
        self.conv_final = nn.Sequential(
            nn.Conv2d(in_channels=self.features, out_channels=channels, kernel_size=3, stride=1, padding=1)
        )

        # combine all layers to create the network
        self.layers = nn.Sequential(
            self.conv_1,
            nn.ModuleList(self.middle_layers),
            self.conv_final
        )

    def forward(self, x):
        # pass the input x through all the layers
        return self.layers(x)