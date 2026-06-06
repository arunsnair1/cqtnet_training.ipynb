import torch.nn as nn
import torch.nn.functional as F

class CQTNet(nn.Module):
    """
    CNN Architecture designed for Cover Song Identification using CQT spectrograms.
    """
    def __init__(self, num_classes):
        super(CQTNet, self).__init__()
        
        # Block 1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(12, 3), padding=(5, 1))
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d((2, 1))
        
        # Block 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(13, 3), padding=(6, 1))
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d((2, 1))
        
        # Block 3
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(13, 3), padding=(6, 1))
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d((2, 1))

        # Temporal Pooling
        self.pool_time = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification Head
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool_time(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
