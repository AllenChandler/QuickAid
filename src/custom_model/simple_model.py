import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 32 * 32, 64)  # Adjust input size (see below)
        self.fc2 = nn.Linear(64, num_classes)
        
        # Other layers
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.6)

    def forward(self, x):
        # Pass through convolutional layers
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)  # Reduce spatial dimensions
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)  # Reduce spatial dimensions further

        # Flatten for fully connected layers
        print(f"Shape before flattening: {x.shape}")
        x = x.view(x.size(0), -1)  # Flatten to [batch_size, features]
        print(f"Shape after flattening: {x.shape}")
        
        # Pass through fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
