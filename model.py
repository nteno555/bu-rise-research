import torch
import torch.nn as nn

class PlantDiseaseCNN(nn.Module):
    """
    A 9-layer Convolutional Neural Network for Plant Leaf Disease Classification.
    Based on the architecture principles from S. Geetharamani & A. Pandian (2019).
    
    Structure:
    - 7 Convolutional Layers (with Batch Normalization and ReLU)
    - 4 Max-Pooling Layers
    - 2 Fully Connected Layers (with Batch Normalization, ReLU, and Dropout)
    """
    def __init__(self, num_classes=39, input_size=128):
        super(PlantDiseaseCNN, self).__init__()
        self.input_size = input_size
        
        # Block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 128 -> 64
        
        # Block 2
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 64 -> 32
        
        # Block 3
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # 32 -> 16
        
        # Block 4
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # 16 -> 8
        
        self.relu = nn.ReLU()
        
        # Classifier
        # Calculate feature map size entering FC layer
        # For input size 128, downsampled by 2^4 = 16, resulting in 8x8 feature maps
        fc_input_dim = 256 * (input_size // 16) * (input_size // 16)
        
        self.fc1 = nn.Linear(fc_input_dim, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Block 1
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        
        # Block 2
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        
        # Block 3
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        
        # Block 4
        x = self.relu(self.bn7(self.conv7(x)))
        x = self.pool4(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Classifier
        x = self.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

if __name__ == '__main__':
    # Simple shape validation check
    model = PlantDiseaseCNN(num_classes=39, input_size=128)
    test_tensor = torch.randn(2, 3, 128, 128)
    out = model(test_tensor)
    print("Output shape:", out.shape) # Expected [2, 39]
