"""
model.py - Neural Network Architectures for GTSRB
==================================================
Contains model definitions for German Traffic Sign Recognition.
(This file was refactored from the original, too lengthy train.py)
Models:
    - LeNet: LeNet-based CNN for 48x48 input images
"""

import torch
import torch.nn as nn


class LeNet(nn.Module):
    """
    LeNet-based Convolutional Neural Network for Traffic Sign Classification.
    
    Architecture:
        Input: 48x48x3 RGB images
        
        Conv Block 1:
            - Conv2d: 3 -> 16 channels, 3x3 kernel, padding=1
            - ReLU activation
            - MaxPool2d: 2x2, stride=2
            Output: 24x24x16
        
        Conv Block 2:
            - Conv2d: 16 -> 32 channels, 3x3 kernel, padding=1
            - ReLU activation
            - MaxPool2d: 2x2, stride=2
            Output: 12x12x32
        
        Conv Block 3:
            - Conv2d: 32 -> 64 channels, 3x3 kernel, padding=1
            - ReLU activation
            - MaxPool2d: 2x2, stride=2
            Output: 6x6x64
        
        Fully Connected:
            - Flatten: 6x6x64 = 2304 features
            - FC1: 2304 -> 256, ReLU, Dropout(0.5)
            - FC2: 256 -> num_classes (43)
    
    Args:
        num_classes (int): Number of output classes (default: 43 for GTSRB)
        dropout_rate (float): Dropout probability (default: 0.5)
    
    Input Shape:
        (batch_size, 3, 48, 48)
    
    Output Shape:
        (batch_size, num_classes)
    """
    
    def __init__(self, num_classes=43, dropout_rate=0.5):
        super(LeNet, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, 
                               kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, 
                               kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, 
                               kernel_size=3, padding=1)
        
        # Pooling and activation
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        
        # Fully connected layers
        # After 3 pooling: 48 -> 24 -> 12 -> 6
        # Feature map size: 64 * 6 * 6 = 2304
        self.fc1 = nn.Linear(64 * 6 * 6, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        # Regularization
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # Store configuration
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 48, 48)
        
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        # Convolutional block 1: (B, 3, 48, 48) -> (B, 16, 24, 24)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Convolutional block 2: (B, 16, 24, 24) -> (B, 32, 12, 12)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Convolutional block 3: (B, 32, 12, 12) -> (B, 64, 6, 6)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Flatten: (B, 64, 6, 6) -> (B, 2304)
        x = x.view(x.size(0), -1)
        
        # Fully connected block 1: (B, 2304) -> (B, 256)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Fully connected block 2: (B, 256) -> (B, num_classes)
        x = self.fc2(x)
        
        return x
    
    def get_num_parameters(self):
        """
        Calculate total number of trainable parameters.
        
        Returns:
            int: Total number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self):
        """
        Get model configuration information.
        
        Returns:
            dict: Model configuration details
        """
        return {
            'name': 'LeNet',
            'input_size': (3, 48, 48),
            'num_classes': self.num_classes,
            'dropout_rate': self.dropout_rate,
            'num_parameters': self.get_num_parameters(),
            'architecture': {
                'conv1': '3->16, 3x3',
                'conv2': '16->32, 3x3',
                'conv3': '32->64, 3x3',
                'fc1': '2304->256',
                'fc2': f'256->{self.num_classes}'
            }
        }


def create_model(model_name='lenet', num_classes=43, **kwargs):
    """
    Factory function to create models by name.
    
    Args:
        model_name (str): Name of the model ('lenet')
        num_classes (int): Number of output classes
        **kwargs: Additional model-specific arguments
    
    Returns:
        nn.Module: Initialized model
    
    Example:
        model = create_model('lenet', num_classes=43, dropout_rate=0.5)
    """
    model_name = model_name.lower()
    
    if model_name == 'lenet':
        return LeNet(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}. Available: 'lenet'")


if __name__ == '__main__':
    # Test model creation
    print("Testing LeNet model...")
    
    # Create model
    model = LeNet(num_classes=43)
    
    # Print model info
    info = model.get_model_info()
    print(f"\nModel: {info['name']}")
    print(f"Input size: {info['input_size']}")
    print(f"Number of classes: {info['num_classes']}")
    print(f"Dropout rate: {info['dropout_rate']}")
    print(f"Total parameters: {info['num_parameters']:,}")
    print(f"\nArchitecture:")
    for layer, config in info['architecture'].items():
        print(f"  {layer}: {config}")
    
    # Test forward pass
    print(f"\nTesting forward pass...")
    dummy_input = torch.randn(4, 3, 48, 48)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"\nModel test successful!")