"""
model.py - Neural Network Architectures for GTSRB
==================================================
Contains model definitions for German Traffic Sign Recognition.
(This file was refactored from the original, too lengthy train.py)
Models:
    - LeNet: LeNet-based CNN for 48x48 input images
    - ShallowCNN: A minimal CNN for baseline comparison.
    - MiniVGG: A scaled-down VGG-style architecture.
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
            - FC2: 256 -> num_classes
    
    Args:
        num_classes (int): Number of output classes (default: 43 for GTSRB)
        dropout_rate (float): Dropout probability (default: 0.5)
        num_conv_blocks (int, optional): Ignored, fixed at 3 for this class.
    
    Input Shape: (batch_size, 3, 48, 48)
    Output Shape: (batch_size, num_classes)
    """
    
    def __init__(self, num_classes=43, dropout_rate=0.5, num_conv_blocks=3):
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
        
        # Feature map size: 64 * 6 * 6 = 2304
        self.fc1 = nn.Linear(64 * 6 * 6, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        # Regularization
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # Store configuration
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.num_conv_blocks = 3 # Fixed for this specific LeNet implementation

    
    def forward(self, x):
        """
        Forward pass through the network.
        """
        # Conv Block 1: (B, 3, 48, 48) -> (B, 16, 24, 24)
        x = self.pool(self.relu(self.conv1(x)))
        
        # Conv Block 2: (B, 16, 24, 24) -> (B, 32, 12, 12)
        x = self.pool(self.relu(self.conv2(x)))
        
        # Conv Block 3: (B, 32, 12, 12) -> (B, 64, 6, 6)
        x = self.pool(self.relu(self.conv3(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def get_num_parameters(self):
        """Calculate total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self):
        """Get model configuration information."""
        return {
            'name': 'LeNet (3 Blocks)',
            'type': 'lenet',
            'input_size': (3, 48, 48),
            'num_classes': self.num_classes,
            'dropout_rate': self.dropout_rate,
            'num_conv_blocks': self.num_conv_blocks,
            'num_parameters': self.get_num_parameters(),
            'architecture': {
                'conv1': '3->16, 3x3',
                'conv2': '16->32, 3x3',
                'conv3': '32->64, 3x3',
                'fc1': '2304->256',
                'fc2': f'256->{self.num_classes}'
            }
        }


class ShallowCNN(nn.Module):
    """
    A minimal, shallow CNN architecture for baseline comparison.
    
    Architecture:
        Input: 48x48x3 RGB images
        
        Conv Block 1:
            - Conv2d: 3 -> 32 channels, 5x5 kernel, padding=2
            - ReLU activation
            - MaxPool2d: 2x2, stride=2
            Output: 24x24x32
            
        Conv Block 2:
            - Conv2d: 32 -> 64 channels, 3x3 kernel, padding=1
            - ReLU activation
            - MaxPool2d: 2x2, stride=2
            Output: 12x12x64
            
        Fully Connected:
            - Flatten: 12x12x64 = 9216 features
            - FC1: 9216 -> 128, ReLU, Dropout(p=dropout_rate)
            - FC2: 128 -> num_classes
    
    Args:
        num_classes (int): Number of output classes (default: 43 for GTSRB)
        dropout_rate (float): Dropout probability for the FC layer (default: 0.5)
    
    Input Shape: (batch_size, 3, 48, 48)
    Output Shape: (batch_size, num_classes)
    """
    def __init__(self, num_classes=43, dropout_rate=0.5, **kwargs):
        super(ShallowCNN, self).__init__()
        
        # Layer 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Layer 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Fully Connected Layers (12*12*64 = 9216 features)
        self.fc1 = nn.Sequential(
            nn.Linear(9216, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.fc2 = nn.Linear(128, num_classes)
        
        # Store info
        self.model_name = 'shallow_cnn'
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.num_conv_blocks = 2
        self._initialize_weights()

    def _initialize_weights(self):
        """Standard weight initialization for Conv and Linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1) # Flatten
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def get_num_parameters(self):
        """Calculate total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_info(self):
        """Returns a dictionary with model details."""
        return {
            'name': 'Shallow CNN Baseline',
            'type': self.model_name,
            'input_size': (3, 48, 48),
            'num_classes': self.num_classes,
            'dropout_rate': self.dropout_rate,
            'num_conv_blocks': self.num_conv_blocks,
            'num_parameters': self.get_num_parameters(),
            'architecture': {
                'conv1': '3->32, 5x5',
                'conv2': '32->64, 3x3',
                'fc1': '9216->128',
                'fc2': f'128->{self.num_classes}'
            }
        }


class MiniVGG(nn.Module):
    """
    Mini-VGG style architecture for Traffic Sign Classification.
    
    Architecture:
        Input: 48x48x3 RGB images
        
        VGG Block 1 (Conv-Conv-Pool-Drop):
            - Conv2d: 3 -> 32 channels, 3x3 kernel, padding=1
            - ReLU activation
            - Conv2d: 32 -> 32 channels, 3x3 kernel, padding=1
            - ReLU activation
            - MaxPool2d: 2x2, stride=2
            - Dropout: p=0.25
            Output: 24x24x32
            
        VGG Block 2 (Conv-Conv-Pool-Drop):
            - Conv2d: 32 -> 64 channels, 3x3 kernel, padding=1
            - ReLU activation
            - Conv2d: 64 -> 64 channels, 3x3 kernel, padding=1
            - ReLU activation
            - MaxPool2d: 2x2, stride=2
            - Dropout: p=0.25
            Output: 12x12x64
            
        Fully Connected (Classifier):
            - Flatten: 12x12x64 = 9216 features
            - FC1: 9216 -> 512, ReLU, Dropout(p=dropout_rate)
            - FC2: 512 -> num_classes
    
    Args:
        num_classes (int): Number of output classes (default: 43 for GTSRB)
        dropout_rate (float): Dropout probability for the FC layer (default: 0.5)
    
    Input Shape: (batch_size, 3, 48, 48)
    Output Shape: (batch_size, num_classes)
    """
    def __init__(self, num_classes=43, dropout_rate=0.5, **kwargs):
        super(MiniVGG, self).__init__()
        
        # Block 1: 48x48x3 -> 24x24x32
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )
        
        # Block 2: 24x24x32 -> 12x12x64
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )
        
        # Classifier (Input size: 12 * 12 * 64 = 9216 features)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(12 * 12 * 64, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
        self.model_name = 'minivgg'
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.num_conv_blocks = 2 
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.classifier(x)
        return x
        
    def get_num_parameters(self):
        """Calculate total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_info(self):
        """Returns a dictionary with model details."""
        return {
            'name': 'Mini-VGG (2 Blocks)',
            'type': self.model_name,
            'input_size': (3, 48, 48),
            'num_classes': self.num_classes,
            'dropout_rate': self.dropout_rate,
            'num_conv_blocks': self.num_conv_blocks,
            'num_parameters': self.get_num_parameters(),
            'architecture': {
                'block1_conv': '2x(3x3), 3->32 (24x24)',
                'block2_conv': '2x(3x3), 32->64 (12x12)',
                'fc1': '9216->512',
                'fc2': f'512->{self.num_classes}'
            }
        }

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------
# New Models: MobileNetV1 (width multiplier = 0.25) and ShuffleNetV2 (0.25×)
# ---------------------------------------------------------------------

class DepthwiseSeparableConv(nn.Module):
    """Helper module: Depthwise separable convolution."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                   stride=stride, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels,
                                   kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.depthwise(x)))
        x = self.relu(self.bn2(self.pointwise(x)))
        return x


class MobileNetV1_025(nn.Module):
    """
    MobileNetV1 (width multiplier α=0.25) for Traffic Sign Classification.

    Lightweight convolutional neural network using depthwise separable convolutions.

    Architecture:
        Input: 48x48x3 RGB images

        Initial Conv Layer:
            - Conv2d: 3 -> 8 channels (α=0.25), 3x3 kernel, stride=2, padding=1
            - BatchNorm2d
            - ReLU activation
            Output: 24x24x8

        Depthwise Separable Convolutions:
            - Block 1: Depthwise 8 -> 8, Pointwise 8 -> 16, stride=1
            - Block 2: Depthwise 16 -> 16, Pointwise 16 -> 32, stride=2
            - Block 3: Depthwise 32 -> 32, Pointwise 32 -> 32, stride=1
            - Block 4: Depthwise 32 -> 32, Pointwise 32 -> 64, stride=2
            - Block 5: Depthwise 64 -> 64, Pointwise 64 -> 64, stride=1
            - Block 6: Depthwise 64 -> 64, Pointwise 64 -> 128, stride=2
            - AdaptiveAvgPool2d: global pooling to 1x1

        Fully Connected (Classifier):
            - Dropout(p=0.3)
            - Linear: 128 -> num_classes (43 for GTSRB by default)

    Notes:
        - DepthwiseSeparableConv consists of:
            * Depthwise convolution: per-channel spatial convolution
            * BatchNorm + ReLU
            * Pointwise convolution: 1x1 convolution to combine channels
        - Width multiplier α=0.25 reduces channels, making the model lightweight
        - Dropout is applied only before the final linear layer
        - Suitable for small datasets or real-time inference with limited parameters

    Args:
        num_classes (int): Number of output classes (default=43)
        dropout_rate (float): Dropout probability before classifier (default=0.3)

    Input Shape: (batch_size, 3, 48, 48)
    Output Shape: (batch_size, num_classes)
    """
    def __init__(self, num_classes=43, dropout_rate=0.3):
        super().__init__()
        α = 0.25
        def c(ch): return max(8, int(ch * α))  # ensure nonzero
        
        self.features = nn.Sequential(
            nn.Conv2d(3, c(32), 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c(32)), nn.ReLU(inplace=True),

            DepthwiseSeparableConv(c(32), c(64), stride=1),
            DepthwiseSeparableConv(c(64), c(128), stride=2),
            DepthwiseSeparableConv(c(128), c(128), stride=1),
            DepthwiseSeparableConv(c(128), c(256), stride=2),
            DepthwiseSeparableConv(c(256), c(256), stride=1),
            DepthwiseSeparableConv(c(256), c(512), stride=2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(c(512), num_classes)
        )
        self.model_name = "mobilenetv1_025"

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_info(self):
        return {
            'name': 'MobileNetV1 (α=0.25)',
            'type': self.model_name,
            'input_size': (3, 48, 48),
            'num_classes': self.classifier[-1].out_features,
            'num_parameters': self.get_num_parameters(),
            'dropout_rate': self.classifier[0].p,
            'architecture': 'Depthwise separable CNN, α=0.25'
        }

class InvertedResidual(nn.Module):
    """Inverted Residual Block used in MobileNetV2."""
    def __init__(self, in_channels, out_channels, stride, expand_ratio, activation=nn.ReLU6):
        super().__init__()
        hidden_dim = int(in_channels * expand_ratio)
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            # 1x1 expansion
            layers.append(nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(activation(inplace=True))

        # Depthwise
        layers.append(nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(activation(inplace=True))

        # Pointwise linear projection
        layers.append(nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2_025(nn.Module):
    """
    MobileNetV2 (width multiplier α=0.25) for 48x48 images.

    Args:
        num_classes (int): number of output classes
        dropout_rate (float): dropout rate before classifier
        width_mult (float): width multiplier α
        activation (nn.Module): activation function class, e.g., nn.ReLU or nn.SiLU
    """
    def __init__(self, num_classes=43, dropout_rate=0.3, width_mult=0.25, activation=nn.ReLU6):
        super().__init__()
        self.width_mult = width_mult
        def c(ch): return max(8, int(ch * width_mult))

        # Initial conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, c(16), 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c(16)),
            activation(inplace=True)
        )

        # Bottleneck blocks: t, c, n, s
        inverted_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
        ]

        in_channels = c(16)
        layers = []
        for t, c_out, n, s in inverted_residual_setting:
            out_channels = c(c_out)
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(InvertedResidual(in_channels, out_channels, stride, expand_ratio=t, activation=activation))
                in_channels = out_channels
        self.features = nn.Sequential(*layers)

        # Final conv
        self.conv_last = nn.Sequential(
            nn.Conv2d(in_channels, c(512), 1, 1, 0, bias=False),
            nn.BatchNorm2d(c(512)),
            activation(inplace=True)
        )

        # Pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(c(512), num_classes)

        self.model_name = 'mobilenetv2_025'

    def forward(self, x):
        x = self.conv1(x)
        x = self.features(x)
        x = self.conv_last(x)
        x = self.global_pool(x).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_info(self):
        return {
            'name': 'MobileNetV2 (α={:.2f})'.format(self.width_mult),
            'type': self.model_name,
            'input_size': (3, 48, 48),
            'num_classes': self.fc.out_features,
            'num_parameters': self.get_num_parameters(),
            'dropout_rate': self.dropout.p,
            'architecture': 'Initial conv + inverted residual blocks + final conv + global pool + fc'
        }    
    

# ---------------------------------------------------------------------
# ShuffleNetV2 (0.25×)
# ---------------------------------------------------------------------

def channel_shuffle(x, groups):
    """ShuffleNet channel shuffle operation."""
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


class ShuffleUnit(nn.Module):
    """Basic ShuffleNetV2 unit."""
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.stride = stride
        mid_channels = out_channels // 2
        
        if stride != 1:
            self.branch1 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, stride, 1,
                          groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, mid_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.branch1 = nn.Identity()

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels if stride != 1 else mid_channels, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, stride, 1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(mid_channels, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        out = channel_shuffle(out, 2)
        return out


class ShuffleNetV2_025(nn.Module):
    """
    ShuffleNetV2 with 0.25x width multiplier.
    Extremely fast and compact (<0.4M params).
    """
    def __init__(self, num_classes=43, dropout_rate=0.3):
        super().__init__()
        width_mult = 0.25
        out_channels = {
            0.25: [24, 48, 96, 192],
        }[width_mult]

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, out_channels[0], 3, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(inplace=True)
        )
        self.stage2 = self._make_stage(out_channels[0], out_channels[1], 4)
        self.stage3 = self._make_stage(out_channels[1], out_channels[2], 8)
        self.stage4 = self._make_stage(out_channels[2], out_channels[3], 4)
        self.conv5 = nn.Sequential(
            nn.Conv2d(out_channels[3], 512, 1, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.globalpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        self.model_name = "shufflenetv2_025"

    def _make_stage(self, in_c, out_c, repeat):
        layers = [ShuffleUnit(in_c, out_c, stride=2)]
        for _ in range(repeat - 1):
            layers.append(ShuffleUnit(out_c, out_c, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = self.globalpool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_info(self):
        return {
            'name': 'ShuffleNetV2 (0.25×)',
            'type': self.model_name,
            'input_size': (3, 48, 48),
            'num_classes': self.fc.out_features,
            'num_parameters': self.get_num_parameters(),
            'dropout_rate': self.dropout.p,
            'architecture': 'Channel shuffle blocks, width=0.25×'
        }


# --- Model Factory ---
def create_model(model_name: str, num_classes=43, **kwargs):
    """
    Factory function to create models by name.
    """
    model_name = model_name.lower()
    
    if model_name == 'lenet':
        return LeNet(num_classes=num_classes, **kwargs)
    elif model_name == 'shallow_cnn':
        return ShallowCNN(num_classes=num_classes, **kwargs)
    elif model_name == 'minivgg':
        return MiniVGG(num_classes=num_classes, **kwargs)
    elif model_name in ['mobilenetv1_025', 'mobilenet025']:
        return MobileNetV1_025(num_classes=num_classes, **kwargs)
    elif model_name in ['shufflenetv2_025', 'shufflenet025']:
        return ShuffleNetV2_025(num_classes=num_classes, **kwargs)
    elif model_name in ['mobilenetv2_025', 'mobilenetv2']:
        return MobileNetV2_025(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(
            f"Unknown model: {model_name}. Available: "
            "'lenet', 'shallow_cnn', 'minivgg', 'mobilenetv1_025', "
            "'shufflenetv2_025', 'mobilenetv2_025'"
        )



if __name__ == '__main__':
    # Test model creation
    print("--- Testing Model Factory ---")
    
    # 1. Test LeNet 
    lenet = create_model('lenet', dropout_rate=0.4)
    info_lenet = lenet.get_model_info()
    print(f"\nModel: {info_lenet['type'].upper()} ({info_lenet['name']})")
    print(f"  Total parameters: {info_lenet['num_parameters']:,}")
    print(f"  Conv Blocks: {info_lenet['num_conv_blocks']}")
    
    # 2. Test ShallowCNN
    shallow_cnn = create_model('shallow_cnn', dropout_rate=0.2)
    info_shallow = shallow_cnn.get_model_info()
    print(f"\nModel: {info_shallow['type'].upper()} ({info_shallow['name']})")
    print(f"  Total parameters: {info_shallow['num_parameters']:,}")
    print(f"  Conv Blocks: {info_shallow['num_conv_blocks']}")
    
    # 3. Test MiniVGG
    minivgg = create_model('minivgg', dropout_rate=0.5)
    info_vgg = minivgg.get_model_info()
    print(f"\nModel: {info_vgg['type'].upper()} ({info_vgg['name']})")
    print(f"  Total parameters: {info_vgg['num_parameters']:,}")
    print(f"  Conv Blocks: {info_vgg['num_conv_blocks']}")
    
    # Test forward pass with MiniVGG
    input_tensor = torch.randn(1, 3, 48, 48)
    output = minivgg(input_tensor)
    print(f"\nMiniVGG forward pass check: {output.shape}")
    
    # 4. Test MobileNetV1_025
    mobilenet = create_model('mobilenetv1_025', dropout_rate=0.3)
    info_mobilenet = mobilenet.get_model_info()
    print(f"\nModel: {info_mobilenet['type'].upper()} ({info_mobilenet['name']})")
    print(f"  Total parameters: {info_mobilenet['num_parameters']:,}")
    print(f"  Dropout rate: {info_mobilenet.get('dropout_rate', 'N/A')}")

    # MobileNet forward pass check
    input_tensor = torch.randn(1, 3, 48, 48)
    out_mobilenet = mobilenet(input_tensor)
    print(f"\nMobileNetV1_025 forward pass check: {out_mobilenet.shape}")

    # 5. Test ShuffleNetV2_025
    shufflenet = create_model('shufflenetv2_025', dropout_rate=0.3)
    info_shufflenet = shufflenet.get_model_info()
    print(f"\nModel: {info_shufflenet['type'].upper()} ({info_shufflenet['name']})")
    print(f"  Total parameters: {info_shufflenet['num_parameters']:,}")
    print(f"  Dropout rate: {info_shufflenet.get('dropout_rate', 'N/A')}")

    # ShuffleNet forward pass check
    input_tensor = torch.randn(1, 3, 48, 48)
    out_shufflenet = shufflenet(input_tensor)
    print(f"\nShuffleNetV2_025 forward pass check: {out_shufflenet.shape}")

    # 6. Test MobileNetV2_025
    mobilenetv2 = create_model('mobilenetv2_025', dropout_rate=0.3)
    info_mobilenetv2 = mobilenetv2.get_model_info()
    print(f"\nModel: {info_mobilenetv2['type'].upper()} ({info_mobilenetv2['name']})")
    print(f"  Total parameters: {info_mobilenetv2['num_parameters']:,}")
    print(f"  Dropout rate: {0.3}")

    # MobileNetV2 forward pass check
    input_tensor = torch.randn(1, 3, 48, 48)
    out_mobilenetv2 = mobilenetv2(input_tensor)
    print(f"\nMobileNetV2_025 forward pass check: {out_mobilenetv2.shape}")