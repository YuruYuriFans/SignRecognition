"""
model.py - Neural Network Architectures for GTSRB
==================================================
Contains model definitions for German Traffic Sign Recognition.

Models (kept in this file):
    - LeNet: LeNet-based CNN for 48x48 input images
    - MiniVGG: A scaled-down VGG-style architecture
    - MobileNetV2_025: Inverted residual blocks (width=0.25)
    - MobileNetV4: State-of-the-art mobile architecture (small/medium/large variants)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LeNet(nn.Module):
    """
    LeNet-based Convolutional Neural Network for Traffic Sign Classification.
    
    Architecture:
        Input: 48x48x3 RGB images
        Conv Block 1: 3->16, 3x3 -> MaxPool -> 24x24x16
        Conv Block 2: 16->32, 3x3 -> MaxPool -> 12x12x32
        Conv Block 3: 32->64, 3x3 -> MaxPool -> 6x6x64
        FC1: 2304->256, ReLU, Dropout(0.5)
        FC2: 256->num_classes
    
    Args:
        num_classes (int): Number of output classes (default: 43)
        dropout_rate (float): Dropout probability (default: 0.5)
    """
    
    def __init__(self, num_classes=43, dropout_rate=0.5, **kwargs):
        super(LeNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        
        self.fc1 = nn.Linear(64 * 6 * 6, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        self.dropout = nn.Dropout(p=dropout_rate)
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.model_name = 'lenet'
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self):
        return {
            'name': 'LeNet (3 Blocks)',
            'type': self.model_name,
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


def build_lenet_variant(num_conv_layers=3,
                        conv_channels=None,
                        kernel_sizes=None,
                        fc_sizes=None,
                        activation='relu',
                        dropout=0.5,
                        dropout_rate=None,
                        num_classes=43,
                        **kwargs):
    """Build a parametric LeNet-like model.

    Args:
        num_conv_layers (int): number of conv blocks to include (1-3+)
        conv_channels (list[int]): output channels per conv block
        kernel_sizes (list[int]): kernel sizes per conv block
        fc_sizes (list[int]): list of fully-connected hidden sizes (final layer to num_classes)
        activation (str): 'relu'|'leakyrelu'|'elu'|'tanh'
        dropout (float): dropout probability used after FC layers
        num_classes (int): final number of classes

    Returns:
        nn.Module: a LeNet-like network instance
    """
    # Backwards-compatible handling: some callers (predict/create_model)
    # pass `dropout_rate` while the ablation/training scripts use `dropout`.
    # Accept both and prefer an explicitly provided `dropout_rate`.
    if dropout_rate is not None:
        dropout = dropout_rate

    if conv_channels is None:
        conv_channels = [16, 32, 64][:num_conv_layers]
    if kernel_sizes is None:
        kernel_sizes = [3] * num_conv_layers
    if fc_sizes is None:
        fc_sizes = [256]

    act_map = {
        'relu': nn.ReLU,
        'leakyrelu': lambda: nn.LeakyReLU(0.1),
        'elu': nn.ELU,
        'tanh': nn.Tanh,
    }
    Act = act_map.get(activation, nn.ReLU)

    class LeNetVariant(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            in_ch = 3
            for i in range(num_conv_layers):
                out_ch = conv_channels[i]
                k = kernel_sizes[i]
                layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=k//2))
                # instantiate activation (callable) if needed
                if callable(Act):
                    layers.append(Act() if Act is not nn.ReLU else nn.ReLU())
                else:
                    layers.append(Act)
                layers.append(nn.MaxPool2d(2))
                in_ch = out_ch

            self.feature_extractor = nn.Sequential(*layers)

            # Infer flattened feature size with a dummy forward pass
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 48, 48)
                feat = self.feature_extractor(dummy)
                flat_size = feat.view(1, -1).shape[1]

            fc_layers = []
            prev = flat_size
            for sz in fc_sizes:
                fc_layers.append(nn.Linear(prev, sz))
                # activation and dropout
                if callable(Act):
                    fc_layers.append(Act() if Act is not nn.ReLU else nn.ReLU())
                else:
                    fc_layers.append(Act)
                fc_layers.append(nn.Dropout(dropout))
                prev = sz
            fc_layers.append(nn.Linear(prev, num_classes))

            self.classifier = nn.Sequential(*fc_layers)

            # metadata
            self.model_name = 'lenet_variant'
            self.num_classes = num_classes
            self.dropout_rate = dropout

        def forward(self, x):
            x = self.feature_extractor(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x

        def get_num_parameters(self):
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

    return LeNetVariant()


# ShallowCNN removed - keeping the codebase focused on the selected models.


class MiniVGG(nn.Module):
    """
    Mini-VGG style architecture.
    
    Architecture:
        Block1: 2x(Conv 3x3, ReLU) -> MaxPool -> Dropout(0.25)
        Block2: 2x(Conv 3x3, ReLU) -> MaxPool -> Dropout(0.25)
        Classifier: FC 9216->512 -> Dropout -> FC 512->num_classes
    """
    
    def __init__(self, num_classes=43, dropout_rate=0.5, **kwargs):
        super(MiniVGG, self).__init__()
        
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )
        
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
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.classifier(x)
        return x
        
    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_info(self):
        return {
            'name': 'Mini-VGG (2 Blocks)',
            'type': self.model_name,
            'input_size': (3, 48, 48),
            'num_classes': self.num_classes,
            'dropout_rate': self.dropout_rate,
            'num_parameters': self.get_num_parameters()
        }


# MobileNetV1 and depthwise separable conv helper removed to keep the repository

class InvertedResidual(nn.Module):
    """Inverted Residual Block for MobileNetV2."""
    
    def __init__(self, in_channels, out_channels, stride, expand_ratio, activation=nn.ReLU6):
        super().__init__()
        hidden_dim = int(in_channels * expand_ratio)
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(activation(inplace=True))

        layers.append(nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(activation(inplace=True))

        layers.append(nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2_025(nn.Module):
    """MobileNetV2 with width multiplier 0.25 for 48x48 images."""
    
    def __init__(self, num_classes=43, dropout_rate=0.3, width_mult=0.25, **kwargs):
        super().__init__()
        self.width_mult = width_mult
        def c(ch): return max(8, int(ch * width_mult))

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, c(16), 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c(16)),
            nn.ReLU6(inplace=True)
        )

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
                layers.append(InvertedResidual(in_channels, out_channels, stride, expand_ratio=t))
                in_channels = out_channels
        self.features = nn.Sequential(*layers)

        self.conv_last = nn.Sequential(
            nn.Conv2d(in_channels, c(512), 1, 1, 0, bias=False),
            nn.BatchNorm2d(c(512)),
            nn.ReLU6(inplace=True)
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(c(512), num_classes)

        self.model_name = 'mobilenetv2_025'
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

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
            'name': f'MobileNetV2 (alpha={self.width_mult})',
            'type': self.model_name,
            'input_size': (3, 48, 48),
            'num_classes': self.num_classes,
            'num_parameters': self.get_num_parameters(),
            'dropout_rate': self.dropout_rate
        }


# ShuffleNetV2 and related helpers removed per request to keep only LeNet, MiniVGG,
# MobileNetV2_025 and MobileNetV4 in the codebase.


def make_divisible(value, divisor, min_value=None, round_down_protect=True):
    """Make value divisible by divisor (MobileNetV4 utility)."""
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if round_down_protect and new_value < 0.9 * value:
        new_value += divisor
    return new_value


class ConvBN(nn.Module):
    """Conv-BN-ReLU block for MobileNetV4."""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                     (kernel_size - 1)//2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.block(x)


class UniversalInvertedBottleneck(nn.Module):
    """Universal Inverted Bottleneck block for MobileNetV4."""
    
    def __init__(self, in_channels, out_channels, expand_ratio,
                 start_dw_kernel_size, middle_dw_kernel_size, stride,
                 middle_dw_downsample=True, use_layer_scale=False,
                 layer_scale_init_value=1e-5):
        super().__init__()
        self.start_dw_kernel_size = start_dw_kernel_size
        self.middle_dw_kernel_size = middle_dw_kernel_size

        if start_dw_kernel_size:
           self.start_dw_conv = nn.Conv2d(in_channels, in_channels, start_dw_kernel_size, 
                                          stride if not middle_dw_downsample else 1,
                                          (start_dw_kernel_size - 1) // 2,
                                          groups=in_channels, bias=False)
           self.start_dw_norm = nn.BatchNorm2d(in_channels)
        
        expand_channels = make_divisible(in_channels * expand_ratio, 8)
        self.expand_conv = nn.Conv2d(in_channels, expand_channels, 1, 1, bias=False)
        self.expand_norm = nn.BatchNorm2d(expand_channels)
        self.expand_act = nn.ReLU(inplace=True)

        if middle_dw_kernel_size:
           self.middle_dw_conv = nn.Conv2d(expand_channels, expand_channels, middle_dw_kernel_size,
                                           stride if middle_dw_downsample else 1,
                                           (middle_dw_kernel_size - 1) // 2,
                                           groups=expand_channels, bias=False)
           self.middle_dw_norm = nn.BatchNorm2d(expand_channels)
           self.middle_dw_act = nn.ReLU(inplace=True)
        
        self.proj_conv = nn.Conv2d(expand_channels, out_channels, 1, 1, bias=False)
        self.proj_norm = nn.BatchNorm2d(out_channels)

        if use_layer_scale:
            self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((out_channels)), 
                                     requires_grad=True)

        self.use_layer_scale = use_layer_scale
        self.identity = stride == 1 and in_channels == out_channels

    def forward(self, x):
        shortcut = x

        if self.start_dw_kernel_size:
            x = self.start_dw_conv(x)
            x = self.start_dw_norm(x)

        x = self.expand_conv(x)
        x = self.expand_norm(x)
        x = self.expand_act(x)

        if self.middle_dw_kernel_size:
            x = self.middle_dw_conv(x)
            x = self.middle_dw_norm(x)
            x = self.middle_dw_act(x)

        x = self.proj_conv(x)
        x = self.proj_norm(x)

        if self.use_layer_scale:
            x = self.gamma * x

        return x + shortcut if self.identity else x


class MobileNetV4(nn.Module):
    """
    MobileNetV4 - Universal Models for the Mobile Ecosystem.
    
    State-of-the-art mobile architecture with Universal Inverted Bottleneck blocks.
    Adapted for 48x48 GTSRB images.
    
    Args:
        variant (str): Model variant ('small', 'medium', 'large')
        num_classes (int): Number of output classes
        dropout_rate (float): Dropout rate before classifier
    """
    
    def __init__(self, variant='small', num_classes=43, dropout_rate=0.3, **kwargs):
        super().__init__()
        
        block_specs = self._get_block_specs(variant)
        
        c = 3
        layers = []
        for block_type, *block_cfg in block_specs:
            if block_type == 'conv_bn':
                k, s, f = block_cfg
                layers.append(ConvBN(c, f, k, s))
            elif block_type == 'uib':
                start_k, middle_k, s, f, e = block_cfg
                layers.append(UniversalInvertedBottleneck(c, f, e, start_k, middle_k, s))
            else:
                raise NotImplementedError
            c = f
        
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        hidden_channels = 640 if variant == 'small' else 960
        self.conv = ConvBN(c, hidden_channels, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_channels, num_classes)

        self._initialize_weights()
        
        self.model_name = f'mobilenetv4_{variant}'
        self.variant = variant
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

    def _get_block_specs(self, variant):
        """Get architecture specifications for each variant."""
        if variant == 'small':
            return [
                ('conv_bn', 3, 2, 32),
                ('conv_bn', 3, 2, 32),
                ('conv_bn', 1, 1, 32),
                ('conv_bn', 3, 2, 96),
                ('conv_bn', 1, 1, 64),
                ('uib', 5, 5, 2, 96, 3.0),
                ('uib', 0, 3, 1, 96, 2.0),
                ('uib', 0, 3, 1, 96, 2.0),
                ('uib', 0, 3, 1, 96, 2.0),
                ('uib', 0, 3, 1, 96, 2.0),
                ('uib', 3, 0, 1, 96, 4.0),
                ('uib', 3, 3, 2, 128, 6.0),
                ('uib', 5, 5, 1, 128, 4.0),
                ('uib', 0, 5, 1, 128, 4.0),
                ('uib', 0, 5, 1, 128, 3.0),
                ('uib', 0, 3, 1, 128, 4.0),
                ('uib', 0, 3, 1, 128, 4.0),
                ('conv_bn', 1, 1, 640),
            ]
        # elif variant == 'medium':
        #     return [
        #         ('conv_bn', 3, 2, 32),
        #         ('conv_bn', 3, 2, 128),
        #         ('conv_bn', 1, 1, 48),
        #         ('uib', 3, 5, 2, 80, 4.0),
        #         ('uib', 3, 3, 1, 80, 2.0),
        #         ('uib', 3, 5, 2, 160, 6.0),
        #         ('uib', 3, 3, 1, 160, 4.0),
        #         ('uib', 3, 0, 1, 160, 4.0),
        #         ('uib', 0, 0, 1, 160, 2.0),
        #         ('uib', 3, 0, 1, 160, 4.0),
        #         ('uib', 5, 5, 2, 256, 6.0),
        #         ('uib', 5, 5, 1, 256, 4.0),
        #         ('uib', 3, 5, 1, 256, 4.0),
        #         ('uib', 3, 5, 1, 256, 4.0),
        #         ('uib', 0, 0, 1, 256, 4.0),
        #         ('uib', 3, 0, 1, 256, 4.0),
        #         ('uib', 3, 5, 1, 256, 2.0),
        #         ('uib', 5, 5, 1, 256, 4.0),
        #         ('uib', 0, 0, 1, 256, 4.0),
        #         ('uib', 0, 0, 1, 256, 4.0),
        #         ('uib', 5, 0, 1, 256, 2.0),
        #         ('conv_bn', 1, 1, 960),
        #     ]
        # elif variant == 'large':
        #     return [
        #         ('conv_bn', 3, 2, 24),
        #         ('conv_bn', 3, 2, 96),
        #         ('conv_bn', 1, 1, 48),
        #         ('uib', 3, 5, 2, 96, 4.0),
        #         ('uib', 3, 3, 1, 96, 4.0),
        #         ('uib', 3, 5, 2, 192, 4.0),
        #         ('uib', 3, 3, 1, 192, 4.0),
        #         ('uib', 3, 3, 1, 192, 4.0),
        #         ('uib', 3, 3, 1, 192, 4.0),
        #         ('uib', 3, 5, 1, 192, 4.0),
        #         ('uib', 5, 3, 1, 192, 4.0),
        #         ('uib', 5, 3, 1, 192, 4.0),
        #         ('uib', 5, 3, 1, 192, 4.0),
        #         ('uib', 5, 3, 1, 192, 4.0),
        #         ('uib', 5, 3, 1, 192, 4.0),
        #         ('uib', 3, 0, 1, 192, 4.0),
        #         ('uib', 5, 5, 2, 512, 4.0),
        #         ('uib', 5, 5, 1, 512, 4.0),
        #         ('uib', 5, 5, 1, 512, 4.0),
        #         ('uib', 5, 5, 1, 512, 4.0),
        #         ('uib', 5, 0, 1, 512, 4.0),
        #         ('uib', 5, 3, 1, 512, 4.0),
        #         ('uib', 5, 0, 1, 512, 4.0),
        #         ('uib', 5, 0, 1, 512, 4.0),
        #         ('uib', 5, 3, 1, 512, 4.0),
        #         ('uib', 5, 5, 1, 512, 4.0),
        #         ('uib', 5, 0, 1, 512, 4.0),
        #         ('uib', 5, 0, 1, 512, 4.0),
        #         ('uib', 5, 0, 1, 512, 4.0),
        #         ('conv_bn', 1, 1, 960),
        #     ]
        else:
            raise ValueError(f"Unknown variant: {variant}. Choose 'small' (other variants are commented out in source)")
        

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_info(self):
        return {
            'name': f'MobileNetV4-{self.variant.capitalize()}',
            'type': self.model_name,
            'input_size': (3, 48, 48),
            'num_classes': self.num_classes,
            'num_parameters': self.get_num_parameters(),
            'dropout_rate': self.dropout_rate,
            'variant': self.variant
        }


def create_model(model_name, num_classes=43, **kwargs):
    """
    Factory function to create models by name.

    Keeps only the selected models:
      - 'lenet'
      - 'minivgg'
      - 'mobilenetv2_025'
      - 'mobilenetv4_small|medium|large'

    Example:
        model = create_model('lenet', num_classes=43, dropout_rate=0.5)
        model = create_model('mobilenetv4_small', num_classes=43)
    """
    model_name = model_name.lower()

    if model_name == 'lenet':
        return LeNet(num_classes=num_classes, **kwargs)
    elif model_name == 'minivgg':
        return MiniVGG(num_classes=num_classes, **kwargs)
    elif model_name in ['mobilenetv2_025', 'mobilenetv2']:
        return MobileNetV2_025(num_classes=num_classes, **kwargs)
    elif model_name == 'mobilenetv4_small':
        return MobileNetV4(variant='small', num_classes=num_classes, **kwargs)
    # elif model_name == 'mobilenetv4_medium':
    #     return MobileNetV4(variant='medium', num_classes=num_classes, **kwargs)
    # elif model_name == 'mobilenetv4_large':
    #     return MobileNetV4(variant='large', num_classes=num_classes, **kwargs)
    elif model_name == 'lenet_variant':
        return build_lenet_variant(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(
            f"Unknown model: {model_name}. Available: 'lenet', 'minivgg', "
            f"'mobilenetv2_025', 'mobilenetv4_small' (other variants are commented out)"
        )


if __name__ == '__main__':
    print("Testing Model Factory")
    print("=" * 60)
    
    models_to_test = [
        ('lenet', {}),
        ('minivgg', {}),
        ('mobilenetv2_025', {}),
        ('mobilenetv4_small', {}),
        # ('mobilenetv4_medium', {}),
        # ('mobilenetv4_large', {}),
    ]
    
    for model_name, kwargs in models_to_test:
        print(f"\nTesting: {model_name}")
        try:
            model = create_model(model_name, num_classes=43, **kwargs)
            info = model.get_model_info()
            
            print(f"  Name: {info['name']}")
            print(f"  Type: {info['type']}")
            print(f"  Parameters: {info['num_parameters']:,}")
            print(f"  Dropout: {info.get('dropout_rate', 'N/A')}")
            
            dummy_input = torch.randn(2, 3, 48, 48)
            output = model(dummy_input)
            print(f"  Forward pass: {dummy_input.shape} -> {output.shape}")
            print(f"  Status: OK")
            
        except Exception as e:
            print(f"  Status: ERROR - {e}")
    
    print("\n" + "=" * 60)
    print("Model factory test complete!")