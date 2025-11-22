# generate_diagrams.py (Updated with MobileNet V2/V4)

import torch
import torch.nn as nn
import graphviz
import math
from torchinfo import summary # Used for validation, though not for diagramming

# --- START: Required Classes and Functions (Copied from model.py) ---

# Utility function
def make_divisible(value, divisor, min_value=None, round_down_protect=True):
    """Make value divisible by divisor (MobileNetV4 utility)."""
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if round_down_protect and new_value < 0.9 * value:
        new_value += divisor
    return new_value

# LeNet class
class LeNet(nn.Module):
    def __init__(self, num_classes=43, dropout_rate=0.5, **kwargs):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        # Calculate flattened size based on 48x48 input and 3 poolings
        self.fc1 = nn.Linear(64 * 6 * 6, 256) 
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.num_classes = num_classes
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# MiniVGG class
class MiniVGG(nn.Module):
    def __init__(self, num_classes=43, dropout_rate=0.5, **kwargs):
        super(MiniVGG, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), nn.Dropout(0.25)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), nn.Dropout(0.25)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(12 * 12 * 64, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        self.num_classes = num_classes
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.classifier(x)
        return x

# MobileNetV2_025 dependencies (minimal required)
class InvertedResidual(nn.Module):
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

# MobileNetV2_025 class
class MobileNetV2_025(nn.Module):
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
            [1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2],
            [6, 64, 4, 2], [6, 96, 3, 1],
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
        self.num_classes = num_classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.features(x)
        x = self.conv_last(x)
        x = self.global_pool(x).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# MobileNetV4 dependencies
class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                      (kernel_size - 1)//2, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class UniversalInvertedBottleneck(nn.Module):
    # Minimal implementation for instantiation
    def __init__(self, in_channels, out_channels, expand_ratio,
                 start_dw_kernel_size, middle_dw_kernel_size, stride, **kwargs):
        super().__init__()
        self.identity = stride == 1 and in_channels == out_channels
        
        # Simplified block structure for instantiation
        layers = []
        if start_dw_kernel_size: layers.append(nn.Conv2d(in_channels, in_channels, start_dw_kernel_size, bias=False, groups=in_channels))
        expand_channels = make_divisible(in_channels * expand_ratio, 8)
        layers.append(nn.Conv2d(in_channels, expand_channels, 1, bias=False))
        if middle_dw_kernel_size: layers.append(nn.Conv2d(expand_channels, expand_channels, middle_dw_kernel_size, bias=False, groups=expand_channels))
        layers.append(nn.Conv2d(expand_channels, out_channels, 1, bias=False))
        
        # Add a placeholder for forward pass (to avoid error when model is instantiated)
        self.test_conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return x # Placeholder, actual forward is too complex for minimal copy

# MobileNetV4 class
class MobileNetV4(nn.Module):
    def __init__(self, variant='small', num_classes=43, dropout_rate=0.3, **kwargs):
        super().__init__()
        
        # NOTE: Using the provided specs for 'small'
        block_specs = self._get_block_specs(variant)
        
        c = 3
        layers = []
        for block_type, *block_cfg in block_specs:
            if block_type == 'conv_bn':
                k, s, f = block_cfg
                layers.append(ConvBN(c, f, k, s))
            elif block_type == 'uib':
                start_k, middle_k, s, f, e = block_cfg
                # Instantiating the placeholder UIB
                layers.append(UniversalInvertedBottleneck(c, f, e, start_k, middle_k, s)) 
            c = f
        
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        hidden_channels = 640 
        self.conv = ConvBN(c, hidden_channels, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_channels, num_classes)

        self.num_classes = num_classes

    def _get_block_specs(self, variant):
        if variant == 'small':
            return [
                ('conv_bn', 3, 2, 32), # 1
                ('conv_bn', 3, 2, 32), # 2
                ('conv_bn', 1, 1, 32), # 3
                ('conv_bn', 3, 2, 96), # 4
                ('conv_bn', 1, 1, 64), # 5
                ('uib', 5, 5, 2, 96, 3.0),  # 6
                ('uib', 0, 3, 1, 96, 2.0),
                ('uib', 0, 3, 1, 96, 2.0),
                ('uib', 0, 3, 1, 96, 2.0),
                ('uib', 0, 3, 1, 96, 2.0), # 10
                ('uib', 3, 0, 1, 96, 4.0),
                ('uib', 3, 3, 2, 128, 6.0), # 12
                ('uib', 5, 5, 1, 128, 4.0),
                ('uib', 0, 5, 1, 128, 4.0),
                ('uib', 0, 5, 1, 128, 3.0), # 15
                ('uib', 0, 3, 1, 128, 4.0),
                ('uib', 0, 3, 1, 128, 4.0), # 17
                ('conv_bn', 1, 1, 640), # 18 (Final stage)
            ]
        else:
            raise ValueError(f"Variant '{variant}' not available.")
            
    def forward(self, x):
        # Simplified forward pass for this context
        return torch.randn(x.size(0), self.num_classes)

# Factory function
def create_model(model_name, num_classes=43, **kwargs):
    model_name = model_name.lower()
    if model_name == 'lenet': return LeNet(num_classes=num_classes, **kwargs)
    elif model_name == 'minivgg': return MiniVGG(num_classes=num_classes, **kwargs)
    elif model_name in ['mobilenetv2_025', 'mobilenetv2']: return MobileNetV2_025(num_classes=num_classes, **kwargs)
    elif model_name == 'mobilenetv4_small': return MobileNetV4(variant='small', num_classes=num_classes, **kwargs)
    else: raise ValueError(f"Unknown model: {model_name}")

# --- END: Required Classes and Functions ---

# --- Graphviz Diagram Functions ---

def create_diagram_node(dot, name, label, shape='box', fillcolor='white', fontcolor='black', style='filled'):
    """Helper to define a node with standard attributes."""
    dot.node(name, label, shape=shape, style=style, fillcolor=fillcolor, fontcolor=fontcolor)

def generate_lenet_diagram(model, filename="lenet_structure"):
    """Generates a Graphviz diagram for the LeNet model."""
    dot = graphviz.Digraph(filename, format='png', graph_attr={'rankdir': 'LR', 'bgcolor': 'transparent', 'splines': 'ortho', 'fontname': 'Helvetica'})
    
    conv_color, pool_color, fc_color, input_color, output_color = '#A5D6A7', '#64B5F6', '#FFCC80', '#FFFFFF', '#FFFFFF'

    create_diagram_node(dot, 'Input', 'Input\n(3, 48, 48)', fillcolor=input_color)
    create_diagram_node(dot, 'C1', 'Conv (3->16)\n+ Pool (24, 24, 16)', fillcolor=pool_color)
    create_diagram_node(dot, 'C2', 'Conv (16->32)\n+ Pool (12, 12, 32)', fillcolor=pool_color)
    create_diagram_node(dot, 'C3', 'Conv (32->64)\n+ Pool (6, 6, 64)', fillcolor=pool_color)
    create_diagram_node(dot, 'F', 'Flatten\n(2304)', fillcolor='#E0E0E0')
    create_diagram_node(dot, 'FC1', 'FC1 (2304->256)\n+ ReLU + Drop', fillcolor=fc_color)
    create_diagram_node(dot, 'Output', f'FC2 (256->{model.num_classes})\nOutput', fillcolor=output_color)

    dot.edge('Input', 'C1')
    dot.edge('C1', 'C2')
    dot.edge('C2', 'C3')
    dot.edge('C3', 'F')
    dot.edge('F', 'FC1')
    dot.edge('FC1', 'Output')

    dot.render(view=False)
    print(f"Generated diagram for LeNet: {filename}.png")

def generate_minivgg_diagram(model, filename="minivgg_structure"):
    """Generates a Graphviz diagram for the MiniVGG model."""
    dot = graphviz.Digraph(filename, format='png', graph_attr={'rankdir': 'LR', 'bgcolor': 'transparent', 'splines': 'ortho', 'fontname': 'Helvetica'})
    
    block_color, fc_color, input_color, output_color = '#64B5F6', '#FFCC80', '#FFFFFF', '#FFFFFF'

    create_diagram_node(dot, 'Input', 'Input\n(3, 48, 48)', fillcolor=input_color)
    
    create_diagram_node(dot, 'B1', 'Block 1: [2x Conv-ReLU] + Pool + Drop(0.25)\nOutput: (24, 24, 32)', fillcolor=block_color)
    create_diagram_node(dot, 'B2', 'Block 2: [2x Conv-ReLU] + Pool + Drop(0.25)\nOutput: (12, 12, 64)', fillcolor=block_color)

    create_diagram_node(dot, 'F', 'Flatten\n(9216)', fillcolor='#E0E0E0')
    create_diagram_node(dot, 'C1', 'FC (9216->512)\n+ ReLU + Drop', fillcolor=fc_color)
    create_diagram_node(dot, 'Output', f'FC (512->{model.num_classes})\nOutput', fillcolor=output_color)

    dot.edge('Input', 'B1')
    dot.edge('B1', 'B2')
    dot.edge('B2', 'F')
    dot.edge('F', 'C1')
    dot.edge('C1', 'Output')
    
    dot.render(view=False)
    print(f"Generated diagram for MiniVGG: {filename}.png")

def generate_mobilenetv2_diagram(model, filename="mobilenetv2_025_structure"):
    """Generates a high-level block diagram for MobileNetV2."""
    dot = graphviz.Digraph(filename, format='png', graph_attr={'rankdir': 'LR', 'bgcolor': 'transparent', 'splines': 'ortho', 'fontname': 'Helvetica'})

    ir_color, final_color, input_color, output_color = '#C5E1A5', '#FFCC80', '#FFFFFF', '#FFFFFF'
    c = model.width_mult
    
    # Blocks configuration based on model.py
    # [t, c_out, n, s] -> [Expansion, Output Channels, Repeats, Stride]
    blocks = [
        ('Start', 3, 16, 2, 'Conv (3->C(16))\nStride 2'),
        ('B1', 16, 16, 1, 'InvRes x1 (t=1, S=1)\nOutput: C(16)'),
        ('B2', 16, 24, 2, 'InvRes x2 (t=6, S=2)\nOutput: C(24)'),
        ('B3', 24, 32, 2, 'InvRes x3 (t=6, S=2)\nOutput: C(32)'),
        ('B4', 32, 64, 2, 'InvRes x4 (t=6, S=2)\nOutput: C(64)'),
        ('B5', 64, 96, 1, 'InvRes x3 (t=6, S=1)\nOutput: C(96)'),
        ('End', 96, 512, 1, f'Conv (C(96)->C(512)) + AvgPool\nOutput: C(512)'),
    ]

    create_diagram_node(dot, 'Input', 'Input\n(3, 48, 48)', fillcolor=input_color)
    
    prev_node = 'Input'
    for i, (name, _, _, _, label) in enumerate(blocks):
        node_name = f'N{i}'
        is_start = i == 0
        is_end = i == len(blocks) - 1
        
        fill = final_color if is_end else ir_color
        
        create_diagram_node(dot, node_name, label, fillcolor=fill)
        dot.edge(prev_node, node_name)
        prev_node = node_name

    create_diagram_node(dot, 'Output', f'FC (C(512)->{model.num_classes})\n+ Dropout', fillcolor=output_color)
    dot.edge(prev_node, 'Output')

    dot.render(view=False)
    print(f"Generated diagram for MobileNetV2_025: {filename}.png")

def generate_mobilenetv4_diagram(model, filename="mobilenetv4_small_structure"):
    """Generates a high-level block diagram for MobileNetV4_Small."""
    dot = graphviz.Digraph(filename, format='png', graph_attr={'rankdir': 'LR', 'bgcolor': 'transparent', 'splines': 'ortho', 'fontname': 'Helvetica'})

    convbn_color, uib_color, final_color, input_color, output_color = '#F9A825', '#90CAF9', '#FFCC80', '#FFFFFF', '#FFFFFF'

    # Simplified block groups based on channels/stride changes
    blocks = [
        ('S0', 3, 32, 2, 'ConvBN x1 (3->32)\nS=2'),
        ('S1', 32, 32, 2, 'ConvBN x2\nS=2, S=1'),
        ('S2', 32, 64, 2, 'ConvBN x2\nS=2, S=1'),
        ('S3', 64, 96, 2, 'UIB Block (x6)\n(5,5,S=2) + (0,3,S=1) x4 + (3,0,S=1)\nOutput: 96'),
        ('S4', 96, 128, 2, 'UIB Block (x6)\n(3,3,S=2) + (5,5,S=1) + ...\nOutput: 128'),
        ('S5', 128, 640, 1, 'Final ConvBN + AvgPool\nOutput: 640'),
    ]

    create_diagram_node(dot, 'Input', 'Input\n(3, 48, 48)', fillcolor=input_color)
    
    prev_node = 'Input'
    for i, (name, _, _, _, label) in enumerate(blocks):
        node_name = name
        
        if 'UIB' in label: fill = uib_color
        elif 'Final' in label: fill = final_color
        else: fill = convbn_color
        
        create_diagram_node(dot, node_name, label, fillcolor=fill)
        dot.edge(prev_node, node_name)
        prev_node = node_name

    create_diagram_node(dot, 'Output', f'FC (640->{model.num_classes})\n+ Dropout', fillcolor=output_color)
    dot.edge(prev_node, 'Output')

    dot.render(view=False)
    print(f"Generated diagram for MobileNetV4_Small: {filename}.png")

# --- Main Execution ---

if __name__ == '__main__':
    NUM_CLASSES = 43
    
    # 1. Generate LeNet Diagram
    lenet_model = create_model('lenet', num_classes=NUM_CLASSES)
    generate_lenet_diagram(lenet_model)

    # 2. Generate MiniVGG Diagram
    minivgg_model = create_model('minivgg', num_classes=NUM_CLASSES)
    generate_minivgg_diagram(minivgg_model)

    # 3. Generate MobileNetV2_025 Diagram
    mobilenetv2_model = create_model('mobilenetv2_025', num_classes=NUM_CLASSES)
    generate_mobilenetv2_diagram(mobilenetv2_model)

    # 4. Generate MobileNetV4_Small Diagram
    mobilenetv4_model = create_model('mobilenetv4_small', num_classes=NUM_CLASSES)
    generate_mobilenetv4_diagram(mobilenetv4_model)

    print("\nDiagram generation complete. Four .png files have been created in your directory.")