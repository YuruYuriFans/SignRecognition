"""
German Traffic Sign Recognition using LeNet Architecture
=======================================================
This script implements a LeNet-based CNN for classifying German traffic signs.

Directory Structure Required:
    project_folder/
    ├── train.py (this file)
    ├── Final_Training/
    │   └── Images/
    │       ├── 00000/  (class 0 images)
    │       ├── 00001/  (class 1 images)
    │       └── ... (up to 00042)
    └── Final_Test/
        └── Images/
            ├── 00000.ppm
            ├── 00001.ppm
            └── ...

How to Run:
    1. Ensure your data is in the structure above
    2. Run: python train.py
    3. The best model will be saved as 'best_lenet_model.pth'
    4. Training progress will be displayed in the terminal

Requirements:
    - PyTorch
    - torchvision
    - PIL
    - numpy
    - tqdm
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import os
from tqdm import tqdm


class LeNet(nn.Module):
    """
    LeNet-based Convolutional Neural Network for Traffic Sign Classification.
    
    Architecture:
        - Conv Layer 1: 3 -> 16 channels, 3x3 kernel, ReLU, MaxPool
        - Conv Layer 2: 16 -> 32 channels, 3x3 kernel, ReLU, MaxPool
        - Conv Layer 3: 32 -> 64 channels, 3x3 kernel, ReLU, MaxPool
        - Fully Connected 1: 1024 -> 128, ReLU, Dropout
        - Fully Connected 2: 128 -> 43 (output classes)
    
    Input: 32x32x3 RGB images
    Output: 43 class predictions (logits)
    
    Args:
        num_classes (int): Number of traffic sign classes (default: 43)
    """
    
    def __init__(self, num_classes=43):
        super(LeNet, self).__init__()
        
        # First convolutional block
        # Input: 32x32x3 -> Output: 16x16x16
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, 
                               kernel_size=3, padding=1)
        
        # Second convolutional block
        # Input: 16x16x16 -> Output: 8x8x32
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, 
                               kernel_size=3, padding=1)
        
        # Third convolutional block
        # Input: 8x8x32 -> Output: 4x4x64
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, 
                               kernel_size=3, padding=1)
        
        # Max pooling layer (2x2 window, stride 2)
        # Reduces spatial dimensions by half
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # ReLU activation function
        self.relu = nn.ReLU()
        
        # First fully connected layer
        # After 3 pooling operations: 32/2/2/2 = 4
        # So input size is 64 channels * 4 * 4 = 1024
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        
        # Output layer (128 -> 43 classes)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Dropout for regularization (prevents overfitting)
        # Randomly drops 50% of neurons during training
        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (Tensor): Input images of shape (batch_size, 3, 32, 32)
        
        Returns:
            Tensor: Class logits of shape (batch_size, 43)
        """
        # Convolutional block 1: Conv -> ReLU -> Pool
        # Shape: (batch, 3, 32, 32) -> (batch, 16, 16, 16)
        x = self.pool(self.relu(self.conv1(x)))
        
        # Convolutional block 2: Conv -> ReLU -> Pool
        # Shape: (batch, 16, 16, 16) -> (batch, 32, 8, 8)
        x = self.pool(self.relu(self.conv2(x)))
        
        # Convolutional block 3: Conv -> ReLU -> Pool
        # Shape: (batch, 32, 8, 8) -> (batch, 64, 4, 4)
        x = self.pool(self.relu(self.conv3(x)))
        
        # Flatten the tensor for fully connected layers
        # Shape: (batch, 64, 4, 4) -> (batch, 1024)
        x = x.view(-1, 64 * 4 * 4)
        
        # Fully connected layers with dropout
        # Shape: (batch, 1024) -> (batch, 128)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)  # Only active during training
        
        # Output layer (no activation - raw logits)
        # Shape: (batch, 128) -> (batch, 43)
        x = self.fc2(x)
        
        return x


class GTSRBDataset(Dataset):
    """
    Custom PyTorch Dataset for German Traffic Sign Recognition Benchmark.
    
    Handles loading of PPM images from the hierarchical directory structure.
    
    Args:
        root_dir (str): Root directory ('Final_Training' or 'Final_Test')
        transform (callable, optional): Optional transform to apply to images
        is_train (bool): If True, load training data with labels; 
                        if False, load test data without labels
    
    Training Structure:
        root_dir/Images/00000/*.ppm (class 0)
        root_dir/Images/00001/*.ppm (class 1)
        ...
        root_dir/Images/00042/*.ppm (class 42)
    
    Test Structure:
        root_dir/Images/00000.ppm
        root_dir/Images/00001.ppm
        ...
    """
    
    def __init__(self, root_dir, transform=None, is_train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        self.images = []  # List of image file paths
        self.labels = []  # List of corresponding labels (only for training)
        
        if is_train:
            # Load training data from class subdirectories
            images_dir = os.path.join(root_dir, 'Images')
            
            # Iterate through all 43 classes (00000 to 00042)
            for class_id in range(43):
                class_dir = os.path.join(images_dir, f'{class_id:05d}')
                
                if os.path.exists(class_dir):
                    # Load all PPM images in this class directory
                    for img_name in os.listdir(class_dir):
                        if img_name.endswith('.ppm'):
                            img_path = os.path.join(class_dir, img_name)
                            self.images.append(img_path)
                            self.labels.append(class_id)
                else:
                    print(f"Warning: Directory {class_dir} not found")
        else:
            # Load test data (no class subdirectories)
            images_dir = os.path.join(root_dir, 'Images')
            
            if os.path.exists(images_dir):
                # Sort to maintain consistent ordering
                for img_name in sorted(os.listdir(images_dir)):
                    if img_name.endswith('.ppm'):
                        img_path = os.path.join(images_dir, img_name)
                        self.images.append(img_path)
            else:
                print(f"Warning: Directory {images_dir} not found")
        
        print(f"Loaded {len(self.images)} images from {root_dir}")
    
    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Get a single image (and label if training).
        
        Args:
            idx (int): Index of the image to retrieve
        
        Returns:
            If training: (image_tensor, label)
            If testing: (image_tensor, image_path)
        """
        img_path = self.images[idx]
        
        # Load image and convert to RGB (in case of grayscale)
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms (resize, augmentation, normalization)
        if self.transform:
            image = self.transform(image)
        
        if self.is_train:
            # Return image and its class label
            label = self.labels[idx]
            return image, label
        else:
            # Return image and its file path (for test submission)
            return image, img_path


def train_model(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model (nn.Module): The neural network model
        train_loader (DataLoader): DataLoader for training data
        criterion: Loss function (CrossEntropyLoss)
        optimizer: Optimizer (Adam)
        device: Device to train on (cuda or cpu)
    
    Returns:
        tuple: (average_loss, accuracy_percentage)
    """
    # Set model to training mode (enables dropout, batch norm training mode)
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Iterate through batches with progress bar
    for images, labels in tqdm(train_loader, desc='Training'):
        # Move data to GPU if available
        images = images.to(device)
        labels = labels.to(device)
        
        # Zero the gradients from previous iteration
        optimizer.zero_grad()
        
        # Forward pass: compute predictions
        outputs = model(images)
        
        # Compute loss
        loss = criterion(outputs, labels)
        
        # Backward pass: compute gradients
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Accumulate statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)  # Get class with highest score
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    # Calculate average metrics
    accuracy = 100. * correct / total
    avg_loss = running_loss / len(train_loader)
    
    return avg_loss, accuracy


def validate_model(model, val_loader, criterion, device):
    """
    Validate the model on validation set.
    
    Args:
        model (nn.Module): The neural network model
        val_loader (DataLoader): DataLoader for validation data
        criterion: Loss function (CrossEntropyLoss)
        device: Device to validate on (cuda or cpu)
    
    Returns:
        tuple: (average_loss, accuracy_percentage)
    """
    # Set model to evaluation mode (disables dropout, batch norm uses running stats)
    model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Disable gradient computation for efficiency
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validation'):
            # Move data to GPU if available
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass only
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Accumulate statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    # Calculate average metrics
    accuracy = 100. * correct / total
    avg_loss = running_loss / len(val_loader)
    
    return avg_loss, accuracy


def main():
    """
    Main training function.
    
    This function:
    1. Sets up the device (CUDA if available)
    2. Creates data transforms and loaders
    3. Initializes the model, loss function, and optimizer
    4. Trains the model for specified epochs
    5. Saves the best performing model
    """
    
    # ===== HYPERPARAMETERS =====
    batch_size = 64          # Number of images per batch
    num_epochs = 30          # Number of complete passes through dataset
    learning_rate = 0.001    # Step size for optimizer
    
    # ===== DEVICE CONFIGURATION =====
    # Automatically use CUDA (GPU) if available, otherwise use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Print GPU information if using CUDA
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'CUDA Version: {torch.version.cuda}')
        print(f'Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
    
    # ===== DATA TRANSFORMS =====
    # Training transforms: resize, augmentation, normalization
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize to 32x32 (LeNet input size)
        transforms.RandomRotation(15),  # Random rotation ±15 degrees
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Color variation
        transforms.ToTensor(),  # Convert PIL Image to Tensor [0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet statistics
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Validation transforms: only resize and normalize (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # ===== LOAD DATASETS =====
    print("\nLoading datasets...")
    train_dataset = GTSRBDataset('Final_Training', 
                                 transform=train_transform, 
                                 is_train=True)
    
    # Split training data: 90% train, 10% validation
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")
    
    # ===== CREATE DATA LOADERS =====
    # DataLoader handles batching, shuffling, and parallel loading
    train_loader = DataLoader(train_dataset, 
                             batch_size=batch_size, 
                             shuffle=True,  # Shuffle training data
                             num_workers=2,  # Parallel data loading
                             pin_memory=True if torch.cuda.is_available() else False)
    
    val_loader = DataLoader(val_dataset, 
                           batch_size=batch_size, 
                           shuffle=False,  # Don't shuffle validation
                           num_workers=2,
                           pin_memory=True if torch.cuda.is_available() else False)
    
    # ===== INITIALIZE MODEL =====
    print("\nInitializing model...")
    model = LeNet(num_classes=43).to(device)  # Move model to GPU
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    # ===== LOSS FUNCTION AND OPTIMIZER =====
    # CrossEntropyLoss: combines LogSoftmax and NLLLoss
    criterion = nn.CrossEntropyLoss()
    
    # Adam optimizer: adaptive learning rate for each parameter
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler: reduce LR by 10x every 10 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # ===== TRAINING LOOP =====
    print("\nStarting training...")
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'\n{"="*60}')
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'{"="*60}')
        
        # Train for one epoch
        train_loss, train_acc = train_model(model, train_loader, criterion, 
                                           optimizer, device)
        
        # Validate
        val_loss, val_acc = validate_model(model, val_loader, criterion, device)
        
        # Print results
        print(f'\nResults:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Step the learning rate scheduler
        scheduler.step()
        
        # Save best model based on validation accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_acc,
            }, 'best_lenet_model.pth')
            print(f'  ✓ Best model saved! (Accuracy: {best_acc:.2f}%)')
    
    # ===== TRAINING COMPLETE =====
    print(f'\n{"="*60}')
    print(f'Training completed!')
    print(f'Best validation accuracy: {best_acc:.2f}%')
    print(f'Model saved as: best_lenet_model.pth')
    print(f'{"="*60}')


if __name__ == '__main__':
    # Entry point of the script
    main()