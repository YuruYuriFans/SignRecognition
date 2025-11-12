"""
German Traffic Sign Recognition - Configurable Data Augmentation
================================================================
Training script with command-line configurable augmentation strategies,
now utilizing refactored modules for model, dataset, and augmentation.

Usage:
    python train.py --aug none
    python train.py --aug basic
    python train.py --aug advanced
    python train.py --aug none basic advanced --compare

Arguments:
    --aug: Augmentation strategy (none, basic, advanced)
           Can specify multiple for comparison
    --compare: Run all specified strategies and compare results
    --epochs: Number of training epochs (default: 30)
    --batch_size: Batch size (default: 64)
    --lr: Learning rate (default: 0.001)

Examples:
    # Train with basic augmentation
    python train.py --aug basic
    
    # Compare all three strategies
    python train.py --aug none basic advanced --compare
    
    # Train with custom parameters
    python train.py --aug advanced --epochs 40 --batch_size 128
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
import argparse
import json
from datetime import datetime

# Import refactored components
# Assuming model.py, dataset.py, and augmentation.py are in the same directory
from model import LeNet
from dataset import GTSRBDataset
from augmentation import get_augmentation_transforms


def train_model(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(train_loader, desc='Training', leave=False):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = running_loss / len(train_loader)
    
    return avg_loss, accuracy


def validate_model(model, val_loader, criterion, device):
    """Validate on validation set."""
    model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validation', leave=False):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = running_loss / len(val_loader)
    
    return avg_loss, accuracy


def train_with_augmentation(aug_type, args, device):
    """
    Train model with specified augmentation strategy.
    
    Args:
        aug_type (str): Augmentation type
        args: Command line arguments
        device: Training device
    
    Returns:
        dict: Training results
    """
    print(f"\n{'='*60}")
    print(f"Training with {aug_type.upper()} augmentation")
    print(f"{'='*60}")
    
    # Get transforms from augmentation.py
    # Note: LeNet in model.py is designed for 48x48 input, which is the default in augmentation.py
    train_transform, val_transform, description = get_augmentation_transforms(aug_type, input_size=48)
    print(f"Strategy: {description}")
    
    # Load datasets using GTSRBDataset from dataset.py
    print(f"\nLoading datasets...")
    # NOTE: The default 'Final_Training' directory is hardcoded in the example main function of dataset.py 
    # and is assumed here.
    train_dataset_full = GTSRBDataset(root_dir='Final_Training', 
                                      transform=train_transform, 
                                      is_train=True)
    
    print(f"Total images: {len(train_dataset_full)}")
    
    # Split dataset
    train_size = int(0.9 * len(train_dataset_full))
    val_size = len(train_dataset_full) - train_size
    train_dataset, val_dataset_split = torch.utils.data.random_split(
        train_dataset_full, [train_size, val_size]
    )
    
    # IMPORTANT: The validation split needs to use the validation transform.
    # We create a new GTSRBDataset instance for the validation split's indices 
    # to ensure the correct transform is applied *only* to the validation data.
    # A simpler way when using random_split:
    val_dataset_split.dataset.transform = val_transform

    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset_split,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Initialize model from model.py
    print(f"\nInitializing model...")
    # The LeNet in model.py defaults to 43 classes and 0.5 dropout, which matches the original train.py logic.
    model = LeNet(num_classes=43, dropout_rate=0.5).to(device) 
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Training loop (rest remains the same)
    print(f"\nStarting training for {args.epochs} epochs...")
    
    best_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'epochs': []
    }
    
    for epoch in range(args.epochs):
        print(f'\nEpoch [{epoch+1}/{args.epochs}]')
        
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_model(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['epochs'].append(epoch + 1)
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'Gap: {train_acc - val_acc:.2f}%')
        
        scheduler.step()
        
        if val_acc > best_acc:
            best_acc = val_acc
            model_name = f'best_lenet_model_{aug_type}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_acc,
                'augmentation': aug_type,
                'input_size': 48
            }, model_name)
            print(f'Best model saved: {model_name} (Val Acc: {best_acc:.2f}%)')
    
    # Final results
    final_gap = history['train_acc'][-1] - history['val_acc'][-1]
    
    print(f"\n{'='*60}")
    print(f"Training completed for {aug_type.upper()}")
    print(f"{'='*60}")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Final train-val gap: {final_gap:.2f}%")
    
    if final_gap < 2:
        print("No overfitting detected")
    elif final_gap < 5:
        print("Slight overfitting (acceptable)")
    else:
        print("Significant overfitting detected")
    
    results = {
        'augmentation': aug_type,
        'description': description,
        'best_val_acc': best_acc,
        'final_train_acc': history['train_acc'][-1],
        'final_val_acc': history['val_acc'][-1],
        'final_gap': final_gap,
        'history': history
    }
    
    return results


def compare_results(all_results):
    """
    Compare results from multiple augmentation strategies.
    
    Args:
        all_results (list): List of result dictionaries
    """
    print(f"\n{'='*80}")
    print("COMPARISON OF AUGMENTATION STRATEGIES")
    print(f"{'='*80}\n")
    
    # Print table header
    print(f"{'Strategy':<15} {'Description':<50} {'Best Val':<10} {'Final Gap':<10}")
    print(f"{'-'*85}")
    
    # Print results for each strategy
    for result in all_results:
        print(f"{result['augmentation']:<15} "
              f"{result['description'][:47]:<50} "
              f"{result['best_val_acc']:>8.2f}% "
              f"{result['final_gap']:>8.2f}%")
    
    print(f"\n{'='*80}")
    
    # Find best strategy
    best_result = max(all_results, key=lambda x: x['best_val_acc'])
    print(f"\nBest Strategy: {best_result['augmentation'].upper()}")
    print(f"  Validation Accuracy: {best_result['best_val_acc']:.2f}%")
    print(f"  Train-Val Gap: {best_result['final_gap']:.2f}%")
    
    # Find strategy with least overfitting
    least_overfit = min(all_results, key=lambda x: x['final_gap'])
    print(f"\nLeast Overfitting: {least_overfit['augmentation'].upper()}")
    print(f"  Train-Val Gap: {least_overfit['final_gap']:.2f}%")
    print(f"  Validation Accuracy: {least_overfit['best_val_acc']:.2f}%")
    
    print(f"\n{'='*80}")
    
    # Save comparison results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'comparison_results_{timestamp}.json'
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")


def main():
    """Main function with command-line argument parsing."""
    
    parser = argparse.ArgumentParser(
        description='Train GTSRB classifier with configurable augmentation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --aug none
  python train.py --aug basic --epochs 40
  python train.py --aug none basic advanced --compare
        """
    )
    
    parser.add_argument('--aug', nargs='+', 
                       choices=['none', 'basic', 'advanced'],
                       default=['basic'],
                       help='Augmentation strategy (can specify multiple)')
    
    parser.add_argument('--compare', action='store_true',
                       help='Compare multiple augmentation strategies')
    
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs (default: 30)')
    
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size (default: 64)')
    
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    
    args = parser.parse_args()
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    print(f"\nConfiguration:")
    print(f"  Augmentation strategies: {', '.join(args.aug)}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Compare mode: {args.compare}")
    
    # Train with each augmentation strategy
    all_results = []
    
    for aug_type in args.aug:
        result = train_with_augmentation(aug_type, args, device)
        all_results.append(result)
    
    # Compare results if multiple strategies or compare flag is set
    if len(all_results) > 1 or args.compare:
        compare_results(all_results)


if __name__ == '__main__':
    main()