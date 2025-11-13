

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
import time

try:
    from thop import profile as thop_profile
    _THOP_AVAILABLE = True
except Exception:
    thop_profile = None
    _THOP_AVAILABLE = False

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except Exception:
    psutil = None
    _PSUTIL_AVAILABLE = False

# Import refactored components
from model import create_model
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


def train_with_augmentation(aug_type, args, device, model_name='lenet'):
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
    
    # Initialize model from model.py using the factory
    print(f"\nInitializing model '{model_name}'...")
    # Use create_model to instantiate the requested architecture
    model = create_model(model_name, num_classes=43, dropout_rate=0.5).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
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
            # Ensure trained_models directory exists
            os.makedirs('trained_models', exist_ok=True)
            save_name = os.path.join('trained_models', f'best_{model_name}_{aug_type}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_acc,
                'augmentation': aug_type,
                'model': model_name,
                'input_size': 48
            }, save_name)
            print(f'Best model saved: {save_name} (Val Acc: {best_acc:.2f}%)')
    
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
        'model': model_name,
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
    results_file = f'./results/comparison_results_{timestamp}.json'
    
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
    
    parser.add_argument('--models', nargs='+',
                       choices=['lenet', 'minivgg', 'mobilenetv2_025', 'mobilenetv4_small',
                                # 'mobilenetv4_medium',
                                # 'mobilenetv4_large',
                                ],
                       help='Which models to train (default: all)')
    
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
    
    # Train with each augmentation strategy and model
    all_results = []

    # Determine model list (default: train all supported models)
    # Place MobileNetV2 first in the default training order
    available_models = [
        'mobilenetv2_025',
        'lenet',
        'minivgg',
        'mobilenetv4_small',
        # 'mobilenetv4_medium',
        # 'mobilenetv4_large',
    ]
    if args.models:
        models_to_run = args.models
    else:
        models_to_run = available_models

    for model_name in models_to_run:
        if model_name not in available_models:
            print(f"Warning: Unknown model '{model_name}' - skipping.")
            continue
        for aug_type in args.aug:
            start_time = time.time()
            result = train_with_augmentation(aug_type, args, device, model_name=model_name)
            end_time = time.time()
            # training time in minutes
            train_time_mins = (end_time - start_time) / 60.0
            result['train_time_mins'] = train_time_mins

            # number of parameters (instantiate model on CPU briefly to count)
            try:
                tmp_model = create_model(model_name, num_classes=43, dropout_rate=0.5)
                num_params = sum(p.numel() for p in tmp_model.parameters() if p.requires_grad)
            except Exception:
                tmp_model = None
                num_params = None

            # FLOPs (MACs) using thop if available
            flops = None
            try:
                if _THOP_AVAILABLE and tmp_model is not None:
                    dummy = torch.randn(1, 3, 48, 48)
                    try:
                        macs, params = thop_profile(tmp_model, inputs=(dummy,), verbose=False)
                        flops = float(macs)
                    except Exception:
                        flops = None
            except Exception:
                flops = None

            # Throughput (images/sec) and peak memory (GPU/CPU)
            throughput = None
            peak_gpu_bytes = None
            peak_cpu_rss = None
            try:
                if tmp_model is not None:
                    tmp_model.eval()
                    # Use a device for throughput measurement
                    measure_dev = device
                    m = tmp_model.to(measure_dev)

                    # Warm-up and timed runs
                    warmup_iters = 10
                    timed_iters = 100
                    batch_size = max(1, int(args.batch_size))
                    inp = torch.randn(batch_size, 3, 48, 48, device=measure_dev)

                    # Reset GPU peak stats if available
                    if measure_dev.type == 'cuda':
                        torch.cuda.reset_peak_memory_stats(measure_dev)

                    # CPU memory sampling setup
                    if _PSUTIL_AVAILABLE:
                        proc = psutil.Process()
                        cpu_peak = 0

                    with torch.no_grad():
                        # warm-up
                        for _ in range(warmup_iters):
                            _ = m(inp)
                        # timed
                        if measure_dev.type == 'cuda':
                            torch.cuda.synchronize(measure_dev)
                        t0 = time.time()
                        for _ in range(timed_iters):
                            out = m(inp)
                            if _PSUTIL_AVAILABLE:
                                try:
                                    rss = proc.memory_info().rss
                                    if rss > cpu_peak:
                                        cpu_peak = rss
                                except Exception:
                                    pass
                        if measure_dev.type == 'cuda':
                            torch.cuda.synchronize(measure_dev)
                        t1 = time.time()

                    duration = t1 - t0 if t1 > t0 else 1e-6
                    throughput = (timed_iters * batch_size) / duration

                    if measure_dev.type == 'cuda':
                        try:
                            peak_gpu_bytes = torch.cuda.max_memory_allocated(measure_dev)
                        except Exception:
                            peak_gpu_bytes = None

                    if _PSUTIL_AVAILABLE:
                        peak_cpu_rss = cpu_peak

                    # cleanup
                    try:
                        m.cpu()
                        del m
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
            except Exception:
                throughput = None
                peak_gpu_bytes = None
                peak_cpu_rss = None

            result['num_parameters'] = num_params
            result['flops_macs'] = flops
            result['throughput_imgs_per_sec'] = throughput
            result['peak_gpu_bytes'] = peak_gpu_bytes
            result['peak_cpu_rss_bytes'] = peak_cpu_rss

            # delete temporary model if present
            try:
                del tmp_model
            except Exception:
                pass

            all_results.append(result)
    
    # Compare results if multiple strategies or compare flag is set
    if len(all_results) > 1 or args.compare:
        compare_results(all_results)

    # After training all models, write a concise summary to records/train_results.txt and print it
    os.makedirs('records', exist_ok=True)

    # Build a nicely aligned table for the training summary
    header_title = f"Training Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    col_fmt = "{:<20} {:<12} {:>14} {:>10} {:>11} {:>13} {:>11} {:>10} {:>15} {:>12} {:>12}"
    cols = ["Model", "Augmentation", "Parameters", "Time(mins)", "BestVal(%)", "FinalTrain(%)", "FinalVal(%)", "FLOPs(G)", "Throughput(img/s)", "PeakGPU_MB", "PeakCPU_MB"]

    summary_lines = [header_title, col_fmt.format(*cols), '-' * 138]

    for r in all_results:
        model_nm = r.get('model', 'unknown')
        aug = r.get('augmentation', 'none')
        params = r.get('num_parameters')
        params_str = f"{params:,}" if params is not None else 'N/A'
        tmins = r.get('train_time_mins')
        tmins_str = f"{tmins:.2f}" if isinstance(tmins, (int, float)) else 'N/A'
        best = r.get('best_val_acc')
        ft = r.get('final_train_acc')
        fv = r.get('final_val_acc')

        best_str = f"{best:.2f}" if isinstance(best, (int, float)) else 'N/A'
        ft_str = f"{ft:.2f}" if isinstance(ft, (int, float)) else 'N/A'
        fv_str = f"{fv:.2f}" if isinstance(fv, (int, float)) else 'N/A'

        # format optional metrics
        flops_val = r.get('flops_macs')
        if isinstance(flops_val, (int, float)):
            flops_g = flops_val / 1e9
            flops_str = f"{flops_g:.3f}"
        else:
            flops_str = 'N/A'

        thr = r.get('throughput_imgs_per_sec')
        thr_str = f"{thr:.2f}" if isinstance(thr, (int, float)) else 'N/A'

        peak_gpu = r.get('peak_gpu_bytes')
        peak_gpu_str = f"{(peak_gpu / (1024**2)):.1f}" if isinstance(peak_gpu, (int, float)) else 'N/A'

        peak_cpu = r.get('peak_cpu_rss_bytes')
        peak_cpu_str = f"{(peak_cpu / (1024**2)):.1f}" if isinstance(peak_cpu, (int, float)) else 'N/A'

        line = col_fmt.format(model_nm, aug, params_str, tmins_str, best_str, ft_str, fv_str, flops_str, thr_str, peak_gpu_str, peak_cpu_str)
        summary_lines.append(line)

    summary_text = "\n".join(summary_lines)
    print(summary_text)
    with open('records/train_results.txt', 'w') as f:
        f.write(summary_text + "\n")


if __name__ == '__main__':
    main()