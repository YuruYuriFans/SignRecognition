"""
predict.py - Model Prediction and Evaluation Tool
==================================================
This script automatically detects all trained models (best_lenet_model_*.pth),
runs predictions and evaluation for each, and provides a comparative summary.

It utilizes the refactored LeNet model from model.py and the GTSRB_Test_Loader
from dataset.py.

Directory Structure Required:
    project_folder/
    ├── predict.py (this file)
    ├── model.py
    ├── dataset.py
    ├── augmentation.py
    ├── best_lenet_model_*.pth (trained models)
    ├── GTSRB_Test_GT.csv (ground truth - optional)
    ├── Final_Test/
    │   └── Images/
    │       ├── 00000.ppm
    │       └── ...
    └── predictions/  (created automatically)

How to Run:
    python predict.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import os
import csv
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, log_loss
from sklearn.metrics import precision_recall_fscore_support
import glob
import re
import json
import time

# Optional profiling dependencies
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

# --- Import Refactored Components ---
# Use the model factory so we can load different architectures saved by the training script
from model import create_model, LeNet
# GTSRB_Test_Loader is in dataset.py, but we will define a simple helper
# to get the test transform needed for the loader.
from dataset import GTSRB_Test_Loader
from augmentation import get_augmentation_transforms


# --- Helper Functions (Re-used/Modified from original) ---

def get_test_transform(input_size=48):
    """
    Get the fixed validation/test transform (resize, ToTensor, Normalize).
    This transform is consistent across all model evaluations.
    """
    # Use the 'none' augmentation strategy's validation transform as the test transform
    _, val_transform, _ = get_augmentation_transforms('none', input_size)
    return val_transform

def load_model(model_path, device):
    """
    Load trained model from checkpoint using the imported LeNet class.
    
    Args:
        model_path (str): Path to saved model
        device: Device to load on
    
    Returns:
        tuple: (model, checkpoint_info)
    """
    print(f"\nLoading model from: {os.path.basename(model_path)}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load checkpoint first so we can infer which architecture to instantiate
    checkpoint = torch.load(model_path, map_location=device)

    # Determine model type
    model_type = checkpoint.get('model') if isinstance(checkpoint, dict) else None
    bn = os.path.basename(model_path).lower()

    if not model_type:
        for candidate in ['lenet', 'shallow_cnn', 'minivgg', 'mobilenetv1_025', 'shufflenetv2_025']:
            if candidate in bn:
                model_type = candidate
                break

    # Detect ablation or variant models
    if isinstance(checkpoint, dict) and checkpoint.get('model') == 'lenet_variant':
        model_type = 'lenet_variant'
    elif "ablation" in bn:
        model_type = 'lenet_variant'  # fallback for older ablations


    if not model_type:
        # fallback to LeNet for backward compatibility
        model_type = 'lenet'

    # Use saved metadata when available
    num_classes = checkpoint.get('num_classes', 43) if isinstance(checkpoint, dict) else 43
    dropout_rate = checkpoint.get('dropout_rate', 0.5) if isinstance(checkpoint, dict) else 0.5

    # Instantiate model via factory
    try:
        # For ablation/variant models, use saved config if available to reconstruct exact architecture
        model_kwargs = {'num_classes': num_classes, 'dropout_rate': dropout_rate}
        if model_type == 'lenet_variant' and isinstance(checkpoint, dict) and 'config' in checkpoint:
            cfg = checkpoint['config']
            model_kwargs.update({
                'num_conv_layers': cfg.get('num_conv_layers', 3),
                'conv_channels': cfg.get('conv_channels'),
                'kernel_sizes': cfg.get('kernel_sizes'),
                'fc_sizes': cfg.get('fc_sizes'),
                'activation': cfg.get('activation', 'relu'),
                'dropout': cfg.get('dropout', 0.5),
            })
        model = create_model(model_type, **model_kwargs).to(device)
    except Exception as e:
        print(f"Warning: could not create model '{model_type}' ({e}), falling back to LeNet")
        model = LeNet(num_classes=num_classes).to(device)

    checkpoint_info = {
        'accuracy': checkpoint.get('accuracy') if isinstance(checkpoint, dict) else None,
        'augmentation': checkpoint.get('augmentation', 'N/A') if isinstance(checkpoint, dict) else 'N/A',
        'input_size': checkpoint.get('input_size', 48) if isinstance(checkpoint, dict) else 48,
        'model': model_type
    }

    # Load state dict: support both wrapped checkpoints and plain state_dicts
    try:
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
            except RuntimeError as e:
                print(f"State dict load (strict) failed: {e}\nRetrying with strict=False...")
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        elif isinstance(checkpoint, dict):
            try:
                model.load_state_dict(checkpoint)
            except RuntimeError as e:
                print(f"State dict load failed: {e}\nAttempting non-strict load...")
                model.load_state_dict(checkpoint, strict=False)
        else:
            model.load_state_dict(checkpoint)
    except Exception as e:
        raise RuntimeError(f"Failed to load model state for '{model_type}' from {model_path}: {e}")

    model.eval()
    return model, checkpoint_info

def predict(model, test_loader, device, has_gt=False):
    """
    Make predictions on test dataset. (Same as original)
    """
    predictions = []
    all_labels = [] if has_gt else None
    all_probs = [] if has_gt else None
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Predicting'):
            if has_gt:
                images, labels, filenames = batch
                all_labels.extend(labels.numpy())
            else:
                images, filenames, indices = batch # indices unused but kept for consistency
            
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidences, predicted = probabilities.max(1)
            
            for i in range(len(filenames)):
                pred_dict = {
                    'filename': filenames[i],
                    'predicted_class': predicted[i].item(),
                    'confidence': confidences[i].item(),
                }
                predictions.append(pred_dict)
            # collect probabilities for log-loss calculation (only if ground truth available)
            if has_gt:
                # probabilities is a tensor [batch, num_classes]
                all_probs.extend(probabilities.cpu().numpy().tolist())
    
    return predictions, all_labels, all_probs


def calculate_metrics(y_true, y_pred, y_prob=None):
    """Calculate core metrics:
    - accuracy
    - macro F1, macro precision, macro recall
    - weighted F1, weighted precision, weighted recall
    - log-loss (if probabilities provided)

    Returns (accuracy, f1_macro, precision_macro, recall_macro,
             f1_weighted, precision_weighted, recall_weighted, logloss)
    """
    accuracy = accuracy_score(y_true, y_pred)

    # Macro-averaged metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=range(43), average='macro', zero_division=0
    )

    # Weighted-averaged metrics
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=range(43), average='weighted', zero_division=0
    )

    # Log loss (cross-entropy) if probabilities provided
    logloss = None
    if y_prob is not None:
        try:
            # y_prob should be shape (n_samples, n_classes)
            logloss = float(log_loss(y_true, y_prob, labels=list(range(y_prob.shape[1]))))
        except Exception:
            logloss = None

    return accuracy, f1_macro, precision_macro, recall_macro, f1_weighted, precision_weighted, recall_weighted, logloss


def plot_confusion_matrix(y_true, y_pred, output_path, num_classes=43):
    """Generate confusion matrix visualization."""
    print("Generating confusion matrix...")
    
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    
    plt.figure(figsize=(20, 18))
    # We use annot=False to prevent clutter on a 43x43 matrix
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                square=True, cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('True Class', fontsize=14)
    plt.xlabel('Predicted Class', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to: {output_path}")


# --- NEW: Single Model Prediction Function ---

def predict_and_evaluate_single(model_path, test_loader, device, has_gt, output_base_dir='predictions'):
    """
    Performs prediction and evaluation for a single model.
    """
    # 1. Load Model
    model, info = load_model(model_path, device)
    
    model_name_safe = os.path.splitext(os.path.basename(model_path))[0]
    output_dir = os.path.join(output_base_dir, model_name_safe)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n--- Starting Prediction for {model_name_safe} ---")
    
    # Profiling: FLOPs, latency, throughput, model size, peak memory
    profiling = {
        'flops_macs': None,
        'latency_sec': None,
        'throughput_imgs_per_sec': None,
        'model_size_bytes': None,
        'peak_gpu_bytes': None,
        'peak_cpu_rss_bytes': None,
    }

    try:
        # Model size (bytes)
        total_bytes = 0
        for p in model.parameters():
            total_bytes += p.numel() * p.element_size()
        profiling['model_size_bytes'] = int(total_bytes)

        # FLOPs via thop (if available)
        if _THOP_AVAILABLE:
            try:
                dummy = torch.randn(1, 3, 48, 48).to(device)
                macs, params = thop_profile(model, inputs=(dummy,), verbose=False)
                profiling['flops_macs'] = float(macs)
            except Exception:
                profiling['flops_macs'] = None

        # Throughput and latency measurement
        batch_size = getattr(test_loader, 'batch_size', 64) or 64
        warmup_iters = 10
        timed_iters = 100
        inp = torch.randn(batch_size, 3, 48, 48, device=device)

        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)

        cpu_peak = 0
        proc = psutil.Process() if _PSUTIL_AVAILABLE else None

        model.eval()
        with torch.no_grad():
            # warm-up
            for _ in range(warmup_iters):
                _ = model(inp)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            t0 = time.time()
            for _ in range(timed_iters):
                _ = model(inp)
                if _PSUTIL_AVAILABLE:
                    try:
                        rss = proc.memory_info().rss
                        if rss > cpu_peak:
                            cpu_peak = rss
                    except Exception:
                        pass
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            t1 = time.time()

        duration = max(1e-6, t1 - t0)
        profiling['throughput_imgs_per_sec'] = (timed_iters * batch_size) / duration
        profiling['latency_sec'] = duration / timed_iters

        if device.type == 'cuda':
            try:
                profiling['peak_gpu_bytes'] = torch.cuda.max_memory_allocated(device)
            except Exception:
                profiling['peak_gpu_bytes'] = None

        if _PSUTIL_AVAILABLE:
            profiling['peak_cpu_rss_bytes'] = cpu_peak

    except Exception as e:
        print(f"Profiling failed for {model_path}: {e}")

    # 2. Make Predictions
    predictions, ground_truth, all_probs = predict(model, test_loader, device, has_gt)
    
    # 3. Save Predictions (simplified version for now)
    csv_path = os.path.join(output_dir, 'predictions.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(['Filename', 'ClassId'])
        for pred in predictions:
            writer.writerow([pred['filename'], pred['predicted_class']])
    print(f"\nPredictions saved to: {csv_path}")

    # 4. Evaluate if ground truth available
    if has_gt and ground_truth is not None:
        y_pred = [p['predicted_class'] for p in predictions]
        # prepare probability matrix if available
        y_prob = None
        if all_probs is not None and len(all_probs) == len(y_pred):
            try:
                y_prob = np.asarray(all_probs)
            except Exception:
                y_prob = None

        # Calculate core metrics (accuracy, macro and weighted metrics, logloss)
        (accuracy, f1_macro, prec_macro, rec_macro,
         f1_weighted, prec_weighted, rec_weighted, logloss) = calculate_metrics(ground_truth, y_pred, y_prob)

        # Save plots/detailed metrics
        plot_confusion_matrix(ground_truth, y_pred, os.path.join(output_dir, 'confusion_matrix.png'))

        # Detailed metrics text file (extended)
        metrics_path = os.path.join(output_dir, 'evaluation_metrics.txt')
        with open(metrics_path, 'w') as f:
            f.write(f"Model: {model_name_safe}\n")
            f.write(f"Augmentation: {info['augmentation']}\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Macro F1: {f1_macro:.4f}\n")
            f.write(f"Macro Precision: {prec_macro:.4f}\n")
            f.write(f"Macro Recall: {rec_macro:.4f}\n")
            f.write(f"Weighted F1: {f1_weighted:.4f}\n")
            f.write(f"Weighted Precision: {prec_weighted:.4f}\n")
            f.write(f"Weighted Recall: {rec_weighted:.4f}\n")
            if logloss is not None:
                f.write(f"Log Loss: {logloss:.6f}\n")
            else:
                f.write("Log Loss: N/A\n")

        print(f"Evaluation metrics saved to: {metrics_path}")

        info.update({
            'test_accuracy': accuracy,
            'test_macro_f1': f1_macro,
            'test_macro_precision': prec_macro,
            'test_macro_recall': rec_macro,
            'test_weighted_f1': f1_weighted,
            'test_weighted_precision': prec_weighted,
            'test_weighted_recall': rec_weighted,
            'test_log_loss': logloss,
            'status': 'Evaluation Complete',
        })
    else:
        info.update({
            'test_accuracy': 'N/A',
            'test_macro_f1': 'N/A',
            'status': 'Prediction Complete (No GT)',
        })

    # Attach profiling info to returned info
    info.update(profiling)

    print(f"--- Completed Prediction for {model_name_safe} ---\n")
    info['model_filename'] = model_name_safe
    return info

# --- NEW: Comparison Function ---

def compare_results(all_results):
    """
    Compare and summarize results from multiple model predictions.
    """
    # Filter results that have been evaluated
    evaluated_results = [r for r in all_results if r.get('test_accuracy') != 'N/A']
    
    print(f"\n{'='*90}")
    print("MODEL PREDICTION COMPARISON SUMMARY")
    print(f"{'='*90}\n")
    
    if not evaluated_results:
        print("No models were evaluated (Ground Truth file not found or not used).")
        return
        
    # Sort by Test Accuracy (highest first)
    evaluated_results.sort(key=lambda x: x['test_accuracy'], reverse=True)
    
    # Print table header
    print(f"{'Model File':<30} {'Augmentation':<15} {'Val Acc (Train)':<18} {'Test Acc':<12} {'Test F1':<8} {'Test Prec':<10} {'Test Rec':<9} {'W-F1':<8} {'W-Prec':<10} {'W-Rec':<9} {'LogLoss':<10}")
    print(f"{'-'*140}")
    
    # Print results for each model
    for result in evaluated_results:
      val_acc_str = f"{result['accuracy']:.2f}%" if result.get('accuracy') is not None else 'N/A'
      test_acc_str = f"{result['test_accuracy']:.4f}"
      test_f1_str = f"{result.get('test_macro_f1', 'N/A'):.4f}" if isinstance(result.get('test_macro_f1'), (int, float)) else 'N/A'
      test_prec_str = f"{result.get('test_macro_precision', 'N/A'):.4f}" if isinstance(result.get('test_macro_precision'), (int, float)) else 'N/A'
      test_rec_str = f"{result.get('test_macro_recall', 'N/A'):.4f}" if isinstance(result.get('test_macro_recall'), (int, float)) else 'N/A'
      wf1 = result.get('test_weighted_f1')
      wf1_str = f"{wf1:.4f}" if isinstance(wf1, (int, float)) else 'N/A'
      wprec = result.get('test_weighted_precision')
      wprec_str = f"{wprec:.4f}" if isinstance(wprec, (int, float)) else 'N/A'
      wrec = result.get('test_weighted_recall')
      wrec_str = f"{wrec:.4f}" if isinstance(wrec, (int, float)) else 'N/A'
      logloss_val = result.get('test_log_loss')
      logloss_str = f"{logloss_val:.6f}" if isinstance(logloss_val, (int, float)) else 'N/A'

      print(f"{result['model_filename']:<30} "
          f"{result['augmentation']:<15} "
          f"{val_acc_str:<18} "
          f"{test_acc_str:<12} "
          f"{test_f1_str:<8} "
          f"{test_prec_str:<10} "
          f"{test_rec_str:<9} "
          f"{wf1_str:<8} "
          f"{wprec_str:<10} "
          f"{wrec_str:<9} "
          f"{logloss_str:<10}")

    print(f"\n{'='*90}")

    # Save comparison results to a JSON file
    timestamp = re.sub(r'[^\d]', '', str(torch.randint(0, 1000000, (1,)).item())) # Simple timestamp-like ID
    results_file = os.path.join('predictions', f'prediction_summary_{timestamp}.json')
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Detailed prediction results saved to: {results_file}")
    print(f"\nBest Model by Test Accuracy: {evaluated_results[0]['model_filename']} ({evaluated_results[0]['test_accuracy']:.4f})")
    print(f"{'='*90}")


# --- MAIN FUNCTION (Updated for batch processing) ---

def main():
    """Main prediction function."""
    
    # Configuration
    model_glob = './trained_models/best_*.pth' # Wildcard for finding all trained models
    test_path = 'Final_Test/Images'
    gt_path = 'GTSRB_Test_GT.csv'
    output_base_dir = 'predictions'
    batch_size = 64
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    # 1. Find all model files
    model_paths = glob.glob(model_glob)
    
    if not model_paths:
        print(f"\nERROR: No trained models found matching '{model_glob}'.")
        print("Please ensure your models are named like 'best_lenet_model_none.pth' and are in the current directory.")
        return
        
    print(f"\n{'='*60}")
    print(f"Found {len(model_paths)} model(s) for prediction.")
    for i, path in enumerate(model_paths):
        print(f"  {i+1}. {os.path.basename(path)}")
    print(f"{'='*60}")

    # 2. Prepare Test Data (only once)
    print(f"\n{'='*60}")
    print(f"Preparing test dataset...")
    print(f"{'='*60}")

    try:
        # Get the fixed test transform (48x48 resize, normalize)
        test_transform = get_test_transform()
        
        # Use the imported GTSRB_Test_Loader with the required transform
        test_dataset = GTSRB_Test_Loader(TEST_PATH=test_path, TEST_GT_PATH=gt_path, transform=test_transform)
        has_gt = test_dataset.has_gt
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )
        print(f"Test dataset loaded. Ground Truth available: {has_gt}")
    except Exception as e:
        print(f"\nError loading test data: {e}")
        return

    # 3. Predict and Evaluate for Each Model
    all_results = []
    
    for model_path in model_paths:
        result = predict_and_evaluate_single(model_path, test_loader, device, has_gt, output_base_dir)
        all_results.append(result)
    
    # 4. Compare and Summarize Results
    if len(all_results) > 1 and has_gt:
        compare_results(all_results)
    elif has_gt:
        print("\nOnly one model found, skipping comparison summary.")
    elif not has_gt:
        print("\nEvaluation metrics are not available as the ground truth file was not found.")
        compare_results(all_results) # Still print the summary table of predictions

    # Write prediction profiling summary to records/predicting_results.txt and print
    os.makedirs('records', exist_ok=True)

    # Nicely formatted prediction summary table
    header_title = f"Prediction Summary - {time.strftime('%Y-%m-%d %H:%M:%S')}"
    col_fmt = "{:<30} {:>12} {:>16} {:>10} {:>14} {:>12} {:>12} {:>10} {:>8} {:>8} {:>8} {:>8} {:>10} {:<20}"
    cols = ["Model", "Latency(ms)", "Throughput(img/s)", "FLOPs(G)", "ModelSize(MB)", "PeakGPU_MB", "PeakCPU_MB",
        "TestAcc", "TestF1", "TestPrec", "TestRec", "W-F1", "W-Prec", "W-Rec", "LogLoss", "Status"]

    summary_lines = [header_title, col_fmt.format(*cols), '-' * 150]
    for r in all_results:
        name = r.get('model_filename', r.get('model', 'unknown'))
        latency = r.get('latency_sec')
        latency_str = f"{latency*1000:.3f}" if isinstance(latency, (int, float)) else 'N/A'
        thr = r.get('throughput_imgs_per_sec')
        thr_str = f"{thr:.2f}" if isinstance(thr, (int, float)) else 'N/A'
        flops = r.get('flops_macs')
        flops_str = f"{(flops/1e9):.3f}" if isinstance(flops, (int, float)) else 'N/A'
        size = r.get('model_size_bytes')
        size_str = f"{(size/(1024**2)):.2f}" if isinstance(size, (int, float)) else 'N/A'
        peak_gpu = r.get('peak_gpu_bytes')
        peak_gpu_str = f"{(peak_gpu/(1024**2)):.1f}" if isinstance(peak_gpu, (int, float)) else 'N/A'
        peak_cpu = r.get('peak_cpu_rss_bytes')
        peak_cpu_str = f"{(peak_cpu/(1024**2)):.1f}" if isinstance(peak_cpu, (int, float)) else 'N/A'

        # Test metrics (macro and weighted)
        test_acc = r.get('test_accuracy')
        test_acc_str = f"{test_acc:.4f}" if isinstance(test_acc, (int, float)) else 'N/A'
        f1 = r.get('test_macro_f1')
        f1_str = f"{f1:.4f}" if isinstance(f1, (int, float)) else 'N/A'
        prec = r.get('test_macro_precision')
        prec_str = f"{prec:.4f}" if isinstance(prec, (int, float)) else 'N/A'
        rec = r.get('test_macro_recall')
        rec_str = f"{rec:.4f}" if isinstance(rec, (int, float)) else 'N/A'

        wf1 = r.get('test_weighted_f1')
        wf1_str = f"{wf1:.4f}" if isinstance(wf1, (int, float)) else 'N/A'
        wprec = r.get('test_weighted_precision')
        wprec_str = f"{wprec:.4f}" if isinstance(wprec, (int, float)) else 'N/A'
        wrec = r.get('test_weighted_recall')
        wrec_str = f"{wrec:.4f}" if isinstance(wrec, (int, float)) else 'N/A'

        logl = r.get('test_log_loss')
        logl_str = f"{logl:.6f}" if isinstance(logl, (int, float)) else 'N/A'

        status = r.get('status', 'N/A')

        line = col_fmt.format(name, latency_str, thr_str, flops_str, size_str, peak_gpu_str, peak_cpu_str,
                      test_acc_str, f1_str, prec_str, rec_str, wf1_str, wprec_str, wrec_str, logl_str, status)
        summary_lines.append(line)

    summary_text = "\n".join(summary_lines)
    print(summary_text)
    with open('records/predicting_results.txt', 'w') as f:
        f.write(summary_text + "\n")

    print(f"\n✓ BATCH PREDICTION AND EVALUATION COMPLETED! All outputs are in the '{output_base_dir}/' directory.")

if __name__ == '__main__':
    main()