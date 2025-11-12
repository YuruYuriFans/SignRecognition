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
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import glob
import re
import json

# --- Import Refactored Components ---
from model import LeNet
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
    
    # Initialize model with 43 classes (default for GTSRB)
    model = LeNet(num_classes=43).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    
    checkpoint_info = {
        'accuracy': checkpoint.get('accuracy'),
        'augmentation': checkpoint.get('augmentation', 'N/A'),
        'input_size': checkpoint.get('input_size', 48)
    }
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint) # Support older save formats
    
    model.eval()
    return model, checkpoint_info


def predict(model, test_loader, device, has_gt=False):
    """
    Make predictions on test dataset. (Same as original)
    """
    predictions = []
    all_labels = [] if has_gt else None
    
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
    
    return predictions, all_labels


def calculate_metrics(y_true, y_pred):
    """Calculate core metrics (accuracy, macro F1)."""
    accuracy = accuracy_score(y_true, y_pred)
    
    # Calculate macro-averaged F1 score
    _, _, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=range(43), average='macro', zero_division=0
    )
    
    return accuracy, f1_macro


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
    
    # 2. Make Predictions
    predictions, ground_truth = predict(model, test_loader, device, has_gt)
    
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
        
        # Calculate core metrics
        accuracy, f1_macro = calculate_metrics(ground_truth, y_pred)
        
        # Save plots/detailed metrics
        plot_confusion_matrix(ground_truth, y_pred, os.path.join(output_dir, 'confusion_matrix.png'))
        
        # Detailed metrics text file (re-using original logic but simplifying for brevity)
        metrics_path = os.path.join(output_dir, 'evaluation_metrics.txt')
        with open(metrics_path, 'w') as f:
            f.write(f"Model: {model_name_safe}\n")
            f.write(f"Augmentation: {info['augmentation']}\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Macro F1: {f1_macro:.4f}\n")
        
        print(f"Evaluation metrics saved to: {metrics_path}")
        
        info.update({
            'test_accuracy': accuracy,
            'test_macro_f1': f1_macro,
            'status': 'Evaluation Complete',
        })
    else:
        info.update({
            'test_accuracy': 'N/A',
            'test_macro_f1': 'N/A',
            'status': 'Prediction Complete (No GT)',
        })

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
    print(f"{'Model File':<30} {'Augmentation':<15} {'Val Acc (Train)':<18} {'Test Acc':<12} {'Test F1':<10}")
    print(f"{'-'*90}")
    
    # Print results for each model
    for result in evaluated_results:
        val_acc_str = f"{result['accuracy']:.2f}%" if result.get('accuracy') is not None else 'N/A'
        test_acc_str = f"{result['test_accuracy']:.4f}"
        test_f1_str = f"{result['test_macro_f1']:.4f}"
        
        print(f"{result['model_filename']:<30} "
              f"{result['augmentation']:<15} "
              f"{val_acc_str:<18} "
              f"{test_acc_str:<12} "
              f"{test_f1_str:<10}")

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
    model_glob = 'best_lenet_model_*.pth' # Wildcard for finding all trained models
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
    
    print(f"\n✓ BATCH PREDICTION AND EVALUATION COMPLETED! All outputs are in the '{output_base_dir}/' directory.")

if __name__ == '__main__':
    main()