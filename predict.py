"""
German Traffic Sign Recognition - Prediction Script
===================================================
This script loads a trained LeNet model and makes predictions on test images.

Directory Structure Required:
    project_folder/
    ├── predict.py (this file)
    ├── best_lenet_model.pth (trained model)
    ├── Final_Test/
    │   └── Images/
    │       ├── 00000.ppm
    │       ├── 00001.ppm
    │       └── ...
    ├── GT-final_test.csv (ground truth labels - OPTIONAL)
    └── predictions/  (will be created automatically)
        ├── predictions.csv
        ├── predictions_detailed.txt
        ├── confusion_matrix.png (if ground truth available)
        └── evaluation_metrics.txt (if ground truth available)

How to Run:
    python predict.py

Output:
    - predictions/predictions.csv: CSV file with predictions (for evaluation)
    - predictions/predictions_detailed.txt: Human-readable predictions with confidence
    - predictions/confusion_matrix.png: Visual confusion matrix (if GT available)
    - predictions/evaluation_metrics.txt: Accuracy, precision, recall, F1 (if GT available)

Requirements:
    - PyTorch
    - torchvision
    - PIL
    - numpy
    - tqdm
    - scikit-learn
    - matplotlib
    - seaborn
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import csv
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support


class LeNet(nn.Module):
    """
    LeNet-based CNN for Traffic Sign Classification.
    
    This must match the architecture used in train.py exactly!
    
    Architecture:
        - Conv1: 3->16 channels, 3x3 kernel
        - Conv2: 16->32 channels, 3x3 kernel
        - Conv3: 32->64 channels, 3x3 kernel
        - FC1: 1024->128
        - FC2: 128->43 classes
    """
    
    def __init__(self, num_classes=43):
        super(LeNet, self).__init__()
        
        # Convolutional layers (same as training)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Pooling and activation
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Dropout (will be disabled in eval mode)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        """Forward pass through the network."""
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class GTSRBTestDataset(Dataset):
    """
    Dataset class for loading test images.
    
    Args:
        root_dir (str): Root directory containing test images
        transform (callable, optional): Transform to apply to images
    
    Returns:
        tuple: (image_tensor, image_filename, image_index)
    """
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        
        # Load test images
        images_dir = os.path.join(root_dir, 'Images')
        
        if not os.path.exists(images_dir):
            raise ValueError(f"Test directory not found: {images_dir}")
        
        # Load all PPM images in sorted order
        for img_name in sorted(os.listdir(images_dir)):
            if img_name.endswith('.ppm'):
                img_path = os.path.join(images_dir, img_name)
                self.images.append((img_path, img_name))
        
        if len(self.images) == 0:
            raise ValueError(f"No PPM images found in {images_dir}")
        
        print(f"Loaded {len(self.images)} test images from {root_dir}")
    
    def __len__(self):
        """Return total number of test images."""
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Get a single test image.
        
        Args:
            idx (int): Index of image to retrieve
        
        Returns:
            tuple: (transformed_image, filename, index)
        """
        img_path, img_name = self.images[idx]
        
        # Load and convert image to RGB
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms (resize and normalize)
        if self.transform:
            image = self.transform(image)
        
        # Extract image number from filename (e.g., "00000.ppm" -> 0)
        img_index = int(img_name.split('.')[0])
        
        return image, img_name, img_index


def load_model(model_path, device):
    """
    Load a trained model from checkpoint.
    
    Args:
        model_path (str): Path to the saved model (.pth file)
        device: Device to load model on (cuda or cpu)
    
    Returns:
        nn.Module: Loaded model in evaluation mode
    """
    print(f"\nLoading model from: {model_path}")
    
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Initialize model architecture
    model = LeNet(num_classes=43).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Load model weights
    if 'model_state_dict' in checkpoint:
        # If saved as dictionary with metadata
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'accuracy' in checkpoint:
            print(f"Model accuracy on validation: {checkpoint['accuracy']:.2f}%")
        if 'epoch' in checkpoint:
            print(f"Model trained for {checkpoint['epoch']+1} epochs")
    else:
        # If saved as state_dict only
        model.load_state_dict(checkpoint)
    
    # Set model to evaluation mode
    # This disables dropout and sets batch norm to use running statistics
    model.eval()
    
    print("Model loaded successfully!")
    return model


def predict(model, test_loader, device):
    """
    Make predictions on test dataset.
    
    Args:
        model (nn.Module): Trained model in eval mode
        test_loader (DataLoader): DataLoader for test images
        device: Device to run predictions on
    
    Returns:
        list: List of tuples (filename, predicted_class, confidence, all_probabilities)
    """
    predictions = []
    
    # Disable gradient computation for faster inference
    with torch.no_grad():
        for images, filenames, indices in tqdm(test_loader, desc='Predicting'):
            # Move images to device
            images = images.to(device)
            
            # Forward pass to get logits
            outputs = model(images)
            
            # Apply softmax to get probabilities
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get predicted class and confidence
            confidences, predicted = probabilities.max(1)
            
            # Store predictions for each image in batch
            for i in range(len(filenames)):
                predictions.append({
                    'filename': filenames[i],
                    'index': indices[i].item(),
                    'predicted_class': predicted[i].item(),
                    'confidence': confidences[i].item(),
                    'probabilities': probabilities[i].cpu().numpy()
                })
    
    # Sort predictions by image index to maintain order
    predictions.sort(key=lambda x: x['index'])
    
    return predictions


def save_predictions(predictions, output_dir):
    """
    Save predictions to CSV and text files.
    
    Args:
        predictions (list): List of prediction dictionaries
        output_dir (str): Directory to save predictions
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # ===== Save CSV file (for evaluation.py) =====
    csv_path = os.path.join(output_dir, 'predictions.csv')
    print(f"\nSaving predictions to: {csv_path}")
    
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        
        # Write header
        writer.writerow(['Filename', 'ClassId'])
        
        # Write predictions
        for pred in predictions:
            writer.writerow([pred['filename'], pred['predicted_class']])
    
    print(f"Saved {len(predictions)} predictions to CSV")
    
    # ===== Save detailed text file (human-readable) =====
    txt_path = os.path.join(output_dir, 'predictions_detailed.txt')
    print(f"Saving detailed predictions to: {txt_path}")
    
    with open(txt_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("German Traffic Sign Recognition - Detailed Predictions\n")
        f.write("=" * 80 + "\n\n")
        
        for pred in predictions:
            f.write(f"Image: {pred['filename']}\n")
            f.write(f"  Predicted Class: {pred['predicted_class']}\n")
            f.write(f"  Confidence: {pred['confidence']:.4f} ({pred['confidence']*100:.2f}%)\n")
            
            # Show top-3 predictions
            top3_indices = pred['probabilities'].argsort()[-3:][::-1]
            f.write(f"  Top 3 predictions:\n")
            for rank, idx in enumerate(top3_indices, 1):
                f.write(f"    {rank}. Class {idx}: {pred['probabilities'][idx]:.4f} "
                       f"({pred['probabilities'][idx]*100:.2f}%)\n")
            f.write("\n")
    
    print(f"Saved detailed predictions to text file")
    
    # ===== Print summary statistics =====
    print("\n" + "=" * 60)
    print("Prediction Summary")
    print("=" * 60)


def load_ground_truth(gt_path='GTSRB_Test_GT.csv'):
    """
    Load ground truth labels from CSV file.
    
    This function tries multiple strategies to find ground truth:
    1. Look for GTSRB_Test_GT.csv in root folder (MAIN FILE)
    2. Look for GT files in test directory
    3. Look for GT-XXXXX.csv files in each class folder (training format)
    
    Args:
        gt_path (str): Path to ground truth CSV file
    
    Returns:
        dict: Dictionary mapping filename to true class ID
              Returns None if file doesn't exist
    
    CSV Format (Test):
        Filename;Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2;ClassId
        00000.ppm;53;54;6;5;48;49;16
    
    CSV Format (Training - per class folder):
        Final_Training/Images/00000/GT-00000.csv:
        Filename;Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2;ClassId
        00000_00000.ppm;29;30;5;6;24;25;0
    """
    # ===== STRATEGY 1: Look for GTSRB_Test_GT.csv in root =====
    primary_paths = [
        'GTSRB_Test_GT.csv',  # Root folder (MAIN FILE)
        gt_path,              # User-specified path
    ]
    
    for path in primary_paths:
        if os.path.exists(path):
            print(f"\nFound ground truth file: {path}")
            result = _load_single_gt_file(path)
            if result is not None:
                return result
    
    # ===== STRATEGY 2: Look for GT files in test directory =====
    test_paths = [
        'Final_Test/GT-final_test.csv',
        'Final_Test/GT-final_test.test.csv',
        'Final_Test/Images/GT-final_test.csv',
        'Final_Test/Images/GT-final_test.test.csv',
        'GT-final_test.csv',
        'GT-final_test.test.csv'
    ]
    
    for path in test_paths:
        if os.path.exists(path):
            print(f"\nFound ground truth file: {path}")
            result = _load_single_gt_file(path)
            if result is not None:
                return result
    
    # ===== STRATEGY 3: Look for GT files in training folders =====
    print(f"\nNo single GT file found. Checking training folder structure...")
    training_path = 'Final_Training/Images'
    
    if os.path.exists(training_path):
        print(f"Found training directory: {training_path}")
        print(f"Attempting to load GT from individual class folders...")
        return _load_training_gt_files(training_path)
    
    # ===== NO GROUND TRUTH FOUND =====
    print(f"\nNote: Ground truth not found in any location.")
    print(f"Tried:")
    print(f"  - GTSRB_Test_GT.csv (MAIN TEST GT FILE)")
    for path in test_paths:
        print(f"  - {path}")
    print(f"  - {training_path}/XXXXX/GT-XXXXX.csv")
    print("\nSkipping evaluation metrics. Predictions will still be saved.")
    return None


def _load_single_gt_file(gt_file):
    """
    Load ground truth from a single CSV file.
    
    Args:
        gt_file (str): Path to ground truth CSV file
    
    Returns:
        dict: Filename -> ClassId mapping, or None if error
    """
    ground_truth = {}
    
    try:
        with open(gt_file, 'r') as f:
            reader = csv.reader(f, delimiter=';')
            header = next(reader)  # Read header
            
            # Check if ClassId column exists
            if 'ClassId' not in header:
                print(f"Warning: File '{gt_file}' does NOT contain ClassId column.")
                print(f"Header found: {';'.join(header)}")
                print(f"Cannot perform evaluation without ClassId labels.")
                return None
            
            # Find ClassId column index
            classid_idx = header.index('ClassId')
            
            # Read data rows
            for row in reader:
                if len(row) > classid_idx:
                    filename = row[0]  # First column: Filename
                    class_id = int(row[classid_idx])  # ClassId column
                    ground_truth[filename] = class_id
        
        print(f"Loaded ground truth for {len(ground_truth)} images from {gt_file}")
        return ground_truth
    
    except Exception as e:
        print(f"Error loading ground truth from {gt_file}: {e}")
        return None


def _load_training_gt_files(training_path):
    """
    Load ground truth from multiple GT-XXXXX.csv files in training folders.
    
    This is used when test GT is not available, to demonstrate functionality
    on training data.
    
    Args:
        training_path (str): Path to Final_Training/Images directory
    
    Returns:
        dict: Filename -> ClassId mapping, or None if error
    """
    ground_truth = {}
    files_loaded = 0
    
    try:
        # Iterate through all class folders (00000 to 00042)
        for class_folder in sorted(os.listdir(training_path)):
            class_path = os.path.join(training_path, class_folder)
            
            # Check if it's a directory and matches class folder pattern
            if os.path.isdir(class_path) and class_folder.isdigit():
                # Look for GT-XXXXX.csv file in this folder
                gt_file = os.path.join(class_path, f'GT-{class_folder}.csv')
                
                if os.path.exists(gt_file):
                    # Load this GT file
                    with open(gt_file, 'r') as f:
                        reader = csv.reader(f, delimiter=';')
                        header = next(reader)  # Skip header
                        
                        for row in reader:
                            if len(row) >= 8:
                                filename = row[0]  # Filename (e.g., 00000_00000.ppm)
                                class_id = int(row[7])  # ClassId
                                ground_truth[filename] = class_id
                    
                    files_loaded += 1
        
        if files_loaded > 0:
            print(f"Loaded ground truth from {files_loaded} class folders")
            print(f"Total images with ground truth: {len(ground_truth)}")
            return ground_truth
        else:
            print(f"No GT-XXXXX.csv files found in {training_path}")
            return None
    
    except Exception as e:
        print(f"Error loading training ground truth: {e}")
        return None


def plot_confusion_matrix(y_true, y_pred, output_dir, num_classes=43):
    """
    Create and save a confusion matrix visualization.
    
    Args:
        y_true (list): True class labels
        y_pred (list): Predicted class labels
        output_dir (str): Directory to save the plot
        num_classes (int): Total number of classes
    
    A confusion matrix shows:
    - Rows: True class (what it actually is)
    - Columns: Predicted class (what model predicted)
    - Diagonal: Correct predictions
    - Off-diagonal: Errors (misclassifications)
    """
    print("\nGenerating confusion matrix...")
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    
    # Create figure
    plt.figure(figsize=(20, 18))
    
    # Plot confusion matrix as heatmap
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                square=True, cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix - Traffic Sign Classification\n' + 
              '(Rows=True Class, Columns=Predicted Class)', fontsize=16, pad=20)
    plt.ylabel('True Class', fontsize=14)
    plt.xlabel('Predicted Class', fontsize=14)
    
    # Save plot
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to: {cm_path}")
    
    return cm


def calculate_metrics(y_true, y_pred, output_dir, num_classes=43):
    """
    Calculate and save comprehensive evaluation metrics.
    
    Metrics calculated:
    - Accuracy: Overall correctness (correct predictions / total predictions)
    - Precision: Of all predicted as class X, how many were actually class X
    - Recall: Of all actual class X, how many were correctly identified
    - F1-Score: Harmonic mean of precision and recall (balanced measure)
    
    Args:
        y_true (list): True class labels
        y_pred (list): Predicted class labels
        output_dir (str): Directory to save metrics
        num_classes (int): Total number of classes
    """
    print("\nCalculating evaluation metrics...")
    
    # ===== Overall Accuracy =====
    accuracy = accuracy_score(y_true, y_pred)
    
    # ===== Per-class metrics =====
    # average=None gives per-class scores
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(num_classes), average=None, zero_division=0
    )
    
    # ===== Macro-averaged metrics (treats all classes equally) =====
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=range(num_classes), average='macro', zero_division=0
    )
    
    # ===== Weighted-averaged metrics (weighted by class frequency) =====
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=range(num_classes), average='weighted', zero_division=0
    )
    
    # ===== Save metrics to file =====
    metrics_path = os.path.join(output_dir, 'evaluation_metrics.txt')
    
    with open(metrics_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("EVALUATION METRICS - German Traffic Sign Recognition\n")
        f.write("=" * 80 + "\n\n")
        
        # Overall metrics
        f.write("OVERALL PERFORMANCE\n")
        f.write("-" * 80 + "\n")
        f.write(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        f.write(f"  → {int(accuracy*len(y_true))} out of {len(y_true)} predictions correct\n\n")
        
        # Macro-averaged metrics
        f.write("MACRO-AVERAGED METRICS (all classes weighted equally)\n")
        f.write("-" * 80 + "\n")
        f.write(f"Precision: {precision_macro:.4f} ({precision_macro*100:.2f}%)\n")
        f.write(f"Recall:    {recall_macro:.4f} ({recall_macro*100:.2f}%)\n")
        f.write(f"F1-Score:  {f1_macro:.4f} ({f1_macro*100:.2f}%)\n\n")
        
        # Weighted-averaged metrics
        f.write("WEIGHTED-AVERAGED METRICS (weighted by class frequency)\n")
        f.write("-" * 80 + "\n")
        f.write(f"Precision: {precision_weighted:.4f} ({precision_weighted*100:.2f}%)\n")
        f.write(f"Recall:    {recall_weighted:.4f} ({recall_weighted*100:.2f}%)\n")
        f.write(f"F1-Score:  {f1_weighted:.4f} ({f1_weighted*100:.2f}%)\n\n")
        
        # Explanation of metrics
        f.write("WHAT DO THESE METRICS MEAN?\n")
        f.write("-" * 80 + "\n")
        f.write("• Accuracy: Percentage of correct predictions overall\n")
        f.write("  - How often is the model right?\n\n")
        
        f.write("• Precision: Of all images predicted as class X, how many were truly X?\n")
        f.write("  - High precision = few false alarms\n")
        f.write("  - Low precision = many false positives\n\n")
        
        f.write("• Recall: Of all images that are truly class X, how many were found?\n")
        f.write("  - High recall = finds most examples\n")
        f.write("  - Low recall = misses many examples\n\n")
        
        f.write("• F1-Score: Balance between precision and recall\n")
        f.write("  - Good when you need both precision and recall\n")
        f.write("  - F1 = 2 × (Precision × Recall) / (Precision + Recall)\n\n")
        
        # Per-class metrics
        f.write("=" * 80 + "\n")
        f.write("PER-CLASS METRICS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'Class':<8} {'Support':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n")
        f.write("-" * 80 + "\n")
        
        for class_id in range(num_classes):
            f.write(f"{class_id:<8} {int(support[class_id]):<10} "
                   f"{precision[class_id]:.4f} ({precision[class_id]*100:5.1f}%)  "
                   f"{recall[class_id]:.4f} ({recall[class_id]*100:5.1f}%)  "
                   f"{f1[class_id]:.4f} ({f1[class_id]*100:5.1f}%)\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("Support: Number of true instances for each class in the test set\n")
        f.write("=" * 80 + "\n")
    
    print(f"Evaluation metrics saved to: {metrics_path}")
    
    # ===== Print summary to console =====
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\n{'Metric':<20} {'Score':<15} {'Percentage'}")
    print("-" * 60)
    print(f"{'Accuracy':<20} {accuracy:.4f}          {accuracy*100:.2f}%")
    print(f"{'Precision (macro)':<20} {precision_macro:.4f}          {precision_macro*100:.2f}%")
    print(f"{'Recall (macro)':<20} {recall_macro:.4f}          {recall_macro*100:.2f}%")
    print(f"{'F1-Score (macro)':<20} {f1_macro:.4f}          {f1_macro*100:.2f}%")
    print("=" * 60)
    
    print("\nMetric Explanations:")
    print("  • Accuracy:  Overall correctness of predictions")
    print("  • Precision: How many predicted positives are actually positive")
    print("  • Recall:    How many actual positives were correctly found")
    print("  • F1-Score:  Balanced measure of precision and recall")
    print("=" * 60)
    
    return {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro
    }
    
    # Calculate average confidence
    # Confidence = the probability assigned to the predicted class
    # Higher confidence means the model is more certain about its prediction
    avg_confidence = sum(p['confidence'] for p in predictions) / len(predictions)
    print(f"Total predictions: {len(predictions)}")
    print(f"Average confidence: {avg_confidence:.4f} ({avg_confidence*100:.2f}%)")
    print(f"\nWhat does confidence mean?")
    print(f"  - Confidence is the probability the model assigns to its prediction")
    print(f"  - {avg_confidence*100:.2f}% means on average, the model is {avg_confidence*100:.2f}% sure")
    print(f"  - High confidence (>90%) = model is very certain")
    print(f"  - Low confidence (<70%) = model is uncertain, might be wrong")
    
    # Count predictions per class
    class_counts = {}
    for pred in predictions:
        class_id = pred['predicted_class']
        class_counts[class_id] = class_counts.get(class_id, 0) + 1
    
    print(f"\nNumber of unique classes predicted: {len(class_counts)}")
    print(f"\nTop 5 most common predictions:")
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    for class_id, count in sorted_classes[:5]:
        print(f"  Class {class_id}: {count} images ({count/len(predictions)*100:.1f}%)")
    
    print("=" * 60)


def main():
    """
    Main prediction function.
    
    Steps:
    1. Configure device (CUDA/CPU)
    2. Load test dataset
    3. Load trained model
    4. Make predictions
    5. Save results to output folder
    6. If ground truth available: generate confusion matrix and metrics
    """
    
    # ===== CONFIGURATION =====
    model_path = 'best_lenet_model.pth'       # Path to trained model
    test_dir = 'Final_Test'                   # Test data directory
    output_dir = 'predictions'                # Output directory for results
    gt_path = 'GTSRB_Test_GT.csv'             # Ground truth labels (MAIN FILE)
    batch_size = 64                           # Batch size for prediction
    
    # ===== DEVICE CONFIGURATION =====
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    
    # ===== DATA TRANSFORMS =====
    # Use same transforms as validation (no augmentation)
    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize to LeNet input size
        transforms.ToTensor(),         # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize
                           std=[0.229, 0.224, 0.225])
    ])
    
    # ===== LOAD TEST DATASET =====
    print("\n" + "=" * 60)
    print("Loading test dataset...")
    print("=" * 60)
    
    try:
        test_dataset = GTSRBTestDataset(test_dir, transform=test_transform)
    except (ValueError, FileNotFoundError) as e:
        print(f"\nError: {e}")
        print("\nPlease ensure:")
        print(f"  1. '{test_dir}' directory exists")
        print(f"  2. '{test_dir}/Images/' contains PPM images")
        return
    
    # Create DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Keep original order
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # ===== LOAD TRAINED MODEL =====
    print("\n" + "=" * 60)
    print("Loading trained model...")
    print("=" * 60)
    
    try:
        model = load_model(model_path, device)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease ensure:")
        print(f"  1. You have trained a model using train.py")
        print(f"  2. '{model_path}' exists in the current directory")
        return
    
    # ===== MAKE PREDICTIONS =====
    print("\n" + "=" * 60)
    print("Making predictions...")
    print("=" * 60)
    
    predictions = predict(model, test_loader, device)
    
    # ===== SAVE RESULTS =====
    print("\n" + "=" * 60)
    print("Saving results...")
    print("=" * 60)
    
    save_predictions(predictions, output_dir)
    
    # ===== LOAD GROUND TRUTH AND EVALUATE (if available) =====
    ground_truth = load_ground_truth(gt_path)
    
    if ground_truth is not None:
        print("\n" + "=" * 60)
        print("Evaluating predictions against ground truth...")
        print("=" * 60)
        
        # Match predictions with ground truth
        y_true = []
        y_pred = []
        matched_count = 0
        unmatched_count = 0
        
        for pred in predictions:
            filename = pred['filename']
            if filename in ground_truth:
                y_true.append(ground_truth[filename])
                y_pred.append(pred['predicted_class'])
                matched_count += 1
            else:
                unmatched_count += 1
        
        print(f"\nMatched {matched_count} predictions with ground truth labels")
        if unmatched_count > 0:
            print(f"Warning: {unmatched_count} predictions had no matching ground truth")
        
        if len(y_true) > 0:
            # Generate confusion matrix
            cm = plot_confusion_matrix(y_true, y_pred, output_dir)
            
            # Calculate and save metrics
            metrics = calculate_metrics(y_true, y_pred, output_dir)
        else:
            print("\nError: No matching predictions found in ground truth file")
            print("Check that filenames in predictions match ground truth filenames")
    
    # ===== COMPLETION MESSAGE =====
    print("\n" + "=" * 60)
    print("✓ Prediction completed successfully!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  1. {output_dir}/predictions.csv - For evaluation")
    print(f"  2. {output_dir}/predictions_detailed.txt - Detailed results")
    
    if ground_truth is not None:
        print(f"  3. {output_dir}/confusion_matrix.png - Visual confusion matrix")
        print(f"  4. {output_dir}/evaluation_metrics.txt - Comprehensive metrics")
        print("\nEvaluation complete! Check the metrics file for detailed performance.")
    else:
        print("\nNote: No ground truth file found. Only predictions saved.")
        print(f"To enable evaluation, place '{gt_path}' in the current directory.")
    
    print("=" * 60 + "\n")


if __name__ == '__main__':
    # Entry point
    main()