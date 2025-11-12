"""
dataset.py - Dataset Classes for GTSRB
=======================================
Contains dataset definitions and data loading utilities.

Classes:
    - GTSRBDataset: Training dataset loader
    - GTSRB_Test_Loader: Test dataset loader with ground truth
"""

import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class GTSRBDataset(Dataset):
    """
    Custom PyTorch Dataset for GTSRB training data.
    
    Loads images from the hierarchical directory structure where each
    class has its own folder (00000 to 00042).
    
    Directory Structure:
        root_dir/Images/
            00000/
                00000_00000.ppm
                00000_00001.ppm
                ...
            00001/
                00001_00000.ppm
                ...
            ...
            00042/
    
    Args:
        root_dir (str): Root directory containing Images folder
        transform (callable, optional): Transform to apply to images
        is_train (bool): Whether this is training data (default: True)
    
    Returns:
        tuple: (image, label) where image is transformed PIL Image
    """
    
    def __init__(self, root_dir, transform=None, is_train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        self.images = []
        self.labels = []
        
        if is_train:
            images_dir = os.path.join(root_dir, 'Images')
            
            if not os.path.exists(images_dir):
                raise ValueError(f"Images directory not found: {images_dir}")
            
            # Load images from each class folder (00000 to 00042)
            for class_id in range(43):
                class_dir = os.path.join(images_dir, f'{class_id:05d}')
                
                if os.path.exists(class_dir):
                    for img_name in os.listdir(class_dir):
                        if img_name.endswith('.ppm'):
                            img_path = os.path.join(class_dir, img_name)
                            self.images.append(img_path)
                            self.labels.append(class_id)
                else:
                    print(f"Warning: Class directory not found: {class_dir}")
    
    def __len__(self):
        """Return total number of images."""
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Get a single image and its label.
        
        Args:
            idx (int): Index of the image
        
        Returns:
            tuple: (transformed_image, label)
        """
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        return image, label


class GTSRB_Test_Loader(Dataset):
    """
    Dataset loader for GTSRB test images with ground truth labels.
    
    Loads test images and their labels from a CSV file. Can work with
    or without ground truth labels.
    
    CSV Format:
        Filename;Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2;ClassId
        00000.ppm;53;54;6;5;48;49;16
        00001.ppm;39;40;5;4;35;36;1
        ...
    
    Args:
        TEST_PATH (str): Path to directory containing test images
        TEST_GT_PATH (str, optional): Path to ground truth CSV file
        transform (callable, optional): Transform to apply to images
    
    Returns:
        tuple: (image, label, filename) if ground truth available
               (image, filename, index) otherwise
    """
    
    def __init__(self, TEST_PATH='Final_Test/Images', 
                 TEST_GT_PATH='GTSRB_Test_GT.csv',
                 transform=None):
        self.TEST_PATH = TEST_PATH
        self.transform = transform
        self.has_gt = False
        
        # Try to load ground truth
        if TEST_GT_PATH and os.path.exists(TEST_GT_PATH):
            try:
                self.df = pd.read_csv(TEST_GT_PATH, sep=';')
                self.has_gt = 'ClassId' in self.df.columns
                
                if self.has_gt:
                    print(f"Loaded {len(self.df)} test images with ground truth")
                else:
                    print(f"Loaded {len(self.df)} test images (no ClassId column)")
            except Exception as e:
                print(f"Warning: Could not load ground truth: {e}")
                self.has_gt = False
        
        # If no ground truth, scan directory
        if not self.has_gt or not hasattr(self, 'df'):
            print(f"Loading test images from {TEST_PATH}")
            if not os.path.exists(TEST_PATH):
                raise ValueError(f"Test directory not found: {TEST_PATH}")
            
            image_files = sorted([f for f in os.listdir(TEST_PATH) 
                                if f.endswith('.ppm')])
            self.df = pd.DataFrame({'Filename': image_files})
            print(f"Loaded {len(self.df)} test images")
    
    def __len__(self):
        """Return total number of test images."""
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Get a single test image with optional label.
        
        Args:
            idx (int): Index of the image
        
        Returns:
            tuple: (image, label, filename) if ground truth available
                   (image, filename, index) otherwise
        """
        row = self.df.iloc[idx]
        
        # Load image
        filename = os.path.join(self.TEST_PATH, row['Filename'])
        image = Image.open(filename).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Return with or without label
        if self.has_gt:
            label = int(row['ClassId'])
            return image, label, row['Filename']
        else:
            return image, row['Filename'], idx


def get_dataset_info(root_dir='Final_Training'):
    """
    Get information about the GTSRB training dataset.
    
    Args:
        root_dir (str): Root directory of training data
    
    Returns:
        dict: Dataset statistics
    """
    images_dir = os.path.join(root_dir, 'Images')
    
    if not os.path.exists(images_dir):
        raise ValueError(f"Images directory not found: {images_dir}")
    
    class_counts = {}
    total_images = 0
    
    for class_id in range(43):
        class_dir = os.path.join(images_dir, f'{class_id:05d}')
        if os.path.exists(class_dir):
            count = len([f for f in os.listdir(class_dir) if f.endswith('.ppm')])
            class_counts[class_id] = count
            total_images += count
    
    return {
        'total_images': total_images,
        'num_classes': len(class_counts),
        'class_counts': class_counts,
        'min_samples': min(class_counts.values()) if class_counts else 0,
        'max_samples': max(class_counts.values()) if class_counts else 0,
        'avg_samples': total_images / len(class_counts) if class_counts else 0
    }


if __name__ == '__main__':
    # Test dataset loading
    print("Testing GTSRB Dataset...")
    
    try:
        info = get_dataset_info('Final_Training')
        print(f"\nDataset Information:")
        print(f"  Total images: {info['total_images']}")
        print(f"  Number of classes: {info['num_classes']}")
        print(f"  Samples per class:")
        print(f"    Min: {info['min_samples']}")
        print(f"    Max: {info['max_samples']}")
        print(f"    Avg: {info['avg_samples']:.1f}")
    except Exception as e:
        print(f"Error: {e}")