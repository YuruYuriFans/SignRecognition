"""
augmentation.py - Data Augmentation Strategies
===============================================
Contains predefined augmentation transforms for GTSRB.

Functions:
    - get_augmentation_transforms: Get train/val transforms by strategy
    - get_available_strategies: List all available strategies
"""

from torchvision import transforms


def get_augmentation_transforms(aug_type='basic', input_size=48):
    """
    Get data augmentation transforms based on strategy.
    
    Strategies:
        - 'none': Only resize and normalize (baseline)
        - 'basic': Rotation and color jitter
        - 'advanced': All transformations including perspective, blur, erasing
    
    Args:
        aug_type (str): Augmentation strategy ('none', 'basic', 'advanced')
        input_size (int): Input image size (default: 48)
    
    Returns:
        tuple: (train_transform, val_transform, description)
    
    Raises:
        ValueError: If aug_type is not recognized
    
    Example:
        train_transform, val_transform, desc = get_augmentation_transforms('basic')
    """
    
    aug_type = aug_type.lower()
    
    if aug_type == 'none':
        train_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        description = "No augmentation (baseline)"
        
    elif aug_type == 'basic':
        train_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        description = "Basic augmentation (rotation, color jitter)"
        
    elif aug_type == 'advanced':
        train_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomRotation(15),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                shear=5
            ),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ], p=0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3))
        ])
        description = "Advanced augmentation (rotation, affine, perspective, color, blur, erasing)"
    
    else:
        raise ValueError(f"Unknown augmentation type: {aug_type}. "
                        f"Available: {get_available_strategies()}")
    
    # Validation transform (always the same, no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    return train_transform, val_transform, description


def get_available_strategies():
    """
    Get list of available augmentation strategies.
    
    Returns:
        list: Available strategy names
    """
    return ['none', 'basic', 'advanced']


def print_augmentation_info():
    """Print information about all augmentation strategies."""
    print("Available Augmentation Strategies:")
    print("="*60)
    
    for strategy in get_available_strategies():
        _, _, desc = get_augmentation_transforms(strategy)
        print(f"\n{strategy.upper()}:")
        print(f"  {desc}")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    # Test augmentation strategies
    print("Testing augmentation strategies...")
    print_augmentation_info()
    
    # Test each strategy
    for strategy in get_available_strategies():
        try:
            train_t, val_t, desc = get_augmentation_transforms(strategy)
            print(f"\n{strategy}: OK")
        except Exception as e:
            print(f"\n{strategy}: ERROR - {e}")