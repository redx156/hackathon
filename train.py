"""
Pneumonia Detection Training Pipeline
=====================================
Fine-tunes pretrained ResNet-18 for binary classification (Normal vs Pneumonia).

Key features:
- Transfer learning: Freezes early layers, trains layer4 + fc only
- Class imbalance handling: BCEWithLogitsLoss with pos_weight
- Recall optimization: Weighted loss penalizes missing pneumonia cases
- Medical-safe augmentations: Mild rotations, flips, contrast normalization
- Grad-CAM compatible: No changes to convolutional structure

Usage:
    python train.py --data_dir /path/to/chest_xrays --epochs 10 --batch_size 32

Expected folder structure:
    data_dir/
        train/
            NORMAL/
            PNEUMONIA/
        val/
            NORMAL/
            PNEUMONIA/
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from pathlib import Path
import json
from datetime import datetime


# ============ CONFIGURATION ============

class Config:
    """Training configuration - modify these for your setup"""
    
    # Model
    MODEL_NAME = "resnet18"
    NUM_CLASSES = 1  # Binary classification (sigmoid output)
    
    # Training
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    EPOCHS = 10
    BATCH_SIZE = 32
    
    # Class imbalance - typical ratio in chest X-ray datasets
    # Higher weight = more penalty for missing pneumonia (false negatives)
    # Adjust based on your dataset's class distribution
    PNEUMONIA_POS_WEIGHT = 2.0  # Increases recall
    
    # Data
    IMAGE_SIZE = 224
    NUM_WORKERS = 4
    
    # Augmentation limits (medical-safe)
    ROTATION_DEGREES = 10  # Mild rotation
    BRIGHTNESS_RANGE = 0.1  # Slight brightness variation
    CONTRAST_RANGE = 0.1  # Slight contrast variation
    
    # Paths
    CHECKPOINT_DIR = "checkpoints"
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============ DATA TRANSFORMS ============

def get_train_transforms(config: Config):
    """
    Medical-safe augmentations for training.
    
    Includes:
    - Mild rotation (±10°): X-rays can have slight positioning differences
    - Horizontal flip: Lungs are roughly symmetric
    - Brightness/contrast: Handles acquisition variations
    - Normalization: ImageNet stats required for pretrained weights
    
    Avoids:
    - Color jitter (X-rays are grayscale)
    - Extreme rotations (anatomically unrealistic)
    - Vertical flips (never clinically relevant)
    """
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=3),  # Ensure 3-channel for ResNet
        transforms.RandomRotation(config.ROTATION_DEGREES),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=config.BRIGHTNESS_RANGE,
            contrast=config.CONTRAST_RANGE
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_val_transforms(config: Config):
    """Validation transforms - no augmentation, just resize and normalize."""
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


# ============ MODEL ============

def create_model(config: Config, freeze_early_layers: bool = True):
    """
    Create ResNet-18 with binary classification head.
    
    Transfer learning strategy:
    1. Load pretrained ImageNet weights
    2. Freeze layers 1-3 (low-level features transfer well)
    3. Train layer4 + fc (high-level features need adaptation)
    
    This preserves Grad-CAM compatibility since we don't modify
    the convolutional structure, only the final classification layer.
    """
    # Load pretrained ResNet-18
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    if freeze_early_layers:
        # Freeze everything first
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze layer4 (last conv block) - these features need adaptation
        for param in model.layer4.parameters():
            param.requires_grad = True
    
    # Replace final FC layer for binary classification
    # Original: 512 -> 1000 (ImageNet classes)
    # New: 512 -> 1 (pneumonia probability after sigmoid)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, config.NUM_CLASSES)
    
    # FC layer is always trainable
    for param in model.fc.parameters():
        param.requires_grad = True
    
    return model.to(config.DEVICE)


def count_parameters(model):
    """Count trainable vs total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


# ============ DATA LOADING ============

def create_dataloaders(data_dir: str, config: Config):
    """
    Create train and validation dataloaders.
    
    Handles class imbalance via WeightedRandomSampler:
    - Oversamples minority class during training
    - Ensures balanced batches for stable gradients
    """
    train_dir = Path(data_dir) / "train"
    val_dir = Path(data_dir) / "val"
    
    # Create datasets
    train_dataset = datasets.ImageFolder(
        train_dir,
        transform=get_train_transforms(config)
    )
    val_dataset = datasets.ImageFolder(
        val_dir,
        transform=get_val_transforms(config)
    )
    
    # Print class mapping
    print(f"Class mapping: {train_dataset.class_to_idx}")
    # Expected: {'NORMAL': 0, 'PNEUMONIA': 1}
    
    # Calculate class weights for balanced sampling
    class_counts = [0, 0]
    for _, label in train_dataset.samples:
        class_counts[label] += 1
    
    print(f"Training class distribution: NORMAL={class_counts[0]}, PNEUMONIA={class_counts[1]}")
    
    # Weight each sample inversely to class frequency
    class_weights = [1.0 / count for count in class_counts]
    sample_weights = [class_weights[label] for _, label in train_dataset.samples]
    
    # Weighted sampler for balanced batches
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        sampler=sampler,  # Balanced sampling
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_dataset.class_to_idx


# ============ TRAINING ============

def train_one_epoch(model, train_loader, criterion, optimizer, config):
    """Train for one epoch, return average loss."""
    model.train()
    running_loss = 0.0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(config.DEVICE)
        labels = labels.float().unsqueeze(1).to(config.DEVICE)  # (B,) -> (B, 1)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Progress every 50 batches
        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    return running_loss / len(train_loader)


def evaluate(model, val_loader, criterion, config, threshold=0.3):
    """
    Evaluate model on validation set.
    
    Returns metrics focused on medical relevance:
    - Recall (sensitivity): % of pneumonia cases correctly detected
    - Precision: % of positive predictions that are correct
    - Loss: For tracking convergence
    
    Uses threshold=0.3 (lower than default 0.5) to prioritize recall.
    In medical AI, missing pneumonia (false negative) is worse than
    a false alarm (false positive).
    """
    model.eval()
    running_loss = 0.0
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(config.DEVICE)
            labels_float = labels.float().unsqueeze(1).to(config.DEVICE)
            
            outputs = model(images)
            loss = criterion(outputs, labels_float)
            running_loss += loss.item()
            
            # Get probabilities and predictions
            probs = torch.sigmoid(outputs).cpu()
            preds = (probs >= threshold).int()
            
            all_probs.extend(probs.squeeze().tolist())
            all_preds.extend(preds.squeeze().tolist())
            all_labels.extend(labels.tolist())
    
    # Calculate metrics
    all_preds = torch.tensor(all_preds)
    all_labels = torch.tensor(all_labels)
    
    # True positives, false positives, false negatives
    tp = ((all_preds == 1) & (all_labels == 1)).sum().item()
    fp = ((all_preds == 1) & (all_labels == 0)).sum().item()
    fn = ((all_preds == 0) & (all_labels == 1)).sum().item()
    tn = ((all_preds == 0) & (all_labels == 0)).sum().item()
    
    # Recall = TP / (TP + FN) - sensitivity for pneumonia
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # Precision = TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # Accuracy (for reference, not primary metric)
    accuracy = (tp + tn) / len(all_labels)
    
    # F1 score
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    avg_loss = running_loss / len(val_loader)
    
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "threshold": threshold
    }


def save_checkpoint(model, optimizer, epoch, metrics, config, filename):
    """
    Save model checkpoint.
    
    Saves ONLY the model state_dict, which is exactly what
    load_model(weights_path=...) expects in the existing backend.
    """
    checkpoint_dir = Path(config.CHECKPOINT_DIR)
    checkpoint_dir.mkdir(exist_ok=True)
    
    filepath = checkpoint_dir / filename
    
    # Save full checkpoint for resuming training
    full_checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": {
            "learning_rate": config.LEARNING_RATE,
            "batch_size": config.BATCH_SIZE,
            "pos_weight": config.PNEUMONIA_POS_WEIGHT
        }
    }
    torch.save(full_checkpoint, filepath.with_suffix('.full.pth'))
    
    # Save ONLY model weights (for production inference)
    # This is what your backend's load_model(weights_path=...) expects
    torch.save(model.state_dict(), filepath)
    
    print(f"💾 Saved checkpoint: {filepath}")
    print(f"   (Full checkpoint with optimizer: {filepath.with_suffix('.full.pth')})")
    
    return str(filepath)


# ============ MAIN TRAINING LOOP ============

def train(data_dir: str, config: Config):
    """
    Main training function.
    
    Training strategy:
    1. Start with pretrained ImageNet weights (proven feature extractors)
    2. Freeze early layers (generic features like edges, textures)
    3. Train layer4 + fc (adapt high-level features for X-rays)
    4. Use pos_weight in loss to penalize missing pneumonia cases
    5. Track recall as primary metric (medical priority)
    """
    print("=" * 60)
    print("🏥 PNEUMONIA DETECTION TRAINING")
    print("=" * 60)
    print(f"Device: {config.DEVICE}")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Pneumonia pos_weight: {config.PNEUMONIA_POS_WEIGHT}")
    print("=" * 60)
    
    # Create model
    print("\n📦 Creating model...")
    model = create_model(config, freeze_early_layers=True)
    trainable, total = count_parameters(model)
    print(f"   Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    
    # Create dataloaders
    print("\n📂 Loading data...")
    train_loader, val_loader, class_to_idx = create_dataloaders(data_dir, config)
    print(f"   Training batches: {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}")
    
    # Loss function with class imbalance handling
    # pos_weight > 1 means we penalize false negatives (missed pneumonia) more
    pos_weight = torch.tensor([config.PNEUMONIA_POS_WEIGHT]).to(config.DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"\n⚖️  Using BCEWithLogitsLoss with pos_weight={config.PNEUMONIA_POS_WEIGHT}")
    print("   (Higher weight = more penalty for missing pneumonia)")
    
    # Optimizer - only for trainable parameters
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )
    
    # Training loop
    best_recall = 0.0
    best_epoch = 0
    history = []
    
    print("\n🚀 Starting training...")
    print("-" * 60)
    
    for epoch in range(1, config.EPOCHS + 1):
        print(f"\n📍 Epoch {epoch}/{config.EPOCHS}")
        
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, config)
        
        # Evaluate (using threshold=0.3 for high recall)
        metrics = evaluate(model, val_loader, criterion, config, threshold=0.3)
        
        # Update scheduler based on recall (our primary metric)
        scheduler.step(metrics['recall'])
        
        # Log metrics
        print(f"   Train Loss: {train_loss:.4f}")
        print(f"   Val Loss: {metrics['loss']:.4f}")
        print(f"   Val Accuracy: {metrics['accuracy']:.4f}")
        print(f"   Val Recall: {metrics['recall']:.4f} ← PRIMARY METRIC")
        print(f"   Val Precision: {metrics['precision']:.4f}")
        print(f"   Val F1: {metrics['f1']:.4f}")
        print(f"   Confusion: TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}, TN={metrics['tn']}")
        
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            **metrics
        })
        
        # Save best model (by recall)
        if metrics['recall'] > best_recall:
            best_recall = metrics['recall']
            best_epoch = epoch
            save_checkpoint(
                model, optimizer, epoch, metrics, config,
                "best_model.pth"
            )
            print(f"   🏆 New best recall: {best_recall:.4f}")
        
        # Save latest checkpoint
        save_checkpoint(
            model, optimizer, epoch, metrics, config,
            "latest_model.pth"
        )
    
    # Training complete
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best recall: {best_recall:.4f} (epoch {best_epoch})")
    print(f"Best model saved to: checkpoints/best_model.pth")
    print("\n📋 To use in your backend:")
    print('   model, device = load_model(weights_path="checkpoints/best_model.pth")')
    
    # Save training history
    history_path = Path(config.CHECKPOINT_DIR) / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n📊 Training history saved to: {history_path}")
    
    return model, history


# ============ CLI ============

def main():
    parser = argparse.ArgumentParser(
        description="Train pneumonia detection model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic training
    python train.py --data_dir ./chest_xrays
    
    # Custom settings
    python train.py --data_dir ./chest_xrays --epochs 20 --batch_size 64 --lr 0.0001
    
    # Higher recall (use higher pos_weight)
    python train.py --data_dir ./chest_xrays --pos_weight 3.0

Expected folder structure:
    data_dir/
        train/
            NORMAL/
                normal1.jpg
                normal2.jpg
                ...
            PNEUMONIA/
                pneumonia1.jpg
                pneumonia2.jpg
                ...
        val/
            NORMAL/
            PNEUMONIA/
        """
    )
    
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to dataset directory (must contain train/ and val/ subdirectories)"
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Number of training epochs (default: 10)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Learning rate (default: 0.0001)"
    )
    parser.add_argument(
        "--pos_weight", type=float, default=2.0,
        help="Positive class weight for BCELoss (higher = more recall, default: 2.0)"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints",
        help="Directory to save checkpoints (default: checkpoints)"
    )
    
    args = parser.parse_args()
    
    # Update config with CLI arguments
    config = Config()
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.lr
    config.PNEUMONIA_POS_WEIGHT = args.pos_weight
    config.CHECKPOINT_DIR = args.checkpoint_dir
    
    # Run training
    train(args.data_dir, config)


if __name__ == "__main__":
    main()
