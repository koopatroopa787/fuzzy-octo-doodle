"""
Optimized training procedure for Fashion-MNIST classification.
Designed to maximize accuracy while maintaining efficiency.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from PIL import Image

from submission.fashion_model import Net


def get_transforms(mode='train'):
    """
    Get data transforms for training or evaluation.

    Args:
        mode: 'train' or 'eval'

    Returns:
        torchvision.transforms.Compose object
    """
    if mode == 'train':
        # Training transforms - only basic preprocessing
        # Note: ToTensor() can handle both PIL Images and numpy arrays
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))  # Fashion-MNIST mean and std
        ])
    elif mode == 'eval':
        # Evaluation transforms - deterministic only
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'train' or 'eval'")


def train_fashion_model(dataset, n_epochs=60, USE_GPU=True, batch_size=128,
                        lr=0.01, validation_split=0.1, verbose=True):
    """
    Train the Fashion-MNIST model with optimized hyperparameters.

    Optimizations:
    - SGD with momentum (better generalization than Adam for CNNs)
    - Cosine annealing learning rate schedule
    - Label smoothing for better generalization
    - Longer training with early stopping
    - Optimized batch size

    Args:
        dataset: Training dataset (FashionMNIST or compatible)
        n_epochs: Number of training epochs (default: 60)
        USE_GPU: Whether to use GPU if available
        batch_size: Batch size for training (default: 128)
        lr: Initial learning rate (default: 0.01)
        validation_split: Fraction of data to use for validation
        verbose: Whether to print training progress

    Returns:
        dict: state_dict of the trained model
    """
    # Device configuration
    device = torch.device('cuda' if USE_GPU and torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"Training on device: {device}")

    # Split dataset into train and validation
    if validation_split > 0:
        val_size = int(len(dataset) * validation_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
    else:
        train_dataset = dataset
        val_dataset = None

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if USE_GPU else False
    )

    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True if USE_GPU else False
        )

    # Initialize model
    model = Net().to(device)

    # Loss function with label smoothing for better generalization
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # SGD with momentum - better for CNNs than Adam
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=5e-4,  # L2 regularization
        nesterov=True  # Nesterov momentum for better convergence
    )

    # Cosine annealing learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=n_epochs,
        eta_min=1e-5
    )

    # Training loop
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_state_dict = None
    patience_counter = 0
    patience = 10  # Early stopping patience

    for epoch in range(n_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            # Handle different data formats (some datasets return numpy arrays)
            if not isinstance(images, torch.Tensor):
                # Convert PIL Images or numpy arrays to tensor
                if isinstance(images[0], Image.Image):
                    images = torch.stack([transforms.ToTensor()(img) for img in images])
                else:
                    images = torch.from_numpy(images).float() / 255.0
                    if images.ndim == 3:
                        images = images.unsqueeze(1)

            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels)

            images, labels = images.to(device), labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = train_loss / len(train_loader)
        train_acc = 100 * correct / total

        # Validation phase
        if val_dataset is not None:
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    # Handle different data formats
                    if not isinstance(images, torch.Tensor):
                        if isinstance(images[0], Image.Image):
                            images = torch.stack([transforms.ToTensor()(img) for img in images])
                        else:
                            images = torch.from_numpy(images).float() / 255.0
                            if images.ndim == 3:
                                images = images.unsqueeze(1)

                    if not isinstance(labels, torch.Tensor):
                        labels = torch.tensor(labels)

                    images, labels = images.to(device), labels.to(device)

                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_loss = val_loss / len(val_loader)
            val_acc = 100 * correct / total

            # Save best model based on validation accuracy (not loss)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_state_dict = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose:
                current_lr = optimizer.param_groups[0]['lr']
                print(f'Epoch [{epoch+1}/{n_epochs}] '
                      f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | '
                      f'LR: {current_lr:.6f}')

            # Early stopping
            if patience_counter >= patience and epoch > 30:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
        else:
            if verbose:
                print(f'Epoch [{epoch+1}/{n_epochs}] '
                      f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')

        # Step the learning rate scheduler
        scheduler.step()

    # Return best state dict if we did validation, otherwise return final state dict
    if best_state_dict is not None:
        if verbose:
            print(f"\nBest validation accuracy: {best_val_acc:.2f}%")
        return best_state_dict
    else:
        return model.state_dict()


if __name__ == "__main__":
    """
    Main training script for Fashion-MNIST.
    Run this to train the model and save weights.
    """
    import torchvision

    # Download and load Fashion-MNIST dataset
    print("Loading Fashion-MNIST dataset...")
    train_dataset = torchvision.datasets.FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=get_transforms(mode='train')
    )

    # Train the model with optimized hyperparameters
    print("\nTraining model with optimized hyperparameters...")
    print("Strategy: SGD with momentum + Cosine annealing + Label smoothing")
    print("Target: >92% accuracy with <80K parameters\n")

    state_dict = train_fashion_model(
        train_dataset,
        n_epochs=60,  # Longer training for better convergence
        USE_GPU=True,
        batch_size=128,
        lr=0.01,  # Higher initial LR for SGD
        validation_split=0.1,
        verbose=True
    )

    # Save the trained weights
    print("\nSaving model weights...")
    torch.save(state_dict, 'submission/model_weights.pth')
    print("Model weights saved to submission/model_weights.pth")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    from submission.fashion_model import Net

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    model.load_state_dict(state_dict)
    model.eval()

    test_dataset = torchvision.datasets.FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=get_transforms(mode='eval')
    )

    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Total Parameters: {num_params:,}")
    print(f"Parameter Limit: 100,000")
    print(f"Parameters Used: {100 * num_params / 100000:.1f}%")
    print(f"Parameter Efficiency: {accuracy / (num_params / 1000):.2f}% per 1K params")
    print(f"{'='*60}")

    if accuracy >= 88:
        print("✓ Minimum accuracy requirement MET (≥88%)")
    else:
        print("✗ WARNING: Below minimum accuracy requirement!")

    if num_params <= 100000:
        print("✓ Parameter limit MET (≤100,000)")
    else:
        print("✗ WARNING: Exceeds parameter limit!")

    if accuracy >= 92:
        print("✓ BONUS: High accuracy achieved (likely top 50th percentile)")

    if num_params <= 70000:
        print("✓ BONUS: Very efficient model (likely bottom 30th percentile)")

    print(f"{'='*60}")
