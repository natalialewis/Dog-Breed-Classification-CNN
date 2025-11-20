import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import os
import time
from model import get_model
from data_loader import get_data_loaders
from visualize import save_training_curves


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Trains the CNN for one epoch.
    """

    # Set model to training mode
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Iterate over data
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    # Compute average loss and accuracy
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, valid_loader, criterion, device):
    """
    Validates the CNN on the validation set.
    """

    # Set model to evaluation mode
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Iterate over data
    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # Compute average loss and accuracy
    epoch_loss = running_loss / len(valid_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def train_model(data_dir='data', num_epochs=30, batch_size=32, learning_rate=0.001, device=None,
                save_dir='models', model_name='dog_breed_cnn'):
    """
    Does the full training of the CNN model.
    Works by loading data, creating the model, training over multiple epochs,
    validating, and saving the best model and training curves.
    """
    
    # Set device: cuda is for GPU, cpu is for CPU
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Load data
    train_loader, valid_loader, test_loader, breed_names, num_classes = get_data_loaders(data_dir=data_dir, batch_size=batch_size)
    
    # Create CNN model
    model = get_model(num_classes=num_classes, device=device)
    print("CNN model created...")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Training history
    train_losses = []
    train_accs = []
    valid_losses = []
    valid_accs = []
    
    best_valid_acc = 0.0
    best_epoch = 0
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print("------------------------------------------------")
    
    start_time = time.time()
    
    # Training loop
    for epoch in range(num_epochs):
        # Track time to see duration of each epoch (thought it would be useful in the report)
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        valid_loss, valid_acc = validate(model, valid_loader, criterion, device)
        
        # Update learning rate; have to update because using StepLR
        scheduler.step()
        
        # Save history
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)
        
        # Save best model
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'valid_acc': valid_acc,
                'breed_names': breed_names,
                'num_classes': num_classes
            }, os.path.join(save_dir, f'{model_name}_best.pth'))
        
        # Print progress
        epoch_time = time.time() - epoch_start
        print(f"Epoch [{epoch+1}/{num_epochs}] ({epoch_time:.2f}s)")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.2f}%")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        print()
    
    total_time = time.time() - start_time
    print("------------------------------------------------")
    print(f"Training completed in {total_time/60:.2f} minutes")
    print(f"Best validation accuracy: {best_valid_acc:.2f}% at epoch {best_epoch+1}")
    
    # Save final model
    torch.save({'epoch': num_epochs, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
        'valid_acc': valid_acc, 'breed_names': breed_names, 'num_classes': num_classes
        }, os.path.join(save_dir, f'{model_name}_final.pth'))
    
    # Save training curves
    save_training_curves(train_losses, train_accs, valid_losses, valid_accs, save_dir)
    
    print(f"\nModel saved to {save_dir}/")
    print(f"Best model: {model_name}_best.pth")
    
    return model, train_losses, train_accs, valid_losses, valid_accs


if __name__ == '__main__':
    # Training configuration
    config = {
        'data_dir': 'data',
        'num_epochs': 30,
        'batch_size': 32,
        'learning_rate': 0.001,
        'save_dir': 'models',
        'model_name': 'dog_breed_cnn'
    }
    
    train_model(**config)

