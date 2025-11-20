import torch
import torch.nn as nn
import os
from model import get_model
from data_loader import get_data_loaders
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


def evaluate_model(model_path, data_dir='data', batch_size=32, device=None,split='test'):
    """
    Loads the trained CNN and evaluates it on the specified data split (test or validation).
    Prints accuracy, loss, classification report, confusion matrix, and per-class accuracy.
    """
    
    # Set device: cuda is for GPU, cpu is for CPU
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint (saved model)
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model and load weights
    num_classes = checkpoint['num_classes']
    breed_names = checkpoint['breed_names']
    model = get_model(num_classes=num_classes, device=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load data
    train_loader, valid_loader, test_loader, _, _ = get_data_loaders(data_dir=data_dir, batch_size=batch_size)
    
    # Decide which data loader to use (test is for final evaluation, valid for tuning)
    if split == 'test':
        data_loader = test_loader
    elif split == 'valid':
        data_loader = valid_loader
    else:
        raise ValueError("split must be 'test' or 'valid'")
    
    # Evaluate the model
    print(f"Evaluating saved model on {split} set...")
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    
    # Iterate over data
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = 100 * correct / total
    avg_loss = running_loss / len(data_loader)
    
    print("\n --------------------------------------------------")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Correct: {correct}/{total}")
    print()
    
    # Classification report
    print("Classification Report:")
    print("--------------------------------------------------")
    print(classification_report(all_labels, all_preds, target_names=breed_names,digits=4))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print("--------------------------------------------------")
    print(f"{'':<15}", end="")
    for breed in breed_names:
        print(f"{breed[:10]:<12}", end="")
    print()
    
    # Print each row of the confusion matrix
    for i, breed in enumerate(breed_names):
        print(f"{breed[:14]:<15}", end="")
        for j in range(len(breed_names)):
            print(f"{cm[i, j]:<12}", end="")
        print()
    
    # Per-class accuracy
    print("\nPer-Class Accuracy:")
    print("--------------------------------------------------")
    for i, breed in enumerate(breed_names):
        class_correct = cm[i, i]
        class_total = cm[i, :].sum()
        class_acc = 100 * class_correct / class_total if class_total > 0 else 0
        print(f"{breed:<20}: {class_acc:.2f}% ({class_correct}/{class_total})")
    
    return accuracy, avg_loss, all_preds, all_labels, breed_names


if __name__ == '__main__':
    # Evaluate the best model
    # model_path = 'models/dog_breed_cnn_best.pth'
    model_path = 'models/fake_model.pth'
    
    # Error handling if model file does not exist (it should though)
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please train the model first using train.py")
    else:
        evaluate_model(model_path, split='test')

