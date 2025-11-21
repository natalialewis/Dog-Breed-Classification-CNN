import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class DogBreedDataset(Dataset):
    """
    Custom Dataset for dog breed images.
    Loads images, applies transforms, and provides labels for CNN training.
    Needed to feed PyTorch DataLoaders efficiently.
    """
    
    def __init__(self, data_dir, transform=None):
        """
        Collects all image paths and their corresponding labels.
        Has an optional transform for data augmentation and normalization.
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # Get all breed names
        self.breeds = sorted([d for d in os.listdir(data_dir) 
                             if os.path.isdir(os.path.join(data_dir, d))])
        
        # Create breed to index mapping
        self.breed_to_idx = {breed: idx for idx, breed in enumerate(self.breeds)}
        self.idx_to_breed = {idx: breed for breed, idx in self.breed_to_idx.items()}
        
        self.images = []
        self.labels = []
        
        # Collect all image paths and their labels
        for breed in self.breeds:
            breed_dir = os.path.join(data_dir, breed)
            for img_name in os.listdir(breed_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(breed_dir, img_name)
                    # Append image path and label
                    self.images.append(img_path)
                    self.labels.append(self.breed_to_idx[breed])
    
    def __len__(self):
        """
        Return the total number of images.
        """
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Return the image and its label at the given index.
        """
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback (this shouldn't happen though)
            image = Image.new('RGB', (224, 224))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_num_classes(self):
        """
        Return the number of unique breeds.
        I will do 9 but I need this just in case I change my mind in the future.
        """
        return len(self.breeds)
    

    def get_breed_names(self):
        """
        Return the list of breed names.
        """
        return self.breeds


def get_data_loaders(data_dir='data', batch_size=32, num_workers=4, image_size=224):
    """
    Create data loaders for train, validation, and test sets.
    Data loaders are needed because they handle batching, shuffling, and parallel loading.
    """
    
    # Data augmentation for training because we want to improve generalization
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.RandomAffine(degrees=0, shear=10, scale=(0.8, 1.2)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # No augmentation for validation and test because we want to evaluate on original images
    val_test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = DogBreedDataset(
        os.path.join(data_dir, 'train'), 
        transform=train_transform
    )
    valid_dataset = DogBreedDataset(
        os.path.join(data_dir, 'valid'), 
        transform=val_test_transform
    )
    test_dataset = DogBreedDataset(
        os.path.join(data_dir, 'test'), 
        transform=val_test_transform
    )
    
    # Get breed names (should be same for all splits)
    breed_names = train_dataset.get_breed_names()
    num_classes = train_dataset.get_num_classes()
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Loaded datasets. Breeds include: {breed_names}")
    
    return train_loader, valid_loader, test_loader, breed_names, num_classes

