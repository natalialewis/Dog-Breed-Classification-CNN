import torch.nn as nn
import torch.nn.functional as F


class DogBreedCNN(nn.Module):
    """
    Architecture of the CNN model for dog breed classification.
    Consists of 4 convolutional blocks (because my images are 224x224).
    Each block uses conv + batch norm + max pooling, with dropout to prevent overfitting.
    """
    
    def __init__(self, num_classes=9):
        """
        Initialize the DogBreedCNN model.
        """

        super(DogBreedCNN, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Third convolutional block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Fourth convolutional block
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)

        # Fifth convolutional block that I added later
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(2, 2)
        
        # Calculates the size of the flattened features to feed into the first fully connected layer
        self.fc1_input_size = 512 * 7 * 7
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.fc1_input_size, 1024)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        """
        Define the forward pass of the model.
        """

        # Convolutional layers with batch norm and pooling
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.pool5(F.relu(self.bn5(self.conv5(x))))
        
        # Flatten for fully connected layers
        x = x.view(-1, self.fc1_input_size)
        
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x


def get_model(num_classes=9, device='cuda'):
    """
    Creates and returns a DogBreedCNN model based on the number of classes and device.
    I used 'cuda' as the device because I will be using an NVIDIA RTX 3060ti GPU for training.
    """
    model = DogBreedCNN(num_classes=num_classes)
    model = model.to(device)
    return model

