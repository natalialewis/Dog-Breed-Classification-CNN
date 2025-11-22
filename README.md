# CS5600 Project 2 - Dog Breed Classification CNN - Natalia Lewis

A Convolutional Neural Network (CNN) for classifying images of dogs by breed, trained on a subset of the Stanford Dogs Dataset.

## Project Overview

This project implements a CNN using PyTorch to classify dog images into 9 different breeds:
- Chihuahua
- Corgi
- German Shepherd
- Golden Retriever
- Great Dane
- Husky
- Pomeranian
- Pug
- Saint Bernard

## Project Structure

```
project2/
├── data/                   # Dataset (train/valid/test splits)
│   ├── train/              # Training images (Default is 70%)
│   ├── valid/              # Validation images (Default is 15%)
│   └── test/               # Test images (Default is 15%)
├── models/                 # Saved model checkpoints (created during training)
├── model.py                # CNN model architecture
├── data_loader.py          # Data loading utilities
├── train.py                # Training script
├── evaluate.py             # Evaluation script (tests how good the model is)
├── visualize.py            # Visualization utilities (plots data on png graphs)
├── split_data.py           # Data preparation script
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Data Preparation

The data should already be split into train/valid/test directories. If you need to re-split the data:

1. Unzip the `full_breeds_dataset.zip` file
2. Run `split_data.py` to create the train/valid/test splits
3. If you want to modify the split percentages, modify the ratios array in `split_data.py`

```bash
python split_data.py
```

## Usage

### Training the Model

To train the CNN model, simply run:

```bash
python train.py
```

This will:
- Load the training and validation data
- Create and train the CNN model
- Save the best model checkpoint (based on validation accuracy)
- Save training curves (loss and accuracy plots)

**Training Configuration:**

You can modify the training parameters in `train.py`:

```python
config = {
    'data_dir': 'data',
    'num_epochs': 30,           # Number of training epochs
    'batch_size': 32,           # Batch size
    'learning_rate': 0.000089,  # Learning rate
    'save_dir': 'models',       # Directory to save models
    'model_name': 'dog_breed_cnn'
}
```

**Training Output:**
- `models/dog_breed_cnn_best.pth` - Best model (highest validation accuracy)
- `models/dog_breed_cnn_final.pth` - Final model after all epochs
- `models/training_curves.png` - Training and validation loss/accuracy curves

### Evaluating the Model

To evaluate the trained model on the test set:

```bash
python evaluate.py
```

This will:
- Load the best model checkpoint
- Evaluate on the test set
- Print classification report, confusion matrix, and per-class accuracy

You can also evaluate on the validation set by modifying `evaluate.py`:

```python
evaluate_model(model_path, split='valid')
```

## Model Architecture

The CNN consists of:
- **5 Convolutional Blocks**: Each with Conv2d → BatchNorm → ReLU → MaxPool
  - Block 1: 3 → 32 channels
  - Block 2: 32 → 64 channels
  - Block 3: 64 → 128 channels
  - Block 4: 128 → 256 channels
  - Block 5: 256 → 512 channels
- **3 Fully Connected Layers**: (512×7×7 = 25,088 inputs) → 1024 → 9 (num_classes)
- **Dropout**: 0.5 dropout rate for regularization
- **Input Size**: 224×224 RGB images

## Training Details

- **Loss Function**: CrossEntropyLoss
- **Optimizer**: AdamW (Decoupled Weight Decay) with weight decay of 0.01
- **Learning Rate**: Stays at a constant 0.000089 throughout training (this value can be modified but is still constant)
- **Data Augmentation**: Random horizontal flips, rotations, and color jitter (training only)
- **Normalization**: ImageNet mean/std normalization

## Expected Results

After training, you should see:
- Training and validation accuracy/loss curves saved as PNG
- Model checkpoints saved in the `models/` directory
- Detailed evaluation metrics including per-class accuracy

## Evaluating Results
After training, run `evaluate.py` to see the model's performance on the test set. You will get:
- Overall accuracy
- Precision, Recall, F1-Score for each breed
- Confusion matrix (both printed and saved as a PNG)

Remember that the accuracy will vary because the test and validation sets are different.

## Hardware Requirements
- **GPU**: Any CUDA-compatible GPU is recommended (code will run on CPU but much slower)
- **RAM**: 8GB+ recommended

The code will automatically use GPU if available, otherwise it will fall back to CPU.

## AI Assistance
While the majority of the code was written by me, I utilized AI tools to help with debugging and strategizing certain sections of the code. The main ones are as follows:
- I couldn't figure out how I needed to pass my images to the PyTorch CNN. I used ChatGPT to help me understand that, as well as the PyTorch documentation.
- I had trouble with some of the plotting functions for visualizing training curves. I used GitHub Copilot to help me write the code for that. (Github Copilot was always on in my IDE throughout the project.)
- I had Github Copilot in my IDE help me to fill in some of the function comments that I had trouble wording myself.
- After I create the code for the CNN, training loop, and evaluation loop, I used ChatGPT to help me evaluate the output to make sure that I had correctly set up my procedures before moving on.
- I used Github Copilot in my IDE to help write the Architecture part of this README because I was having a hard time describing in words what I had built. 