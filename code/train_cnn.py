import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import os
import glob
import torch.nn.functional as F
import csv
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, ConfusionMatrixDisplay
import numpy as np

# Set the model configuration name
model_config_name = "Model with Augmentation"

# Adjustable parameters
batch_size = 64
learning_rate = 0.001
dropout_rate = 0.4
num_epochs = 20

train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

# Define a CSV file to log results
log_file = f'training_log_{model_config_name.replace(" ", "_")}.csv'
log_exists = os.path.isfile(log_file)

# Define directories for checkpoints and ensure they exist
checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

# Open the file in write mode to log headers only if the file doesn't exist
if not log_exists:
    with open(log_file, mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Train Loss", "Train Accuracy", "Val Loss", "Val Accuracy"])



# Step 1: Dataset and DataLoader setup
class ImagenetteDataset(Dataset):
    def __init__(self, labels, image_paths, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

        # Create a mapping from class ID strings to integers
        self.label_to_int = {label: idx for idx, label in enumerate(sorted(set(labels)))}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_str = self.labels[idx]
        label = self.label_to_int[label_str]  # Convert string label to integer using mapping

        # Open and transform the image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label



# Load CSV
# data_df = pd.read_csv('imagenette2-160/noisy_imagenette.csv') #run locally
data_df = pd.read_csv('/content/DNN_HW1_CNN/imagenette2-160/noisy_imagenette.csv')


train_df = data_df[data_df['is_valid'] == False]
val_df = data_df[data_df['is_valid'] == True]

train_transforms = transforms.Compose([
    transforms.Resize((160, 160)),  # Enforce 160x160 for all images
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(),
    transforms.ToTensor(),
])
val_transforms = transforms.Compose([
    transforms.Resize((160, 160)),  # Ensure validation images are also 160x160
    transforms.ToTensor(),
])

# Ensure full paths for images by concatenating root directory with paths from 'path' column
# Local version
# train_paths = [os.path.join('/Users/bruce/Desktop/Romania/DNN/DNN_CNN_HW1/imagenette2-160', p) for p in train_df['path']]
# val_paths = [os.path.join('/Users/bruce/Desktop/Romania/DNN/DNN_CNN_HW1/imagenette2-160', p) for p in val_df['path']]

train_paths = [os.path.join('/content/DNN_HW1_CNN/imagenette2-160', p) for p in train_df['path']] #colab
val_paths = [os.path.join('/content/DNN_HW1_CNN/imagenette2-160', p) for p in val_df['path']] #colab

train_dataset = ImagenetteDataset(train_df['noisy_labels_0'].values, train_paths, transform=train_transforms)
val_dataset = ImagenetteDataset(val_df['noisy_labels_0'].values, val_paths, transform=val_transforms)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Step 2: Define the CNN model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        
        # First convolutional layer with ReLU activation and max pooling
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        #self.bn1 = nn.BatchNorm2d(32)  # Batch normalization for first conv layer
        self.relu1 = nn.ReLU()  # ReLU activation
        
        # Second convolutional layer with ReLU activation and max pooling
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        #self.bn2 = nn.BatchNorm2d(64)  # Batch normalization for second conv layer
        self.relu2 = nn.ReLU()  # ReLU activation

        # Pooling layer to reduce dimensions
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layer and final output layer with Softmax
        self.fc1 = nn.Linear(102400, 128)  # Adjust input size based on pooling output dimensions
        #self.dropout = nn.Dropout(dropout_rate)  # Dropout layer with 50% probability
        self.fc2 = nn.Linear(128, num_classes)  # Output layer for class predictions

    def forward(self, x):
        # Pass through first conv layer, apply ReLU, and then max pooling
        x = self.conv1(x)
        #x = self.bn1(x)  # Apply batch normalization
        x = self.relu1(x)  # Applying ReLU activation
        x = self.pool(x)   # Applying Max Pooling
        
        # Pass through second conv layer, apply ReLU, and then max pooling
        x = self.conv2(x)
        #x = self.bn2(x)  # Apply batch normalization
        x = self.relu2(x)  # Applying ReLU activation
        x = self.pool(x)   # Applying Max Pooling

        # Flatten the tensor for the fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)  # Another ReLU activation in the fully connected layer
        #x = self.dropout(x)  # Apply dropout regularization

        # Output layer with Softmax
        x = self.fc2(x)
        x = F.softmax(x, dim=1)  # Softmax activation for class probabilities
        
        return x

# Step 3: Training and evaluation setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = SimpleCNN(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# Learning Rate Scheduler setup
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Plotting results
def plot_training_results(train_losses, val_losses, train_accuracies, val_accuracies, model_config_name):
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_config_name} - Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{model_config_name} - Accuracy')
    plt.legend()

    plt.savefig(f'/content/DNN_HW1_CNN/{model_config_name}_training_results.png')
    plt.show()


# Confusion Matrix and Metrics with model setting label
def calculate_confusion_matrix_and_metrics(model, val_loader, device, model_config_name):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_preds, average=None, zero_division=1)
    
    # Display and save confusion matrix with label
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(set(all_labels)))
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f'{model_config_name} - Confusion Matrix')
    plt.savefig(f'/content/DNN_HW1_CNN/{model_config_name}_confusion_matrix.png')
    plt.show()
    
    # Output precision, recall, and F1-score for each class
    for idx, label in enumerate(sorted(set(all_labels))):
        print(f"Class {label} - {model_config_name}: Precision = {precision[idx]:.2f}, Recall = {recall[idx]:.2f}, F1-Score = {f1_score[idx]:.2f}")
    
    return cm, precision, recall, f1_score

# Step 4: Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images = images.to(device)  # Move images to the device (e.g., GPU)
        labels = labels.to(device) if isinstance(labels, torch.Tensor) else torch.tensor(labels).to(device)  # Convert labels if needed

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_losses.append(running_loss / total)
    train_accuracies.append(100 * correct / total)

    # Validation loop
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_losses.append(val_loss / val_total)
    val_accuracies.append(100 * val_correct / val_total)

    # Step the learning rate scheduler
    scheduler.step()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.2f}%"
          f", Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accuracies[-1]:.2f}%")

    # End of each epoch - plot results, calculate confusion matrix, etc.
    if epoch == num_epochs - 1:
        # Plot training results and calculate metrics after final epoch
        plot_training_results(train_losses, val_losses, train_accuracies, val_accuracies, model_config_name)
        calculate_confusion_matrix_and_metrics(model, val_loader, device, model_config_name)

    # Log epoch metrics to CSV
    with open(log_file, mode='a') as file:
        writer = csv.writer(file)
        writer.writerow([epoch + 1, train_losses[-1], train_accuracies[-1], val_losses[-1], val_accuracies[-1]])

    '''
    # Save model checkpoint at each epoch
    checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}_valacc_{val_accuracies[-1]:.2f}.pth')
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_losses[-1],
        'train_acc': train_accuracies[-1],
        'val_loss': val_losses[-1],
        'val_acc': val_accuracies[-1]
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")
    '''

# Save final model
final_model_path = 'final_model.pth'
torch.save(model.state_dict(), final_model_path)
print(f"Final model saved at {final_model_path}")



