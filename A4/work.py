import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn


class UNET(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = self.contract_block(in_channels, 32, 7, 3)
        self.conv2 = self.contract_block(32, 64, 3, 1)
        self.conv3 = self.contract_block(64, 128, 3, 1)

        self.upconv3 = self.expand_block(128, 64, 3, 1)
        self.upconv2 = self.expand_block(64*2, 32, 3, 1)
        self.upconv1 = self.expand_block(32*2, out_channels, 3, 1)

    def __call__(self, x):

        # downsampling part
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        upconv3 = self.upconv3(conv3)

        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        return upconv1

    def contract_block(self, in_channels, out_channels, kernel_size, padding):

        contract = nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            #Sine(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            #Sine(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                                 )
        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):

        expand = nn.Sequential(torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
                            torch.nn.BatchNorm2d(out_channels),
                            torch.nn.ReLU(),
                            #Sine(),
                            torch.nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
                            torch.nn.BatchNorm2d(out_channels),
                            torch.nn.ReLU(),
                            #Sine(),
                            torch.nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1) 
                            )    
        return expand    
def dice_coefficient(pred, target, smooth=1e-6):
    """Calculate Dice coefficient for semantic segmentation.
    
    Args:
        pred: Predicted masks (B, C, H, W)
        target: Ground truth masks (B, H, W)
        smooth: Smoothing factor to avoid division by zero
    """
    num_classes = pred.shape[1]
    pred = F.softmax(pred, dim=1)
    pred = torch.argmax(pred, dim=1)
    
    dice_scores = []
    # Calculate dice for each class (excluding background)
    for class_idx in range(num_classes-1):  # Exclude background class
        pred_class = (pred == class_idx)
        target_class = (target == class_idx)
        
        intersection = (pred_class & target_class).sum().float()
        union = pred_class.sum() + target_class.sum()
        
        dice = (2. * intersection + smooth) / (union + smooth)
        dice_scores.append(dice.item())
        
    return np.mean(dice_scores)

class MNISTDDDataset(Dataset):
    def __init__(self, image_paths, target_paths):
        self.image_paths = image_paths
        self.target_paths = target_paths
        self.transforms_image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        # Reshape image to proper dimensions (64, 64, 3)
        image = self.image_paths[index].reshape(64, 64, 3)
        # Reshape mask to (64, 64)
        mask = self.target_paths[index].reshape(64, 64)
        
        # Convert image to tensor and normalize
        image = self.transforms_image(image)
        # Convert mask to tensor (no normalization needed)
        mask = torch.tensor(mask, dtype=torch.long)
        
        return image, mask

    def __len__(self):
        return len(self.image_paths)

def train_model(model, train_loader, valid_loader, device, num_epochs=15):
    """Train the UNET model with progress tracking and metrics."""
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    
    best_valid_dice = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch_idx, (data, target) in enumerate(train_pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # Calculate metrics
            train_loss += loss.item()
            train_dice += dice_coefficient(output, target)
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': f'{train_loss/(batch_idx+1):.4f}',
                'dice': f'{train_dice/(batch_idx+1):.4f}'
            })
        
        # Validation phase
        model.eval()
        valid_loss = 0.0
        valid_dice = 0.0
        
        with torch.no_grad():
            valid_pbar = tqdm(valid_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Valid]')
            for batch_idx, (data, target) in enumerate(valid_pbar):
                data, target = data.to(device), target.to(device)
                output = model(data)
                
                loss = criterion(output, target)
                
                # Calculate metrics
                valid_loss += loss.item()
                valid_dice += dice_coefficient(output, target)
                
                # Update progress bar
                valid_pbar.set_postfix({
                    'loss': f'{valid_loss/(batch_idx+1):.4f}',
                    'dice': f'{valid_dice/(batch_idx+1):.4f}'
                })
        
        # Calculate epoch metrics
        train_loss /= len(train_loader)
        train_dice /= len(train_loader)
        valid_loss /= len(valid_loader)
        valid_dice /= len(valid_loader)
        
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}')
        print(f'Valid Loss: {valid_loss:.4f}, Valid Dice: {valid_dice:.4f}')
        
        # Save best model
        if valid_dice > best_valid_dice:
            best_valid_dice = valid_dice
            torch.save(model.state_dict(), 'best_unet.pt')
            print(f'Saved new best model with dice score: {best_valid_dice:.4f}')

def main():
    # Load data
    train_data = np.load("train.npz")
    valid_data = np.load("valid.npz")
    
    # Create datasets
    train_dataset = MNISTDDDataset(train_data["images"], train_data["semantic_masks"])
    valid_dataset = MNISTDDDataset(valid_data["images"], valid_data["semantic_masks"])
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=4
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=32, shuffle=False, num_workers=4
    )
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create and move model to device
    model = UNET(in_channels=3, out_channels=11).to(device)  # 11 classes (0-9 digits + background)
    
    # Train model
    train_model(model, train_loader, valid_loader, device)

if __name__ == "__main__":
    main()