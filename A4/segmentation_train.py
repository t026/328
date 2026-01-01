import torch
from torch.utils.data.dataset import Dataset  
import torch.nn.functional as F
from torchvision import transforms
import torchvision
from PIL import Image
import glob
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = self.contract_block(in_channels, 32, 7, 3)
        self.conv2 = self.contract_block(32, 64, 3, 1)
        self.conv3 = self.contract_block(64, 128, 3, 1)
        
        self.upconv3 = self.expand_block(128, 64, 3, 1)
        self.upconv2 = self.expand_block(64*2, 32, 3, 1)
        self.upconv1 = self.expand_block(32*2, out_channels, 3, 1)


    def forward(self, x):

        # downsampling part
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        upconv3 = self.upconv3(conv3)
        
        # Add output padding calculation
        output_size = conv2.size()
        upconv3 = F.interpolate(upconv3, size=(output_size[2], output_size[3]), 
                              mode='bilinear', align_corners=True)
        
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        
        # Repeat for next concatenation
        output_size = conv1.size()
        upconv2 = F.interpolate(upconv2, size=(output_size[2], output_size[3]), 
                              mode='bilinear', align_corners=True)

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

        expand = nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
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

class MNISTDDRGB(Dataset):
    def __init__(self, images, labels, bboxes, seg_masks, ins_masks):
        self.images = images
        self.labels = labels
        self.bboxes = bboxes
        self.seg_masks = seg_masks
        self.ins_masks = ins_masks
        
    def __getitem__(self, idx):
        image = self.images[idx, ...].reshape((64, 64, 3),).astype(np.uint8)
        image = image.transpose((2, 0, 1))
        
        
        bbox_1 = self.bboxes[idx, 0, :].squeeze().astype(np.int32)
        bbox_2 = self.bboxes[idx, 1, :].squeeze().astype(np.int32)
        
        seg_mask = self.seg_masks[idx, ...].reshape((64, 64))
        
        def get_patch_and_mask(bbox, image, mask):
            x1, y1, x2, y2 = bbox
            patch = image[:, y1:y2, x1:x2]
            mask_patch = mask[y1:y2, x1:x2]
            patch = patch.transpose((1, 2, 0))
            # Convert to PIL Image for proper resizing
            patch_pil = Image.fromarray(patch)
            patch_resized = patch_pil.resize((28, 28), Image.Resampling.BILINEAR)
            
            # Convert mask to PIL Image and resize
            mask_pil = Image.fromarray(mask_patch.astype(np.uint8))
            mask_resized = mask_pil.resize((28, 28), Image.Resampling.NEAREST)
            
            # Convert back to numpy arrays
            patch = np.array(patch_resized)
            mask_patch = np.array(mask_resized)
            
            return patch, mask_patch
        
        # Extract and resize patches
        patch1, mask1 = get_patch_and_mask(bbox_1, image, seg_mask)
        patch2, mask2 = get_patch_and_mask(bbox_2, image, seg_mask)

        mask1 = (mask1 != 10).astype(float)
        mask2 = (mask2 != 10).astype(float)
        
        # Convert to tensors and normalize images
        patch1 = torch.FloatTensor(patch1.transpose((2, 0, 1))) / 255.0
        patch2 = torch.FloatTensor(patch2.transpose((2, 0, 1))) / 255.0
        mask1 = torch.FloatTensor(mask1)
        mask2 = torch.FloatTensor(mask2)
        
        return (patch1, patch2), (mask1, mask2)
    
    def __len__(self):
        return len(self.images)
    



def calculate_dice_score(pred, target, smooth=1e-7):
    pred = pred > 0.5  # Convert predictions to binary
    pred = pred.float()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    model = model.to(device)
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        train_batches = 0
        
        # Create progress bar for training
        train_pbar = tqdm(total=len(train_loader), desc='Training')
        
        for batch_idx, (patches, masks) in enumerate(train_loader):
            batch_loss = 0.0
            batch_dice = 0.0
            
            # Process both patches in the batch
            for patch, mask in zip(patches, masks):
                patch = patch.to(device)
                mask = mask.to(device)
                
                optimizer.zero_grad()
                output = model(patch)
                
                # Squeeze output to match mask dimensions
                output = output.squeeze(1)
                loss = criterion(output, mask)
                
                loss.backward()
                optimizer.step()
                
                # Calculate metrics
                batch_loss += loss.item()
                batch_dice += calculate_dice_score(output, mask).item()
            
            # Average batch metrics
            batch_loss /= len(patches)
            batch_dice /= len(patches)
            
            train_loss += batch_loss
            train_dice += batch_dice
            train_batches += 1
            
            # Update progress bar with current batch metrics
            train_pbar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'dice': f'{batch_dice:.4f}'
            })
            train_pbar.update()
        
        train_pbar.close()
        
        # Calculate average training metrics
        avg_train_loss = train_loss / train_batches
        avg_train_dice = train_dice / train_batches
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        val_batches = 0
        
        # Create progress bar for validation
        val_pbar = tqdm(total=len(val_loader), desc='Validation')
        
        with torch.no_grad():
            for patches, masks in val_loader:
                batch_loss = 0.0
                batch_dice = 0.0
                
                for patch, mask in zip(patches, masks):
                    patch = patch.to(device)
                    mask = mask.to(device)
                    
                    output = model(patch)
                    output = output.squeeze(1)
                    loss = criterion(output, mask)
                    
                    # Calculate metrics
                    batch_loss += loss.item()
                    batch_dice += calculate_dice_score(output, mask).item()
                
                # Average batch metrics
                batch_loss /= len(patches)
                batch_dice /= len(patches)
                
                val_loss += batch_loss
                val_dice += batch_dice
                val_batches += 1
                
                # Update progress bar with current batch metrics
                val_pbar.set_postfix({
                    'loss': f'{batch_loss:.4f}',
                    'dice': f'{batch_dice:.4f}'
                })
                val_pbar.update()
        
        val_pbar.close()
        
        # Calculate average validation metrics
        avg_val_loss = val_loss / val_batches
        avg_val_dice = val_dice / val_batches
        
        # Print epoch summary
        print(f'\nEpoch Summary:')
        print(f'Training    - Loss: {avg_train_loss:.4f}, Dice Score: {avg_train_dice:.4f}')
        print(f'Validation  - Loss: {avg_val_loss:.4f}, Dice Score: {avg_val_dice:.4f}')
        
        # Save best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_dice': avg_val_dice
            }, 'best_segm_model.pt')
            print(f'Saved new best model with validation loss: {best_val_loss:.4f}')



    
def main():
    # Load data
    train_data = np.load('train.npz')
    val_data = np.load('valid.npz')
    
    # Create datasets
    train_dataset = MNISTDDRGB(
        train_data['images'],
        train_data['labels'],
        train_data['bboxes'],
        train_data['semantic_masks'],
        train_data['instance_masks']
    )
    
    val_dataset = MNISTDDRGB(
        val_data['images'],
        val_data['labels'],
        val_data['bboxes'],
        val_data['semantic_masks'],
        val_data['instance_masks']
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2
    )
    
    # Initialize model, criterion, and optimizer
    model = UNet(in_channels=3, out_channels=1)  # 1 output channel for binary segmentation
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device=device)

if __name__ == '__main__':
    main()