import torch
from torch.utils.data.dataset import Dataset  
import torch.nn.functional as F
from torchvision import transforms
import torchvision
from PIL import Image
import glob
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import os
import yaml

class MNISTDDRGB(Dataset):
    def __init__(self, images, labels, bboxes):
        self.images = images
        self.labels = labels
        self.bboxes = bboxes
        
    def __getitem__(self, idx):
        image = self.images[idx, ...].reshape((64, 64, 3)).astype(np.uint8)
        label_1, label_2 = self.labels[idx]
        
        bbox_1 = self.bboxes[idx, 0, :].squeeze().astype(np.int32)
        bbox_2 = self.bboxes[idx, 1, :].squeeze().astype(np.int32)
        
        # Calculate center coordinates and dimensions for both boxes
        box_1 = [
            (bbox_1[0] + bbox_1[2])/2,  # center x
            (bbox_1[1] + bbox_1[3])/2,  # center y
            bbox_1[2] - bbox_1[0],      # width
            bbox_1[3] - bbox_1[1]       # height
        ]
        
        box_2 = [
            (bbox_2[0] + bbox_2[2])/2,  # center x
            (bbox_2[1] + bbox_2[3])/2,  # center y
            bbox_2[2] - bbox_2[0],      # width
            bbox_2[3] - bbox_2[1]       # height
        ]
        
        return image, (label_1, label_2), box_1, box_2
    
    def __len__(self):
        return len(self.images)


import os
import numpy as np
from PIL import Image
import yaml
from tqdm import tqdm

def convert_npz_to_yolo(train_npz_path, valid_npz_path, output_dir):
    """
    Convert train.npz and valid.npz datasets to YOLOv5 format
    
    Args:
        train_npz_path: Path to train.npz file
        valid_npz_path: Path to valid.npz file
        output_dir: Base directory for YOLOv5 dataset
    """
    # Create directory structure
    os.makedirs(os.path.join(output_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'val'), exist_ok=True)
    
    # Process both training and validation sets
    for split, npz_path in [('train', train_npz_path), ('val', valid_npz_path)]:
        # Load npz file
        data = np.load(npz_path)
        
        # Extract data
        images = data['images']
        labels = data['labels']
        bboxes = data['bboxes']
        
        # Create dataset instance
        dataset = MNISTDDRGB(images, labels, bboxes)
        
        # Process all samples in this split
        for idx in tqdm(range(len(dataset)), desc=f"Converting {split} dataset"):
            # Get data
            image, (label_1, label_2), box_1, box_2 = dataset[idx]
            
            # Convert image to PIL Image
            img = Image.fromarray(image)
            
            # Save image
            img_path = os.path.join(output_dir, 'images', split, f'{split}_{idx:06d}.png')
            img.save(img_path)
            
            # Create YOLO format labels
            labels = []
            for label, box in zip([label_1, label_2], [box_1, box_2]):
                # Normalize coordinates
                x_center = box[0] / 64.0  # divide by image width
                y_center = box[1] / 64.0  # divide by image height
                width = box[2] / 64.0     # divide by image width
                height = box[3] / 64.0    # divide by image height
                
                # Add to labels list
                labels.append(f"{label} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
            # Save labels
            label_path = os.path.join(output_dir, 'labels', split, f'{split}_{idx:06d}.txt')
            with open(label_path, 'w') as f:
                f.write('\n'.join(labels))
    
    # Create YAML file
    yaml_content = {
        'path': output_dir,  # dataset root dir
        'train': os.path.join('images', 'train'),  # train images relative to path
        'val': os.path.join('images', 'val'),      # val images relative to path
        'nc': 10,  # number of classes (assuming MNIST digits 0-9)
        'names': [str(i) for i in range(10)]  # class names
    }
    
    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    # Print statistics
    train_samples = len(os.listdir(os.path.join(output_dir, 'images', 'train')))
    val_samples = len(os.listdir(os.path.join(output_dir, 'images', 'val')))
    
    print(f"\nDataset converted and saved to {output_dir}")
    print(f"Training samples: {train_samples}")
    print(f"Validation samples: {val_samples}")
    print(f"YAML file created at: {yaml_path}")
    
    return yaml_content

# Usage example:
if __name__ == "__main__":
    train_npz = "train.npz"
    valid_npz = "valid.npz"
    output_directory = "yolov5_mnist"
    
    yaml_config = convert_npz_to_yolo(train_npz, valid_npz, output_directory)