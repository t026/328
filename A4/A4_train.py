import numpy as np
import os
import cv2
import yaml
from pathlib import Path
import shutil

class DatasetPreprocessor:
    def __init__(self, base_path='mnist_yolo_dataset'):
        self.base_path = Path(base_path)
        self.images_path = self.base_path / 'images'
        self.labels_path = self.base_path / 'labels'
        
        # Create directory structure
        for split in ['train', 'val']:
            (self.images_path / split).mkdir(parents=True, exist_ok=True)
            (self.labels_path / split).mkdir(parents=True, exist_ok=True)
    
    def process_npz(self, npz_path, split):
        """Process NPZ file and convert to YOLO format"""
        data = np.load(npz_path)
        images = data['images']
        labels = data['labels']
        bboxes = data['bboxes']
        
        num_samples = len(images)
        
        for idx in range(num_samples):
            # Save image
            img = images[idx].reshape(64, 64, 3)
            img_path = self.images_path / split / f'{idx:06d}.png'
            cv2.imwrite(str(img_path), img)
            
            # Convert and save labels
            label_path = self.labels_path / split / f'{idx:06d}.txt'
            with open(label_path, 'w') as f:
                for i in range(2):  # Two digits per image
                    digit = labels[idx][i]
                    x_min, y_min, x_max, y_max = bboxes[idx][i]
                    
                    # Convert to YOLO format: <class> <x_center> <y_center> <width> <height>
                    x_center = (x_min + x_max) / (2 * 64)
                    y_center = (y_min + y_max) / (2 * 64)
                    width = (x_max - x_min) / 64
                    height = (y_max - y_min) / 64
                    
                    f.write(f'{digit} {x_center} {y_center} {width} {height}\n')
    
    def create_data_yaml(self):
        """Create YAML configuration file for YOLOv5"""
        data = {
            'path': str(self.base_path),
            'train': str(self.images_path / 'train'),
            'val': str(self.images_path / 'val'),
            'nc': 10,  # number of classes (digits 0-9)
            'names': [str(i) for i in range(10)]  # class names
        }
        
        with open(self.base_path / 'data.yaml', 'w') as f:
            yaml.dump(data, f)

def setup_training():
    """Setup and start YOLOv5 training"""
    # Clone YOLOv5 if not exists
    if not Path('yolov5').exists():
        os.system('git clone https://github.com/ultralytics/yolov5')
        os.system('pip install -r yolov5/requirements.txt')
    
    # Create custom YOLOv5s config with modified anchor sizes
    config = {
        'nc': 10,
        'depth_multiple': 0.33,
        'width_multiple': 0.50,
        'anchors': [
            [[4,4], [8,8], [12,12]],      # Smaller anchors for digits
            [[16,16], [20,20], [24,24]], 
            [[28,28], [32,32], [36,36]]
        ],
        # ... rest of YOLOv5s config remains the same
    }
    
    with open('yolov5s_custom.yaml', 'w') as f:
        yaml.dump(config, f)

def train_yolo():
    """Main training function"""
    # Process dataset
    preprocessor = DatasetPreprocessor()
    preprocessor.process_npz('train.npz', 'train')
    preprocessor.process_npz('valid.npz', 'val')
    preprocessor.create_data_yaml()
    
    # Setup training
    setup_training()
    
    # Training command
    train_cmd = (
        f"python3 yolov5/train.py "
        f"--img 64 "  # Use original image size
        f"--batch 32 "
        f"--epochs 100 "
        f"--data {str(Path('mnist_yolo_dataset/data.yaml'))} "
        f"--cfg yolov5s_custom.yaml "
        f"--weights yolov5s.pt "
        f"--cache "
        f"--project runs/train "
        f"--name mnist_dd "
        f"--patience 20"
    )
    
    # Start training
    os.system(train_cmd)

if __name__ == '__main__':
    # Clean up previous runs if needed
    if Path('mnist_yolo_dataset').exists():
        shutil.rmtree('mnist_yolo_dataset')
    
    train_yolo()