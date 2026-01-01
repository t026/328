import numpy as np
import os
from pathlib import Path
import cv2

def create_yolo_dataset():
    # Create directory structure
    base_dir = Path('datasets')
    images_dir = base_dir / 'images'
    labels_dir = base_dir / 'labels'
    
    # Create directories if they don't exist
    for dir_path in [base_dir, 
                     images_dir / 'train', images_dir / 'val',
                     labels_dir / 'train', labels_dir / 'val']:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create data.yaml
    yaml_content = """
train: images/train  # train images
val: images/val  # val images

# number of classes
nc: 10

# class names (0 to 9)
names: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    """
    
    with open(base_dir / 'data.yaml', 'w') as f:
        f.write(yaml_content.strip())
    
    # Process train and validation sets
    for split, filename in [('train', 'train.npz'), ('val', 'valid.npz')]:
        # Load the .npz file
        data = np.load(filename)
        
        # Extract data
        images = data['images']  # [N, 12288]
        bboxes = data['bboxes']  # [N, 2, 4]
        labels = data['labels']  # [N, 2]
        
        # Process each image
        for idx in range(len(images)):
            # Reshape and normalize image
            img = images[idx].reshape(64, 64, 3)
            img = (img * 255).astype(np.uint8)  # Assuming images are in [0,1]
            
            # Save image
            img_path = images_dir / split / f'{idx:06d}.png'
            cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            
            # Process bounding boxes and labels
            label_content = []
            img_height, img_width = img.shape[:2]
            
            for box_idx in range(2):  # Process both digits
                # Get bbox coordinates [x_min, y_min, x_max, y_max]
                box = bboxes[idx, box_idx]
                
                # Convert to YOLO format [class_id, x_center, y_center, width, height]
                x_center = (box[0] + box[2]) / 2 / img_width
                y_center = (box[1] + box[3]) / 2 / img_height
                width = (box[2] - box[0]) / img_width
                height = (box[3] - box[1]) / img_height
                
                # Get class label
                class_id = labels[idx, box_idx]
                
                # Add to label content
                label_content.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
            # Save label file
            label_path = labels_dir / split / f'{idx:06d}.txt'
            with open(label_path, 'w') as f:
                f.write('\n'.join(label_content))
        
        print(f"Processed {split} set: {len(images)} images")

if __name__ == "__main__":
    create_yolo_dataset()