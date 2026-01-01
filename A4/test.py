import cv2
import matplotlib.pyplot as plt
import os

def verify_single_pair():
    # Update these paths to match your actual paths
    image_dir = 'dataset/images/train'
    label_dir = 'dataset/labels/train'
    
    # Get first image file
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    if not image_files:
        print("No PNG files found!")
        return
        
    first_image = image_files[0]
    image_path = os.path.join(image_dir, first_image)
    label_path = os.path.join(label_dir, first_image.replace('.png', '.txt'))
    
    # Load and check image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return
    
    print(f"Image shape: {img.shape}")
    
    # Load and check label
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            labels = f.read().strip()
        print(f"\nLabels for {first_image}:")
        print(labels)
    else:
        print(f"No label file found for {first_image}")

verify_single_pair()