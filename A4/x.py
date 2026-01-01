import numpy as np
import cv2
from pathlib import Path
import os

def normalize_npz_images(npz_path, output_dir):
    """
    Load images from NPZ file, normalize them properly to [0,1] range for YOLOv5.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load NPZ file
    data = np.load(npz_path)
    images = data['images']  # adjust key if needed
    
    print(f"Original pixel value range: [{images.min()}, {images.max()}]")
    
    # Process each image
    for idx, img in enumerate(images):
        # Convert to float32 and normalize to [0,1]
        img = img.astype(np.float32) / 255.0
        
        # Verify normalization
        if img.min() < 0 or img.max() > 1:
            print(f"Warning: Image {idx} has values outside [0,1] range")
            img = np.clip(img, 0, 1)
            
        # Convert back to uint8 for saving, scaling to [0,255]
        save_img = (img * 255).astype(np.uint8)
        
        # Save normalized image
        output_path = os.path.join(output_dir, f'image_{idx:05d}.jpg')
        cv2.imwrite(output_path, save_img)
    
    print(f"Processed {len(images)} images")

def verify_dataset(output_dir):
    """
    Verify the normalized dataset.
    """
    images = list(Path(output_dir).glob('*.jpg'))
    
    # Load and check first few images
    sample_size = min(100, len(images))
    samples = np.random.choice(images, sample_size)
    
    min_val, max_val = float('inf'), float('-inf')
    for img_path in samples:
        img = cv2.imread(str(img_path))
        img = img.astype(np.float32) / 255.0  # Convert to [0,1] range
        min_val = min(min_val, img.min())
        max_val = max(max_val, img.max())
    
    print("\nDataset verification:")
    print(f"Total images saved: {len(images)}")
    print(f"Image shape: {img.shape}")
    print(f"Pixel value range (normalized): [{min_val:.3f}, {max_val:.3f}]")
    
    if min_val < 0 or max_val > 1:
        print("❌ WARNING: Values outside expected [0,1] range!")
    else:
        print("✓ Pixel values are properly normalized")


# Example usage
if __name__ == "__main__":
    npz_path = "valid.npz"
    output_directory = "dataset/images/val"
    
    normalize_npz_images(npz_path, output_directory)
    verify_dataset(output_directory)