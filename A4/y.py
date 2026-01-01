import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def view_image(image_dir, image_index=0):
    """
    Display an image from the normalized dataset
    
    Args:
        image_dir: Directory containing the images
        image_index: Index of image to display (default: 0 for first image)
    """
    # Get list of image files
    image_files = sorted(list(Path(image_dir).glob('*.png')))
    
    if not image_files:
        print("No images found in directory! ")
        return
    
    if image_index >= len(image_files):
        print(f"Image index {image_index} out of range. Max index: {len(image_files)-1}")
        return
    
    # Read the image
    img_path = image_files[image_index]
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Plot original image
    ax1.imshow(img)
    ax1.set_title(f'Image {image_index}\nPixel range: [{img.min()}, {img.max()}]')
    ax1.axis('off')
    
    # Plot normalized version (0-1 range)
    img_norm = img.astype(np.float32) / 255.0
    ax2.imshow(img_norm)
    ax2.set_title(f'Normalized\nPixel range: [{img_norm.min():.3f}, {img_norm.max():.3f}]')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return img  # Return the image array in case needed for further analysis

# Example usage
image_dir = "datasets/images/train"  # Update this to your image directory
img = view_image(image_dir, image_index=0)  # View first image
# Change image_index to view different images

# Print additional image information
print(f"\nImage shape: {img.shape}")
print(f"Data type: {img.dtype}")

# To view multiple random images:
def view_random_images(image_dir, num_images=5):
    """Display multiple random images from the dataset"""
    image_files = list(Path(image_dir).glob('*.jpg'))
    indices = np.random.choice(len(image_files), min(num_images, len(image_files)), replace=False)
    
    for idx in indices:
        print(f"\nViewing image {idx}")
        view_image(image_dir, idx)

# Uncomment to view random images:
#view_random_images(image_dir, num_images=5)