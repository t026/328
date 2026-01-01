import torch

# Path to your model
model_path = "best_segmentation_model.pt"

# Load the model
checkpoint = torch.load(model_path)

# Print the keys in the checkpoint to see if loss is included
print(checkpoint.keys())

# If the loss exists
if 'val_loss' in checkpoint:
    print("Loss:", checkpoint['val_loss'])
    print("Dice:", checkpoint['val_dice'])
else:
    print("Loss information not saved in this checkpoint.")
